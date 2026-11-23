"""The Devin inference backend: tenant credentials + Devin-managed routing -> Amazon Bedrock.

Every model call from the agent loop or the OpenAI/Anthropic-compatible API
goes through `InferenceService.converse`:

  1. Devin's router picks an ordered list of Bedrock models for the role.
  2. `ClientFactory` builds a Bedrock client for the caller's org: the org's own
     Bedrock API key (from AgentCore Identity), an assumed role in the org's
     account, or Cognition's account for Devin-managed orgs.
  3. The first model that answers wins; throttling / capacity / model-access
     errors fall through to the next model.
  4. The call is metered against the org, with who pays for the tokens.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import threading
import time
from dataclasses import dataclass

from .bedrock import BedrockClient, ConverseResult, make_client
from .config import Settings
from .identity import MissingCredential, make_vault
from .metering import CallRecord, Ledger
from .router import ModelRouter
from .tenancy import Org, TenantStore

FALLBACK_CODES = {"ThrottlingException", "ServiceUnavailableException", "ModelNotReadyException",
                  "ModelTimeoutException", "InternalServerException", "ServiceQuotaExceededException"}


@dataclass
class CallContext:
    org: Org
    user_id: str
    session_id: str
    source: str
    workload_token: str | None = None  # set inside AgentCore Runtime


class ClientFactory:
    def __init__(self, settings: Settings, vault, base_client=None):
        self.settings, self.vault = settings, vault
        self.base = base_client or make_client(settings)
        self.mock = settings.resolved_mode == "mock" or base_client is not None
        self._lock = threading.Lock()
        self._cache: dict[tuple, tuple[object, float]] = {}

    def auth_label(self, org: Org) -> str:
        return {"customer_api_key": f"customer key · {self.vault.name}",
                "customer_iam_role": "customer role · sts:AssumeRole",
                "devin_managed": "devin-managed"}[org.credential_source]

    def client_for(self, ctx: CallContext):
        org = ctx.org
        if org.credential_source == "customer_iam_role" and not org.role_arn:
            raise MissingCredential("no IAM role ARN on file; add one in org settings")
        if org.credential_source == "customer_api_key":
            key = self.vault.get_api_key(org.provider_name, ctx.user_id, ctx.workload_token)
        if self.mock:
            return self.base
        if org.credential_source == "devin_managed":
            if org.region == self.settings.region:
                return self.base
            return self._cached(("managed", org.region), lambda: (BedrockClient(self.settings, org.region), 1e18))
        if org.credential_source == "customer_api_key":
            cache_key = ("key", org.id, org.region, hashlib.sha256(key.encode()).hexdigest())
            return self._cached(cache_key, lambda: (BedrockClient(self.settings, org.region, api_key=key), 1e18))
        return self._cached(("role", org.id, org.region, org.role_arn), lambda: self._assume(org))

    def _cached(self, key, build):
        with self._lock:
            hit = self._cache.get(key)
            if hit and hit[1] > time.time() + 300:
                return hit[0]
        client, expires = build()
        with self._lock:
            self._cache[key] = (client, expires)
        return client

    def _assume(self, org: Org):
        import boto3

        creds = boto3.client("sts", region_name=org.region).assume_role(
            RoleArn=org.role_arn, RoleSessionName=f"devin-{org.id}"[:64], ExternalId=org.external_id,
            DurationSeconds=3600)["Credentials"]
        sess = boto3.Session(aws_access_key_id=creds["AccessKeyId"], aws_secret_access_key=creds["SecretAccessKey"],
                             aws_session_token=creds["SessionToken"], region_name=org.region)
        return BedrockClient(self.settings, org.region, boto_session=sess), creds["Expiration"].timestamp()


class InferenceService:
    def __init__(self, settings: Settings, router: ModelRouter, ledger: Ledger, factory: ClientFactory):
        self.settings, self.router, self.ledger, self.factory = settings, router, ledger, factory

    def converse(self, ctx: CallContext, role: str, messages: list[dict], system=None, tools=None,
                 max_tokens: int = 4096, temperature: float | None = None,
                 requested: str | None = None) -> tuple[ConverseResult, CallRecord]:
        ctx = dataclasses.replace(ctx, org=copy.deepcopy(ctx.org))  # admins may edit settings mid-call
        if requested is not None:
            role, model = self.router.resolve(requested)
            candidates = [model] if role == "passthrough" else self.router.candidates(role, ctx.org)
        else:
            candidates = self.router.candidates(role, ctx.org)
        client = self.factory.client_for(ctx)
        tried: list[str] = []
        for model in candidates:
            try:
                res = client.converse(model, messages, system=system, tools=tools, max_tokens=max_tokens,
                                      temperature=temperature)
            except Exception as e:
                err = getattr(e, "response", {}).get("Error", {})
                no_access = err.get("Code") == "AccessDeniedException" and "model" in err.get("Message", "").lower()
                if (err.get("Code") in FALLBACK_CODES or no_access) and model != candidates[-1]:
                    tried.append(model)
                    continue
                raise
            rec = self.ledger.record_call(session_id=ctx.session_id, role=role, source=ctx.source, result=res,
                                          org_id=ctx.org.id, auth_method=self.factory.auth_label(ctx.org),
                                          billed_to=ctx.org.billed_to, fallback_from=tried)
            return res, rec
        raise RuntimeError("no Bedrock model candidates")


@dataclass
class Stack:
    settings: Settings
    tenants: TenantStore
    vault: object
    router: ModelRouter
    ledger: Ledger
    inference: InferenceService


def build_stack(settings: Settings, base_client=None, vault=None) -> Stack:
    router, ledger = ModelRouter(settings), Ledger(settings)
    vault = vault or make_vault(settings)
    factory = ClientFactory(settings, vault, base_client)
    return Stack(settings, TenantStore(settings.region), vault, router, ledger,
                 InferenceService(settings, router, ledger, factory))
