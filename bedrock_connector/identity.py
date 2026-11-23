"""Credential vaults for customer Bedrock API keys.

`AgentCoreIdentityVault` keeps each org's key in Amazon Bedrock AgentCore
Identity as an API-key credential provider (backed by its token vault /
Secrets Manager). The agent never sees the key at rest: at run time it trades a
workload access token, issued by AgentCore for the signed-in user, for the key.

`LocalVault` has the same interface and keeps keys in process memory, for demos
and tests without AWS.
"""
from __future__ import annotations

import threading


class MissingCredential(Exception):
    pass


class LocalVault:
    name = "local-vault"

    def __init__(self):
        self._lock = threading.Lock()
        self._keys: dict[str, str] = {}

    def put_api_key(self, provider_name: str, api_key: str) -> None:
        with self._lock:
            self._keys[provider_name] = api_key

    def get_api_key(self, provider_name: str, user_id: str, workload_token: str | None = None) -> str:
        with self._lock:
            key = self._keys.get(provider_name)
        if not key:
            raise MissingCredential(f"no Bedrock API key on file ({provider_name}); add one in org settings")
        return key


class AgentCoreIdentityVault:
    name = "agentcore-identity"

    def __init__(self, region: str, workload_name: str, control=None, data=None):
        import boto3

        self.workload_name = workload_name
        self.control = control or boto3.client("bedrock-agentcore-control", region_name=region)
        self.data = data or boto3.client("bedrock-agentcore", region_name=region)
        self._workload_ready = False

    @staticmethod
    def _code(e: Exception) -> str:
        return getattr(e, "response", {}).get("Error", {}).get("Code", "")

    def _ensure_workload(self) -> None:
        if self._workload_ready:
            return
        try:
            self.control.create_workload_identity(name=self.workload_name)
        except Exception as e:
            if self._code(e) != "ConflictException":
                raise
        self._workload_ready = True

    def put_api_key(self, provider_name: str, api_key: str) -> None:
        try:
            self.control.create_api_key_credential_provider(name=provider_name, apiKey=api_key)
        except Exception as e:
            if self._code(e) != "ConflictException":
                raise
            self.control.update_api_key_credential_provider(name=provider_name, apiKey=api_key)

    def get_api_key(self, provider_name: str, user_id: str, workload_token: str | None = None) -> str:
        """Inside AgentCore Runtime the workload token arrives with the request;
        outside it, the Devin backend requests one for the signed-in user."""
        if not workload_token:
            self._ensure_workload()
            workload_token = self.data.get_workload_access_token_for_user_id(
                workloadName=self.workload_name, userId=user_id)["workloadAccessToken"]
        try:
            return self.data.get_resource_api_key(workloadIdentityToken=workload_token,
                                                  resourceCredentialProviderName=provider_name)["apiKey"]
        except Exception as e:
            if self._code(e) == "ResourceNotFoundException":
                raise MissingCredential(f"no Bedrock API key on file ({provider_name}); add one in org settings") from e
            raise


def make_vault(settings):
    if settings.credential_vault == "agentcore":
        return AgentCoreIdentityVault(settings.region, settings.agentcore_workload_name)
    return LocalVault()
