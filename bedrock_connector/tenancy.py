"""Orgs, users and login sessions: the Devin side of multi-tenant Bedrock.

Login is simulated here (a stand-in for IAM Identity Center / SAML / OIDC SSO).
Each org picks whose AWS account serves its inference:

  customer_api_key   the org's own Amazon Bedrock API key, stored in AgentCore
                     Identity (the default for a new org)
  customer_iam_role  a role in the customer's account that Devin assumes with an ExternalId
  devin_managed      Cognition's Bedrock account, paid for through ACUs

Devin owns model routing either way; the credential only decides which account is billed.
"""
from __future__ import annotations

import hashlib
import re
import secrets
import threading
from dataclasses import asdict, dataclass, field

CREDENTIAL_SOURCES = ("customer_api_key", "customer_iam_role", "devin_managed")
ROLE_ARN_RE = re.compile(r"^arn:aws[a-z-]*:iam::\d{12}:role/[\w+=,.@/-]{1,512}$")


@dataclass
class Org:
    id: str
    name: str
    domain: str
    region: str = "us-east-1"
    credential_source: str = "customer_api_key"
    key_last4: str = ""  # display hint only; the key lives in the credential vault
    role_arn: str = ""
    external_id: str = ""  # stable per org, so customer role trust policies survive restarts
    devin_managed_routing: bool = True
    routing_overrides: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self):
        if not self.external_id:
            self.external_id = "devin-" + hashlib.sha256(f"devin-external-id:{self.id}".encode()).hexdigest()[:16]

    @property
    def provider_name(self) -> str:
        """AgentCore Identity API-key credential provider holding this org's Bedrock key."""
        return f"devin-{self.id}-bedrock"

    @property
    def billed_to(self) -> str:
        return "cognition" if self.credential_source == "devin_managed" else "customer"

    def public(self) -> dict:
        return asdict(self) | {"provider_name": self.provider_name, "billed_to": self.billed_to,
                               "has_api_key": bool(self.key_last4)}


@dataclass
class User:
    email: str
    org_id: str
    is_admin: bool = True


class TenantStore:
    def __init__(self, region: str = "us-east-1"):
        self._lock = threading.Lock()
        self.orgs = {
            "acme": Org("acme", "Acme Financial", "acme.com", region),
            "globex": Org("globex", "Globex Retail", "globex.com", region, credential_source="customer_iam_role"),
            "demo": Org("demo", "Demo (no login)", "", region, credential_source="devin_managed"),
        }
        self.users: dict[str, User] = {}
        self._tokens: dict[str, str] = {}

    @property
    def default_org(self) -> Org:
        return self.orgs["demo"]

    def login(self, email: str) -> tuple[str, User]:
        email = email.strip().lower()
        domain = email.rpartition("@")[2]
        org = next((o for o in self.orgs.values() if o.domain and o.domain == domain), None)
        if "@" not in email or org is None:
            raise KeyError(f"no Devin org for {email!r}")
        with self._lock:
            user = self.users.setdefault(email, User(email, org.id))
            token = "devin_sess_" + secrets.token_urlsafe(24)
            self._tokens[token] = email
        return token, user

    def logout(self, token: str) -> None:
        with self._lock:
            self._tokens.pop(token, None)

    def user_for(self, token: str | None) -> User | None:
        if not token or not token.startswith("devin_sess_"):
            return None
        email = self._tokens.get(token)
        return self.users.get(email) if email else None

    def update_org(self, org: Org, body: dict) -> Org:
        src = body.get("credential_source", org.credential_source)
        if src not in CREDENTIAL_SOURCES:
            raise ValueError(f"credential_source must be one of {CREDENTIAL_SOURCES}")
        role_arn = body.get("role_arn", org.role_arn).strip()
        if role_arn and not ROLE_ARN_RE.match(role_arn):
            raise ValueError("role_arn must look like arn:aws:iam::123456789012:role/Name")
        overrides = body.get("routing_overrides", org.routing_overrides)
        if not isinstance(overrides, dict) or not all(
                isinstance(v, list) and all(isinstance(m, str) for m in v) for v in overrides.values()):
            raise ValueError("routing_overrides must map role -> list of Bedrock model ids")
        region = body.get("region", org.region).strip()
        if not re.fullmatch(r"[a-z]{2}(-gov)?-[a-z]+-\d", region):
            raise ValueError("region must be an AWS region such as us-east-1")
        with self._lock:
            org.credential_source, org.role_arn, org.region = src, role_arn, region
            org.routing_overrides = {r: list(dict.fromkeys(ms)) for r, ms in overrides.items()}
            org.devin_managed_routing = bool(body.get("devin_managed_routing", org.devin_managed_routing))
        return org
