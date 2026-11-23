"""Runtime configuration, read from environment variables."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

# Bedrock on-demand list prices, USD per 1M tokens (input, output).
# Illustrative defaults; override with BEDROCK_PRICING_JSON and verify against
# https://aws.amazon.com/bedrock/pricing/ for your region.
DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    "anthropic.claude-opus-4": (15.0, 75.0),
    "anthropic.claude-opus-4-5": (5.0, 25.0),
    "anthropic.claude-sonnet-4": (3.0, 15.0),
    "anthropic.claude-haiku-4-5": (1.0, 5.0),
    "anthropic.claude-3-5-haiku": (0.8, 4.0),
    "amazon.nova-pro": (0.8, 3.2),
    "amazon.nova-lite": (0.06, 0.24),
    "amazon.nova-micro": (0.035, 0.14),
    "meta.llama": (0.72, 0.72),
    "mistral": (2.0, 6.0),
    "deepseek": (1.35, 5.4),
    "openai.gpt-oss": (0.15, 0.6),
    "mock": (3.0, 15.0),
}

DEFAULT_ROLE_MODELS = {
    "planner": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "coder": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "reviewer": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "fast": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
}


def _f(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


@dataclass
class Settings:
    region: str = field(default_factory=lambda: os.environ.get("AWS_REGION")
                        or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1")
    # "auto" -> live Bedrock if credentials are present, otherwise mock.
    mode: str = field(default_factory=lambda: os.environ.get("BEDROCK_MODE", "auto"))
    role_models: dict[str, str] = field(default_factory=dict)
    pricing: dict[str, tuple[float, float]] = field(default_factory=dict)
    guardrail_id: str | None = field(default_factory=lambda: os.environ.get("BEDROCK_GUARDRAIL_ID"))
    guardrail_version: str = field(default_factory=lambda: os.environ.get("BEDROCK_GUARDRAIL_VERSION", "DRAFT"))
    gateway_api_key: str | None = field(default_factory=lambda: os.environ.get("GATEWAY_API_KEY"))
    ledger_path: str | None = field(default_factory=lambda: os.environ.get("LEDGER_PATH"))

    # ACU economics (illustrative; see docs/economics.md).
    acu_price_usd: float = field(default_factory=lambda: _f("ACU_PRICE_USD", 2.25))
    aws_cogs_per_acu_usd: float = field(default_factory=lambda: _f("AWS_COGS_PER_ACU_USD", 0.90))
    aws_revenue_share: float = field(default_factory=lambda: _f("AWS_REVENUE_SHARE", 0.20))
    runtime_usd_per_hour: float = field(default_factory=lambda: _f("RUNTIME_USD_PER_HOUR", 0.40))
    other_aws_pct_of_infra: float = field(default_factory=lambda: _f("OTHER_AWS_PCT_OF_INFRA", 0.10))

    def __post_init__(self):
        self.role_models = {
            role: os.environ.get(f"BEDROCK_MODEL_{role.upper()}", model)
            for role, model in DEFAULT_ROLE_MODELS.items()
        } | self.role_models
        pricing = dict(DEFAULT_PRICING)
        if os.environ.get("BEDROCK_PRICING_JSON"):
            pricing.update({k: tuple(v) for k, v in json.loads(os.environ["BEDROCK_PRICING_JSON"]).items()})
        self.pricing = pricing | self.pricing

    @property
    def auth_method(self) -> str:
        if self.resolved_mode == "mock":
            return "mock"
        if os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
            return "bedrock-api-key"
        return "iam"

    @property
    def resolved_mode(self) -> str:
        if self.mode in ("live", "mock"):
            return self.mode
        if os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
            return "live"
        try:
            import boto3

            return "live" if boto3.Session().get_credentials() else "mock"
        except Exception:
            return "mock"
