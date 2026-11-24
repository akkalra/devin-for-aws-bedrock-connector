"""Per-call metering: Bedrock tokens -> USD -> ACUs, plus the value-pool split.

The ACU model here is illustrative, made for a partnership demo. It is not
Cognition's real ACU formula. An ACU is treated as a normalized unit of AWS
cost of goods (Bedrock inference + agent runtime + other AWS services):

    acus = (bedrock_usd + runtime_usd) * (1 + other_aws_pct) / aws_cogs_per_acu

Bedrock calls made with the customer's own credentials (their Bedrock API key
or IAM role) are billed by AWS straight to the customer's account, so they are
not part of Cognition's cost or of ACUs. The money flows are kept apart so
nothing is counted twice:
    customer_revenue      = acus * acu_price                 (customer -> AWS bill)
    aws_revenue_share     = customer_revenue * share          (AWS keeps)
    cognition_aws_infra   = Devin-managed bedrock + runtime + other AWS  (Cognition -> AWS)
    customer_bedrock      = Bedrock on the customer's own account        (customer -> AWS)
    aws_total_capture     = aws_revenue_share + cognition_aws_infra + customer_bedrock
    cognition_margin      = customer_revenue - aws_revenue_share - cognition_aws_infra
    ppa_drawdown          = customer_revenue + customer_bedrock   (both land on the customer's AWS bill)
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field

from .config import Settings


def price_for(model_id: str, pricing: dict[str, tuple[float, float]]) -> tuple[float, float]:
    """Find the longest pricing key contained in the model id
    (handles `us.`/`global.` inference-profile prefixes and version suffixes)."""
    best = None
    for key in pricing:
        if key in model_id and (best is None or len(key) > len(best)):
            best = key
    return pricing[best] if best else pricing["mock"]


@dataclass
class CallRecord:
    session_id: str
    role: str
    source: str
    model_id: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    bedrock_usd: float
    acus: float
    request_id: str
    auth_method: str
    org_id: str = "demo"
    billed_to: str = "cognition"
    fallback_from: list = field(default_factory=list)
    ts: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])


class Ledger:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.Lock()
        self.calls: list[CallRecord] = []
        self.runtime_seconds: dict[str, float] = {}
        self.session_org: dict[str, str] = {}

    def acus_for_usd(self, aws_usd: float) -> float:
        s = self.settings
        return aws_usd * (1 + s.other_aws_pct_of_infra) / s.aws_cogs_per_acu_usd

    def record_call(self, *, session_id: str, role: str, source: str, result, org_id: str = "demo",
                    auth_method: str | None = None, billed_to: str = "cognition",
                    fallback_from: list | None = None) -> CallRecord:
        p_in, p_out = price_for(result.model_id, self.settings.pricing)
        usd = result.input_tokens / 1e6 * p_in + result.output_tokens / 1e6 * p_out
        rec = CallRecord(session_id=session_id, role=role, source=source, model_id=result.model_id,
                         input_tokens=result.input_tokens, output_tokens=result.output_tokens,
                         latency_ms=result.latency_ms, bedrock_usd=usd,
                         acus=self.acus_for_usd(usd) if billed_to == "cognition" else 0.0,
                         request_id=result.request_id, auth_method=auth_method or self.settings.auth_method,
                         org_id=org_id, billed_to=billed_to, fallback_from=list(fallback_from or []))
        self._append(rec)
        return rec

    def ingest(self, calls: list[dict]) -> None:
        """Add call records produced elsewhere (e.g. by an AgentCore Runtime session)."""
        for c in calls:
            self._append(CallRecord(**c))

    def _append(self, rec: CallRecord) -> None:
        with self._lock:
            self.calls.append(rec)
        if self.settings.ledger_path:
            with open(self.settings.ledger_path, "a") as f:
                f.write(json.dumps(asdict(rec)) + "\n")

    def record_runtime(self, session_id: str, seconds: float, org_id: str = "demo") -> None:
        with self._lock:
            self.session_org[session_id] = org_id
            self.runtime_seconds[session_id] = self.runtime_seconds.get(session_id, 0.0) + seconds

    def summary(self, session_id: str | None = None, org_id: str | None = None) -> dict:
        s = self.settings
        with self._lock:
            calls = [c for c in self.calls if (session_id is None or c.session_id == session_id)
                     and (org_id is None or c.org_id == org_id)]
            runtime_s = sum(v for k, v in self.runtime_seconds.items()
                            if (session_id is None or k == session_id)
                            and (org_id is None or self.session_org.get(k) == org_id))
        bedrock = sum(c.bedrock_usd for c in calls if c.billed_to == "cognition")
        customer_bedrock = sum(c.bedrock_usd for c in calls if c.billed_to != "cognition")
        runtime = runtime_s / 3600 * s.runtime_usd_per_hour
        other = (bedrock + runtime) * s.other_aws_pct_of_infra
        infra = bedrock + runtime + other
        acus = infra / s.aws_cogs_per_acu_usd
        revenue = acus * s.acu_price_usd
        rev_share = revenue * s.aws_revenue_share
        tokens = sum(c.input_tokens + c.output_tokens for c in calls)
        # Tokens per ACU as if every Bedrock call were Devin-managed, so BYO-credential orgs still map tokens to ACUs.
        acu_basis = (bedrock + customer_bedrock + runtime) * (1 + s.other_aws_pct_of_infra) / s.aws_cogs_per_acu_usd
        by_model: dict[str, dict] = {}
        for c in calls:
            m = by_model.setdefault(c.model_id, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "bedrock_usd": 0.0})
            m["calls"] += 1
            m["input_tokens"] += c.input_tokens
            m["output_tokens"] += c.output_tokens
            m["bedrock_usd"] += c.bedrock_usd
        return {
            "calls": len(calls),
            "fallbacks": sum(1 for c in calls if c.fallback_from),
            "input_tokens": sum(c.input_tokens for c in calls),
            "output_tokens": sum(c.output_tokens for c in calls),
            "runtime_seconds": runtime_s,
            "acus": acus,
            "tokens_per_acu": tokens / acu_basis if acu_basis else 0.0,
            "by_model": by_model,
            "commerce": {
                "aws_bill_usd": revenue + customer_bedrock,
                "ppa_commit_usd": s.ppa_commit_usd,
                "ppa_drawdown_usd": revenue + customer_bedrock,
                "seller_quota_retired_usd": revenue,
            },
            "value_pools": {
                "customer_revenue_usd": revenue,
                "aws_revenue_share_usd": rev_share,
                "cognition_aws_infra_usd": infra,
                "of_which_bedrock_usd": bedrock,
                "of_which_runtime_usd": runtime,
                "of_which_other_aws_usd": other,
                "customer_bedrock_direct_usd": customer_bedrock,
                "bedrock_total_usd": bedrock + customer_bedrock,
                "aws_total_capture_usd": rev_share + infra + customer_bedrock,
                "cognition_contribution_usd": revenue - rev_share - infra,
            },
            "assumptions": {
                "acu_price_usd": s.acu_price_usd,
                "aws_cogs_per_acu_usd": s.aws_cogs_per_acu_usd,
                "aws_revenue_share": s.aws_revenue_share,
                "runtime_usd_per_hour": s.runtime_usd_per_hour,
                "other_aws_pct_of_infra": s.other_aws_pct_of_infra,
                "ppa_commit_usd": s.ppa_commit_usd,
            },
        }

    def recent(self, n: int = 50, org_id: str | None = None, session_id: str | None = None) -> list[dict]:
        with self._lock:
            calls = [c for c in self.calls if (org_id is None or c.org_id == org_id)
                     and (session_id is None or c.session_id == session_id)]
        return [asdict(c) for c in calls[-n:]][::-1]
