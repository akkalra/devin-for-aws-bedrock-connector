"""Per-call metering: Bedrock tokens -> USD -> ACUs, plus the value-pool split.

The ACU model here is illustrative, made for a partnership demo. It is not
Cognition's real ACU formula. An ACU is treated as a normalized unit of AWS
cost of goods (Bedrock inference + agent runtime + other AWS services):

    acus = (bedrock_usd + runtime_usd) * (1 + other_aws_pct) / aws_cogs_per_acu

The four money flows are kept apart so nothing is counted twice:
    customer_revenue      = acus * acu_price                 (customer -> AWS bill)
    aws_revenue_share     = customer_revenue * share          (AWS keeps)
    cognition_aws_infra   = bedrock + runtime + other AWS     (Cognition -> AWS)
    aws_total_capture     = aws_revenue_share + cognition_aws_infra
    cognition_margin      = customer_revenue - aws_revenue_share - cognition_aws_infra
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
    ts: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])


class Ledger:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.Lock()
        self.calls: list[CallRecord] = []
        self.runtime_seconds: dict[str, float] = {}

    def acus_for_usd(self, aws_usd: float) -> float:
        s = self.settings
        return aws_usd * (1 + s.other_aws_pct_of_infra) / s.aws_cogs_per_acu_usd

    def record_call(self, *, session_id: str, role: str, source: str, result) -> CallRecord:
        p_in, p_out = price_for(result.model_id, self.settings.pricing)
        usd = result.input_tokens / 1e6 * p_in + result.output_tokens / 1e6 * p_out
        rec = CallRecord(session_id=session_id, role=role, source=source, model_id=result.model_id,
                         input_tokens=result.input_tokens, output_tokens=result.output_tokens,
                         latency_ms=result.latency_ms, bedrock_usd=usd, acus=self.acus_for_usd(usd),
                         request_id=result.request_id, auth_method=self.settings.auth_method)
        with self._lock:
            self.calls.append(rec)
        if self.settings.ledger_path:
            with open(self.settings.ledger_path, "a") as f:
                f.write(json.dumps(asdict(rec)) + "\n")
        return rec

    def record_runtime(self, session_id: str, seconds: float) -> None:
        with self._lock:
            self.runtime_seconds[session_id] = self.runtime_seconds.get(session_id, 0.0) + seconds

    def summary(self, session_id: str | None = None) -> dict:
        s = self.settings
        with self._lock:
            calls = [c for c in self.calls if session_id is None or c.session_id == session_id]
            runtime_s = (sum(self.runtime_seconds.values()) if session_id is None
                         else self.runtime_seconds.get(session_id, 0.0))
        bedrock = sum(c.bedrock_usd for c in calls)
        runtime = runtime_s / 3600 * s.runtime_usd_per_hour
        other = (bedrock + runtime) * s.other_aws_pct_of_infra
        infra = bedrock + runtime + other
        acus = infra / s.aws_cogs_per_acu_usd
        revenue = acus * s.acu_price_usd
        rev_share = revenue * s.aws_revenue_share
        by_model: dict[str, dict] = {}
        for c in calls:
            m = by_model.setdefault(c.model_id, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "bedrock_usd": 0.0})
            m["calls"] += 1
            m["input_tokens"] += c.input_tokens
            m["output_tokens"] += c.output_tokens
            m["bedrock_usd"] += c.bedrock_usd
        return {
            "calls": len(calls),
            "input_tokens": sum(c.input_tokens for c in calls),
            "output_tokens": sum(c.output_tokens for c in calls),
            "runtime_seconds": runtime_s,
            "acus": acus,
            "by_model": by_model,
            "value_pools": {
                "customer_revenue_usd": revenue,
                "aws_revenue_share_usd": rev_share,
                "cognition_aws_infra_usd": infra,
                "of_which_bedrock_usd": bedrock,
                "of_which_runtime_usd": runtime,
                "of_which_other_aws_usd": other,
                "aws_total_capture_usd": rev_share + infra,
                "cognition_contribution_usd": revenue - rev_share - infra,
            },
            "assumptions": {
                "acu_price_usd": s.acu_price_usd,
                "aws_cogs_per_acu_usd": s.aws_cogs_per_acu_usd,
                "aws_revenue_share": s.aws_revenue_share,
                "runtime_usd_per_hour": s.runtime_usd_per_hour,
                "other_aws_pct_of_infra": s.other_aws_pct_of_infra,
            },
        }

    def recent(self, n: int = 50) -> list[dict]:
        with self._lock:
            return [asdict(c) for c in self.calls[-n:]][::-1]
