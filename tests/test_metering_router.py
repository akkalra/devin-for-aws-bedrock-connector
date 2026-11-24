from bedrock_connector.bedrock import ConverseResult
from bedrock_connector.config import Settings
from bedrock_connector.metering import Ledger, price_for
from bedrock_connector.router import ModelRouter


def test_price_lookup_handles_inference_profiles():
    s = Settings()
    assert price_for("us.anthropic.claude-sonnet-4-5-20250929-v1:0", s.pricing) == (3.0, 15.0)
    assert price_for("global.anthropic.claude-haiku-4-5-20251001-v1:0", s.pricing) == (1.0, 5.0)
    assert price_for("anthropic.claude-opus-4-5-20251101-v1:0", s.pricing) == (5.0, 25.0)


def test_value_pools_do_not_double_count():
    s = Settings(aws_cogs_per_acu_usd=1.0, acu_price_usd=2.0, aws_revenue_share=0.25, other_aws_pct_of_infra=0.0)
    ledger = Ledger(s)
    res = ConverseResult({"role": "assistant", "content": []}, "end_turn", 1_000_000, 0, 1,
                         "us.anthropic.claude-sonnet-4-5-20250929-v1:0", "r")
    ledger.record_call(session_id="x", role="coder", source="test", result=res)  # $3 of Bedrock
    vp = ledger.summary()["value_pools"]
    assert ledger.summary()["acus"] == 3.0
    assert vp["customer_revenue_usd"] == 6.0
    assert vp["aws_revenue_share_usd"] == 1.5
    assert vp["cognition_aws_infra_usd"] == 3.0
    assert vp["aws_total_capture_usd"] == 4.5
    assert vp["cognition_contribution_usd"] == 1.5


def test_router_keeps_unknown_models_on_bedrock():
    r = ModelRouter(Settings())
    assert r.resolve("gpt-4o")[1].startswith("us.anthropic.")
    assert r.resolve("gpt-4o-mini")[0] == "fast"
    assert r.resolve("devin-planner")[0] == "planner"
    assert r.resolve("amazon.nova-pro-v1:0") == ("passthrough", "amazon.nova-pro-v1:0")
    assert r.resolve("us.meta.llama3-3-70b-instruct-v1:0")[0] == "passthrough"
