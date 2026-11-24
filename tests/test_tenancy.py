import io
import json
import time

import boto3
import pytest
from botocore.stub import Stubber
from fastapi.testclient import TestClient

from bedrock_connector.bedrock import BedrockClient
from bedrock_connector.config import Settings
from bedrock_connector.gateway import create_app
from bedrock_connector.identity import AgentCoreIdentityVault
from bedrock_connector.runtime import AgentCoreDispatcher

KEY = "ABSKQmVkcm9ja0FQSUtleS1kZW1vLWtleS0xMjM0"


def login(c, email="dev@acme.com"):
    r = c.post("/v1/auth/login", json={"email": email})
    assert r.status_code == 200, r.text
    return {"x-devin-session": r.json()["session_token"]}


def wait(c, sid, h):
    for _ in range(200):
        s = c.get(f"/v1/agent/sessions/{sid}", headers=h).json()
        if s["status"] in ("finished", "unverified", "error"):
            return s
        time.sleep(0.05)
    raise AssertionError(s)


def test_login_defaults_to_customer_bedrock_key_and_never_returns_it():
    c = TestClient(create_app(Settings()))
    h = login(c)
    me = c.get("/v1/me", headers=h).json()
    assert me["org"]["credential_source"] == "customer_api_key" and not me["org"]["has_api_key"]
    assert c.post("/v1/org/test-connection", headers=h).json()["ok"] is False  # no key yet
    r = c.put("/v1/org/bedrock-key", headers=h, json={"api_key": KEY})
    assert r.status_code == 200 and KEY not in r.text and r.json()["org"]["key_last4"] == KEY[-4:]
    t = c.post("/v1/org/test-connection", headers=h).json()
    assert t["ok"] and t["billed_to"] == "customer" and "customer key" in t["credential"]
    assert KEY not in c.get("/v1/stats", headers=h).text


def test_byok_session_is_billed_to_customer_and_scoped_to_org():
    c = TestClient(create_app(Settings()))
    h = login(c)
    c.put("/v1/org/bedrock-key", headers=h, json={"api_key": KEY})
    s = wait(c, c.post("/v1/agent/sessions", headers=h, json={}).json()["id"], h)
    assert s["status"] == "unverified" and s["org_id"] == "acme" and s["user_id"] == "dev@acme.com"
    assert {st["billed_to"] for st in s["steps"] if st["type"] == "model"} == {"customer"}
    vp = s["usage"]["value_pools"]
    assert vp["customer_bedrock_direct_usd"] > 0 and vp["of_which_bedrock_usd"] == 0
    assert abs(vp["aws_total_capture_usd"] - (vp["aws_revenue_share_usd"] + vp["cognition_aws_infra_usd"]
                                               + vp["customer_bedrock_direct_usd"])) < 1e-9
    c.post("/v1/chat/completions", json={"model": "x", "messages": [{"role": "user", "content": "hi"}]})  # demo org
    g = login(c, "dev@globex.com")
    assert c.get("/v1/stats", headers=g).json()["summary"]["calls"] == 0
    assert c.get(f"/v1/agent/sessions/{s['id']}", headers=g).status_code == 404
    assert {r["org_id"] for r in c.get("/v1/stats", headers=h).json()["recent"]} == {"acme"}


def test_missing_key_fails_session_with_clear_error():
    c = TestClient(create_app(Settings()))
    h = login(c)
    s = wait(c, c.post("/v1/agent/sessions", headers=h, json={}).json()["id"], h)
    assert s["status"] == "error" and "no Bedrock API key" in s["steps"][-1]["text"]
    r = c.post("/v1/messages", headers=h, json={"model": "x", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 412


def test_devin_managed_routing_falls_back_and_org_can_pin(monkeypatch):
    settings = Settings()
    monkeypatch.setattr(settings, "mock_unavailable_models", {settings.role_models["planner"]})
    c = TestClient(create_app(settings))
    h = login(c)
    c.put("/v1/org/bedrock-key", headers=h, json={"api_key": KEY})
    s = wait(c, c.post("/v1/agent/sessions", headers=h, json={}).json()["id"], h)
    planner = next(st for st in s["steps"] if st["type"] == "model" and st["role"] == "planner")
    assert planner["fallback_from"] == [settings.role_models["planner"]]
    assert planner["model_id"] == settings.fallbacks["planner"][0]
    r = c.put("/v1/org/settings", headers=h, json={"devin_managed_routing": False,
                                                    "routing_overrides": {"coder": ["us.amazon.nova-pro-v1:0"]}})
    assert r.json()["routing"]["coder"] == ["us.amazon.nova-pro-v1:0"]
    assert c.put("/v1/org/settings", headers=h, json={"role_arn": "nope"}).status_code == 400
    assert c.put("/v1/org/settings", json={"region": "us-west-2"}).status_code == 403  # not signed in


def test_gateway_key_still_guards_login(monkeypatch):
    monkeypatch.setenv("GATEWAY_API_KEY", "s3cret")
    c = TestClient(create_app(Settings()))
    assert c.post("/v1/auth/login", json={"email": "dev@acme.com"}).status_code == 401
    tok = c.post("/v1/auth/login", headers={"x-api-key": "s3cret"}, json={"email": "dev@acme.com"}).json()
    assert c.get("/v1/me", headers={"x-api-key": "s3cret", "x-devin-session": tok["session_token"]}
                 ).json()["org"]["id"] == "acme"


def test_agentcore_identity_vault_calls():
    control = boto3.client("bedrock-agentcore-control", region_name="us-east-1")
    data = boto3.client("bedrock-agentcore", region_name="us-east-1")
    with Stubber(control) as sc, Stubber(data) as sd:
        sc.add_client_error("create_api_key_credential_provider", "ConflictException",
                            expected_params={"name": "devin-acme-bedrock", "apiKey": KEY})
        sc.add_response("update_api_key_credential_provider",
                        {"apiKeySecretArn": {"secretArn": "arn:aws:secretsmanager:us-east-1:111122223333:secret:x"},
                         "name": "devin-acme-bedrock",
                         "credentialProviderArn": "arn:aws:bedrock-agentcore:us-east-1:111122223333:token-vault/default/apikeycredentialprovider/devin-acme-bedrock",
                         "createdTime": 0, "lastUpdatedTime": 0},
                        {"name": "devin-acme-bedrock", "apiKey": KEY})
        sc.add_response("create_workload_identity", {"name": "devin-agent", "workloadIdentityArn":
                        "arn:aws:bedrock-agentcore:us-east-1:111122223333:workload-identity-directory/default/workload-identity/devin-agent"},
                        {"name": "devin-agent"})
        sd.add_response("get_workload_access_token_for_user_id", {"workloadAccessToken": "wat"},
                        {"workloadName": "devin-agent", "userId": "dev@acme.com"})
        sd.add_response("get_resource_api_key", {"apiKey": KEY},
                        {"workloadIdentityToken": "wat", "resourceCredentialProviderName": "devin-acme-bedrock"})
        sd.add_response("get_resource_api_key", {"apiKey": KEY},
                        {"workloadIdentityToken": "runtime-token", "resourceCredentialProviderName": "devin-acme-bedrock"})
        v = AgentCoreIdentityVault("us-east-1", "devin-agent", control=control, data=data)
        v.put_api_key("devin-acme-bedrock", KEY)
        assert v.get_api_key("devin-acme-bedrock", "dev@acme.com") == KEY
        assert v.get_api_key("devin-acme-bedrock", "dev@acme.com", workload_token="runtime-token") == KEY


def test_bedrock_client_sends_tenant_key_as_bearer(monkeypatch):
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    client = BedrockClient(Settings(), "us-west-2", api_key=KEY)
    seen = {}

    def capture(request, **kw):
        seen["auth"], seen["url"] = request.headers["Authorization"], request.url
        raise RuntimeError("stop")
    client._client.meta.events.register("before-send.bedrock-runtime.Converse", capture)
    try:
        client.converse("m", [{"role": "user", "content": [{"text": "x"}]}])
    except RuntimeError:
        pass
    assert seen["auth"] == f"Bearer {KEY}".encode() and "us-west-2" in seen["url"]


def test_agentcore_runtime_dispatch(monkeypatch):
    from bedrock_connector import agentcore_app

    app = create_app(Settings())
    c = TestClient(app)
    h = login(c)
    c.put("/v1/org/bedrock-key", headers=h, json={"api_key": KEY})
    stack, agent = app.state.stack, app.state.agent
    sess = agent.create_session("fix", str(agentcore_app.SAMPLE_REPO), org=stack.tenants.orgs["acme"],
                                user_id="dev@acme.com")

    class FakeRuntime:
        def invoke_agent_runtime(self, **req):
            assert len(req["runtimeSessionId"]) >= 33 and req["runtimeUserId"] == "dev@acme.com"
            payload = json.loads(req["payload"])
            assert KEY not in req["payload"].decode() and payload["org"]["id"] == "acme"
            # what the container does, with the vault the runtime would reach through AgentCore Identity
            monkeypatch.setattr(agentcore_app, "build_stack", lambda st: _stack_with_vault(st, stack.vault))
            out = agentcore_app.handle(payload, workload_token="runtime-token")
            return {"response": io.BytesIO(json.dumps(out).encode())}

    d = AgentCoreDispatcher("arn:aws:bedrock-agentcore:us-east-1:111122223333:runtime/devin-x", "us-east-1",
                            stack.ledger, client=FakeRuntime())
    d.run(sess)
    assert sess.status == "unverified" and sess.runtime == "agentcore"
    s = c.get(f"/v1/agent/sessions/{sess.id}", headers=h).json()
    assert s["usage"]["calls"] >= 3 and s["usage"]["runtime_seconds"] > 0


def _stack_with_vault(settings, vault):
    from bedrock_connector.inference import build_stack
    return build_stack(settings, vault=vault)


def test_agentcore_runtime_deploy_request_matches_api():
    import importlib.util
    spec = importlib.util.spec_from_file_location("deploy", "deploy/agentcore/deploy.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    req = mod.runtime_request("devin_bedrock_agent", "111122223333.dkr.ecr.us-east-1.amazonaws.com/devin:latest",
                              "arn:aws:iam::111122223333:role/DevinAgentCoreRuntime", "us-east-1")
    ctl = boto3.client("bedrock-agentcore-control", region_name="us-east-1")
    with Stubber(ctl) as st:  # Stubber validates the request against the AgentCore API model
        st.add_response("create_agent_runtime", {
            "agentRuntimeArn": "arn:aws:bedrock-agentcore:us-east-1:111122223333:runtime/devin_bedrock_agent-abc",
            "agentRuntimeId": "devin_bedrock_agent-abc", "agentRuntimeVersion": "1", "createdAt": 0,
            "status": "CREATING", "workloadIdentityDetails": {"workloadIdentityArn": "arn:x"}}, req)
        ctl.create_agent_runtime(**req)


def test_anonymous_callers_only_see_demo_org(monkeypatch):
    c = TestClient(create_app(Settings(mode="mock")))
    tok = c.post("/v1/auth/login", json={"email": "dev@acme.com"}).json()["session_token"]
    c.put("/v1/org/bedrock-key", json={"api_key": "ABSK" + "x" * 40}, headers={"x-devin-session": tok})
    sid = c.post("/v1/agent/sessions", json={}, headers={"x-devin-session": tok}).json()["id"]
    assert c.get(f"/v1/agent/sessions/{sid}").status_code == 404
    assert all(s["org_id"] == "demo" for s in c.get("/v1/stats").json()["sessions"])
    assert all(r["org_id"] == "demo" for r in c.get("/v1/stats").json()["recent"])


def test_assumed_role_client_signs_sigv4_despite_process_api_key(monkeypatch):
    from bedrock_connector.bedrock import BedrockClient
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "process-key")
    sess = boto3.Session(aws_access_key_id="AKIAEXAMPLE", aws_secret_access_key="x", aws_session_token="t",
                         region_name="us-east-1")
    client, seen = BedrockClient(Settings(mode="live"), "us-east-1", boto_session=sess), {}

    def grab(request, **kw):
        seen["auth"] = request.headers["Authorization"]
        raise RuntimeError("stop")
    client._client.meta.events.register("before-send", grab)
    with pytest.raises(RuntimeError):
        client.converse("m", [{"role": "user", "content": [{"text": "hi"}]}])
    assert seen["auth"].startswith(b"AWS4-HMAC-SHA256")


def test_external_id_is_stable_across_restarts():
    from bedrock_connector.tenancy import TenantStore
    assert TenantStore().orgs["globex"].external_id == TenantStore().orgs["globex"].external_id


def test_acu_budget_blocks_new_sessions_and_commerce_is_reported():
    c = TestClient(create_app(Settings()))
    h = login(c, "dev@globex.com")
    assert c.put("/v1/org/settings", headers=h, json={"acu_budget": -1}).status_code == 400
    assert c.put("/v1/org/settings", headers=h, json={"acu_budget": "10"}).status_code == 400
    me = c.put("/v1/org/settings", headers=h, json={"credential_source": "devin_managed", "acu_budget": 0.000001}).json()
    assert me["org"]["acu_budget"] == 0.000001
    r = c.post("/v1/chat/completions", headers=h, json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    r = c.post("/v1/agent/sessions", headers=h, json={"task": "fix tests"})
    assert r.status_code == 402 and "budget" in r.json()["detail"]
    s = c.get("/v1/stats", headers=h).json()["summary"]
    assert s["commerce"]["ppa_drawdown_usd"] == pytest.approx(s["value_pools"]["customer_revenue_usd"])
    assert s["tokens_per_acu"] > 0
