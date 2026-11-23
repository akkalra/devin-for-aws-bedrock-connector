import time

from fastapi.testclient import TestClient

from bedrock_connector.config import Settings
from bedrock_connector.gateway import create_app


def test_openai_and_anthropic_endpoints_are_metered():
    c = TestClient(create_app(Settings()))
    assert c.get("/healthz").json()["inference_provider"] == "amazon-bedrock"
    r = c.post("/v1/chat/completions", json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]})
    assert r.status_code == 200 and "mock Bedrock" in r.json()["choices"][0]["message"]["content"]
    r = c.post("/v1/messages", json={"model": "claude", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]})
    assert r.json()["type"] == "message"
    s = c.get("/v1/stats").json()["summary"]
    assert s["calls"] == 2 and s["acus"] > 0


def test_streaming_responses():
    c = TestClient(create_app(Settings()))
    r = c.post("/v1/messages", json={"model": "x", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
    assert "event: message_stop" in r.text
    r = c.post("/v1/chat/completions", json={"model": "x", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
    assert r.text.strip().endswith("data: [DONE]")


def test_gateway_api_key(monkeypatch):
    monkeypatch.setenv("GATEWAY_API_KEY", "s3cret")
    c = TestClient(create_app(Settings()))
    assert c.post("/v1/messages", json={"messages": []}).status_code == 401
    ok = c.post("/v1/messages", headers={"x-api-key": "s3cret"},
                json={"model": "x", "messages": [{"role": "user", "content": "hi"}]})
    assert ok.status_code == 200


def test_agent_session_runs_through_bedrock():
    c = TestClient(create_app(Settings()))
    sid = c.post("/v1/agent/sessions", json={"task": "fix tests"}).json()["id"]
    for _ in range(100):
        s = c.get(f"/v1/agent/sessions/{sid}").json()
        if s["status"] in ("finished", "unverified", "error"):
            break
        time.sleep(0.05)
    assert s["status"] == "unverified", s["steps"]  # mock client does not fix the bugs
    roles = [st["role"] for st in s["steps"] if st["type"] == "model"]
    assert roles[0] == "planner" and roles[-1] == "reviewer" and "coder" in roles
    assert {st["name"] for st in s["steps"] if st["type"] == "tool"} >= {"list_files", "run_command", "finish"}
    assert s["usage"]["runtime_seconds"] > 0
