"""Amazon Bedrock Converse client. All inference in this project goes through here."""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any

from .config import Settings


@dataclass
class ConverseResult:
    message: dict[str, Any]          # {"role": "assistant", "content": [...]} (Converse format)
    stop_reason: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    model_id: str
    request_id: str

    @property
    def text(self) -> str:
        return "".join(b.get("text", "") for b in self.message.get("content", []))

    @property
    def tool_uses(self) -> list[dict[str, Any]]:
        return [b["toolUse"] for b in self.message.get("content", []) if "toolUse" in b]


class _StaticToken:
    """botocore token provider that serves one Bedrock API key to one client,
    so tenants never share a process-wide AWS_BEARER_TOKEN_BEDROCK."""
    METHOD = "static"

    def __init__(self, token: str):
        self._token = token

    def load_token(self, **kwargs):
        from botocore.tokens import FrozenAuthToken

        return FrozenAuthToken(self._token)


class BedrockClient:
    """Thin wrapper over `bedrock-runtime.converse`.

    Auth comes from the standard AWS chain. With `AWS_BEARER_TOKEN_BEDROCK` set
    (a Bedrock API key), botocore sends it as a bearer token; otherwise IAM
    credentials (instance role, IAM Identity Center/SSO, env keys) are used.
    """

    def __init__(self, settings: Settings, region: str | None = None, api_key: str | None = None,
                 boto_session=None):
        """`api_key` (a customer's Bedrock API key) or `boto_session` (assumed-role
        credentials) scope this client to one tenant; otherwise the process's own
        AWS credentials are used."""
        import boto3
        from botocore.config import Config

        self.settings = settings
        cfg = Config(retries={"max_attempts": 6, "mode": "adaptive"}, read_timeout=300)
        region = region or settings.region
        if api_key:
            import botocore.session

            bs = botocore.session.Session()
            bs.register_component("token_provider", _StaticToken(api_key))
            self._client = boto3.Session(botocore_session=bs).client(
                "bedrock-runtime", region_name=region, config=cfg.merge(Config(signature_version="bearer")))
        else:
            self._client = (boto_session or boto3).client("bedrock-runtime", region_name=region, config=cfg)

    def converse(self, model_id: str, messages: list[dict], system: list[dict] | None = None,
                 tools: list[dict] | None = None, max_tokens: int = 4096,
                 temperature: float | None = None) -> ConverseResult:
        req: dict[str, Any] = {
            "modelId": model_id,
            "messages": messages,
            "inferenceConfig": {"maxTokens": max_tokens},
        }
        if temperature is not None:
            req["inferenceConfig"]["temperature"] = temperature
        if system:
            req["system"] = system
        if tools:
            req["toolConfig"] = {"tools": tools}
        if self.settings.guardrail_id:
            req["guardrailConfig"] = {
                "guardrailIdentifier": self.settings.guardrail_id,
                "guardrailVersion": self.settings.guardrail_version,
            }
        t0 = time.perf_counter()
        resp = self._client.converse(**req)
        usage = resp.get("usage", {})
        return ConverseResult(
            message=resp["output"]["message"],
            stop_reason=resp.get("stopReason", "end_turn"),
            input_tokens=usage.get("inputTokens", 0),
            output_tokens=usage.get("outputTokens", 0),
            latency_ms=resp.get("metrics", {}).get("latencyMs") or int((time.perf_counter() - t0) * 1000),
            model_id=model_id,
            request_id=resp.get("ResponseMetadata", {}).get("RequestId", ""),
        )


def _approx_tokens(obj: Any) -> int:
    return max(1, len(json.dumps(obj, default=str)) // 4)


class MockBedrockClient:
    """Offline stand-in with the same interface, for tests and demos without AWS access.

    For tool-using agents it follows a short scripted policy: inspect the
    repo, run the tests, then finish. It never edits code.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    def converse(self, model_id: str, messages: list[dict], system: list[dict] | None = None,
                 tools: list[dict] | None = None, max_tokens: int = 4096,
                 temperature: float | None = None) -> ConverseResult:
        if model_id in self.settings.mock_unavailable_models:
            from botocore.exceptions import ClientError

            raise ClientError({"Error": {"Code": "ThrottlingException",
                                         "Message": f"[mock] {model_id} is out of capacity"}}, "Converse")
        tool_names = {t["toolSpec"]["name"] for t in (tools or [])}
        n_results = sum(1 for m in messages for b in m["content"] if "toolResult" in b)
        script = [n for n in ("list_files", "run_command", "finish") if n in tool_names]
        if script:
            name = script[min(n_results, len(script) - 1)]
            args = {"list_files": {"path": "."},
                    "run_command": {"command": "python -m pytest -q"},
                    "finish": {"summary": "[mock] Inspected the repo and ran the tests. "
                                          "Set a Bedrock API key for real fixes."}}[name]
            content = [{"text": f"[mock] calling {name}"},
                       {"toolUse": {"toolUseId": f"tooluse_{uuid.uuid4().hex[:12]}", "name": name, "input": args}}]
            stop = "tool_use"
        else:
            last = next((b.get("text", "") for b in reversed(messages[-1]["content"]) if "text" in b), "")
            content = [{"text": f"[mock Bedrock response from {model_id}] You said: {last[:200]}"}]
            stop = "end_turn"
        msg = {"role": "assistant", "content": content}
        return ConverseResult(msg, stop, _approx_tokens([system, messages, tools]), _approx_tokens(content),
                              5, model_id, f"mock-{uuid.uuid4().hex[:8]}")


def make_client(settings: Settings):
    return BedrockClient(settings) if settings.resolved_mode == "live" else MockBedrockClient(settings)
