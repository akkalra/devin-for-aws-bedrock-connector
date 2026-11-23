"""Translate OpenAI Chat Completions and Anthropic Messages payloads to/from Bedrock Converse."""
from __future__ import annotations

import json
import time
import uuid
from typing import Any


def _merge_same_role(messages: list[dict]) -> list[dict]:
    """Converse requires alternating roles; merge consecutive same-role turns."""
    out: list[dict] = []
    for m in messages:
        if not m["content"]:
            continue
        if out and out[-1]["role"] == m["role"]:
            out[-1]["content"].extend(m["content"])
        else:
            out.append({"role": m["role"], "content": list(m["content"])})
    return out


def _text_of(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict))


# ---------------------------------------------------------------- OpenAI


def openai_to_converse(body: dict) -> dict:
    system, messages = [], []
    for m in body.get("messages", []):
        role = m.get("role")
        if role in ("system", "developer"):
            if t := _text_of(m.get("content")):
                system.append({"text": t})
        elif role == "user":
            messages.append({"role": "user", "content": [{"text": _text_of(m.get("content")) or " "}]})
        elif role == "assistant":
            content = []
            if t := _text_of(m.get("content")):
                content.append({"text": t})
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                args = fn.get("arguments") or "{}"
                content.append({"toolUse": {"toolUseId": tc.get("id"), "name": fn.get("name"),
                                            "input": json.loads(args) if isinstance(args, str) else args}})
            messages.append({"role": "assistant", "content": content})
        elif role == "tool":
            messages.append({"role": "user", "content": [{"toolResult": {
                "toolUseId": m.get("tool_call_id"), "content": [{"text": _text_of(m.get("content")) or " "}]}}]})
    tools = [{"toolSpec": {"name": t["function"]["name"],
                           "description": t["function"].get("description") or t["function"]["name"],
                           "inputSchema": {"json": t["function"].get("parameters") or {"type": "object", "properties": {}}}}}
             for t in body.get("tools") or [] if t.get("type", "function") == "function"]
    return {
        "messages": _merge_same_role(messages),
        "system": system or None,
        "tools": tools or None,
        "max_tokens": body.get("max_completion_tokens") or body.get("max_tokens") or 4096,
        "temperature": body.get("temperature"),
    }


_OPENAI_FINISH = {"end_turn": "stop", "stop_sequence": "stop", "tool_use": "tool_calls",
                  "max_tokens": "length", "guardrail_intervened": "content_filter",
                  "content_filtered": "content_filter"}


def converse_to_openai(result, requested_model: str) -> dict:
    tool_calls = [{"id": tu["toolUseId"], "type": "function",
                   "function": {"name": tu["name"], "arguments": json.dumps(tu["input"])}}
                  for tu in result.tool_uses]
    message: dict[str, Any] = {"role": "assistant", "content": result.text or None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": requested_model,
        "choices": [{"index": 0, "message": message,
                     "finish_reason": _OPENAI_FINISH.get(result.stop_reason, "stop")}],
        "usage": {"prompt_tokens": result.input_tokens, "completion_tokens": result.output_tokens,
                  "total_tokens": result.input_tokens + result.output_tokens},
        "x_bedrock": {"model_id": result.model_id, "request_id": result.request_id},
    }


def openai_stream_chunks(resp: dict):
    """Re-emit a finished completion as SSE chunks (single delta) for streaming clients."""
    base = {k: resp[k] for k in ("id", "created", "model")} | {"object": "chat.completion.chunk"}
    msg = resp["choices"][0]["message"]
    delta: dict[str, Any] = {"role": "assistant", "content": msg.get("content") or ""}
    if msg.get("tool_calls"):
        delta["tool_calls"] = [tc | {"index": i} for i, tc in enumerate(msg["tool_calls"])]
    yield base | {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
    yield base | {"choices": [{"index": 0, "delta": {}, "finish_reason": resp["choices"][0]["finish_reason"]}],
                  "usage": resp["usage"]}


# ---------------------------------------------------------------- Anthropic


def anthropic_to_converse(body: dict) -> dict:
    sys_field = body.get("system")
    system = [{"text": sys_field}] if isinstance(sys_field, str) and sys_field else \
        [{"text": b["text"]} for b in sys_field or [] if b.get("type") == "text"]
    messages = []
    for m in body.get("messages", []):
        content = m.get("content")
        blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content or []
        out = []
        for b in blocks:
            t = b.get("type")
            if t == "text" and b.get("text"):
                out.append({"text": b["text"]})
            elif t == "tool_use":
                out.append({"toolUse": {"toolUseId": b["id"], "name": b["name"], "input": b.get("input") or {}}})
            elif t == "tool_result":
                rc = b.get("content")
                text = rc if isinstance(rc, str) else _text_of(rc)
                tr = {"toolUseId": b["tool_use_id"], "content": [{"text": text or " "}]}
                if b.get("is_error"):
                    tr["status"] = "error"
                out.append({"toolResult": tr})
        messages.append({"role": m["role"], "content": out})
    tools = [{"toolSpec": {"name": t["name"], "description": t.get("description") or t["name"],
                           "inputSchema": {"json": t.get("input_schema") or {"type": "object", "properties": {}}}}}
             for t in body.get("tools") or [] if "name" in t and "input_schema" in t]
    return {
        "messages": _merge_same_role(messages),
        "system": system or None,
        "tools": tools or None,
        "max_tokens": body.get("max_tokens") or 4096,
        "temperature": body.get("temperature"),
    }


def converse_to_anthropic(result, requested_model: str) -> dict:
    content = []
    for b in result.message.get("content", []):
        if "text" in b:
            content.append({"type": "text", "text": b["text"]})
        elif "toolUse" in b:
            tu = b["toolUse"]
            content.append({"type": "tool_use", "id": tu["toolUseId"], "name": tu["name"], "input": tu["input"]})
    stop = {"end_turn": "end_turn", "tool_use": "tool_use", "max_tokens": "max_tokens",
            "stop_sequence": "stop_sequence"}.get(result.stop_reason, "end_turn")
    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": requested_model,
        "content": content,
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {"input_tokens": result.input_tokens, "output_tokens": result.output_tokens},
    }


def anthropic_stream_events(resp: dict):
    """Re-emit a finished message as Anthropic SSE events."""
    start = resp | {"content": [], "stop_reason": None, "usage": {"input_tokens": resp["usage"]["input_tokens"], "output_tokens": 0}}
    yield "message_start", {"type": "message_start", "message": start}
    for i, block in enumerate(resp["content"]):
        if block["type"] == "text":
            yield "content_block_start", {"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}}
            yield "content_block_delta", {"type": "content_block_delta", "index": i, "delta": {"type": "text_delta", "text": block["text"]}}
        else:
            yield "content_block_start", {"type": "content_block_start", "index": i,
                                          "content_block": {"type": "tool_use", "id": block["id"], "name": block["name"], "input": {}}}
            yield "content_block_delta", {"type": "content_block_delta", "index": i,
                                          "delta": {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}}
        yield "content_block_stop", {"type": "content_block_stop", "index": i}
    yield "message_delta", {"type": "message_delta", "delta": {"stop_reason": resp["stop_reason"], "stop_sequence": None},
                            "usage": {"output_tokens": resp["usage"]["output_tokens"]}}
    yield "message_stop", {"type": "message_stop"}
