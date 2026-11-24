from bedrock_connector.bedrock import ConverseResult
from bedrock_connector.translate import (anthropic_to_converse, converse_to_anthropic, converse_to_openai,
                                         openai_to_converse)


def _result(content, stop="end_turn"):
    return ConverseResult({"role": "assistant", "content": content}, stop, 10, 5, 1, "m", "rid")


def test_openai_roundtrip_with_tools():
    body = {"model": "gpt-4o", "max_tokens": 100, "messages": [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "get", "arguments": "{\"city\": \"SEA\"}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "rainy"},
        {"role": "user", "content": "thanks"},
    ], "tools": [{"type": "function", "function": {"name": "get", "parameters": {"type": "object"}}}]}
    conv = openai_to_converse(body)
    assert conv["system"] == [{"text": "be brief"}]
    assert [m["role"] for m in conv["messages"]] == ["user", "assistant", "user"]
    assert conv["messages"][1]["content"][0]["toolUse"]["input"] == {"city": "SEA"}
    assert conv["messages"][2]["content"][0]["toolResult"]["toolUseId"] == "c1"
    assert conv["tools"][0]["toolSpec"]["name"] == "get"

    out = converse_to_openai(_result([{"toolUse": {"toolUseId": "t", "name": "get", "input": {"a": 1}}}], "tool_use"), "gpt-4o")
    assert out["choices"][0]["finish_reason"] == "tool_calls"
    assert out["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'
    assert out["usage"]["total_tokens"] == 15


def test_anthropic_roundtrip_with_tools():
    body = {"model": "claude-sonnet", "max_tokens": 50, "system": "sys", "messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "tu1", "name": "ls", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu1", "content": "a.py", "is_error": True}]},
    ], "tools": [{"name": "ls", "description": "list", "input_schema": {"type": "object"}}]}
    conv = anthropic_to_converse(body)
    assert conv["system"] == [{"text": "sys"}]
    assert conv["messages"][2]["content"][0]["toolResult"]["status"] == "error"
    out = converse_to_anthropic(_result([{"text": "ok"}]), "claude-sonnet")
    assert out["content"] == [{"type": "text", "text": "ok"}]
    assert out["usage"] == {"input_tokens": 10, "output_tokens": 5}
