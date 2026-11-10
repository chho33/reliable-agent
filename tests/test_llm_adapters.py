"""Adapter request building / response parsing with a mocked ``urlopen`` (no network)."""

import io
import json
import urllib.error

import pytest

from llm.anthropic import AnthropicChat
from llm.factory import cost_usd, make_chat, resolve
from llm.openai_compat import OpenAICompatChat
from llm.scripted import ScriptedChat, turn
from llm.types import ChatError, Message, ToolCall, ToolSpec, Usage

TOOLS = [ToolSpec("get_order", "Get an order", {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]})]
HISTORY = [
    Message("user", "where is my order?"),
    Message("assistant", "Checking.", [ToolCall("t1", "get_order", {"order_id": "ORD-1"}), ToolCall("t2", "get_order", {"order_id": "ORD-2"})]),
    Message("tool", '{"ok": true}', tool_call_id="t1"),
    Message("tool", '{"ok": false}', tool_call_id="t2"),
    Message("assistant", "It shipped."),
    Message("user", "thanks"),
]


class FakeResp:
    def __init__(self, payload):
        self._b = json.dumps(payload).encode()

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture
def capture(monkeypatch):
    """Patch urlopen; returns a list of captured requests. Set ``capture.responses``."""
    class Seen(list):
        state: dict

    seen = Seen()
    state = {"responses": []}

    def fake_urlopen(req, timeout=None):
        seen.append({"url": req.full_url, "headers": {k.lower(): v for k, v in req.header_items()}, "body": json.loads(req.data), "timeout": timeout})
        item = state["responses"].pop(0)
        if isinstance(item, Exception):
            raise item
        return FakeResp(item)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("llm.http.time.sleep", lambda s: None)
    seen.state = state
    return seen


# ---------- Anthropic ----------
def test_anthropic_request_shape(capture, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    capture.state["responses"].append({"content": [{"type": "text", "text": "hi"}], "usage": {"input_tokens": 11, "output_tokens": 3}, "stop_reason": "end_turn"})
    chat = AnthropicChat("some-model", max_tokens=256, temperature=0.3)
    resp = chat.chat("SYS", HISTORY, TOOLS)
    req = capture[0]
    assert req["url"] == "https://api.anthropic.com/v1/messages"
    assert req["headers"]["x-api-key"] == "sk-test" and req["headers"]["anthropic-version"] == "2023-06-01"
    b = req["body"]
    assert b["model"] == "some-model" and b["system"] == "SYS" and b["max_tokens"] == 256 and b["temperature"] == 0.3
    assert b["tools"] == [{"name": "get_order", "description": "Get an order", "input_schema": TOOLS[0].input_schema}]
    roles = [m["role"] for m in b["messages"]]
    assert roles == ["user", "assistant", "user", "assistant", "user"]
    assistant = b["messages"][1]["content"]
    assert assistant[0] == {"type": "text", "text": "Checking."} and assistant[1]["type"] == "tool_use" and assistant[1]["input"] == {"order_id": "ORD-1"}
    # both tool results are merged into one user turn
    results = b["messages"][2]["content"]
    assert [r["tool_use_id"] for r in results] == ["t1", "t2"] and all(r["type"] == "tool_result" for r in results)
    assert resp.text == "hi" and resp.usage == Usage(11, 3)


def test_anthropic_tool_result_then_user_text_are_one_turn():
    msgs = AnthropicChat.convert_messages([
        Message("assistant", "", [ToolCall("a", "x", {})]), Message("tool", "r", tool_call_id="a"), Message("user", "next"),
    ])
    assert [m["role"] for m in msgs] == ["assistant", "user"]
    assert [b["type"] for b in msgs[1]["content"]] == ["tool_result", "text"]


def test_anthropic_omits_optional_fields():
    body = AnthropicChat("m").build_request("s", [Message("user", "x")], [])
    assert "tools" not in body and "temperature" not in body


def test_anthropic_parse_tool_use():
    r = AnthropicChat.parse_response({
        "content": [{"type": "text", "text": "ok"}, {"type": "tool_use", "id": "tu1", "name": "get_order", "input": {"order_id": "ORD-9"}}],
        "usage": {"input_tokens": 5, "output_tokens": 7}, "stop_reason": "tool_use",
    })
    assert r.text == "ok" and r.tool_calls == [ToolCall("tu1", "get_order", {"order_id": "ORD-9"})] and r.stop_reason == "tool_use"


def test_anthropic_requires_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ChatError, match="ANTHROPIC_API_KEY"):
        AnthropicChat("m").chat("s", [Message("user", "x")], [])


def test_retry_on_429_then_success(capture, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    err = urllib.error.HTTPError("u", 429, "rate", {}, io.BytesIO(b"slow down"))
    capture.state["responses"] += [err, {"content": [{"type": "text", "text": "ok"}], "usage": {}}]
    assert AnthropicChat("m").chat("s", [Message("user", "x")], []).text == "ok"
    assert len(capture) == 2


def test_non_retryable_error_raises_without_key_leak(capture, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    capture.state["responses"].append(urllib.error.HTTPError("u", 400, "bad", {}, io.BytesIO(b'{"error":"bad request"}')))
    with pytest.raises(ChatError) as ei:
        AnthropicChat("m").chat("s", [Message("user", "x")], [])
    assert "400" in str(ei.value) and "sk-secret" not in str(ei.value) and len(capture) == 1


def test_gives_up_after_retries(capture, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    capture.state["responses"] += [urllib.error.URLError("down")] * 4
    with pytest.raises(ChatError, match="giving up"):
        AnthropicChat("m").chat("s", [Message("user", "x")], [])


# ---------- OpenAI-compatible ----------
def test_openai_request_shape(capture, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oai")
    capture.state["responses"].append({"choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 20, "completion_tokens": 4}})
    chat = OpenAICompatChat("gpt-x", base_url="http://localhost:8000/v1/", max_tokens=99)
    resp = chat.chat("SYS", HISTORY, TOOLS)
    req = capture[0]
    assert req["url"] == "http://localhost:8000/v1/chat/completions"
    assert req["headers"]["authorization"] == "Bearer sk-oai"
    b = req["body"]
    assert b["model"] == "gpt-x" and b["max_tokens"] == 99
    assert b["messages"][0] == {"role": "system", "content": "SYS"}
    assert b["tools"][0] == {"type": "function", "function": {"name": "get_order", "description": "Get an order", "parameters": TOOLS[0].input_schema}}
    asst = b["messages"][2]
    assert asst["role"] == "assistant" and asst["tool_calls"][0]["function"]["arguments"] == '{"order_id": "ORD-1"}'
    tool_msgs = [m for m in b["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["t1", "t2"]
    assert resp.text == "hello" and resp.usage == Usage(20, 4)


def test_openai_max_tokens_field_configurable():
    body = OpenAICompatChat("m", max_tokens_field="max_completion_tokens", max_tokens=5).build_request("s", [], [])
    assert body["max_completion_tokens"] == 5 and "max_tokens" not in body


def test_openai_extra_body_is_sent_and_cannot_override_core_fields():
    tools = [ToolSpec("get_order", "look up an order", {"type": "object", "properties": {}})]
    body = OpenAICompatChat("m", extra_body={"reasoning_effort": "none"}).build_request("s", [], tools)
    assert body["reasoning_effort"] == "none" and body["tools"][0]["function"]["name"] == "get_order"
    with pytest.raises(ValueError, match="extra_body may not set"):
        OpenAICompatChat("m", max_tokens_field="max_completion_tokens", extra_body={"max_completion_tokens": 9})


def test_make_chat_passes_extra_body_from_config():
    config = {"models": {"openai:gpt-6-luna": {"max_tokens_field": "max_completion_tokens", "max_tokens": 2048,
                                               "extra_body": {"reasoning_effort": "none"},
                                               "input_per_mtok": 0.1, "output_per_mtok": 0.5}}}
    chat = make_chat("openai:gpt-6-luna", config)
    body = chat.build_request("s", [], [])
    assert body["model"] == "gpt-6-luna" and body["max_completion_tokens"] == 2048
    assert body["reasoning_effort"] == "none" and "max_tokens" not in body
    assert cost_usd(resolve("openai:gpt-6-luna", config), Usage(1_000_000, 1_000_000)) == pytest.approx(0.6)


def test_openai_parse_tool_calls_and_bad_json():
    r = OpenAICompatChat.parse_response({
        "choices": [{"message": {"content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "get_order", "arguments": '{"order_id": "ORD-1"}'}},
            {"id": "c2", "type": "function", "function": {"name": "get_order", "arguments": "{oops"}},
        ]}, "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2},
    })
    assert r.text == "" and r.tool_calls[0].arguments == {"order_id": "ORD-1"} and r.tool_calls[0].parse_error is None
    assert r.tool_calls[1].arguments == {} and "invalid JSON" in r.tool_calls[1].parse_error


def test_openai_bad_shape_raises():
    with pytest.raises(ChatError):
        OpenAICompatChat.parse_response({"error": "x"})


def test_openai_keyless_local_server(capture, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    capture.state["responses"].append({"choices": [{"message": {"content": "ok"}}]})
    OpenAICompatChat("m", base_url="http://x/v1", require_key=False).chat("s", [Message("user", "x")], [])
    assert "authorization" not in capture[0]["headers"]


# ---------- scripted + factory ----------
def test_scripted_chat_replays_and_records_then_exhausts():
    chat = ScriptedChat([turn("a"), turn("", ("get_order", {"order_id": "X"}))])
    assert chat.chat("sys", [Message("user", "hi")], TOOLS).text == "a"
    assert chat.chat("sys", [], []).tool_calls[0].name == "get_order"
    assert chat.requests[0]["tools"] == ["get_order"] and chat.exhausted
    with pytest.raises(ChatError):
        chat.chat("sys", [], [])


def test_resolve_and_cost():
    cfg = {"defaults": {"max_tokens": 10}, "models": {"anthropic:m": {"input_per_mtok": 3.0, "output_per_mtok": 15.0, "temperature": 0.5}}}
    mc = resolve("anthropic:m", cfg)
    assert mc.options == {"max_tokens": 10, "temperature": 0.5}
    assert cost_usd(mc, Usage(1_000_000, 100_000)) == pytest.approx(4.5)
    assert cost_usd(resolve("anthropic:unpriced", cfg), Usage(5, 5)) is None


def test_make_chat_types_and_bad_spec():
    assert isinstance(make_chat("anthropic:m", {}), AnthropicChat)
    chat = make_chat("openai:m", {"models": {"openai:m": {"base_url": "http://h/v1"}}})
    assert isinstance(chat, OpenAICompatChat) and chat.base_url == "http://h/v1"
    with pytest.raises(ValueError):
        make_chat("nocolon", {})
    with pytest.raises(ValueError):
        make_chat("gemini:x", {})
