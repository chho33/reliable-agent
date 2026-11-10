"""OpenAI-compatible Chat Completions adapter (OpenAI, vLLM, Ollama, OpenRouter, ...)."""

from __future__ import annotations

import json
import os
from typing import Any, Sequence

from llm.http import post_json
from llm.types import ChatError, ChatResponse, Message, ToolCall, ToolSpec, Usage


class OpenAICompatChat:
    def __init__(
        self,
        model: str,
        *,
        base_url: str = "https://api.openai.com/v1",
        api_key_env: str = "OPENAI_API_KEY",
        max_tokens: int = 1024,
        max_tokens_field: str = "max_tokens",
        temperature: float | None = None,
        timeout: float = 120.0,
        require_key: bool = True,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.max_tokens = max_tokens
        self.max_tokens_field = max_tokens_field  # some newer OpenAI models want "max_completion_tokens"
        self.temperature = temperature
        self.timeout = timeout
        self.require_key = require_key  # local servers (Ollama/vLLM) often need no key
        # Provider-specific request fields from config (e.g. {"reasoning_effort": "none"}).
        # They cannot replace the fields this adapter builds itself.
        reserved = {"model", "messages", "tools", max_tokens_field}
        clash = reserved & set(extra_body or {})
        if clash:
            raise ValueError(f"extra_body may not set {sorted(clash)}; use the dedicated options instead")
        self.extra_body = dict(extra_body or {})

    @staticmethod
    def convert_messages(system: str, messages: Sequence[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for m in messages:
            if m.role == "user":
                out.append({"role": "user", "content": m.content})
            elif m.role == "tool":
                out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
            else:
                msg: dict[str, Any] = {"role": "assistant", "content": m.content or None}
                if m.tool_calls:
                    msg["tool_calls"] = [
                        {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                        for c in m.tool_calls
                    ]
                out.append(msg)
        return out

    def build_request(self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": self.convert_messages(system, messages),
            self.max_tokens_field: self.max_tokens,
        }
        if tools:
            body["tools"] = [
                {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.input_schema}}
                for t in tools
            ]
        if self.temperature is not None:
            body["temperature"] = self.temperature
        body.update(self.extra_body)
        return body

    @staticmethod
    def parse_response(data: dict[str, Any]) -> ChatResponse:
        try:
            choice = data["choices"][0]
            msg = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ChatError(f"unexpected response shape: {str(data)[:300]}") from exc
        calls: list[ToolCall] = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            raw = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw)
                err = None if isinstance(args, dict) else "arguments must be a JSON object"
                if err:
                    args = {}
            except json.JSONDecodeError as exc:
                args, err = {}, f"invalid JSON arguments: {exc}"
            calls.append(ToolCall(tc.get("id", ""), fn.get("name", ""), args, parse_error=err))
        usage = data.get("usage") or {}
        return ChatResponse(
            text=msg.get("content") or "", tool_calls=calls,
            usage=Usage(int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))),
            stop_reason=choice.get("finish_reason") or "",
        )

    def chat(self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> ChatResponse:
        headers: dict[str, str] = {}
        key = os.environ.get(self.api_key_env)
        if key:
            headers["authorization"] = f"Bearer {key}"
        elif self.require_key:
            raise ChatError(f"environment variable {self.api_key_env} is not set")
        data = post_json(f"{self.base_url}/chat/completions", headers, self.build_request(system, messages, tools), timeout=self.timeout)
        return self.parse_response(data)
