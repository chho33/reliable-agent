"""Anthropic Messages API adapter (stdlib only)."""

from __future__ import annotations

import os
from typing import Any, Sequence

from llm.http import post_json
from llm.types import ChatError, ChatResponse, Message, ToolCall, ToolSpec, Usage

API_VERSION = "2023-06-01"


class AnthropicChat:
    def __init__(
        self,
        model: str,
        *,
        api_key_env: str = "ANTHROPIC_API_KEY",
        base_url: str = "https://api.anthropic.com",
        max_tokens: int = 1024,
        temperature: float | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.api_key_env = api_key_env
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

    # ---- request building ----
    @staticmethod
    def convert_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
        """Neutral messages -> Anthropic format.

        Tool results become ``tool_result`` blocks in a user turn; consecutive user-side
        content (tool results followed by the next user text) is merged into one turn.
        """
        out: list[dict[str, Any]] = []

        def add_user_blocks(blocks: list[dict[str, Any]]) -> None:
            if out and out[-1]["role"] == "user":
                out[-1]["content"].extend(blocks)
            else:
                out.append({"role": "user", "content": blocks})

        for m in messages:
            if m.role == "user":
                add_user_blocks([{"type": "text", "text": m.content}])
            elif m.role == "tool":
                add_user_blocks([{"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content}])
            else:
                blocks: list[dict[str, Any]] = []
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                blocks += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments} for c in m.tool_calls]
                if blocks:  # the API rejects empty assistant turns
                    out.append({"role": "assistant", "content": blocks})
        return out

    def build_request(self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": self.convert_messages(messages),
        }
        if tools:
            body["tools"] = [{"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in tools]
        if self.temperature is not None:
            body["temperature"] = self.temperature
        return body

    # ---- response parsing ----
    @staticmethod
    def parse_response(data: dict[str, Any]) -> ChatResponse:
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in data.get("content", []):
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                calls.append(ToolCall(block["id"], block["name"], block.get("input") or {}))
        usage = data.get("usage") or {}
        return ChatResponse(
            text="".join(text_parts), tool_calls=calls,
            usage=Usage(int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))),
            stop_reason=data.get("stop_reason") or "",
        )

    def chat(self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> ChatResponse:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise ChatError(f"environment variable {self.api_key_env} is not set")
        data = post_json(
            f"{self.base_url}/v1/messages",
            {"x-api-key": key, "anthropic-version": API_VERSION},
            self.build_request(system, messages, tools),
            timeout=self.timeout,
        )
        return self.parse_response(data)
