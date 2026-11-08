"""Scripted chat model for offline tests: replays a fixed list of responses."""

from __future__ import annotations

import itertools
from typing import Any, Callable, Sequence, Union

from llm.types import ChatError, ChatResponse, Message, ToolCall, ToolSpec, Usage

_ids = itertools.count(1)

ScriptItem = Union[ChatResponse, Callable[[str, Sequence[Message], Sequence[ToolSpec]], ChatResponse]]


def turn(text: str = "", *calls: tuple[str, dict[str, Any]]) -> ChatResponse:
    """Build a scripted model response: optional text plus (tool_name, args) calls."""
    return ChatResponse(
        text=text,
        tool_calls=[ToolCall(f"scripted_{next(_ids)}", name, args) for name, args in calls],
        usage=Usage(100, 20),
        stop_reason="tool_use" if calls else "end_turn",
    )


class ScriptedChat:
    """Returns ``script`` items in order; records every request for assertions."""

    def __init__(self, script: Sequence[ScriptItem], model: str = "scripted") -> None:
        self.model = model
        self._script = list(script)
        self._pos = 0
        self.requests: list[dict[str, Any]] = []

    def chat(self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> ChatResponse:
        self.requests.append({"system": system, "messages": list(messages), "tools": [t.name for t in tools]})
        if self._pos >= len(self._script):
            raise ChatError(f"ScriptedChat exhausted after {self._pos} calls")
        item = self._script[self._pos]
        self._pos += 1
        return item(system, messages, tools) if callable(item) else item

    @property
    def exhausted(self) -> bool:
        return self._pos >= len(self._script)
