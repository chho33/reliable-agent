"""Minimal provider-neutral chat types. No SDKs: adapters speak HTTP via the stdlib."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, Sequence


class ChatError(RuntimeError):
    """Raised by adapters on transport/API errors (never includes credentials)."""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    parse_error: str | None = None  # set if the model produced unparseable JSON arguments


@dataclass
class Message:
    """role=user: ``content``. role=assistant: ``content`` and/or ``tool_calls``.
    role=tool: ``content`` is the result for ``tool_call_id``."""

    role: Literal["user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(self.input_tokens + other.input_tokens, self.output_tokens + other.output_tokens)


@dataclass
class ChatResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    stop_reason: str = ""


class ChatModel(Protocol):
    model: str

    def chat(self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> ChatResponse: ...
