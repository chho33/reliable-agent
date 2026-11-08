"""Agent state. Plain mutable dataclasses: nodes mutate the state and return it."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from agent.policy_guard import Decision, WriteAction
from llm.types import Message, ToolCall, Usage

Variant = Literal["guarded", "control"]


@dataclass
class PendingAction:
    """A policy-approved write waiting for the customer's explicit confirmation."""

    call: ToolCall
    action: WriteAction
    description: str
    turn: int


@dataclass
class Trigger:
    """What the ``act`` node hands to the rest of the graph this step.

    ``propose``: the model called a write tool. ``confirm``: the model called
    ``confirm_pending`` (execute the stored pending action).
    """

    kind: Literal["propose", "confirm"]
    call: ToolCall


@dataclass
class ActionRecord:
    """One write executed against the backend, with the ground truth needed to audit it.

    ``audit`` is the tool server's ``_audit`` block: verified customer, today, and the
    order's facts *before* the write. ``auto`` marks system-initiated writes (auto CRM
    note, guard-triggered handoff) as opposed to model-initiated ones.
    """

    seq: int
    turn: int
    tool: str
    args: dict[str, Any]
    ok: bool
    result: dict[str, Any]
    audit: dict[str, Any] | None
    auto: bool = False
    confirmed: bool = False


@dataclass
class State:
    variant: Variant
    messages: list[Message] = field(default_factory=list)
    verified_customer_id: str | None = None
    pending: PendingAction | None = None
    trigger: Trigger | None = None
    unanswered: ToolCall | None = None  # trigger call still lacking a tool result
    action: WriteAction | None = None
    decision: Decision | None = None
    last_record: ActionRecord | None = None
    consecutive_failures: int = 0
    action_log: list[ActionRecord] = field(default_factory=list)
    handed_off: bool = False
    done: bool = False
    termination: str | None = None
    steps: int = 0  # number of model calls
    turn: int = 0
    usage: Usage = field(default_factory=Usage)
    reply: str | None = None  # set when the current turn is over
