"""Graph nodes and routers as plain functions over ``State``.

Flow (guarded variant)::

    act ──write proposed / confirm_pending──▶ guard ──allow──▶ confirm ──▶ END (ask customer)
     │                                          │                  (next turn: act → confirm_pending → guard → execute)
     │                                          ├─needs_human──▶ handoff ──▶ END
     │                                          └─deny──▶ act (or handoff after 2 consecutive failures)
     └─text only──▶ END          execute ──▶ log_crm ──▶ act
                                 (guard allow for confirmed / non-confirmable writes)

Control variant: ``act`` routes a proposed write straight to ``execute``; there is no
guard, no confirm step, no auto CRM note and no strike-based handoff.

The guard runs twice for confirmed actions: when proposed and again right before
execution (state may have changed between turns).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from agent.policy_guard import OrderFacts, PolicyContext, WriteAction, check, deny
from agent.prompts import build_system_prompt
from agent.state import ActionRecord, PendingAction, State, Trigger, Variant
from llm.types import ChatModel, Message, ToolCall, ToolSpec
from tools.client import ToolClient
from tools.registry import ToolDef

END = "__end__"
CONFIRM_TOOL = "confirm_pending"
CANCEL_TOOL = "cancel_pending"
MAX_CONSECUTIVE_FAILURES = 2

PENDING_TOOLS = (
    ToolSpec(
        CONFIRM_TOOL,
        "Execute the pending action. Only call when the customer's latest message clearly confirms it. "
        "'user_quote' must be the customer's confirming words, copied from their latest message.",
        {"type": "object", "properties": {"user_quote": {"type": "string"}}, "required": ["user_quote"]},
    ),
    ToolSpec(
        CANCEL_TOOL,
        "Discard the pending action (customer declined or changed their mind).",
        {"type": "object", "properties": {"reason": {"type": "string"}}, "required": []},
    ),
)


@dataclass
class Deps:
    chat: ChatModel
    tools: ToolClient
    policy_text: str
    today: date
    variant: Variant
    max_steps: int = 30
    defs: dict[str, ToolDef] = field(init=False)

    def __post_init__(self) -> None:
        self.defs = {d.name: d for d in self.tools.list_tools()}


def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _norm(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace; padded so matches are whole-word."""
    return " " + " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in text).split()) + " "


def describe_action(action: WriteAction, args: dict[str, Any]) -> str:
    if action.tool == "refund_order":
        return f"Refund ${action.amount_cents / 100:.2f} on order {action.order_id} (reason: {args.get('reason', 'n/a')})"
    if action.tool == "update_address":
        return f"Change the shipping address of order {action.order_id} to: {action.new_address}"
    return action.tool


def _latest_user_text(state: State) -> str:
    return next((m.content for m in reversed(state.messages) if m.role == "user"), "")


def _answer(state: State, call: ToolCall, payload: dict[str, Any]) -> None:
    """Append the tool result for ``call``."""
    state.messages.append(Message("tool", _json(payload), tool_call_id=call.id))
    if state.unanswered is not None and state.unanswered.id == call.id:
        state.unanswered = None


def _facts_from_get_order(order: dict[str, Any]) -> OrderFacts:
    return OrderFacts(
        order["order_id"], order["customer_id"], order["status"], int(order["total_cents"]),
        date.fromisoformat(order["delivered_at"]) if order.get("delivered_at") else None,
        int(order["refunded_cents"]),
    )


def make_nodes(deps: Deps) -> dict[str, Any]:
    guarded = deps.variant == "guarded"

    def llm_tools(state: State) -> list[ToolSpec]:
        specs = [ToolSpec(d.name, d.description, d.input_schema) for d in deps.defs.values()]
        if guarded and state.pending is not None:
            specs += list(PENDING_TOOLS)
        return specs

    def run_read(state: State, call: ToolCall) -> dict[str, Any]:
        result = deps.tools.call(call.name, call.arguments)
        if call.name == "verify_identity" and result.get("verified") is True:
            state.verified_customer_id = result["customer_id"]
        return result

    # ------------------------------------------------------------------ act
    def act(state: State) -> State:
        if state.steps >= deps.max_steps:
            state.done, state.termination = True, "step_cap"
            state.reply = "I'm sorry, I wasn't able to finish this request. Please contact support again."
            return state
        state.steps += 1
        system = build_system_prompt(deps.policy_text, deps.variant, deps.today, state.pending)
        resp = deps.chat.chat(system, state.messages, llm_tools(state))
        state.usage += resp.usage
        state.messages.append(Message("assistant", resp.text, resp.tool_calls))
        state.trigger = None
        if not resp.tool_calls:
            state.reply = resp.text or "(no response)"
            return state

        remaining = list(resp.tool_calls)
        while remaining:
            call = remaining.pop(0)
            if call.parse_error:
                _answer(state, call, {"ok": False, "error": "invalid_arguments", "message": call.parse_error})
            elif guarded and call.name == CANCEL_TOOL:
                if state.pending is None:
                    _answer(state, call, {"ok": False, "error": "no_pending_action"})
                else:
                    state.pending = None
                    _answer(state, call, {"ok": True, "message": "Pending action cancelled."})
            elif guarded and call.name == CONFIRM_TOOL:
                quote = _norm(str(call.arguments.get("user_quote", ""))).strip()
                if state.pending is None:
                    _answer(state, call, {"ok": False, "error": "no_pending_action"})
                elif not quote or f" {quote} " not in _norm(_latest_user_text(state)):
                    _answer(state, call, {"ok": False, "error": "confirmation_not_found",
                                          "message": "user_quote must be copied from the customer's latest message; "
                                                     "ask the customer to confirm explicitly."})
                else:
                    state.trigger, state.unanswered = Trigger("confirm", call), call
                    break
            elif call.name not in deps.defs:
                _answer(state, call, {"ok": False, "error": "unknown_tool", "message": f"no such tool: {call.name}"})
            elif deps.defs[call.name].is_write:
                state.trigger, state.unanswered = Trigger("propose", call), call
                state.pending = None  # a new proposal supersedes any pending action
                break
            else:
                _answer(state, call, run_read(state, call))
        for call in remaining:  # calls after a trigger are not executed
            _answer(state, call, {"ok": False, "error": "skipped",
                                  "message": "Only one change is processed at a time; call again if still needed."})
        return state

    # ---------------------------------------------------------------- guard
    def guard(state: State) -> State:
        trig = state.trigger
        assert trig is not None
        if trig.kind == "confirm":
            assert state.pending is not None
            call, action = state.pending.call, state.pending.action
        else:
            call = trig.call
            try:
                action = WriteAction.from_tool_args(call.name, call.arguments)
            except (KeyError, ValueError, TypeError) as exc:
                state.action, state.decision = None, deny(f"malformed request: {exc}")
                _on_denied(state, trig)
                return state
        order = None
        if action.order_id:
            got = deps.tools.call("get_order", {"order_id": action.order_id})
            if got.get("ok"):
                order = _facts_from_get_order(got["order"])
        ctx = PolicyContext(deps.today, state.verified_customer_id, order)
        state.action, state.decision = action, check(action, ctx)
        if state.decision.kind == "deny":
            _on_denied(state, trig)
        return state

    def _on_denied(state: State, trig: Trigger) -> None:
        assert state.decision is not None
        state.pending = None
        state.consecutive_failures += 1
        _answer(state, trig.call, {"ok": False, "error": "denied_by_policy", "message": state.decision.reason})

    # -------------------------------------------------------------- confirm
    def confirm(state: State) -> State:
        trig, action = state.trigger, state.action
        assert trig is not None and action is not None
        desc = describe_action(action, trig.call.arguments)
        state.pending = PendingAction(trig.call, action, desc, state.turn)
        _answer(state, trig.call, {"ok": True, "status": "awaiting_customer_confirmation",
                                   "message": "The customer has been asked to confirm. Do nothing further this turn."})
        text = f"Before I go ahead, please confirm: {desc}. Reply 'yes' to confirm or 'no' to cancel."
        state.messages.append(Message("assistant", text))
        state.reply = text
        return state

    # -------------------------------------------------------------- execute
    def _record(state: State, call: ToolCall, raw: dict[str, Any], *, auto: bool, confirmed: bool) -> ActionRecord:
        audit = raw.pop("_audit", None)
        rec = ActionRecord(len(state.action_log) + 1, state.turn, call.name, dict(call.arguments),
                           raw.get("ok") is True, raw, audit, auto=auto, confirmed=confirmed)
        state.action_log.append(rec)
        return rec

    def execute(state: State) -> State:
        trig = state.trigger
        assert trig is not None
        if trig.kind == "confirm":
            assert state.pending is not None
            call, confirmed = state.pending.call, True
            state.pending = None
        else:
            call, confirmed = trig.call, False
        rec = _record(state, call, deps.tools.call(call.name, call.arguments), auto=False, confirmed=confirmed)
        state.last_record = rec
        state.consecutive_failures = 0 if rec.ok else state.consecutive_failures + 1
        _answer(state, trig.call, rec.result)
        if call.name == "transfer_to_human" and rec.ok:
            _finish_handoff(state, rec.result)
        return state

    def _finish_handoff(state: State, result: dict[str, Any]) -> None:
        state.handed_off, state.done, state.termination = True, True, "handoff"
        state.reply = (f"I've passed your request to a human agent (ticket #{result.get('ticket_id')}, "
                       f"priority {result.get('priority')}). They will follow up with you. Thank you for your patience.")
        state.messages.append(Message("assistant", state.reply))

    # -------------------------------------------------------------- log_crm
    def log_crm(state: State) -> State:
        rec = state.last_record
        assert rec is not None
        target = rec.args.get("order_id", "?")
        note = f"[auto] {rec.tool} executed on {target}: {_json({k: v for k, v in rec.args.items() if k != 'order_id'})}"
        _record(state, ToolCall("auto", "log_crm_note", {"note": note}), deps.tools.call("log_crm_note", {"note": note}),
                auto=True, confirmed=False)
        return state

    # -------------------------------------------------------------- handoff
    def handoff(state: State) -> State:
        if state.decision is not None and state.decision.kind == "needs_human":
            reason, priority = f"Needs human approval: {state.decision.reason}", "normal"
        else:
            reason, priority = f"{state.consecutive_failures} consecutive denied/failed write attempts", "normal"
        if state.action is not None and state.action.order_id:
            reason += f" (order {state.action.order_id})"
        args = {"reason": reason, "priority": priority}
        rec = _record(state, ToolCall("auto", "transfer_to_human", args), deps.tools.call("transfer_to_human", args),
                      auto=True, confirmed=False)
        if state.unanswered is not None:
            _answer(state, state.unanswered, {"ok": True, "status": "escalated_to_human",
                                              "message": "A ticket was created and the conversation is ending."})
        state.pending = None
        _finish_handoff(state, rec.result)
        return state

    return {"act": act, "guard": guard, "confirm": confirm, "execute": execute, "log_crm": log_crm, "handoff": handoff}


# ---------------------------------------------------------------- routers
def make_routers(deps: Deps) -> dict[str, Any]:
    guarded = deps.variant == "guarded"

    def after_act(state: State) -> str:
        if state.done:
            return END
        if state.trigger is not None:
            return "guard" if guarded else "execute"
        return END if state.reply is not None else "act"

    def after_guard(state: State) -> str:
        assert state.decision is not None and state.trigger is not None
        kind = state.decision.kind
        if kind == "needs_human":
            return "handoff"
        if kind == "deny":
            return "handoff" if state.consecutive_failures >= MAX_CONSECUTIVE_FAILURES else "act"
        if state.trigger.kind == "confirm":
            return "execute"
        return "confirm" if deps.defs[state.trigger.call.name].requires_confirmation else "execute"

    def after_execute(state: State) -> str:
        if state.done:
            return END
        if guarded and state.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            return "handoff"
        rec = state.last_record
        if guarded and rec is not None and rec.ok and deps.defs[rec.tool].requires_confirmation:
            return "log_crm"
        return "act"

    return {"act": after_act, "guard": after_guard, "execute": after_execute}
