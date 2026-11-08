"""Agent graph behaviour with scripted models (fully offline)."""

import json

import pytest

from agent.build import build_agent
from llm.scripted import ScriptedChat, turn

EMAIL = "maya.silva@example.com"  # customer 13, owns ORD-1013 (delivered, $35.00)
VERIFY = ("verify_identity", {"email": EMAIL, "order_id": "ORD-1013"})
REFUND_OK = ("refund_order", {"order_id": "ORD-1013", "amount": 35.0, "reason": "damaged"})


def run(variant, script, user_msgs):
    chat = ScriptedChat(script)
    agent, backend = build_agent(chat, variant)
    state = agent.new_state()
    replies = []
    for m in user_msgs:
        if state.done:
            break
        replies.append(agent.run_turn(state, m))
    return agent, backend, state, chat, replies


def tool_results(state):
    return [json.loads(m.content) for m in state.messages if m.role == "tool"]


def test_guarded_happy_path_confirm_execute_and_auto_crm_note():
    script = [
        turn("", VERIFY), turn("", REFUND_OK),                     # turn 1: verify, propose refund -> confirm prompt
        turn("", ("confirm_pending", {"user_quote": "yes please"})),  # turn 2: customer says yes
        turn("Done, $35.00 refunded."),
    ]
    _, backend, state, chat, replies = run("guarded", script, ["refund my order ORD-1013", "yes please"])
    assert "please confirm: Refund $35.00 on order ORD-1013" in replies[0]
    assert replies[1] == "Done, $35.00 refunded."
    snap = backend.snapshot()
    assert [(r["order_id"], r["amount_cents"]) for r in snap["refunds"][-1:]] == [("ORD-1013", 3500)]
    assert len(snap["crm_notes"]) == 1 and snap["crm_notes"][0]["note"].startswith("[auto] refund_order")
    assert [(r.tool, r.auto, r.confirmed) for r in state.action_log] == [("refund_order", False, True), ("log_crm_note", True, False)]
    assert state.action_log[0].audit["order"]["refunded_cents"] == 0  # pre-state captured
    assert "_audit" not in json.dumps(tool_results(state))
    # pending tools only offered while an action is pending
    assert "confirm_pending" not in chat.requests[0]["tools"]
    assert "confirm_pending" in chat.requests[2]["tools"]
    assert "confirm_pending" not in chat.requests[3]["tools"]


def test_nothing_executes_before_confirmation():
    script = [turn("", VERIFY), turn("", REFUND_OK)]
    _, backend, state, _, _ = run("guarded", script, ["refund ORD-1013"])
    assert state.pending is not None and state.action_log == []
    assert backend.snapshot()["refunds"] == backend_seed_refunds()


def backend_seed_refunds():
    from backends.mock import MockBackend

    return MockBackend().snapshot()["refunds"]


def test_confirm_requires_quote_from_latest_user_message():
    script = [
        turn("", VERIFY), turn("", REFUND_OK),
        turn("", ("confirm_pending", {"user_quote": "I definitely said yes"})),  # not in the user's message
        turn("Could you confirm explicitly?"),
    ]
    _, backend, state, _, replies = run("guarded", script, ["refund ORD-1013", "hmm, what is the timeline?"])
    assert state.action_log == [] and state.pending is not None
    assert tool_results(state)[-1]["error"] == "confirmation_not_found"
    assert replies[-1] == "Could you confirm explicitly?"


def test_cancel_pending_clears_and_nothing_executes():
    script = [turn("", VERIFY), turn("", REFUND_OK), turn("", ("cancel_pending", {"reason": "no"})), turn("Okay, cancelled.")]
    _, _, state, _, replies = run("guarded", script, ["refund ORD-1013", "no, never mind"])
    assert state.pending is None and state.action_log == [] and replies[-1] == "Okay, cancelled."


def test_guarded_deny_explains_then_second_deny_hands_off():
    addr = ("update_address", {"order_id": "ORD-1004", "new_address": "1 Elm St"})  # shipped order (customer 4)
    v = ("verify_identity", {"email": "dana.kaya@example.com", "order_id": "ORD-1004"})
    script = [turn("", v), turn("", addr), turn("Sorry, shipped orders cannot be changed."), turn("", addr)]
    agent, backend, state, _, replies = run("guarded", script, ["change address on ORD-1004", "please try again"])
    assert tool_results(state)[1]["error"] == "denied_by_policy"
    assert state.handed_off and state.done and state.termination == "handoff"
    assert "ticket #1" in replies[-1]
    snap = backend.snapshot()
    assert snap["tickets"][0]["summary"].startswith("2 consecutive denied/failed write attempts")
    assert snap["orders"]["ORD-1004"]["shipping_address"] != "1 Elm St"


def test_success_resets_strike_counter():
    addr_bad = ("update_address", {"order_id": "ORD-1004", "new_address": "1 Elm St"})
    script = [
        turn("", ("verify_identity", {"email": "dana.kaya@example.com", "order_id": "ORD-1004"})),
        turn("", addr_bad), turn("denied, sorry"),            # strike 1
        turn("", ("log_crm_note", {"note": "customer asked about address"})),  # success -> reset
        turn("", addr_bad), turn("denied again"),             # strike 1 again, no handoff
    ]
    _, _, state, _, _ = run("guarded", script, ["a", "b"])
    assert not state.handed_off and state.consecutive_failures == 1


def test_guarded_needs_human_hands_off_without_executing():
    refund = ("refund_order", {"order_id": "ORD-1002", "amount": 80.0, "reason": "late"})  # delivered 31 days ago
    script = [turn("", ("verify_identity", {"email": "bruno.ivanova@example.com", "order_id": "ORD-1002"})), turn("", refund)]
    _, backend, state, _, replies = run("guarded", script, ["refund ORD-1002"])
    assert state.handed_off and "human agent" in replies[0]
    snap = backend.snapshot()
    assert not [r for r in snap["refunds"] if r["order_id"] == "ORD-1002"]
    assert snap["tickets"][0]["customer_id"] == "C002" and "delivered 31 days ago" in snap["tickets"][0]["summary"]


def test_guarded_blocks_unverified_write():
    script = [turn("", REFUND_OK), turn("I need to verify you first.")]
    _, backend, state, _, _ = run("guarded", script, ["refund ORD-1013"])
    assert tool_results(state)[0]["error"] == "denied_by_policy" and state.action_log == []


def test_guarded_blocks_other_customers_order():
    script = [turn("", ("verify_identity", {"email": EMAIL, "order_id": "ORD-1013"})),
              turn("", ("refund_order", {"order_id": "ORD-1021", "amount": 10.0, "reason": "x"})), turn("no")]
    _, _, state, _, _ = run("guarded", script, ["refund ORD-1021"])
    assert tool_results(state)[1]["error"] == "denied_by_policy" and state.action_log == []


def test_guard_rechecked_at_execution_time():
    """State changes between proposal and confirmation: the re-run guard must catch it."""
    script = [turn("", VERIFY), turn("", REFUND_OK), turn("", ("confirm_pending", {"user_quote": "yes"})), turn("Sorry.")]
    chat = ScriptedChat(script)
    agent, backend = build_agent(chat, "guarded")
    state = agent.new_state()
    agent.run_turn(state, "refund ORD-1013")
    backend.add_refund("ORD-1013", 3500, "someone else refunded it meanwhile")  # now fully refunded
    agent.run_turn(state, "yes")
    assert state.action_log == [] and tool_results(state)[-1]["error"] == "denied_by_policy"
    assert "fully refunded" in tool_results(state)[-1]["message"]


def test_user_requested_human_creates_ticket_and_ends_in_both_variants():
    for variant in ("guarded", "control"):
        script = [turn("", ("transfer_to_human", {"reason": "customer wants a person", "priority": "high"}))]
        _, backend, state, _, replies = run(variant, script, ["let me talk to a human"])
        assert state.handed_off and state.done and backend.snapshot()["tickets"][0]["priority"] == "high"
        assert "human agent" in replies[0]


def test_control_executes_immediately_without_confirmation_or_crm_note():
    script = [turn("", VERIFY), turn("", REFUND_OK), turn("Refunded.")]
    _, backend, state, chat, replies = run("control", script, ["refund ORD-1013"])
    assert replies == ["Refunded."]
    snap = backend.snapshot()
    assert snap["refunds"][-1]["amount_cents"] == 3500 and snap["crm_notes"] == []
    assert all("confirm_pending" not in r["tools"] for r in chat.requests)


def test_multiple_tool_calls_with_a_write_skip_the_rest():
    script = [turn("", VERIFY, REFUND_OK, ("get_order", {"order_id": "ORD-1013"})), turn("ok")]
    _, _, state, _, _ = run("guarded", script, ["refund ORD-1013"])
    by_id = {m.tool_call_id: json.loads(m.content) for m in state.messages if m.role == "tool"}
    first = state.messages[1].tool_calls  # the assistant message with three calls
    assert by_id[first[0].id]["verified"] is True
    assert by_id[first[1].id]["status"] == "awaiting_customer_confirmation"
    assert by_id[first[2].id]["error"] == "skipped"
    # every tool_use id has exactly one result
    ids = [c.id for m in state.messages for c in m.tool_calls]
    assert sorted(ids) == sorted(m.tool_call_id for m in state.messages if m.role == "tool")


def test_malformed_arguments_do_not_crash_guarded():
    script = [turn("", VERIFY), turn("", ("refund_order", {"order_id": "ORD-1013", "amount": "lots", "reason": "x"})), turn("sorry")]
    _, _, state, _, _ = run("guarded", script, ["refund"])
    assert tool_results(state)[1]["error"] == "denied_by_policy" and "malformed" in tool_results(state)[1]["message"]


def test_step_cap_terminates_episode():
    loop = [turn("", ("get_order", {"order_id": "ORD-1013"}))] * 5
    chat = ScriptedChat(loop)
    agent, _ = build_agent(chat, "guarded", max_steps=3)
    state = agent.new_state()
    agent.run_turn(state, "hi")
    assert state.done and state.termination == "step_cap" and state.steps == 3


def test_system_prompt_contains_policy_date_and_pending_action():
    script = [turn("", VERIFY), turn("", REFUND_OK), turn("ok")]
    _, _, _, chat, _ = run("guarded", script, ["refund", "what's the status?"])
    first, last = chat.requests[0]["system"], chat.requests[-1]["system"]
    assert "Today's date is 2026-11-01" in first and "2. **Address changes.**" in first
    assert "Pending action awaiting" not in first and "Pending action awaiting" in last
    assert "Refund $35.00 on order ORD-1013" in last


def test_unknown_tool_and_ended_conversation():
    chat = ScriptedChat([turn("", ("transfer_to_human", {"reason": "r", "priority": "low"}))])
    agent, _ = build_agent(chat, "control")
    state = agent.new_state()
    agent.run_turn(state, "human please")
    with pytest.raises(RuntimeError):
        agent.run_turn(state, "hello?")
