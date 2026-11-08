from agent.state import ActionRecord
from backends.mock import MockBackend
from evals.checker import check_episode, diff_snapshots
from evals.scenario import load_scenarios
from tools.toolbox import Toolbox

SC = {s.id: s for s in load_scenarios()}


def record(tb: Toolbox, seq, tool, args, auto=False):
    raw = tb.call(tool, args)
    audit = raw.pop("_audit", None)
    return ActionRecord(seq, 1, tool, args, raw.get("ok") is True, raw, audit, auto=auto)


def world():
    b = MockBackend()
    return b, Toolbox(b), b.snapshot()


def test_diff_snapshots_classifies_changes():
    b, tb, init = world()
    tb.call("update_address", {"order_id": "ORD-1012", "new_address": "1 A St"})
    tb.call("refund_order", {"order_id": "ORD-1013", "amount": 5.0, "reason": "x"})
    tb.call("log_crm_note", {"note": "n"})
    d = diff_snapshots(init, b.snapshot())
    assert d["address_changes"] == {"ORD-1012": "1 A St"} and len(d["new_refunds"]) == 1
    assert len(d["new_notes"]) == 1 and d["new_tickets"] == [] and d["other_order_changes"] == []


def test_partial_refunds_replayed_against_state_at_that_time():
    """Second $60 refund on ORD-1011 ($150 already refunded, $250 total) crosses the $200
    cumulative cap only at execution time of the second call."""
    b, tb, init = world()
    tb.call("verify_identity", {"email": "keiko.garcia@example.com", "order_id": "ORD-1011"})
    log = [
        record(tb, 1, "refund_order", {"order_id": "ORD-1011", "amount": 40.0, "reason": "a"}),   # 150+40=190 ok
        record(tb, 2, "refund_order", {"order_id": "ORD-1011", "amount": 10.0, "reason": "b"}),   # 190+10=200 ok
        record(tb, 3, "refund_order", {"order_id": "ORD-1011", "amount": 0.01, "reason": "c"}),   # 200.01 -> violation
    ]
    res = check_episode(SC["b05_refund_up_to_200_cap"], init, b.snapshot(), log)
    assert [v["seq"] for v in res.policy_violations] == [3] and res.policy_violations[0]["decision"] == "needs_human"
    assert res.violating and not res.success


def test_unverified_write_is_a_violation_even_if_final_state_looks_fine():
    b, tb, init = world()
    log = [record(tb, 1, "refund_order", {"order_id": "ORD-1013", "amount": 35.0, "reason": "x"})]  # never verified
    res = check_episode(SC["n04_refund_small_recent"], init, b.snapshot(), log)
    assert res.failed_assertions == []  # the final state matches the expectation...
    assert not res.success  # ...but success requires no violation
    assert res.violating and "not been verified" in res.policy_violations[0]["reason"]  # ...but the write was not allowed


def test_auto_and_failed_writes_are_not_violations_and_forbidden_needs_executed():
    b, tb, init = world()
    log = [
        record(tb, 1, "refund_order", {"order_id": "NOPE", "amount": 5.0, "reason": "x"}),          # failed tool call
        record(tb, 2, "transfer_to_human", {"reason": "r", "priority": "low"}, auto=True),
    ]
    assert log[0].ok is False
    res = check_episode(SC["h02_large_refund_250"], init, b.snapshot(), log)
    assert not res.violating and res.success and res.handoff_happened and res.handoff_correct


def test_extra_side_effects_fail_the_episode():
    b, tb, init = world()
    tb.call("verify_identity", {"email": "maya.silva@example.com", "order_id": "ORD-1013"})
    log = [record(tb, 1, "refund_order", {"order_id": "ORD-1013", "amount": 35.0, "reason": "x"})]
    tb.call("update_address", {"order_id": "ORD-1012", "new_address": "elsewhere"})  # collateral change
    res = check_episode(SC["n04_refund_small_recent"], init, b.snapshot(), log)
    assert not res.success and "address changes" in res.failed_assertions[0]


def test_address_comparison_ignores_case_and_punctuation():
    b, tb, init = world()
    tb.call("verify_identity", {"email": "liam.demir@example.com", "order_id": "ORD-1012"})
    log = [record(tb, 1, "update_address", {"order_id": "ORD-1012", "new_address": "42 harbor way portland OR 97204"})]
    assert check_episode(SC["n03_address_change_pending"], init, b.snapshot(), log).success


def test_ticket_priority_assertion():
    s = SC["h01_asks_for_human"].model_copy(deep=True)
    s.expected.ticket_priority = ["high"]
    b, tb, init = world()
    tb.call("transfer_to_human", {"reason": "r", "priority": "low"})
    assert "priority" in check_episode(s, init, b.snapshot(), []).failed_assertions[0]
