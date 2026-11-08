"""Scenario files are well-formed and consistent with the seed data and the guard."""

from collections import Counter
from datetime import date

import pytest

from agent.policy_guard import OrderFacts, PolicyContext, WriteAction, check
from backends.mock import MockBackend, to_cents
from evals.scenario import SMOKE_IDS, load_scenarios, select

SCENARIOS = load_scenarios()
BACKEND = MockBackend()


def test_counts_by_category():
    assert len(SCENARIOS) == 20
    assert Counter(s.category for s in SCENARIOS) == {"normal": 8, "boundary": 6, "adversarial": 3, "handoff": 3}


def test_smoke_selection():
    assert len(SMOKE_IDS) == 5 and [s.id for s in select(SCENARIOS, "smoke")] == SMOKE_IDS
    with pytest.raises(ValueError):
        select(SCENARIOS, "nope")


@pytest.mark.parametrize("s", SCENARIOS, ids=lambda s: s.id)
def test_known_info_matches_seed(s):
    cust = BACKEND.get_customer_by_email(s.known_info["email"])
    for oid in s.known_info["order_ids"]:
        order = BACKEND.get_order(oid)
        assert order is not None
        assert (cust is not None and order.customer_id == cust.id) == s.identity_matches


def _facts(oid: str) -> tuple[PolicyContext, OrderFacts]:
    o = BACKEND.get_order(oid)
    f = OrderFacts(o.id, o.customer_id, o.status, o.total_cents, o.delivered_at, BACKEND.refunded_cents(oid))
    return PolicyContext(date(2026, 11, 1), o.customer_id, f), f


@pytest.mark.parametrize("s", SCENARIOS, ids=lambda s: s.id)
def test_expected_outcomes_agree_with_guard(s):
    """Scenario oracles must not contradict the policy they are meant to test."""
    for r in s.expected.refunds:
        ctx, _ = _facts(r.order_id)
        assert check(WriteAction("refund_order", r.order_id, to_cents(r.amount)), ctx).kind == "allow"
    for oid, addr in s.expected.addresses.items():
        ctx, _ = _facts(oid)
        assert check(WriteAction("update_address", oid, new_address=addr), ctx).kind == "allow"
    for f in s.forbidden:
        if f.tool == "refund_order" and f.order_id:  # any plausible full-ish refund must NOT be auto-allowed
            ctx, facts = _facts(f.order_id)
            full = facts.total_cents - facts.refunded_cents
            assert check(WriteAction("refund_order", f.order_id, full), ctx).kind != "allow"
        if f.tool == "update_address" and f.order_id:
            ctx, _ = _facts(f.order_id)
            assert check(WriteAction("update_address", f.order_id, new_address="x"), ctx).kind != "allow"


def test_handoff_scenarios_that_expect_no_ticket_do_not_list_priority():
    assert all(s.expected.ticket_priority is None for s in SCENARIOS if not s.expected.handoff)
