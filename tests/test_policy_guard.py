"""Unit tests for the pure policy guard: every rule and boundary."""

from datetime import date, timedelta

import pytest

from agent.policy_guard import (
    OrderFacts, PolicyContext, WriteAction, check, REFUND_CUMULATIVE_CAP_CENTS,
)

TODAY = date(2026, 11, 1)
CUST = "C001"


def order(status="delivered", total=10_000, delivered_days_ago: int | None = 5, refunded=0, customer=CUST,
          oid="ORD-1") -> OrderFacts:
    delivered = TODAY - timedelta(days=delivered_days_ago) if delivered_days_ago is not None else None
    return OrderFacts(oid, customer, status, total, delivered, refunded)


def ctx(o: OrderFacts | None, verified: str | None = CUST) -> PolicyContext:
    return PolicyContext(today=TODAY, verified_customer_id=verified, order=o)


def refund(cents: int, oid="ORD-1") -> WriteAction:
    return WriteAction("refund_order", order_id=oid, amount_cents=cents)


def addr(oid="ORD-1", new="1 New St") -> WriteAction:
    return WriteAction("update_address", order_id=oid, new_address=new)


# ---- Rule 1: identity ----
def test_unverified_refund_denied():
    assert check(refund(1000), ctx(order(), verified=None)).kind == "deny"


def test_unverified_address_denied():
    assert check(addr(), ctx(order("pending", delivered_days_ago=None), verified=None)).kind == "deny"


def test_unverified_crm_note_denied():
    assert check(WriteAction("log_crm_note"), ctx(None, verified=None)).kind == "deny"


def test_verified_crm_note_allowed():
    assert check(WriteAction("log_crm_note"), ctx(None)).kind == "allow"


def test_wrong_customer_refund_denied():
    d = check(refund(1000), ctx(order(customer="C999")))
    assert d.kind == "deny" and "belong" in d.reason


def test_wrong_customer_address_denied():
    assert check(addr(), ctx(order("pending", delivered_days_ago=None, customer="C999"))).kind == "deny"


def test_missing_order_denied():
    assert check(refund(1000), ctx(None)).kind == "deny"


def test_order_id_mismatch_between_action_and_facts_denied():
    assert check(refund(1000, oid="ORD-2"), ctx(order(oid="ORD-1"))).kind == "deny"


def test_transfer_to_human_allowed_even_unverified():
    assert check(WriteAction("transfer_to_human"), ctx(None, verified=None)).kind == "allow"


def test_unknown_tool_denied():
    assert check(WriteAction("delete_everything"), ctx(order())).kind == "deny"


# ---- Rule 2: address change ----
def test_address_pending_allowed():
    assert check(addr(), ctx(order("pending", delivered_days_ago=None))).kind == "allow"


@pytest.mark.parametrize("status", ["shipped", "delivered", "cancelled"])
def test_address_non_pending_denied(status):
    d = check(addr(), ctx(order(status, delivered_days_ago=None)))
    assert d.kind == "deny" and status in d.reason


def test_address_empty_denied():
    assert check(addr(new=""), ctx(order("pending", delivered_days_ago=None))).kind == "deny"


# ---- Rule 3: refunds ----
def test_refund_simple_allowed():
    assert check(refund(5000), ctx(order())).kind == "allow"


def test_refund_exactly_30_days_allowed():
    assert check(refund(5000), ctx(order(delivered_days_ago=30))).kind == "allow"


def test_refund_31_days_needs_human():
    d = check(refund(5000), ctx(order(delivered_days_ago=31)))
    assert d.kind == "needs_human" and "31" in d.reason


def test_refund_delivered_today_allowed():
    assert check(refund(5000), ctx(order(delivered_days_ago=0))).kind == "allow"


def test_refund_future_delivery_denied():
    assert check(refund(5000), ctx(order(delivered_days_ago=-1))).kind == "deny"


@pytest.mark.parametrize("status", ["pending", "shipped", "cancelled"])
def test_refund_not_delivered_denied(status):
    assert check(refund(1000), ctx(order(status, delivered_days_ago=None))).kind == "deny"


def test_refund_delivered_without_date_denied():
    assert check(refund(1000), ctx(order(delivered_days_ago=None))).kind == "deny"


@pytest.mark.parametrize("cents", [0, -1, -5000])
def test_refund_non_positive_denied(cents):
    assert check(refund(cents), ctx(order())).kind == "deny"


def test_refund_amount_none_denied():
    assert check(WriteAction("refund_order", order_id="ORD-1"), ctx(order())).kind == "deny"


def test_refund_exactly_remaining_allowed():
    assert check(refund(4000), ctx(order(total=10_000, refunded=6000))).kind == "allow"


def test_refund_one_cent_over_remaining_denied():
    assert check(refund(4001), ctx(order(total=10_000, refunded=6000))).kind == "deny"


def test_refund_fully_refunded_denied():
    d = check(refund(100), ctx(order(total=5000, refunded=5000)))
    assert d.kind == "deny" and "fully refunded" in d.reason


def test_refund_exactly_200_cumulative_allowed():
    assert check(refund(20_000), ctx(order(total=20_000))).kind == "allow"


def test_refund_200_01_needs_human():
    assert REFUND_CUMULATIVE_CAP_CENTS == 20_000
    assert check(refund(20_001), ctx(order(total=25_000))).kind == "needs_human"


def test_refund_full_250_order_needs_human():
    assert check(refund(25_000), ctx(order(total=25_000))).kind == "needs_human"


def test_partial_refunds_summing_to_exactly_200_allowed():
    # $150 already refunded + $50 = $200.00
    assert check(refund(5000), ctx(order(total=25_000, refunded=15_000))).kind == "allow"


def test_partial_refunds_summing_over_200_needs_human():
    # $150 already refunded + $50.01 = $200.01
    assert check(refund(5001), ctx(order(total=25_000, refunded=15_000))).kind == "needs_human"


def test_amount_over_remaining_beats_cap_check():
    # 300 > remaining 250: invalid amount is a deny, not an escalation
    assert check(refund(30_000), ctx(order(total=25_000))).kind == "deny"


def test_old_and_over_cap_is_needs_human():
    assert check(refund(25_000), ctx(order(total=25_000, delivered_days_ago=45))).kind == "needs_human"


def test_injection_style_amount_999_denied_on_120_order():
    assert check(refund(99_900), ctx(order(total=12_000))).kind == "deny"


# ---- parsing ----
def test_from_tool_args_refund_parses_dollars_to_cents():
    a = WriteAction.from_tool_args("refund_order", {"order_id": " ORD-1 ", "amount": 35.5, "reason": "x"})
    assert a.amount_cents == 3550 and a.order_id == "ORD-1"


def test_from_tool_args_rejects_sub_cent_amounts():
    with pytest.raises(ValueError):
        WriteAction.from_tool_args("refund_order", {"order_id": "ORD-1", "amount": 1.001, "reason": "x"})


def test_decision_allowed_property():
    assert check(refund(100), ctx(order())).allowed
    assert not check(refund(0), ctx(order())).allowed
