"""Deterministic policy guard: a pure function over (action, context).

No I/O, no clock reads, no LLM. Everything it needs arrives in ``PolicyContext``
(including "today"), so the same function is used (a) live by the guarded agent and
(b) by the offline checker to re-evaluate executed writes against the state captured
at execution time. Money is integer cents.

Policy summary (mirrors ``policy.md``):
  1. Identity: every write except ``transfer_to_human`` needs a verified customer, and
     order-targeting writes need an order owned by that customer.
  2. Address change: only while the order is ``pending``.
  3. Refund: auto-allowed only if delivered, delivered <= 30 days ago (inclusive),
     0 < amount <= remaining refundable, and cumulative refunds on the order <= $200.
     Over the cap or past the window -> ``needs_human``; invalid/ineligible -> ``deny``.
  "Cumulative" is per order (prior refunds on that order + this one).

Rules 4 (confirmation) and 6 (handoff) are graph behaviour, not guard logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Literal, Mapping

from backends.mock.money import to_cents

REFUND_WINDOW_DAYS = 30
REFUND_CUMULATIVE_CAP_CENTS = 20_000  # $200.00

WRITE_TOOLS = ("update_address", "refund_order", "log_crm_note", "transfer_to_human")


@dataclass(frozen=True)
class OrderFacts:
    """Authoritative order state as of the moment of the decision."""

    order_id: str
    customer_id: str
    status: str
    total_cents: int
    delivered_on: date | None
    refunded_cents: int  # sum of prior refunds on this order

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id, "customer_id": self.customer_id, "status": self.status,
            "total_cents": self.total_cents,
            "delivered_on": self.delivered_on.isoformat() if self.delivered_on else None,
            "refunded_cents": self.refunded_cents,
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> OrderFacts:
        return OrderFacts(
            order_id=d["order_id"], customer_id=d["customer_id"], status=d["status"],
            total_cents=int(d["total_cents"]),
            delivered_on=date.fromisoformat(d["delivered_on"]) if d.get("delivered_on") else None,
            refunded_cents=int(d["refunded_cents"]),
        )


@dataclass(frozen=True)
class PolicyContext:
    today: date
    verified_customer_id: str | None
    order: OrderFacts | None  # None if the target order is unknown / not accessible


@dataclass(frozen=True)
class WriteAction:
    """A proposed write, normalised. Unused fields are None."""

    tool: str
    order_id: str | None = None
    amount_cents: int | None = None
    new_address: str | None = None

    @staticmethod
    def from_tool_args(tool: str, args: Mapping[str, Any]) -> WriteAction:
        """Parse raw tool arguments. Raises ``ValueError`` on malformed input."""
        if tool == "update_address":
            return WriteAction(tool, order_id=str(args["order_id"]).strip(), new_address=str(args["new_address"]).strip())
        if tool == "refund_order":
            return WriteAction(tool, order_id=str(args["order_id"]).strip(), amount_cents=to_cents(args["amount"]))
        if tool in ("log_crm_note", "transfer_to_human"):
            return WriteAction(tool)
        raise ValueError(f"not a write tool: {tool}")


@dataclass(frozen=True)
class Decision:
    kind: Literal["allow", "deny", "needs_human"]
    reason: str = ""

    @property
    def allowed(self) -> bool:
        return self.kind == "allow"


def allow(reason: str = "") -> Decision:
    return Decision("allow", reason)


def deny(reason: str) -> Decision:
    return Decision("deny", reason)


def needs_human(reason: str) -> Decision:
    return Decision("needs_human", reason)


def _usd(cents: int) -> str:
    return f"${cents / 100:.2f}"


def check(action: WriteAction, ctx: PolicyContext) -> Decision:
    """Decide whether ``action`` may execute automatically."""
    if action.tool == "transfer_to_human":
        return allow("escalation is always available")
    if action.tool not in WRITE_TOOLS:
        return deny(f"unknown write tool {action.tool!r}")

    # Rule 1: identity.
    if ctx.verified_customer_id is None:
        return deny("customer identity has not been verified (email + order id)")
    if action.tool == "log_crm_note":
        return allow()

    order = ctx.order
    if order is None or order.order_id != action.order_id:
        return deny("order not found or not accessible")
    if order.customer_id != ctx.verified_customer_id:
        return deny("order does not belong to the verified customer")

    if action.tool == "update_address":
        # Rule 2.
        if not action.new_address:
            return deny("new address is empty")
        if order.status != "pending":
            return deny(f"address can only be changed on pending orders; this order is {order.status}")
        return allow()

    # Rule 3: refund.
    assert action.tool == "refund_order"
    if order.status != "delivered":
        return deny(f"only delivered orders can be refunded; this order is {order.status}")
    if order.delivered_on is None:
        return deny("order is marked delivered but has no delivery date")
    remaining = order.total_cents - order.refunded_cents
    if remaining <= 0:
        return deny("order has already been fully refunded")
    amount = action.amount_cents
    if amount is None or amount <= 0:
        return deny("refund amount must be greater than zero")
    if amount > remaining:
        return deny(f"refund {_usd(amount)} exceeds the remaining refundable amount {_usd(remaining)}")
    age_days = (ctx.today - order.delivered_on).days
    if age_days < 0:
        return deny("delivery date is in the future")
    if age_days > REFUND_WINDOW_DAYS:
        return needs_human(f"delivered {age_days} days ago; automatic refunds are limited to {REFUND_WINDOW_DAYS} days")
    cumulative = order.refunded_cents + amount
    if cumulative > REFUND_CUMULATIVE_CAP_CENTS:
        return needs_human(
            f"cumulative refunds on this order would be {_usd(cumulative)}, over the {_usd(REFUND_CUMULATIVE_CAP_CENTS)} automatic limit"
        )
    return allow()
