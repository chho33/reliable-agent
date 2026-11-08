"""Deterministic mock e-commerce + CRM backend (SQLite)."""

from backends.mock.clock import SEED_TODAY, Clock, FixedClock
from backends.mock.money import fmt_usd, to_cents
from backends.mock.store import Customer, MockBackend, Order

__all__ = ["SEED_TODAY", "Clock", "FixedClock", "Customer", "MockBackend", "Order", "fmt_usd", "to_cents"]
