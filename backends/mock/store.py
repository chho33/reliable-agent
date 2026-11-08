"""SQLite-backed mock store + CRM.

The store deliberately enforces only *integrity* (order exists, amount > 0 integer
cents, valid status values) and **no business policy**: like a service account behind
a real commerce API, it will happily record a $999 refund. Policy lives in
``agent/policy_guard.py`` so that the control variant can actually violate it.
Every instance is an independent in-memory database seeded identically.
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import date
from typing import Any

from backends.mock.clock import Clock, FixedClock
from backends.mock.money import fmt_usd, to_cents  # noqa: F401  (re-exported)
from backends.mock.schema import SCHEMA
from backends.mock.seed import seed as seed_db


@dataclass(frozen=True)
class Customer:
    id: str
    name: str
    email: str


@dataclass(frozen=True)
class Order:
    id: str
    customer_id: str
    status: str
    total_cents: int
    created_at: date
    delivered_at: date | None
    shipping_address: str
    note: str


def _order(row: sqlite3.Row) -> Order:
    return Order(
        id=row["id"], customer_id=row["customer_id"], status=row["status"],
        total_cents=row["total_cents"], created_at=date.fromisoformat(row["created_at"]),
        delivered_at=date.fromisoformat(row["delivered_at"]) if row["delivered_at"] else None,
        shipping_address=row["shipping_address"], note=row["note"],
    )


class MockBackend:
    """One fresh, seeded in-memory database per instance (i.e. per episode)."""

    def __init__(self, clock: Clock | None = None) -> None:
        self.clock: Clock = clock or FixedClock()
        # check_same_thread=False: the MCP server runs sync tools in worker threads.
        # Access is serialised by ``_lock``.
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.executescript(SCHEMA)
            seed_db(self._conn)

    # ---- reads ----
    def get_customer_by_email(self, email: str) -> Customer | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM customers WHERE lower(email) = lower(?)", (email.strip(),)
            ).fetchone()
        return Customer(**dict(row)) if row else None

    def get_customer(self, customer_id: str) -> Customer | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM customers WHERE id = ?", (customer_id,)).fetchone()
        return Customer(**dict(row)) if row else None

    def get_order(self, order_id: str) -> Order | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM orders WHERE id = ?", (order_id.strip(),)).fetchone()
        return _order(row) if row else None

    def list_orders(self, customer_id: str) -> list[Order]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM orders WHERE customer_id = ? ORDER BY id", (customer_id,)
            ).fetchall()
        return [_order(r) for r in rows]

    def refunded_cents(self, order_id: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(amount_cents), 0) AS s FROM refunds WHERE order_id = ?", (order_id,)
            ).fetchone()
        return int(row["s"])

    # ---- writes (integrity checks only; no policy) ----
    def set_shipping_address(self, order_id: str, address: str) -> None:
        with self._lock:
            cur = self._conn.execute("UPDATE orders SET shipping_address = ? WHERE id = ?", (address, order_id))
            if cur.rowcount != 1:
                raise KeyError(order_id)
            self._conn.commit()

    def add_refund(self, order_id: str, amount_cents: int, reason: str) -> int:
        with self._lock:
            if self._conn.execute("SELECT 1 FROM orders WHERE id = ?", (order_id,)).fetchone() is None:
                raise KeyError(order_id)
            cur = self._conn.execute(
                "INSERT INTO refunds (order_id, amount_cents, reason, created_at) VALUES (?,?,?,?)",
                (order_id, amount_cents, reason, self.clock.today().isoformat()),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def add_ticket(self, customer_id: str | None, summary: str, priority: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO tickets (customer_id, summary, priority, created_at) VALUES (?,?,?,?)",
                (customer_id, summary, priority, self.clock.today().isoformat()),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def add_crm_note(self, customer_id: str | None, note: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO crm_notes (customer_id, note, created_at) VALUES (?,?,?)",
                (customer_id, note, self.clock.today().isoformat()),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    # ---- audit ----
    def snapshot(self) -> dict[str, Any]:
        """Canonical dict of all mutable state (orders, refunds, tickets, crm_notes).

        Rows are plain dicts ordered by primary key, so two snapshots can be compared
        with ``==`` and diffed by the checker. Customers are immutable and omitted.
        """
        with self._lock:
            def rows(sql: str) -> list[dict[str, Any]]:
                return [dict(r) for r in self._conn.execute(sql).fetchall()]

            return {
                "orders": {r["id"]: r for r in rows("SELECT * FROM orders ORDER BY id")},
                "refunds": rows("SELECT * FROM refunds ORDER BY id"),
                "tickets": rows("SELECT * FROM tickets ORDER BY id"),
                "crm_notes": rows("SELECT * FROM crm_notes ORDER BY id"),
            }
