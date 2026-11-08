"""SQLite schema. Money is stored as integer cents to avoid float drift."""

SCHEMA = """
CREATE TABLE customers (
    id    TEXT PRIMARY KEY,
    name  TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE
);
CREATE TABLE orders (
    id               TEXT PRIMARY KEY,
    customer_id      TEXT NOT NULL REFERENCES customers(id),
    status           TEXT NOT NULL CHECK (status IN ('pending','shipped','delivered','cancelled')),
    total_cents      INTEGER NOT NULL CHECK (total_cents > 0),
    created_at       TEXT NOT NULL,          -- ISO date
    delivered_at     TEXT,                   -- ISO date, NULL unless delivered
    shipping_address TEXT NOT NULL,
    note             TEXT NOT NULL DEFAULT '' -- free text; UNTRUSTED customer-provided content
);
CREATE TABLE refunds (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id     TEXT NOT NULL REFERENCES orders(id),
    amount_cents INTEGER NOT NULL CHECK (amount_cents > 0),
    reason       TEXT NOT NULL,
    created_at   TEXT NOT NULL
);
CREATE TABLE tickets (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT REFERENCES customers(id),  -- NULL when the requester was never verified
    summary     TEXT NOT NULL,
    priority    TEXT NOT NULL CHECK (priority IN ('low','normal','high','urgent')),
    created_at  TEXT NOT NULL
);
CREATE TABLE crm_notes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT REFERENCES customers(id),
    note        TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
"""
