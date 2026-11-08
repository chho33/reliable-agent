"""Money helpers: amounts are integer cents internally."""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal, InvalidOperation


def to_cents(amount: float | int | str) -> int:
    """Convert a dollar amount to integer cents; reject more than 2 decimal places."""
    try:
        d = Decimal(str(amount))
    except InvalidOperation as exc:
        raise ValueError(f"not a number: {amount!r}") from exc
    if not d.is_finite():
        raise ValueError(f"not a finite number: {amount!r}")
    cents = (d * 100).quantize(Decimal(1), rounding=ROUND_HALF_UP)
    if cents != d * 100:
        raise ValueError(f"amount has more than 2 decimal places: {amount!r}")
    return int(cents)


def fmt_usd(cents: int) -> str:
    return f"{cents / 100:.2f}"
