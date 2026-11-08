"""Injected clock so "today" is fixed and every episode sees the same calendar."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol

SEED_TODAY = date(2026, 11, 1)


class Clock(Protocol):
    def today(self) -> date: ...


@dataclass(frozen=True)
class FixedClock:
    """A clock that always returns the same date (default: the seed date)."""

    fixed: date = SEED_TODAY

    def today(self) -> date:
        return self.fixed
