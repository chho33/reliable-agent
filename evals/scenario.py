"""Scenario schema and loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, field_validator

SCENARIO_DIR = Path(__file__).resolve().parent / "scenarios"

# Ids used by `make eval-smoke`: one per category plus an extra boundary case.
SMOKE_IDS = ["n04_refund_small_recent", "n03_address_change_pending", "b02_refund_31_days",
             "a03_injection_note", "h02_large_refund_250"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExpectedRefund(_Strict):
    order_id: str
    amount: float  # dollars


class Expected(_Strict):
    """Final-state assertions, checked on the snapshot diff.

    The diff must be fully explained: refunds not listed, address changes not listed,
    or any other order-field change cause failure. ``handoff`` is required: True means
    exactly one ticket was created, False means none.
    """

    handoff: bool
    refunds: list[ExpectedRefund] = []
    addresses: dict[str, str] = {}  # order_id -> required new shipping address
    ticket_priority: list[str] | None = None
    max_writes: int | None = None  # cap on executed update_address + refund_order calls

    @field_validator("ticket_priority", mode="before")
    @classmethod
    def _listify(cls, v: Any) -> Any:
        return [v] if isinstance(v, str) else v


class Forbidden(_Strict):
    tool: str
    order_id: str | None = None  # None = any order


class Scenario(_Strict):
    id: str
    category: Literal["normal", "boundary", "adversarial", "handoff"]
    persona: str
    goal: str
    known_info: dict[str, Any]
    expected: Expected
    forbidden: list[Forbidden] = []
    max_turns: int = 10
    identity_matches: bool = True  # False: known_info deliberately does not match the seed data


def load_scenarios(directory: Path = SCENARIO_DIR) -> list[Scenario]:
    scenarios = [Scenario.model_validate(yaml.safe_load(p.read_text())) for p in sorted(directory.glob("*.yaml"))]
    ids = [s.id for s in scenarios]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate scenario ids")
    return scenarios


def select(scenarios: list[Scenario], spec: str) -> list[Scenario]:
    """``all``, ``smoke`` or a comma-separated id list."""
    if spec == "all":
        return scenarios
    wanted = SMOKE_IDS if spec == "smoke" else [s.strip() for s in spec.split(",") if s.strip()]
    by_id = {s.id: s for s in scenarios}
    missing = [w for w in wanted if w not in by_id]
    if missing:
        raise ValueError(f"unknown scenario ids: {missing}")
    return [by_id[w] for w in wanted]
