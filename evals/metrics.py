"""Metrics: pass^k, violation rate, handoff accuracy, cost, latency, bootstrap CIs.

pass^k (tau-bench): probability that ALL k i.i.d. trials of a task succeed. With ``c``
successes observed in ``n >= k`` trials of a task, the unbiased estimator is
C(c, k) / C(n, k); we average it over tasks. (With n == k it is the fraction of tasks
solved in every trial.) pass^1 = mean of c/n.

Bootstrap: resample *scenarios* with replacement (episodes within a scenario stay
together), recompute the statistic, take the 2.5 / 97.5 percentiles. Seeded.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import comb
from typing import Callable, Iterable, Sequence

import numpy as np

from evals.episode import EpisodeResult


@dataclass(frozen=True)
class ScenarioStats:
    scenario_id: str
    category: str
    n: int  # trials
    successes: int
    violating: int  # episodes with >= 1 policy violation or forbidden hit
    handoff_correct: int


def pass_hat_k(c: int, n: int, k: int) -> float:
    if k < 1 or n < k:
        raise ValueError(f"need n >= k >= 1 (n={n}, k={k})")
    if not 0 <= c <= n:
        raise ValueError(f"need 0 <= c <= n (c={c}, n={n})")
    return comb(c, k) / comb(n, k)


def scenario_stats(episodes: Iterable[EpisodeResult]) -> list[ScenarioStats]:
    groups: dict[str, list[EpisodeResult]] = defaultdict(list)
    for e in episodes:
        groups[e.scenario_id].append(e)
    return [
        ScenarioStats(
            sid, eps[0].category, len(eps), sum(e.success for e in eps), sum(e.violating for e in eps),
            sum(e.handoff_correct for e in eps),
        )
        for sid, eps in sorted(groups.items())
    ]


def mean_pass_hat_k(stats: Sequence[ScenarioStats], k: int) -> float:
    return float(np.mean([pass_hat_k(s.successes, s.n, k) for s in stats]))


def violation_rate(stats: Sequence[ScenarioStats]) -> float:
    return sum(s.violating for s in stats) / sum(s.n for s in stats)


def handoff_accuracy(stats: Sequence[ScenarioStats]) -> float:
    return sum(s.handoff_correct for s in stats) / sum(s.n for s in stats)


Stat = Callable[[Sequence[ScenarioStats]], float]


def bootstrap_ci(
    stats: Sequence[ScenarioStats], stat: Stat, *, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(stats), size=(n_boot, len(stats)))
    vals = [stat([stats[i] for i in row]) for row in idx]
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_delta_ci(
    a: Sequence[ScenarioStats], b: Sequence[ScenarioStats], stat: Stat, *, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05,
) -> tuple[float, float, float]:
    """(point, lo, hi) of stat(a) - stat(b), resampling the same scenarios for both variants."""
    if [s.scenario_id for s in a] != [s.scenario_id for s in b]:
        raise ValueError("paired comparison needs identical scenario sets")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    vals = [stat([a[i] for i in row]) - stat([b[i] for i in row]) for row in idx]
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return stat(a) - stat(b), float(lo), float(hi)


def p95(values: Sequence[float]) -> float:
    return float(np.percentile(values, 95)) if values else float("nan")


def variant_summary(episodes: Sequence[EpisodeResult], k: int, *, n_boot: int = 2000, seed: int = 0) -> dict:
    stats = scenario_stats(episodes)
    pk: Stat = lambda s: mean_pass_hat_k(s, k)  # noqa: E731
    p1: Stat = lambda s: mean_pass_hat_k(s, 1)  # noqa: E731
    costs = [e.cost_usd for e in episodes]
    return {
        "episodes": len(episodes),
        "scenarios": len(stats),
        "trials_per_scenario": min(s.n for s in stats),
        "errored_episodes": sum(1 for e in episodes if e.error),
        "pass_1": {"value": p1(stats), "ci95": bootstrap_ci(stats, p1, n_boot=n_boot, seed=seed)},
        f"pass_{k}": {"value": pk(stats), "ci95": bootstrap_ci(stats, pk, n_boot=n_boot, seed=seed)},
        "violation_rate": {"value": violation_rate(stats), "ci95": bootstrap_ci(stats, violation_rate, n_boot=n_boot, seed=seed)},
        "handoff_accuracy": {"value": handoff_accuracy(stats), "ci95": bootstrap_ci(stats, handoff_accuracy, n_boot=n_boot, seed=seed)},
        "mean_cost_usd": None if any(c is None for c in costs) else float(np.mean(costs)),
        "mean_input_tokens": float(np.mean([e.input_tokens for e in episodes])),
        "mean_output_tokens": float(np.mean([e.output_tokens for e in episodes])),
        "mean_turns": float(np.mean([e.turns for e in episodes])),
        "p95_latency_s": p95([e.latency_s for e in episodes]),
    }
