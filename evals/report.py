"""Render results.md / summary.json / episodes.csv and (optionally) update the README."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Sequence

from evals.episode import EpisodeResult
from evals.metrics import (
    ScenarioStats, handoff_accuracy, mean_pass_hat_k, paired_delta_ci, scenario_stats, variant_summary, violation_rate,
)

README = Path(__file__).resolve().parents[1] / "README.md"
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def _pct(v: float) -> str:
    return f"{100 * v:.1f}%"


def _ci(m: dict) -> str:
    lo, hi = m["ci95"]
    return f"{_pct(m['value'])} [{_pct(lo)}, {_pct(hi)}]"


def build_summary(episodes: Sequence[EpisodeResult], k: int, meta: dict[str, Any], *, n_boot: int = 2000, seed: int = 0) -> dict:
    variants = sorted({e.variant for e in episodes})
    summary: dict[str, Any] = {"meta": {**meta, "k": k, "bootstrap": {"n_boot": n_boot, "seed": seed}}, "variants": {}}
    for v in variants:
        summary["variants"][v] = variant_summary([e for e in episodes if e.variant == v], k, n_boot=n_boot, seed=seed)
    if {"guarded", "control"} <= set(variants):
        g = scenario_stats([e for e in episodes if e.variant == "guarded"])
        c = scenario_stats([e for e in episodes if e.variant == "control"])
        summary["paired_delta_guarded_minus_control"] = {
            f"pass_{k}": paired_delta_ci(g, c, lambda s: mean_pass_hat_k(s, k), n_boot=n_boot, seed=seed),
            "violation_rate": paired_delta_ci(g, c, violation_rate, n_boot=n_boot, seed=seed),
            "handoff_accuracy": paired_delta_ci(g, c, handoff_accuracy, n_boot=n_boot, seed=seed),
        }
    return summary


def render_markdown(summary: dict, episodes: Sequence[EpisodeResult]) -> str:
    meta, k = summary["meta"], summary["meta"]["k"]
    lines = [
        f"_Run date: {meta.get('date')}. Agent model: `{meta.get('agent_model')}`. User-simulator model: "
        f"`{meta.get('user_model')}`. Engine: {meta.get('engine')}. Scenarios: {meta.get('n_scenarios')}, "
        f"trials per scenario: {meta.get('trials')}, k = {k}. Intervals are 95% bootstrap CIs over scenarios "
        f"(seed {summary['meta']['bootstrap']['seed']}); with this few scenarios they are wide._",
        "",
        f"| Variant | pass^1 | pass^{k} | Violating episodes | Handoff accuracy | Mean cost / episode | p95 latency (s) | Errored |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, m in summary["variants"].items():
        cost = "n/a (set prices in config/models.yaml)" if m["mean_cost_usd"] is None else f"${m['mean_cost_usd']:.4f}"
        lines.append(
            f"| {name} | {_ci(m['pass_1'])} | {_ci(m[f'pass_{k}'])} | {_ci(m['violation_rate'])} | "
            f"{_ci(m['handoff_accuracy'])} | {cost} | {m['p95_latency_s']:.1f} | {m['errored_episodes']} |"
        )
    delta = summary.get("paired_delta_guarded_minus_control")
    if delta:
        lines += ["", "Paired difference, guarded minus control (percentage points, 95% bootstrap CI over scenarios):", "",
                  "| Metric | Difference | 95% CI |", "|---|---|---|"]
        for name, (pt, lo, hi) in delta.items():
            lines.append(f"| {name} | {100 * pt:+.1f} pp | [{100 * lo:+.1f}, {100 * hi:+.1f}] |")
    lines += ["", "Per scenario (successes / trials; violating episodes in parentheses):", "",
              "| Scenario | Category | " + " | ".join(summary["variants"]) + " |",
              "|---|---|" + "---|" * len(summary["variants"])]
    per_variant = {v: {s.scenario_id: s for s in scenario_stats([e for e in episodes if e.variant == v])} for v in summary["variants"]}
    any_variant = next(iter(per_variant.values()))
    for sid, s in any_variant.items():
        cells = []
        for v in summary["variants"]:
            t: ScenarioStats = per_variant[v][sid]
            cells.append(f"{t.successes}/{t.n} ({t.violating})")
        lines.append(f"| {sid} | {s.category} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_outputs(out: Path, episodes: Sequence[EpisodeResult], summary: dict) -> str:
    out.mkdir(parents=True, exist_ok=True)
    rows = [e.csv_row() for e in episodes]
    with open(out / "episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "transcripts.json").write_text(json.dumps(
        [{"scenario": e.scenario_id, "variant": e.variant, "trial": e.trial, "success": e.success, **e.detail} for e in episodes], indent=1))
    md = render_markdown(summary, episodes)
    (out / "results.md").write_text(md)
    return md


def update_readme(markdown: str, readme: Path = README) -> None:
    text = readme.read_text()
    block = f"{START}\n{markdown}{END}"
    new, n = re.subn(re.escape(START) + r".*?" + re.escape(END), lambda _: block, text, flags=re.S)
    if n != 1:
        raise RuntimeError("README results markers not found")
    readme.write_text(new)
