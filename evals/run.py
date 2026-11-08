"""Eval CLI.

    python -m evals.run --variant both --agent-model anthropic:<id> --user-model anthropic:<id> \
        --k 4 --scenarios all --out results/

Runs scenario x variant x trial episodes (each in a fresh seeded world), writes
episodes.csv, transcripts.json, summary.json and results.md, and with --update-readme
fills the README results block.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Sequence

from evals.episode import EpisodeResult, run_episode
from evals.report import build_summary, update_readme, write_outputs
from evals.scenario import Scenario, load_scenarios, select
from evals.user import LLMUser, UserModel
from llm.factory import cost_usd, load_config, load_dotenv, make_chat, resolve
from llm.types import ChatModel, Usage


@dataclass(frozen=True)
class EpisodeSpec:
    scenario: Scenario
    variant: str
    trial: int


AgentFactory = Callable[[EpisodeSpec], ChatModel]
UserFactory = Callable[[EpisodeSpec], UserModel]


def run_experiment(
    scenarios: Sequence[Scenario],
    variants: Sequence[str],
    trials: int,
    agent_factory: AgentFactory,
    user_factory: UserFactory,
    *,
    engine: str = "builtin",
    workers: int = 1,
    max_steps: int = 30,
    agent_cost: Callable[[Usage], float | None] = lambda u: None,
    user_cost: Callable[[Usage], float | None] = lambda u: None,
    progress: Callable[[EpisodeResult], None] | None = None,
) -> list[EpisodeResult]:
    specs = [EpisodeSpec(s, v, t) for s in scenarios for v in variants for t in range(trials)]

    def one(spec: EpisodeSpec) -> EpisodeResult:
        res = run_episode(
            spec.scenario, spec.variant, spec.trial, agent_factory(spec), user_factory(spec),  # type: ignore[arg-type]
            engine=engine, max_steps=max_steps, agent_cost=agent_cost, user_cost=user_cost,
        )
        if progress:
            progress(res)
        return res

    if workers <= 1:
        return [one(s) for s in specs]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, specs))  # map preserves input order


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m evals.run", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", choices=["guarded", "control", "both"], default="both")
    p.add_argument("--agent-model", default=None, help="provider:model-id (default: $AGENT_MODEL)")
    p.add_argument("--user-model", default=None, help="provider:model-id (default: $USER_MODEL)")
    p.add_argument("--k", type=int, default=4, help="k in pass^k")
    p.add_argument("--trials", type=int, default=None, help="trials per scenario (default: k)")
    p.add_argument("--scenarios", default="all", help="all | smoke | comma-separated ids")
    p.add_argument("--out", default="results")
    p.add_argument("--engine", choices=["builtin", "langgraph"], default="builtin")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    p.add_argument("--update-readme", action="store_true", help="write the results table into README.md")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    args = parse_args(argv)
    agent_spec = args.agent_model or os.environ.get("AGENT_MODEL")
    user_spec = args.user_model or os.environ.get("USER_MODEL")
    if not agent_spec or not user_spec:
        print("error: set --agent-model/--user-model or AGENT_MODEL/USER_MODEL (format provider:model-id)", file=sys.stderr)
        return 2
    trials = args.trials or args.k
    if trials < args.k:
        print("error: --trials must be >= --k", file=sys.stderr)
        return 2

    config = load_config()
    agent_mc, user_mc = resolve(agent_spec, config), resolve(user_spec, config)
    scenarios = select(load_scenarios(), args.scenarios)
    variants = ["guarded", "control"] if args.variant == "both" else [args.variant]
    done = []

    def progress(r: EpisodeResult) -> None:
        done.append(r)
        flag = "ok " if r.success else "FAIL"
        print(f"[{len(done)}] {flag} {r.scenario_id} {r.variant} #{r.trial} violating={r.violating} {r.error}", flush=True)

    print(f"{len(scenarios)} scenarios x {len(variants)} variants x {trials} trials", flush=True)
    episodes = run_experiment(
        scenarios, variants, trials,
        agent_factory=lambda spec: make_chat(agent_spec, config),
        user_factory=lambda spec: LLMUser(make_chat(user_spec, config), spec.scenario),
        engine=args.engine, workers=args.workers, max_steps=args.max_steps,
        agent_cost=partial(cost_usd, agent_mc), user_cost=partial(cost_usd, user_mc), progress=progress,
    )
    meta = {"date": dt.date.today().isoformat(), "agent_model": agent_spec, "user_model": user_spec,
            "engine": args.engine, "n_scenarios": len(scenarios), "trials": trials, "variants": variants}
    summary = build_summary(episodes, args.k, meta, n_boot=args.bootstrap, seed=args.seed)
    markdown = write_outputs(Path(args.out), episodes, summary)
    if args.update_readme:
        update_readme(markdown)
    print(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
