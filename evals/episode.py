"""Run one episode: simulated user <-> agent over a fresh seeded world, then check it."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Sequence

from agent.build import build_agent
from agent.state import State, Variant
from evals.checker import CheckResult, check_episode
from evals.scenario import Scenario
from evals.user import STOP, UserModel
from llm.types import ChatModel, ChatResponse, Message, ToolSpec, Usage

CostFn = Callable[[Usage], "float | None"]


class TimedChat:
    """Wraps a ChatModel to accumulate time spent in model calls."""

    def __init__(self, inner: ChatModel) -> None:
        self.inner, self.model, self.seconds = inner, inner.model, 0.0

    def chat(self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> ChatResponse:
        t0 = time.perf_counter()
        try:
            return self.inner.chat(system, messages, tools)
        finally:
            self.seconds += time.perf_counter() - t0


@dataclass
class EpisodeResult:
    scenario_id: str
    category: str
    variant: str
    trial: int
    success: bool
    violating: bool
    n_policy_violations: int
    n_forbidden: int
    handoff_expected: bool
    handoff_happened: bool
    handoff_correct: bool
    n_writes: int
    turns: int
    steps: int
    termination: str
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    user_cost_usd: float | None
    latency_s: float
    agent_latency_s: float
    error: str
    failed_assertions: str
    # Not written to CSV:
    detail: dict = field(default_factory=dict, repr=False)

    def csv_row(self) -> dict:
        row = asdict(self)
        row.pop("detail")
        return row


def run_episode(
    scenario: Scenario,
    variant: Variant,
    trial: int,
    agent_chat: ChatModel,
    user: UserModel,
    *,
    engine: str = "builtin",
    max_steps: int = 30,
    agent_cost: CostFn = lambda u: None,
    user_cost: CostFn = lambda u: None,
) -> EpisodeResult:
    chat = TimedChat(agent_chat)
    agent, backend = build_agent(chat, variant, engine=engine, max_steps=max_steps)
    initial = backend.snapshot()
    state: State = agent.new_state()
    error, termination, turns = "", "", 0
    t0 = time.perf_counter()
    try:
        message = user.next_message(None)
        termination = "max_turns"
        for _ in range(scenario.max_turns):
            if STOP in message:
                termination = "user_stop"
                break
            reply = agent.run_turn(state, message)
            turns += 1
            if state.done:
                termination = state.termination or "done"
                break
            message = user.next_message(reply)
    except Exception as exc:  # provider/network failure: record, count as a failed episode
        error, termination = f"{type(exc).__name__}: {exc}"[:300], "error"
    latency = time.perf_counter() - t0

    result: CheckResult = check_episode(scenario, initial, backend.snapshot(), state.action_log)
    return EpisodeResult(
        scenario_id=scenario.id, category=scenario.category, variant=variant, trial=trial,
        success=result.success and not error, violating=result.violating,
        n_policy_violations=len(result.policy_violations), n_forbidden=len(result.forbidden_hits),
        handoff_expected=scenario.expected.handoff, handoff_happened=result.handoff_happened,
        handoff_correct=result.handoff_correct, n_writes=result.n_writes, turns=turns, steps=state.steps,
        termination=termination, input_tokens=state.usage.input_tokens, output_tokens=state.usage.output_tokens,
        cost_usd=agent_cost(state.usage), user_cost_usd=user_cost(user.usage), latency_s=round(latency, 3),
        agent_latency_s=round(chat.seconds, 3), error=error, failed_assertions=" | ".join(result.failed_assertions),
        detail={"violations": result.policy_violations, "forbidden": result.forbidden_hits,
                "transcript": [(m.role, m.content) for m in state.messages if m.role != "tool" and m.content]},
    )
