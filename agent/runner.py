"""Built-in graph runner (no LangGraph needed) and the ``Agent`` facade."""

from __future__ import annotations

from typing import Any

from agent.graph import GraphSpec, build_spec, validate_spec
from agent.nodes import END, Deps
from agent.state import State
from llm.types import Message


def run_graph(spec: GraphSpec, state: State, *, max_visits: int = 500) -> State:
    """Execute nodes from the entry until END. ``max_visits`` is a safety net only;
    the real bound is ``Deps.max_steps`` on model calls."""
    node = spec.entry
    for _ in range(max_visits):
        state = spec.nodes[node](state)
        if node in spec.static_edges:
            node = spec.static_edges[node]
        else:
            node = spec.routes[node][0](state)
        if node == END:
            return state
    raise RuntimeError(f"graph exceeded {max_visits} node visits")


class Agent:
    """One conversation = one ``State``. ``run_turn`` consumes a user message and returns
    the text shown to the customer (a turn ends when the agent replies, asks for
    confirmation, or hands off)."""

    def __init__(self, deps: Deps, engine: str = "builtin") -> None:
        self.deps = deps
        self.spec = build_spec(deps)
        validate_spec(self.spec)
        self._lg: Any = None
        if engine == "langgraph":
            from agent.graph_langgraph import compile_graph  # lazy: langgraph is optional

            self._lg = compile_graph(self.spec)
        elif engine != "builtin":
            raise ValueError(f"unknown engine {engine!r}")

    def new_state(self) -> State:
        return State(variant=self.deps.variant)

    def run_turn(self, state: State, user_text: str) -> str:
        if state.done:
            raise RuntimeError("conversation has ended")
        state.turn += 1
        state.reply, state.trigger = None, None
        state.messages.append(Message("user", user_text))
        if self._lg is not None:
            self._lg.invoke({"state": state}, config={"recursion_limit": 1000})
        else:
            run_graph(self.spec, state)
        return state.reply or ""
