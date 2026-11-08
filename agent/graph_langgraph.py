"""Thin LangGraph wiring over the same node functions the built-in runner uses.

LangGraph is imported lazily so the rest of the package works without it. NOT exercised
in the offline sandbox this repo was written in; ``tests/test_langgraph.py`` runs it
only when langgraph is installed.

The graph state is a one-key TypedDict holding our mutable ``State`` object; each node
wrapper runs the plain node function (which mutates the state) and returns it under that
key. This keeps all logic in ``agent.nodes`` and the wiring obviously correct.
"""

from __future__ import annotations

from typing import Any, TypedDict

from agent.graph import GraphSpec
from agent.nodes import END as OUR_END


class GraphState(TypedDict):
    state: Any  # agent.state.State


def compile_graph(spec: GraphSpec) -> Any:
    from langgraph.graph import END, START, StateGraph

    def lg(target: str) -> str:
        return END if target == OUR_END else target

    def wrap(fn: Any) -> Any:
        def node(gs: GraphState) -> dict[str, Any]:
            return {"state": fn(gs["state"])}

        return node

    graph = StateGraph(GraphState)
    for name, fn in spec.nodes.items():
        graph.add_node(name, wrap(fn))
    graph.add_edge(START, spec.entry)
    for src, dst in spec.static_edges.items():
        graph.add_edge(src, lg(dst))
    for src, (router, targets) in spec.routes.items():
        graph.add_conditional_edges(src, lambda gs, r=router: r(gs["state"]), {t: lg(t) for t in targets})
    return graph.compile()
