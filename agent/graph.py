"""Declarative graph spec shared by the built-in runner and the LangGraph wiring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from agent.nodes import END, Deps, make_nodes, make_routers
from agent.state import State

NodeFn = Callable[[State], State]


@dataclass(frozen=True)
class GraphSpec:
    entry: str
    nodes: dict[str, NodeFn]
    static_edges: dict[str, str]  # node -> next node (or END)
    routes: dict[str, tuple[Callable[[State], str], tuple[str, ...]]]  # node -> (router, possible targets)


def build_spec(deps: Deps) -> GraphSpec:
    routers = make_routers(deps)
    return GraphSpec(
        entry="act",
        nodes=make_nodes(deps),
        static_edges={"confirm": END, "handoff": END, "log_crm": "act"},
        routes={
            "act": (routers["act"], ("guard", "execute", "act", END)),
            "guard": (routers["guard"], ("confirm", "execute", "handoff", "act")),
            "execute": (routers["execute"], ("log_crm", "handoff", "act", END)),
        },
    )


def validate_spec(spec: GraphSpec) -> None:
    """Every node has exactly one outgoing mechanism and every target exists."""
    names = set(spec.nodes)
    assert spec.entry in names
    assert set(spec.static_edges) | set(spec.routes) == names, "each node needs an edge or a router"
    assert not set(spec.static_edges) & set(spec.routes)
    targets = set(spec.static_edges.values()) | {t for _, ts in spec.routes.values() for t in ts}
    assert targets <= names | {END}, targets - names - {END}
