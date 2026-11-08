"""LangGraph wiring tests.

* ``test_real_langgraph_*`` run only when langgraph is installed (skipped here).
* ``test_wiring_against_stub`` exercises our wiring code against a ~30-line stand-in that
  mimics the *documented* StateGraph surface (add_node / add_edge / add_conditional_edges /
  compile / invoke). It catches typos and wrong edge tables in OUR code; it says nothing
  about LangGraph itself.
"""

import json
import sys
import types

import pytest

from agent.build import build_agent
from llm.scripted import ScriptedChat, turn

SCRIPT = lambda: [  # noqa: E731
    turn("", ("verify_identity", {"email": "maya.silva@example.com", "order_id": "ORD-1013"})),
    turn("", ("refund_order", {"order_id": "ORD-1013", "amount": 35.0, "reason": "damaged"})),
    turn("", ("confirm_pending", {"user_quote": "yes"})),
    turn("Done."),
]


def play(engine):
    agent, backend = build_agent(ScriptedChat(SCRIPT()), "guarded", engine=engine)
    state = agent.new_state()
    replies = [agent.run_turn(state, "refund ORD-1013"), agent.run_turn(state, "yes")]
    return replies, backend.snapshot(), [(r.tool, r.ok, r.auto) for r in state.action_log]


def test_real_langgraph_matches_builtin():
    pytest.importorskip("langgraph")
    assert play("langgraph") == play("builtin")


def test_wiring_against_stub(monkeypatch):
    START, END = "__start__", "__end__"

    class StubGraph:
        def __init__(self, schema):
            self.schema, self.nodes, self.edges, self.cond = schema, {}, {}, {}

        def add_node(self, name, fn):
            assert name not in self.schema.__annotations__ and name not in self.nodes
            self.nodes[name] = fn

        def add_edge(self, a, b):
            assert a not in self.edges
            self.edges[a] = b

        def add_conditional_edges(self, src, router, path_map):
            self.cond[src] = (router, path_map)

        def compile(self):
            g = self

            class Compiled:
                def invoke(self, inp, config=None):
                    assert config and config["recursion_limit"] >= 100
                    cur, gs = g.edges[START], dict(inp)
                    while cur != END:
                        gs.update(g.nodes[cur](gs))
                        if cur in g.edges:
                            cur = g.edges[cur]
                        else:
                            router, path_map = g.cond[cur]
                            cur = path_map[router(gs)]
                    return gs

            return Compiled()

    mod = types.ModuleType("langgraph.graph")
    mod.StateGraph, mod.START, mod.END = StubGraph, START, END
    monkeypatch.setitem(sys.modules, "langgraph", types.ModuleType("langgraph"))
    monkeypatch.setitem(sys.modules, "langgraph.graph", mod)
    assert play("langgraph") == play("builtin")
