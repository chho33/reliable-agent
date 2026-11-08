"""Convenience assembly: backend + toolbox + in-process tools + agent."""

from __future__ import annotations

from pathlib import Path

from agent.nodes import Deps
from agent.runner import Agent
from agent.state import Variant
from backends.mock import MockBackend
from llm.types import ChatModel
from tools.client import InProcessToolClient, ToolClient
from tools.toolbox import Toolbox

POLICY_PATH = Path(__file__).resolve().parents[1] / "policy.md"


def load_policy(path: Path = POLICY_PATH) -> str:
    return path.read_text()


def build_agent(
    chat: ChatModel,
    variant: Variant,
    *,
    backend: MockBackend | None = None,
    tools: ToolClient | None = None,
    engine: str = "builtin",
    max_turn_steps: int = 8,
    max_episode_steps: int = 60,
) -> tuple[Agent, MockBackend]:
    """Create an agent over a fresh seeded backend (or the given backend/tool client)."""
    backend = backend or MockBackend()
    tools = tools or InProcessToolClient(Toolbox(backend))
    deps = Deps(chat, tools, load_policy(), backend.clock.today(), variant, max_turn_steps, max_episode_steps)
    return Agent(deps, engine=engine), backend
