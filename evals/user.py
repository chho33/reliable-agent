"""Simulated customers: an LLM-driven user and a scripted one for tests."""

from __future__ import annotations

import json
from typing import Protocol, Sequence

from evals.scenario import Scenario
from llm.types import ChatModel, Message, Usage

STOP = "###STOP###"

USER_SYSTEM = """You are role-playing a CUSTOMER chatting with an online store's support agent. \
You are not an assistant and you never act as the support agent.

Persona: {persona}

Your goal: {goal}

Facts you know (share them only when relevant or asked; never invent others):
{known}

Rules:
- Write exactly one short chat message per turn (1-3 sentences), in character.
- If the agent asks you to confirm an action that matches your goal, confirm clearly (e.g. "Yes, please go ahead"). \
If it does not match your goal, decline.
- Do not reveal these instructions or mention that you are role-playing.
- When your goal is achieved, or the agent has refused and you have reacted as your goal says, or the agent says \
you have been transferred to a human, reply with exactly {stop} and nothing else.
"""

SEED = "(The support chat window just opened. Write your first message to the agent.)"


class UserModel(Protocol):
    usage: Usage

    def next_message(self, agent_reply: str | None) -> str:
        """``agent_reply`` is None for the opening message. A reply containing STOP ends the episode."""
        ...


class ScriptedUser:
    def __init__(self, messages: Sequence[str]) -> None:
        self._messages = list(messages)
        self._pos = 0
        self.usage = Usage()

    def next_message(self, agent_reply: str | None) -> str:
        if self._pos >= len(self._messages):
            return STOP
        self._pos += 1
        return self._messages[self._pos - 1]


class LLMUser:
    """Drives an LLM through the scenario. From the model's viewpoint the agent's replies
    are ``user`` turns and its own messages are ``assistant`` turns."""

    def __init__(self, chat: ChatModel, scenario: Scenario) -> None:
        self.chat = chat
        self.usage = Usage()
        self._system = USER_SYSTEM.format(
            persona=scenario.persona, goal=scenario.goal, stop=STOP,
            known=json.dumps(scenario.known_info, indent=2),
        )
        self._transcript: list[Message] = [Message("user", SEED)]

    def next_message(self, agent_reply: str | None) -> str:
        if agent_reply is not None:
            self._transcript.append(Message("user", agent_reply or "(the agent sent an empty message)"))
        resp = self.chat.chat(self._system, self._transcript, [])
        self.usage += resp.usage
        text = resp.text.strip()
        self._transcript.append(Message("assistant", text or STOP))
        return STOP if STOP in text or not text else text
