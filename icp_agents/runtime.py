"""Async runtime that lets agents work at the same time and talk to each other.

Each agent has its own inbox and runs its own loop. `send` routes a message
to one agent or the whole team. `ask` sends a request and suspends the
asking agent until the other agent replies (or a timeout passes), so an agent
can genuinely wait on a colleague's answer before it decides.

Every message is written to the transcript and shown live on the console.
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Callable

from . import log as logs
from .messages import BROADCAST, REPLY, REQUEST, AgentMessage

log = logs.get("messages")

if TYPE_CHECKING:
    from .agents.base import BaseAgent


class Runtime:
    def __init__(self, on_message: Callable[[AgentMessage], None] | None = None,
                 ask_timeout: float = 300.0, max_messages: int = 20_000):
        self.agents: dict[str, BaseAgent] = {}
        self.transcript: list[AgentMessage] = []
        self.on_message = on_message
        self.ask_timeout = ask_timeout
        self.max_messages = max_messages
        self._pending: dict[int, asyncio.Future] = {}
        self._tasks: list[asyncio.Task] = []
        self.finished = asyncio.Event()

    # --- wiring ---------------------------------------------------------------
    def register(self, agent: "BaseAgent") -> None:
        self.agents[agent.name] = agent

    def start(self) -> None:
        for agent in self.agents.values():
            self._tasks.append(asyncio.create_task(agent.run(), name=f"agent:{agent.name}"))

    async def stop(self, grace: float = 30.0) -> None:
        for fut in self._pending.values():  # nobody will answer now
            if not fut.done():
                fut.set_result(None)
        try:
            await asyncio.wait_for(asyncio.gather(*(a.drain() for a in self.agents.values())), grace)
        except asyncio.TimeoutError:
            pass
        for agent in self.agents.values():
            for t in list(agent._work):
                t.cancel()
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    # --- messaging ------------------------------------------------------------
    async def send(self, msg: AgentMessage) -> AgentMessage:
        if len(self.transcript) >= self.max_messages:
            raise RuntimeError("message budget exceeded - agents may be looping")
        self.transcript.append(msg)
        log.debug("#%d %s -> %s [%s] %s%s | %s", msg.id, msg.sender, msg.recipient, msg.performative, msg.subject,
                  f" (re #{msg.in_reply_to})" if msg.in_reply_to else "", msg.text[:500].replace("\n", " "))
        if self.on_message:
            try:
                self.on_message(msg)
            except Exception:  # a display problem must never break the agents' conversation
                self.on_message = None

        # A reply to someone who is waiting goes straight to the waiting agent.
        if msg.performative == REPLY and msg.in_reply_to in self._pending:
            fut = self._pending.pop(msg.in_reply_to)
            if not fut.done():
                fut.set_result(msg)
            return msg

        if msg.recipient == BROADCAST:
            for name, agent in self.agents.items():
                if name != msg.sender:
                    agent.inbox.put_nowait(msg)
        elif msg.recipient in self.agents:
            self.agents[msg.recipient].inbox.put_nowait(msg)
        else:
            raise KeyError(f"unknown agent '{msg.recipient}'")
        return msg

    async def ask(self, msg: AgentMessage, timeout: float | None = None) -> AgentMessage | None:
        """Send a REQUEST and wait for the REPLY that answers it."""
        assert msg.performative == REQUEST
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[msg.id] = fut
        await self.send(msg)
        try:
            return await asyncio.wait_for(fut, timeout or self.ask_timeout)
        except asyncio.TimeoutError:
            self._pending.pop(msg.id, None)
            return None

    def thread_of(self, message_id: int) -> list[AgentMessage]:
        return [m for m in self.transcript if m.id == message_id or m.in_reply_to == message_id]

    def stats(self) -> dict[str, Any]:
        by_perf: dict[str, int] = {}
        for m in self.transcript:
            by_perf[m.performative] = by_perf.get(m.performative, 0) + 1
        return {"messages": len(self.transcript), "by_type": by_perf,
                "questions_answered": sum(1 for m in self.transcript if m.performative == REPLY)}
