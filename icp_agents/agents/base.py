"""Base class every agent builds on."""
from __future__ import annotations

import asyncio
import contextlib
import itertools
import time
import traceback
from typing import Any

from .. import log as logs
from ..messages import INFORM, REPLY, REQUEST, AgentMessage
from ..runtime import Runtime

_keys = itertools.count(1)


class BaseAgent:
    name = "Agent"
    role = ""

    def __init__(self, runtime: Runtime, llm=None, settings=None):
        self.runtime = runtime
        self.llm = llm
        self.settings = settings
        self.inbox: asyncio.Queue[AgentMessage] = asyncio.Queue()
        self.briefing: dict | None = None
        self.activities: dict[int, tuple[str, float]] = {}  # what this agent is doing right now
        self.log = logs.get(self.name)
        self._work: set[asyncio.Task] = set()
        runtime.register(self)

    # --- what am I doing? (shown by the heartbeat) -------------------------------------
    @contextlib.contextmanager
    def doing(self, what: str):
        key = next(_keys)
        self.activities[key] = (what, time.monotonic())
        self.log.debug("start: %s", what)
        try:
            yield
        finally:
            _, started = self.activities.pop(key, (what, time.monotonic()))
            self.log.debug("done:  %s (%.1fs)", what, time.monotonic() - started)

    # --- loop -------------------------------------------------------------------
    async def run(self) -> None:
        while True:
            msg = await self.inbox.get()
            # Each message is handled in its own task, so an agent can keep
            # answering colleagues' questions while it waits on its own.
            task = asyncio.create_task(self._safe_handle(msg))
            self._work.add(task)
            task.add_done_callback(self._work.discard)

    async def drain(self) -> None:
        while self._work:
            await asyncio.gather(*list(self._work), return_exceptions=True)

    async def idle(self) -> bool:
        return self.inbox.empty() and not self._work

    async def _safe_handle(self, msg: AgentMessage) -> None:
        if msg.subject == "vendor_briefing":  # the Researcher's briefing, shared with everyone
            self.briefing = msg.data
            return
        try:
            await self.handle(msg)
        except Exception as exc:  # an agent failing must not kill the team
            detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            self.log.error("failed handling '%s' from %s: %s", msg.subject, msg.sender, detail)
            self.log.debug("traceback:\n%s", traceback.format_exc())
            if msg.performative == REQUEST:
                await self.reply(msg, f"I couldn't answer that: {detail}", {"error": detail})
            await self.say("Coordinator", "error",
                           f"I hit a problem while handling '{msg.subject}' from {msg.sender}: {detail}",
                           {"error": detail, "account_id": msg.thread}, thread=msg.thread)

    def vendor(self) -> str:
        """How agents refer to the company whose ICP they qualify against."""
        name = getattr(self.settings, "vendor_name", "") if self.settings else ""
        return name or "our client"

    def briefing_text(self) -> str:
        from ..vendor_profile import VENDOR_BASELINE
        b = getattr(self, "briefing", None)
        if not b or b.get("source") != "live":
            return VENDOR_BASELINE
        return (f"{b.get('offering','')}\nProducts / agents: {', '.join(b.get('agents', []))}\n"
                f"Verticals: {', '.join(b.get('target_verticals', []))}\n"
                f"Buyer personas: {', '.join(b.get('buyer_personas', []))}\n"
                f"Fit signals: {'; '.join(b.get('icp_signals', []))}\n"
                f"Value claims: {'; '.join(b.get('value_claims', []))}\n"
                f"Competitors: {', '.join(b.get('competitors', []))}")

    async def handle(self, msg: AgentMessage) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    # --- speaking ---------------------------------------------------------------
    async def say(self, to: str, subject: str, text: str, data: dict[str, Any] | None = None,
                  performative: str = INFORM, thread: str | None = None) -> AgentMessage:
        return await self.runtime.send(AgentMessage(self.name, to, performative, subject, text, data or {},
                                                    thread=thread))

    async def ask(self, to: str, subject: str, text: str, data: dict[str, Any] | None = None,
                  thread: str | None = None, timeout: float | None = None) -> AgentMessage | None:
        with self.doing(f"waiting for {to} to answer ({subject}{', ' + thread if thread else ''})"):
            reply = await self.runtime.ask(AgentMessage(self.name, to, REQUEST, subject, text, data or {},
                                                        thread=thread), timeout)
        if reply is None:
            self.log.warning("%s did not answer '%s' in time", to, subject)
        return reply

    async def reply(self, msg: AgentMessage, text: str, data: dict[str, Any] | None = None) -> AgentMessage:
        return await self.runtime.send(AgentMessage(self.name, msg.sender, REPLY, msg.subject, text, data or {},
                                                    in_reply_to=msg.id, thread=msg.thread))

    async def challenge(self, to: str, subject: str, text: str, data: dict[str, Any] | None = None,
                        thread: str | None = None) -> AgentMessage | None:
        """Disagree with a colleague and wait for them to respond."""
        with self.doing(f"waiting for {to} to respond to a challenge ({thread})"):
            return await self.runtime.ask(AgentMessage(self.name, to, REQUEST, subject, text,
                                                       {**(data or {}), "challenge": True}, thread=thread))

    def batch_size(self) -> int:
        """Accounts per model call: only in fast mode (several tasks share one request)."""
        s = self.settings
        if not s or not getattr(s, "fast", False):
            return 1
        return max(1, int(getattr(s, "batch_size", 4) or 1))
