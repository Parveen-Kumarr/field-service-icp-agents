"""Field-service ICP agent team: live, conversational multi-agent ICP validation."""
from __future__ import annotations

import asyncio

from . import log as logs
from .agents.analyst import AnalystAgent
from .agents.coordinator import CoordinatorAgent
from .agents.reporter import ReporterAgent
from .agents.researcher import ResearcherAgent
from .agents.scout import ScoutAgent
from .agents.strategist import StrategistAgent
from .config import Settings
from .llm import LLM
from .monitor import heartbeat
from .providers import make_llm
from .runtime import Runtime

_log = logs.get("team")


def build_team(settings: Settings, llm: LLM, on_message=None, fetcher=None) -> tuple[Runtime, dict]:
    rt = Runtime(on_message=on_message)
    team = {
        "coordinator": CoordinatorAgent(rt, llm, settings),
        "scout": ScoutAgent(rt, llm, settings, fetcher=fetcher),
        "researcher": ResearcherAgent(rt, llm, settings),
        "analyst": AnalystAgent(rt, llm, settings),
        "strategist": StrategistAgent(rt, llm, settings),
        "reporter": ReporterAgent(rt, llm, settings),
    }
    return rt, team


async def run_team(settings: Settings, llm: LLM, on_message=None, fetcher=None,
                   timeout: float | None = None) -> tuple[Runtime, dict]:
    rt, team = build_team(settings, llm, on_message, fetcher)
    coord = team["coordinator"]
    rt.start()
    starter = asyncio.create_task(coord.start())

    def _starter_done(task: asyncio.Task) -> None:
        # If the Coordinator itself fails, stop the run with the reason instead of waiting forever.
        if not task.cancelled() and task.exception() is not None and not rt.finished.is_set():
            coord.failed = f"Coordinator failed: {task.exception()!r}"
            _log.error(coord.failed)
            rt.finished.set()

    starter.add_done_callback(_starter_done)
    beat = asyncio.create_task(heartbeat(rt, llm, getattr(settings, "heartbeat", 15.0)))
    try:
        await asyncio.wait_for(rt.finished.wait(), timeout)
        if not starter.done():
            await asyncio.wait([starter], timeout=5)
    finally:
        beat.cancel()
        await rt.stop()
    return rt, team


__all__ = ["Settings", "LLM", "Runtime", "build_team", "run_team", "make_llm"]
