"""Heartbeat: a periodic status report so a long run never looks frozen.

Every few seconds it logs elapsed time, messages, accounts finished, every
model call in flight (which agent, for what, how long), and what each busy
agent is doing. Anything waiting longer than `slow_after` is flagged.
"""
from __future__ import annotations

import asyncio
import time

from . import log as logs

log = logs.get("heartbeat")


def _fmt(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 60}m{seconds % 60:02d}s" if seconds >= 60 else f"{seconds}s"


def status_lines(runtime, llm, slow_after: float = 120.0, started: float | None = None) -> list[str]:
    now = time.monotonic()
    coord = runtime.agents.get("Coordinator")
    expected = len(coord.expected) if coord and coord.expected is not None else None
    done = sum(1 for st in (coord.status.values() if coord else []) if st.get("stage") in ("done", "error"))
    inflight = sorted(getattr(llm, "inflight", {}).values(), key=lambda e: e["started"]) if llm else []
    calling = [e for e in inflight if e["state"] == "calling"]
    queued = [e for e in inflight if e["state"] == "queued"]
    head = (f"{_fmt(now - started) + ' elapsed | ' if started else ''}{len(runtime.transcript)} messages | "
            f"accounts {done}/{expected if expected is not None else '?'} done | model: {len(calling)} in flight, "
            f"{len(queued)} queued, {getattr(llm, 'failures', 0)} failed")
    lines = [head]
    for e in calling:
        waited = now - e["started"]
        flag = "  SLOW" if waited > slow_after else ""
        retry = f", attempt {e['attempt']}" if e["attempt"] > 1 else ""
        lines.append(f"  model <- {e['agent']}: {e['purpose']} - {_fmt(waited)}{retry}{flag}")
    for name, agent in runtime.agents.items():
        for what, since in sorted(agent.activities.values(), key=lambda x: x[1]):
            waited = now - since
            flag = "  SLOW" if waited > slow_after and not what.startswith(("queued", "waiting for the vendor")) else ""
            lines.append(f"  {name}: {what} - {_fmt(waited)}{flag}")
    if hasattr(llm, "status_line"):
        lines.append("  " + llm.status_line())
    if len(lines) == 1:
        lines.append("  (all agents idle)")
    return lines


async def heartbeat(runtime, llm, every: float = 15.0, slow_after: float = 120.0) -> None:
    if every <= 0:
        return
    started = time.monotonic()
    while not runtime.finished.is_set():
        try:
            await asyncio.wait_for(runtime.finished.wait(), every)
            return
        except asyncio.TimeoutError:
            pass
        for line in status_lines(runtime, llm, slow_after, started):
            log.info(line)
