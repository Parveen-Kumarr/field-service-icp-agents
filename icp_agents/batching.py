"""Micro-batching: several agents' tasks answered by ONE model call.

Free model tiers limit requests per minute (Cerebras: 5 per model) more tightly than
tokens. When many accounts are in flight, the Researcher, Analyst and Strategist each
collect the tasks that arrive within a short window (or until a batch is full) and send
them to the model together, e.g. "research these 4 companies". Every account still gets
its own message to the next agent, so the conversation is unchanged; only the model
call is shared.

If a batch call fails, or the model leaves an item out, that item's caller gets None
and falls back to a normal single call.
"""
from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from . import log as logs

log = logs.get("batching")


class MicroBatcher:
    def __init__(self, run_batch: Callable[[list[tuple[str, Any]]], Awaitable[dict[str, Any]]],
                 max_size: int = 4, window: float = 0.5, name: str = "batch"):
        self.run_batch, self.max_size, self.window, self.name = run_batch, max(1, max_size), window, name
        self._pending: list[tuple[str, Any, asyncio.Future]] = []
        self._timer: asyncio.TimerHandle | None = None
        self.stats = {"batches": 0, "items": 0, "fallbacks": 0}

    async def submit(self, key: str, item: Any) -> Any | None:
        """Queue one task; returns its result, or None if the caller should do it alone."""
        if self.max_size <= 1:
            return None
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending.append((key, item, fut))
        if len(self._pending) >= self.max_size:
            self._flush()
        elif self._timer is None:
            self._timer = loop.call_later(self.window, self._flush)
        return await fut

    def _flush(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        while self._pending:
            batch, self._pending = self._pending[:self.max_size], self._pending[self.max_size:]
            asyncio.ensure_future(self._run(batch))

    async def _run(self, batch: list[tuple[str, Any, asyncio.Future]]) -> None:
        results: dict[str, Any] = {}
        if len(batch) == 1:  # nothing to share: let the caller make its normal call
            results = {}
        else:
            try:
                results = await self.run_batch([(k, item) for k, item, _ in batch]) or {}
                self.stats["batches"] += 1
                self.stats["items"] += sum(1 for k, _, _ in batch if k in results)
            except Exception as exc:  # noqa: BLE001 - every caller falls back to a single call
                log.warning("%s: a batch of %d failed (%s: %s) - doing them one by one", self.name, len(batch),
                            type(exc).__name__, str(exc)[:160])
        for key, _, fut in batch:
            if fut.done():
                continue
            if key not in results:
                self.stats["fallbacks"] += 1 if len(batch) > 1 else 0
            fut.set_result(results.get(key))


def norm_id(value: Any) -> str:
    return " ".join(str(value or "").lower().split())
