"""Gemini tool-use loop shared by every thinking agent (Gemini Interactions API).

An agent gives Gemini a task, built-in tools (Google Search and URL context,
run by Google), its own function tools (e.g. `ask_researcher`, which really
messages another agent and waits for the answer), and one `submit_*` function
whose JSON schema is the agent's structured output. The loop runs until
Gemini calls the submit function.

The loop is stateful: each follow-up sends only the new function results and
`previous_interaction_id`, so Google keeps the history and thought signatures.
"""
from __future__ import annotations

import asyncio
import itertools
import json
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from . import log as logs

log = logs.get("gemini")
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}

HINTS = {
    400: "bad request - check ICP_LLM_MODEL is a valid Gemini 3 model name",
    401: "API key rejected - check GEMINI_API_KEY",
    403: "permission denied - the key may not have Gemini API access or the model isn't enabled for it",
    404: "model not found - set ICP_LLM_MODEL to a model your key can use (e.g. gemini-3.8-flash)",
    429: "quota / rate limit hit - lower --concurrency, wait, or check your plan's limits",
    500: "Gemini server error", 502: "Gemini gateway error", 503: "Gemini overloaded", 504: "Gemini gateway timeout",
}


def describe_error(exc: Exception) -> tuple[str, bool]:
    """Plain-English reason for a failed call, and whether retrying may help."""
    status = getattr(exc, "status_code", None)
    name = type(exc).__name__
    transient = status in RETRYABLE_STATUS or (status is None and any(
        w in name for w in ("Connect", "Timeout", "Network", "RemoteProtocol", "ReadError", "WriteError")))
    detail = str(exc).strip().splitlines()[0][:300] if str(exc).strip() else name
    hint = HINTS.get(status, "network problem reaching Gemini" if transient else "")
    if status == 429 and re.search(r"per day|daily|RPD", str(exc), re.I):
        # a daily quota won't come back in seconds: retrying only wastes time
        return (f"HTTP 429: daily quota exhausted for this model - wait for the daily reset, enable billing, or "
                f"run locally with ICP_LLM_PROVIDER=ollama ({detail})"), False
    return (f"HTTP {status}: {hint} ({detail})" if status else f"{name}: {hint + ' - ' if hint else ''}{detail}"), transient


def search_tool(max_uses: int | None = None) -> dict:
    """Web search. Gemini: Google Search grounding (count steered by the prompt).
    Local models: our own web_search tool, hard-capped at `max_uses` calls per task."""
    tool = {"type": "google_search"}
    if max_uses:
        tool["_max_uses"] = max_uses  # private marker, stripped before calling Gemini
    return tool


def url_tool() -> dict:
    """URL context: Gemini reads pages whose URLs appear in the prompt or in search results."""
    return {"type": "url_context"}


@dataclass
class ClientTool:
    name: str
    description: str
    input_schema: dict
    handler: Callable[[dict], Awaitable[Any]] | None = None  # None for submit tools

    def spec(self) -> dict:
        return {"type": "function", "name": self.name, "description": self.description,
                "parameters": self.input_schema}


@dataclass
class LLMResult:
    output: dict
    sources: list[dict] = field(default_factory=list)
    turns: int = 0
    searches: int = 0
    fetches: int = 0
    tool_calls: list[str] = field(default_factory=list)


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    thought_tokens: int = 0
    web_searches: int = 0
    url_fetches: int = 0

    def add(self, other: "Usage") -> None:
        for k in vars(self):
            setattr(self, k, getattr(self, k) + getattr(other, k))


class LLMError(RuntimeError):
    pass


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def text_part(text: str) -> dict:
    return {"type": "text", "text": text}


def image_part(b64: str, mime_type: str) -> dict:
    return {"type": "image", "data": b64, "mime_type": mime_type}


class GeminiLLM:
    label = "Gemini"
    supports_vision = True
    can_fetch_remotely = True  # URL context reads pages from Google's servers

    def __init__(self, model: str, client: Any = None, max_parallel: int = 6, max_retries: int = 4,
                 timeout: float = 180.0, poll_interval: float = 2.0):
        if client is None:
            from google import genai
            client = genai.Client()  # reads GEMINI_API_KEY (or GOOGLE_API_KEY)
        self.client = client
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.usage: dict[str, Usage] = {}
        self.inflight: dict[int, dict] = {}  # what is being waited on right now (for the heartbeat)
        self.failures = 0
        self._seq = itertools.count(1)
        self._gate = asyncio.Semaphore(max_parallel)

    def total_usage(self) -> Usage:
        total = Usage()
        for u in self.usage.values():
            total.add(u)
        return total

    async def _await_terminal(self, agent: str, purpose: str, resp: Any, started: float) -> Any:
        """Some interactions come back 'in_progress'/'queued' (long searches): poll until they finish."""
        while _get(resp, "status") in ("in_progress", "queued"):
            if time.monotonic() - started > self.timeout:
                raise asyncio.TimeoutError()
            log.debug("[%s] %s still %s after %.0fs - polling", agent, purpose, _get(resp, "status"),
                      time.monotonic() - started)
            await asyncio.sleep(self.poll_interval)
            resp = await asyncio.wait_for(self.client.aio.interactions.get(id=_get(resp, "id")), self.timeout)
        return resp

    async def _create(self, agent: str, purpose: str = "", **body) -> Any:
        delay = 2.0
        for attempt in range(1, self.max_retries + 2):
            cid = next(self._seq)
            started = time.monotonic()
            entry = {"agent": agent, "purpose": purpose, "started": started, "attempt": attempt, "state": "queued"}
            self.inflight[cid] = entry
            log.debug("[%s] Gemini call #%d start: %s (attempt %d)", agent, cid, purpose, attempt)
            error: str | None = None
            transient = False
            try:
                async with self._gate:
                    entry["state"], entry["started"] = "calling", time.monotonic()
                    resp = await asyncio.wait_for(
                        self.client.aio.interactions.create(model=self.model, **body), self.timeout)
                    resp = await self._await_terminal(agent, purpose, resp, entry["started"])
            except asyncio.TimeoutError:
                error, transient = f"no response within {self.timeout:.0f}s", True
            except Exception as exc:
                error, transient = describe_error(exc)
            finally:
                self.inflight.pop(cid, None)

            if error is None:
                took = time.monotonic() - entry["started"]
                self._account(agent, resp)
                status = _get(resp, "status")
                usage = _get(resp, "usage")
                searches = sum((_get(g, "count", 0) or 0) for g in (_get(usage, "grounding_tool_count", None) or [])
                               if _get(g, "type") == "google_search") if usage is not None else 0
                log.info("[%s] Gemini ok: %s | %.1fs | %s in / %s out tokens%s | status %s", agent, purpose, took,
                         f"{(_get(usage, 'total_input_tokens', 0) or 0):,}" if usage is not None else "?",
                         f"{(_get(usage, 'total_output_tokens', 0) or 0):,}" if usage is not None else "?",
                         f" | {searches} searches" if searches else "", status)
                if status in ("failed", "cancelled"):
                    errors = _get(resp, "errors") or []
                    self.failures += 1
                    raise LLMError(f"{agent}: interaction {status}: {[_get(e, 'message', e) for e in errors]}")
                return resp

            if not transient or attempt > self.max_retries:
                self.failures += 1
                log.error("[%s] Gemini call failed: %s - %s", agent, purpose, error)
                raise LLMError(f"{agent}: Gemini call failed ({purpose}): {error}")
            wait = delay + random.random()
            log.warning("[%s] Gemini call problem: %s - %s. Retry %d/%d in %.0fs", agent, purpose, error,
                        attempt, self.max_retries, wait)
            await asyncio.sleep(wait)
            delay = min(delay * 2, 30)
        raise LLMError("unreachable")

    async def preflight(self) -> float:
        """One tiny call to prove the key, model and quota work before the team starts."""
        started = time.monotonic()
        saved = self.max_retries
        self.max_retries = 1
        try:
            await self._create("Preflight", "check key, model and quota",
                               input=[text_part("Reply with the single word OK.")],
                               generation_config={"max_output_tokens": 16})
        finally:
            self.max_retries = saved
        return time.monotonic() - started

    def _account(self, agent: str, resp: Any) -> None:
        u = self.usage.setdefault(agent, Usage())
        u.calls += 1
        usage = _get(resp, "usage")
        if usage is None:
            return
        u.input_tokens += (_get(usage, "total_input_tokens", 0) or 0) + (_get(usage, "total_tool_use_tokens", 0) or 0)
        u.output_tokens += _get(usage, "total_output_tokens", 0) or 0
        u.thought_tokens += _get(usage, "total_thought_tokens", 0) or 0
        for g in _get(usage, "grounding_tool_count", None) or []:
            if _get(g, "type") == "google_search":
                u.web_searches += _get(g, "count", 0) or 0

    @staticmethod
    def _collect(steps: list, sink: list[dict], agent_usage: Usage) -> tuple[int, int]:
        """Sources: URL citations on the model's text, and pages read via URL context."""
        searches = fetches = 0
        seen = {s["url"] for s in sink}

        def add(url, title, via, **extra):
            if url and url not in seen:
                seen.add(url)
                sink.append({"url": url, "title": title or "", "via": via, **extra})

        for st in steps:
            t = _get(st, "type")
            if t == "google_search_call":
                searches += len(_get(_get(st, "arguments"), "queries", None) or []) or 1
            elif t == "url_context_call":
                fetches += len(_get(_get(st, "arguments"), "urls", None) or [])
            elif t == "url_context_result":
                for r in _get(st, "result", None) or []:
                    if _get(r, "status") == "success":
                        add(_get(r, "url"), "", "url_context")
            elif t == "model_output":
                for c in _get(st, "content", None) or []:
                    for a in _get(c, "annotations", None) or []:
                        if _get(a, "type") == "url_citation":
                            add(_get(a, "url"), _get(a, "title"), "google_search")
        agent_usage.url_fetches += fetches
        return searches, fetches

    async def run(self, *, agent: str, system: str, prompt: str | list, submit: ClientTool,
                  tools: list[ClientTool] | None = None, server_tools: list[dict] | None = None,
                  max_turns: int = 14, max_tokens: int = 8192, force_submit: bool = False,
                  purpose: str = "") -> LLMResult:
        tools = tools or []
        by_name = {t.name: t for t in tools}
        tool_specs = [*[{k: v for k, v in st.items() if not k.startswith("_")} for st in (server_tools or [])],
                      *[t.spec() for t in tools], submit.spec()]
        forced = {"allowed_tools": {"mode": "any", "tools": [submit.name]}}
        base = {"system_instruction": system, "tools": tool_specs}

        def config(force: bool) -> dict:
            cfg: dict = {"max_output_tokens": max_tokens}
            if force:
                cfg["tool_choice"] = forced
            return cfg

        purpose = purpose or submit.name.replace("submit_", "")
        user_input = [text_part(prompt)] if isinstance(prompt, str) else prompt
        result = LLMResult(output={})
        resp = await self._create(agent, f"{purpose} (turn 1)", input=user_input,
                                  generation_config=config(force_submit), **base)

        for turn in range(1, max_turns + 1):
            result.turns = turn
            steps = list(_get(resp, "steps", None) or [])
            s, f = self._collect(steps, result.sources, self.usage[agent])
            result.searches += s
            result.fetches += f

            calls = [st for st in steps if _get(st, "type") == "function_call"]
            submitted = next((c for c in calls if _get(c, "name") == submit.name), None)
            if submitted is not None:
                from .schema_tools import fill_required
                result.output, filled = fill_required(dict(_get(submitted, "arguments") or {}), submit.input_schema)
                if filled:
                    log.warning("[%s] %s: the model left out %s - filled with neutral defaults", agent, purpose,
                                ", ".join(filled))
                result.tool_calls.append(submit.name)
                return result
            if turn == max_turns:
                break

            if not calls:
                # Gemini answered in prose without submitting: ask for the structured result.
                log.debug("[%s] %s: answered in prose, asking for %s", agent, purpose, submit.name)
                resp = await self._create(agent, f"{purpose} (turn {turn + 1}, asking for the structured answer)",
                                          previous_interaction_id=_get(resp, "id"),
                                          input=[text_part(f"Now call {submit.name} with your final answer.")],
                                          generation_config=config(True), **base)
                continue

            results = []
            for call in calls:
                name, cid, args = _get(call, "name"), _get(call, "id"), _get(call, "arguments") or {}
                result.tool_calls.append(name)
                log.debug("[%s] %s: calling tool %s(%s)", agent, purpose, name, json.dumps(args, ensure_ascii=False)[:200])
                tool = by_name.get(name)
                if tool is None or tool.handler is None:
                    results.append({"type": "function_result", "call_id": cid, "name": name, "is_error": True,
                                    "result": f"Unknown tool {name}"})
                    continue
                try:
                    out = await tool.handler(args)
                    results.append({"type": "function_result", "call_id": cid, "name": name,
                                    "result": out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)})
                except Exception as exc:
                    results.append({"type": "function_result", "call_id": cid, "name": name, "is_error": True,
                                    "result": f"{type(exc).__name__}: {exc}"})
            resp = await self._create(agent, f"{purpose} (turn {turn + 1}, after {', '.join(_get(c, 'name') for c in calls)})",
                                      previous_interaction_id=_get(resp, "id"), input=results,
                                      generation_config=config(turn >= max_turns - 2), **base)

        raise LLMError(f"{agent}: no {submit.name} after {max_turns} turns")


LLM = GeminiLLM  # backwards-compatible name
