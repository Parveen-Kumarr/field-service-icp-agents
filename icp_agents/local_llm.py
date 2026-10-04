"""Local model client: Ollama (e.g. Gemma 4 on your GPU), same interface as the Gemini client.

The agents call `run(agent=..., system=..., prompt=..., submit=..., tools=..., server_tools=...)`
exactly as they do with Gemini. Differences handled here:

- Server tools (Google Search / URL context) become local `web_search` / `fetch_page`
  tools (websearch.py) that run on this machine, hard-capped per task.
- Ollama has no tool_choice, so a submit is forced with structured output: the final
  request drops the tools and constrains the reply to the submit function's JSON schema.
- Ollama is stateless: every request carries the whole conversation.
- Images (logo reading) go in the message's `images` field.
"""
from __future__ import annotations

import asyncio
import itertools
import json
import random
import time
from typing import Any

from . import log as logs
from .llm import ClientTool, LLMError, LLMResult, Usage
from .schema_tools import fill_required, repair_json
from .websearch import LocalWebTools, ToolBudget

log = logs.get("ollama")
RETRYABLE = {408, 429, 500, 502, 503, 504}
IMAGE_TYPES = ("image/png", "image/jpeg")


class OllamaError(RuntimeError):
    def __init__(self, message: str, transient: bool):
        super().__init__(message)
        self.transient = transient


class OllamaLLM:
    label = "Local model"
    supports_vision = True
    can_fetch_remotely = False  # no cloud fetch: pages are read from this machine
    image_types = IMAGE_TYPES

    def __init__(self, model: str, base_url: str = "http://localhost:11434", web: LocalWebTools | None = None,
                 num_ctx: int = 16384, think: bool = False, timeout: float = 300.0, max_parallel: int = 2,
                 max_retries: int = 2, http=None, max_output: int = 2048):
        import httpx

        # Cap on tokens a single reply may generate. Local models can loop and write thousands of tokens
        # (one call in a real run wrote 6,000 tokens in 150s and cut its JSON off); a cap keeps calls short.
        self.max_output = max_output

        self.model = model
        self.base_url = base_url.rstrip("/")
        self.web = web
        self.num_ctx = num_ctx
        self.think = think
        self.timeout = timeout
        self.max_retries = max_retries
        self.http = http or httpx.AsyncClient(base_url=self.base_url, timeout=httpx.Timeout(timeout, connect=10))
        self.usage: dict[str, Usage] = {}
        self.inflight: dict[int, dict] = {}
        self.failures = 0
        self._seq = itertools.count(1)
        self._gate = asyncio.Semaphore(max_parallel)
        self.label = f"Local model ({model})"

    def total_usage(self) -> Usage:
        total = Usage()
        for u in self.usage.values():
            total.add(u)
        return total

    # --- one request ------------------------------------------------------------------
    def _explain(self, exc: Exception) -> OllamaError:
        name = type(exc).__name__
        if "Connect" in name:
            return OllamaError(f"can't reach Ollama at {self.base_url} - start the Ollama app (or run "
                               f"'ollama serve') and try again", False)
        if "Timeout" in name:
            return OllamaError(f"no response within {self.timeout:.0f}s - the model may be loading or the GPU "
                               f"is overloaded", True)
        return OllamaError(f"{name}: {exc}", "Read" in name or "Protocol" in name)

    async def _post_chat(self, body: dict) -> dict:
        try:
            r = await self.http.post("/api/chat", json=body)
        except Exception as exc:
            raise self._explain(exc) from exc
        if r.status_code == 200:
            return r.json()
        try:
            err = r.json().get("error", r.text)
        except Exception:
            err = r.text
        if r.status_code == 404 and "not found" in str(err).lower():
            raise OllamaError(f"model '{self.model}' is not downloaded - run: ollama pull {self.model}", False)
        hint = " (GPU out of memory? lower ICP_OLLAMA_NUM_CTX or ICP_CONCURRENCY)" if "memory" in str(err).lower() else ""
        raise OllamaError(f"HTTP {r.status_code}: {str(err)[:300]}{hint}", r.status_code in RETRYABLE)

    async def _chat(self, agent: str, purpose: str, body: dict) -> dict:
        delay = 2.0
        for attempt in range(1, self.max_retries + 2):
            cid = next(self._seq)
            entry = {"agent": agent, "purpose": purpose, "started": time.monotonic(), "attempt": attempt,
                     "state": "queued"}
            self.inflight[cid] = entry
            error = None
            transient = False
            try:
                async with self._gate:
                    entry["state"], entry["started"] = "calling", time.monotonic()
                    data = await asyncio.wait_for(self._post_chat(body), self.timeout)
            except asyncio.TimeoutError:
                error, transient = f"no response within {self.timeout:.0f}s", True
            except OllamaError as exc:
                error, transient = str(exc), exc.transient
            finally:
                self.inflight.pop(cid, None)

            if error is None:
                took = time.monotonic() - entry["started"]
                u = self.usage.setdefault(agent, Usage())
                u.calls += 1
                u.input_tokens += data.get("prompt_eval_count", 0) or 0
                u.output_tokens += data.get("eval_count", 0) or 0
                msg = data.get("message") or {}
                if msg.get("thinking"):
                    u.thought_tokens += len(msg["thinking"]) // 4  # Ollama doesn't report thinking tokens separately
                calls = [c.get("function", {}).get("name") for c in msg.get("tool_calls") or []]
                log.info("[%s] %s ok: %s | %.1fs | %s in / %s out tokens%s", agent, self.model, purpose, took,
                         f"{data.get('prompt_eval_count', 0):,}", f"{data.get('eval_count', 0):,}",
                         f" | calls {', '.join(calls)}" if calls else "")
                return data
            if not transient or attempt > self.max_retries:
                self.failures += 1
                log.error("[%s] %s failed: %s - %s", agent, self.model, purpose, error)
                raise LLMError(f"{agent}: local model call failed ({purpose}): {error}")
            wait = delay + random.random()
            log.warning("[%s] %s problem: %s - %s. Retry %d/%d in %.0fs", agent, self.model, purpose, error,
                        attempt, self.max_retries, wait)
            await asyncio.sleep(wait)
            delay = min(delay * 2, 30)
        raise LLMError("unreachable")

    # --- checks -------------------------------------------------------------------------
    async def preflight(self) -> float:
        """Ollama running? Model downloaded? Then one tiny chat to load it onto the GPU."""
        started = time.monotonic()
        try:
            r = await self.http.get("/api/tags")
        except Exception as exc:
            raise LLMError(str(self._explain(exc))) from exc
        names = {m.get("name") for m in (r.json().get("models") or [])} if r.status_code == 200 else set()
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        if wanted not in names:
            have = ", ".join(sorted(n for n in names if n)) or "none"
            raise LLMError(f"model '{self.model}' is not downloaded (installed: {have}) - run: ollama pull {self.model}")
        log.info("Loading %s into memory (the first load can take a minute) ...", self.model)
        await self._chat("Preflight", "load the model and check it answers",
                         {"model": self.model, "stream": False,
                          "messages": [{"role": "user", "content": "Reply with the single word OK."}],
                          "options": {"num_ctx": self.num_ctx, "num_predict": 8}, "keep_alive": "30m"})
        return time.monotonic() - started

    # --- the agent loop -----------------------------------------------------------------
    @staticmethod
    def _user_message(prompt: str | list, image_types: tuple[str, ...]) -> dict:
        if isinstance(prompt, str):
            return {"role": "user", "content": prompt}
        texts, images = [], []
        for part in prompt:
            if part.get("type") == "text":
                texts.append(part["text"])
            elif part.get("type") == "image" and part.get("mime_type") in image_types:
                images.append(part["data"])
                texts.append(f"[image {len(images)}]")
        msg = {"role": "user", "content": "\n".join(texts)}
        if images:
            msg["images"] = images
        return msg

    def _body(self, messages: list, tools: list | None, max_tokens: int, fmt: dict | None = None) -> dict:
        body: dict[str, Any] = {"model": self.model, "messages": messages, "stream": False, "keep_alive": "30m",
                                "options": {"num_ctx": self.num_ctx, "num_predict": min(max_tokens, self.max_output),
                                            "temperature": 0.3, "repeat_penalty": 1.15}}
        if tools:
            body["tools"] = [{"type": "function", "function": {"name": t.name, "description": t.description,
                                                              "parameters": t.input_schema}} for t in tools]
        if fmt is not None:
            body["format"] = fmt
        if self.think:  # only sent when wanted: some models reject the parameter entirely
            body["think"] = True
        return body

    @staticmethod
    def _args(raw: Any) -> dict:
        if isinstance(raw, dict):
            return raw
        try:
            return json.loads(raw or "{}")
        except (TypeError, json.JSONDecodeError):
            return {}

    @staticmethod
    def _complete(output: dict, schema: dict) -> bool:
        return all(k in output for k in schema.get("required", []))

    async def _structured(self, agent: str, purpose: str, messages: list, submit: ClientTool,
                          max_tokens: int, turn: int) -> dict:
        """Force the final answer: no tools, reply constrained to the submit schema.

        If the reply is cut off by the output cap, ask once more for a shorter answer; if that is cut off too,
        repair the JSON (keep the complete fields, drop the unfinished one)."""
        concise = ("Be concise: keep every text field under 40 words and every list to at most 5 items. ")
        best: dict | None = None
        for attempt in (1, 2):
            ask = {"role": "user", "content": (f"Now give your final answer for {submit.name}: {submit.description} "
                                               + (concise if attempt == 2 else "")
                                               + "Reply with only the JSON object, using what you have found so far.")}
            label = f"{purpose} (turn {turn}, structured answer{', shorter retry' if attempt == 2 else ''})"
            data = await self._chat(agent, label, self._body(messages + [ask], None, max_tokens,
                                                             fmt=submit.input_schema))
            content = (data.get("message") or {}).get("content") or ""
            truncated = data.get("done_reason") == "length"
            parsed = repair_json(content)
            if parsed is not None and not truncated:
                return parsed
            if parsed is not None and (best is None or len(parsed) > len(best)):
                best = parsed
            log.warning("[%s] %s: answer was %s - %s", agent, purpose,
                        "cut off at the output limit" if truncated else "not valid JSON",
                        "asking for a shorter one" if attempt == 1 else "keeping the complete part")
        if best is not None:
            return best
        raise LLMError(f"{agent}: the model did not return valid JSON for {submit.name}")

    async def run(self, *, agent: str, system: str, prompt: str | list, submit: ClientTool,
                  tools: list[ClientTool] | None = None, server_tools: list[dict] | None = None,
                  max_turns: int = 10, max_tokens: int = 4096, force_submit: bool = False,
                  purpose: str = "") -> LLMResult:
        purpose = purpose or submit.name.replace("submit_", "")
        tools = list(tools or [])
        budget = ToolBudget()
        want_search = want_fetch = False
        for st in server_tools or []:
            if st.get("type") == "google_search":
                want_search, budget.max_searches = True, st.get("_max_uses") or 2
            elif st.get("type") == "url_context":
                want_fetch = True
        if (want_search or want_fetch) and self.web is None:
            raise LLMError(f"{agent}: web tools requested but no local web tools are configured")
        if self.web is not None:
            tools += self.web.tools(budget, want_search, want_fetch)
        by_name = {t.name: t for t in tools}
        result = LLMResult(output={})

        system_msg = system + (f"\n\nWhen you are ready, call {submit.name} with your final answer. "
                               "Only call the tools listed; never invent tool results.")
        messages: list[dict] = [{"role": "system", "content": system_msg},
                                self._user_message(prompt, self.image_types)]

        def finish(output: dict) -> LLMResult:
            output, filled = fill_required(output, submit.input_schema)
            if filled:
                log.warning("[%s] %s: the model left out %s - filled with neutral defaults", agent, purpose,
                            ", ".join(filled))
            result.output = output
            result.sources = budget.sources
            result.searches, result.fetches = budget.searches, budget.fetches
            u = self.usage.setdefault(agent, Usage())
            u.web_searches += budget.searches
            u.url_fetches += budget.fetches
            return result

        if force_submit and not tools:
            return finish(await self._structured(agent, purpose, messages, submit, max_tokens, 1))

        for turn in range(1, max_turns + 1):
            result.turns = turn
            data = await self._chat(agent, f"{purpose} (turn {turn})",
                                    self._body(messages, [*tools, submit], max_tokens))
            msg = data.get("message") or {}
            calls = msg.get("tool_calls") or []
            messages.append({"role": "assistant", "content": msg.get("content") or "",
                             **({"tool_calls": calls} if calls else {})})
            submitted = next((c for c in calls if c.get("function", {}).get("name") == submit.name), None)
            if submitted is not None:
                output = self._args(submitted["function"].get("arguments"))
                result.tool_calls.append(submit.name)
                if self._complete(output, submit.input_schema):
                    return finish(output)
                log.debug("[%s] %s: incomplete %s arguments, asking for a structured answer", agent, purpose,
                          submit.name)
                return finish(await self._structured(agent, purpose, messages, submit, max_tokens, turn + 1))
            if not calls or turn == max_turns:
                log.debug("[%s] %s: answered in prose, asking for a structured answer", agent, purpose)
                return finish(await self._structured(agent, purpose, messages, submit, max_tokens, turn + 1))

            for call in calls:
                fn = call.get("function", {})
                name, args = fn.get("name"), self._args(fn.get("arguments"))
                result.tool_calls.append(name)
                log.debug("[%s] %s: calling tool %s(%s)", agent, purpose, name, json.dumps(args)[:200])
                tool = by_name.get(name)
                if tool is None or tool.handler is None:
                    content = f"Unknown tool {name}"
                else:
                    try:
                        out = await tool.handler(args)
                        content = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)
                    except Exception as exc:
                        content = f"Tool error: {type(exc).__name__}: {exc}"
                messages.append({"role": "tool", "tool_name": name, "content": content})
        raise LLMError(f"{agent}: no {submit.name} after {max_turns} turns")
