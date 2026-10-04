"""Model pool: spread the agents' model calls across several free LLM providers.

Every provider here speaks the OpenAI chat-completions API (Groq, Cerebras, Mistral,
GitHub Models, OpenRouter, Gemini's OpenAI-compatible endpoint, and a local Ollama).
Each becomes a `Backend` with its own limiter (requests/min, requests/day, tokens/min,
tokens/day, concurrent requests). For every model *turn*, the pool picks the backend
that can take it soonest, so the free quotas of all providers are used together:

- A 429 sets a cool-down from the Retry-After header; a daily-quota 429 retires that
  backend for the day. Server errors and timeouts move the turn to another backend.
- The conversation is kept in the OpenAI message format with our own portable
  tool-call ids, so a multi-turn agent task can continue on a different provider.
- Image input (logo reading) only goes to backends marked as vision-capable.

The pool implements the same `run(...)` interface as the Gemini and Ollama clients,
so the agents don't change.
"""
from __future__ import annotations

import asyncio
import itertools
import json
import re
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from . import log as logs
from .llm import ClientTool, LLMError, LLMResult, Usage
from .schema_tools import fill_required, repair_json
from .websearch import LocalWebTools, ToolBudget

log = logs.get("pool")
DAILY_RE = re.compile(r"per day|daily|\bRPD\b|\bTPD\b", re.I)
RETRY_IN_RE = re.compile(r"(?:retry|try again) in ([\d.]+)\s*(ms|s|sec|seconds|m|min)?", re.I)


# --- provider presets (free-tier limits as published; override any of them in .env) ---------------
@dataclass
class Preset:
    base_url: str
    key_env: str
    model: str
    rpm: int = 0
    rpd: int = 0
    tpm: int = 0
    tpd: int = 0
    concurrency: int = 2
    max_input_tokens: int = 0
    vision: bool = False
    needs_key: bool = True
    extra_body: dict = field(default_factory=dict)
    note: str = ""
    retired: str = ""        # provider shut this service down: skipped with a message
    discover_free: bool = False  # if the default model is gone, pick a current free tool-capable one
    discover_more: bool = False  # free limits are per MODEL: add the provider's other tool-capable models
    strict_tools: bool = False   # provider rejects tool calls that miss 'required' fields (so don't send them)
    own_tool_history: bool = False  # can only continue tool conversations it started itself (Gemini 3)
    header_style: str = ""       # which rate-limit headers the provider sends: "groq" | "cerebras"


PRESETS: dict[str, Preset] = {
    # Groq free tier (console.groq.com/docs/rate-limits): 30 RPM, 1K RPD, 8K TPM, 200K TPD. Very fast inference.
    "groq": Preset("https://api.groq.com/openai/v1", "GROQ_API_KEY", "openai/gpt-oss-120b", rpm=30, rpd=1000,
                   tpm=8000, tpd=200_000, concurrency=4, extra_body={"reasoning_effort": "low"},
                   discover_more=True, strict_tools=True, header_style="groq"),
    # Cerebras free trial (inference-docs.cerebras.ai/support/rate-limits): 5 RPM, 30K TPM, 1M TPD. Very fast.
    "cerebras": Preset("https://api.cerebras.ai/v1", "CEREBRAS_API_KEY", "gpt-oss-120b", rpm=5, tpm=30_000,
                       tpd=1_000_000, concurrency=3, extra_body={"reasoning_effort": "low"},
                       discover_more=True, header_style="cerebras"),
    # Mistral free "Experiment" plan: limits shown at console.mistral.ai/limits (defaults here are conservative).
    "mistral": Preset("https://api.mistral.ai/v1", "MISTRAL_API_KEY", "mistral-small-latest", rpm=30, tpm=50_000,
                      concurrency=3, note="free plan requires opting in to data training"),
    # GitHub Models was retired on 30 July 2026 (its inference API is gone); kept only to explain why it's skipped.
    "github": Preset("https://models.github.ai/inference", "GITHUB_TOKEN", "openai/gpt-4.1-mini", rpm=15, rpd=150,
                     concurrency=5, max_input_tokens=8000,
                     retired="GitHub retired GitHub Models on 30 July 2026 - remove 'github' from ICP_POOL"),
    # OpenRouter free models (':free'): 20 RPM, 50 RPD (1,000 RPD after a one-time $10 credit purchase).
    # The free line-up changes often, so if this model is gone the pool picks a current free one itself.
    "openrouter": Preset("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "openai/gpt-oss-120b:free", rpm=20,
                         rpd=50, concurrency=3, discover_free=True),
    # Gemini via Google's OpenAI-compatible endpoint. Free tier on this model: ~20 RPD.
    "gemini": Preset("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY", "gemini-3.8-flash",
                     rpm=5, rpd=20, concurrency=2, vision=True, own_tool_history=True),
    # Your own GPU through Ollama's OpenAI-compatible endpoint: no limits, but slower. Used as overflow.
    "ollama": Preset("http://localhost:11434/v1", "", "gemma4:12b", concurrency=1, vision=True, needs_key=False),
}
DEFAULT_ORDER = ["groq", "cerebras", "mistral", "openrouter", "gemini", "ollama"]
# Extra models worth adding from a provider's own list (tool calling, good enough for scoring), and ones to skip.
EXTRA_MODEL_WANT = ("gpt-oss", "llama-3.3-70b", "llama-4", "qwen3", "qwen-3", "kimi-k2", "deepseek", "glm")
EXTRA_MODEL_SKIP = ("guard", "whisper", "tts", "embed", "compound", "playai", "orpheus", "allam", "saba", "distil",
                    "8b", "instant", "audio", "image", "vision-preview")
MAX_EXTRA_MODELS = 3
FREE_MODEL_PREFERENCE = ("gpt-oss-120b", "qwen3", "deepseek", "llama-3.3-70b", "llama-4", "gemma", "mistral")


# --- errors ------------------------------------------------------------------------------------------
class RateLimited(Exception):
    def __init__(self, message: str, retry_after: float, daily: bool):
        super().__init__(message)
        self.retry_after, self.daily = retry_after, daily


class Transient(Exception):
    pass


class BadRequest(Exception):
    pass


class Fatal(Exception):
    """This backend can't be used at all (bad key, unknown model)."""


# --- one provider -----------------------------------------------------------------------------------
class Backend:
    def __init__(self, name: str, preset: Preset, api_key: str = "", http=None, timeout: float = 120.0):
        import httpx

        self.name, self.p = name, preset
        self.model = preset.model
        self.api_key = api_key
        self.extra_body = dict(preset.extra_body)
        self.timeout = timeout
        self.http = http or httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10))
        self.minute: deque = deque()  # [timestamp, tokens]
        self.on_free = None  # set by the pool: wakes queued calls when a slot frees up
        self.auto_model = True  # False when the user chose the model in .env
        self.provider = name    # preset name (several backends can share one provider, one per model)
        self.overridden: set = set()  # limits the user set in .env (never replaced by learned ones)
        self.consecutive_429 = 0
        self.learned = False
        self.day = date.today()
        self.day_requests = 0
        self.day_tokens = 0
        self.inflight = 0
        self.cooldown_until = 0.0
        self.exhausted = ""   # reason, for the rest of the day
        self.disabled = ""    # reason, for the rest of the run
        self.latencies: deque = deque(maxlen=20)
        self.stats = {"calls": 0, "rate_limited": 0, "errors": 0, "in_tokens": 0, "out_tokens": 0, "seconds": 0.0}

    @property
    def label(self) -> str:
        return f"{self.name}:{self.model}"

    def clone(self, model: str) -> "Backend":
        """Another model of the same provider as its own pool member: free limits are counted per model."""
        from dataclasses import replace
        short = model.split("/")[-1]
        p = replace(self.p, model=model, discover_more=False,
                    extra_body=dict(self.p.extra_body) if "gpt-oss" in model else {})
        b = Backend(f"{self.provider}/{short}", p, self.api_key, http=self.http, timeout=self.timeout)
        b.provider, b.auto_model, b.overridden = self.provider, False, set(self.overridden)
        b.is_extra = True
        return b

    def _roll_day(self) -> None:
        if date.today() != self.day:
            self.day, self.day_requests, self.day_tokens, self.exhausted = date.today(), 0, 0, ""

    def usable(self, est_tokens: int, est_input: int, vision: bool) -> str:
        """Empty string if this backend can serve the request at all today, else the reason it can't."""
        self._roll_day()
        if self.disabled:
            return self.disabled
        if self.exhausted:
            return self.exhausted
        if vision and not self.p.vision:
            return "no image input"
        if self.p.rpd and self.day_requests >= self.p.rpd:
            self.exhausted = f"daily request limit ({self.p.rpd}) used up"
            return self.exhausted
        if self.p.tpd and self.day_tokens + est_tokens > self.p.tpd:
            return "daily token limit nearly used up"
        if self.p.max_input_tokens and est_input > self.p.max_input_tokens:
            return f"request too large for its {self.p.max_input_tokens}-token input limit"
        if self.p.tpm and est_tokens > self.p.tpm:
            return f"request larger than its {self.p.tpm} tokens/min limit"
        return ""

    def wait_for(self, est_tokens: int, now: float) -> float:
        """Seconds until this backend has capacity for the request (0 = now)."""
        while self.minute and now - self.minute[0][0] >= 60:
            self.minute.popleft()
        waits = [self.cooldown_until - now]
        if self.inflight >= self.p.concurrency:
            waits.append(0.5)
        if self.p.rpm and len(self.minute) >= self.p.rpm:
            waits.append(self.minute[-self.p.rpm][0] + 60 - now)
        if self.p.tpm:
            used = sum(t for _, t in self.minute)
            if used + est_tokens > self.p.tpm:
                freed, at = 0, now
                for ts, tok in self.minute:
                    freed += tok
                    at = ts + 60
                    if used - freed + est_tokens <= self.p.tpm:
                        break
                waits.append(at - now)
        return max(0.0, *waits)

    def reserve(self, est_tokens: int) -> list:
        rec = [time.monotonic(), est_tokens]
        self.minute.append(rec)
        self.inflight += 1
        self.day_requests += 1
        self.day_tokens += est_tokens
        return rec

    def settle(self, rec: list, actual_tokens: int | None) -> None:
        self.inflight = max(0, self.inflight - 1)
        if self.on_free:
            self.on_free()
        if actual_tokens is not None:
            self.day_tokens += actual_tokens - rec[1]
            rec[1] = actual_tokens

    def per_minute_capacity(self, tokens_per_call: int = 1600) -> float:
        """Rough calls/minute this backend can sustain (for the ETA)."""
        lat = (sum(self.latencies) / len(self.latencies)) if self.latencies else (20.0 if self.name == "ollama" else 4.0)
        caps = [self.p.concurrency * 60 / max(lat, 0.5)]
        if self.p.rpm:
            caps.append(self.p.rpm)
        if self.p.tpm:
            caps.append(self.p.tpm / tokens_per_call)
        # a provider with a small daily quota (e.g. 50/day) can't give its full per-minute rate to a whole run:
        # count at most the calls it has left today spread over a ~5-minute run
        caps.append(self.daily_calls_left(tokens_per_call) / 5)
        return min(caps)

    def daily_calls_left(self, tokens_per_call: int = 1600) -> float:
        left = float("inf")
        if self.p.rpd:
            left = min(left, self.p.rpd - self.day_requests)
        if self.p.tpd:
            left = min(left, (self.p.tpd - self.day_tokens) / tokens_per_call)
        return max(0.0, left)

    # --- HTTP -------------------------------------------------------------------------------------
    async def discover_free_model(self) -> str:
        """Ask an OpenRouter-style /models endpoint for a current ':free' model that supports tool calling."""
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        r = await self.http.get(f"{self.p.base_url.rstrip('/')}/models", headers=headers)
        r.raise_for_status()
        models = [m for m in (r.json().get("data") or [])
                  if str(m.get("id", "")).endswith(":free") and m.get("id") != self.model
                  and "tools" in (m.get("supported_parameters") or [])]
        if not models:
            return ""

        def rank(m):
            mid = m["id"].lower()
            pref = next((i for i, k in enumerate(FREE_MODEL_PREFERENCE) if k in mid), len(FREE_MODEL_PREFERENCE))
            return (pref, -(m.get("context_length") or 0))
        return sorted(models, key=rank)[0]["id"]

    async def list_models(self) -> list[str]:
        """Model ids the provider offers this key (OpenAI-compatible GET /models)."""
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        r = await self.http.get(f"{self.p.base_url.rstrip('/')}/models", headers=headers)
        if r.status_code != 200:
            return []
        try:
            data = r.json().get("data") or []
        except (ValueError, AttributeError):
            return []
        return [str(m.get("id")) for m in data if isinstance(m, dict) and m.get("id") and m.get("active", True)]

    def learn_limits(self, headers) -> None:
        """Read the provider's rate-limit headers: true per-model limits, and how much is left right now."""
        style = self.p.header_style
        if not style or headers is None:
            return
        names = {"groq": ("x-ratelimit-limit-requests", "x-ratelimit-limit-tokens",
                          "x-ratelimit-remaining-requests", "x-ratelimit-remaining-tokens", "x-ratelimit-reset-tokens"),
                 "cerebras": ("x-ratelimit-limit-requests-day", "x-ratelimit-limit-tokens-minute",
                              "x-ratelimit-remaining-requests-day", "x-ratelimit-remaining-tokens-minute",
                              "x-ratelimit-reset-tokens-minute")}.get(style)
        if not names:
            return
        rpd, tpm, left_req, left_tok, reset_tok = (_num(headers.get(n)) for n in names)
        changed = []
        if rpd and "rpd" not in self.overridden and int(rpd) != self.p.rpd:
            self.p.rpd = int(rpd)
            changed.append(f"{self.p.rpd}/day")
        if tpm and "tpm" not in self.overridden and int(tpm) != self.p.tpm:
            self.p.tpm = int(tpm)
            changed.append(f"{self.p.tpm // 1000}K tok/min")
        if changed:
            log.info("  %s: limits read from the provider: %s", self.label, ", ".join(changed))
        self.learned = True
        if left_req is not None and left_req <= 0 and rpd:
            self.exhausted = "daily request limit reached (provider headers)"
        if left_tok is not None and left_tok < 1500:
            wait = _duration(headers.get(names[4])) or 10.0
            self.cooldown_until = max(self.cooldown_until, time.monotonic() + wait)

    async def chat(self, body: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.provider == "openrouter":
            headers["X-Title"] = "Field-service ICP agents"
        payload = {"model": self.model, **body, **self.extra_body}
        if self.p.strict_tools and payload.get("tools"):
            payload["tools"] = [_without_required(t) for t in payload["tools"]]
        try:
            r = await self.http.post(f"{self.p.base_url.rstrip('/')}/chat/completions", json=payload, headers=headers)
        except Exception as exc:
            raise Transient(f"{type(exc).__name__}: {str(exc)[:160]}") from exc
        if r.status_code == 200:
            # some endpoints answer 200 with an empty body, an HTML page (a proxy, captive portal or wrong
            # base URL) or an error object; treat all of those as a failed call on this backend, never a crash
            try:
                data = r.json()
            except ValueError:
                kind = r.headers.get("content-type", "unknown type")
                snippet = " ".join(r.text.split())[:120] or "(empty body)"
                raise Transient(f"HTTP 200 but not JSON ({kind}): {snippet} - check ICP_{self.provider.upper()}_BASE_URL"
                                " / any proxy") from None
            if not isinstance(data, dict) or not data.get("choices"):
                err = data.get("error") if isinstance(data, dict) else None
                msg = (err.get("message") if isinstance(err, dict) else err) or str(data)[:160]
                raise Transient(f"HTTP 200 without an answer: {str(msg)[:160]}")
            self.learn_limits(r.headers)
            return data
        self.learn_limits(r.headers)
        text = _error_text(r.text)
        if r.status_code == 429:
            retry = _retry_after(r.headers.get("retry-after"), text)
            # a daily quota won't come back in seconds: retire this backend for the day instead of retrying
            raise RateLimited(f"HTTP 429: {text[:200]}", retry, bool(DAILY_RE.search(text)) and "minute" not in text.lower())
        if r.status_code in (401, 403):
            raise Fatal(f"key rejected (HTTP {r.status_code}) - check {self.p.key_env}: {text[:160]}")
        if r.status_code == 404:
            raise Fatal(f"model '{self.model}' not found (HTTP 404) - set ICP_{self.provider.upper()}_MODEL: {text[:160]}")
        if r.status_code in (400, 413, 422):
            raise BadRequest(f"HTTP {r.status_code}: {text[:300]}")
        raise Transient(f"HTTP {r.status_code}: {text[:200]}")


def _missing_share(answer, schema: dict) -> float:
    """Share of the schema's required fields that an answer leaves out or leaves empty (1.0 = no answer)."""
    required = schema.get("required") or []
    if not required:
        return 0.0
    if not isinstance(answer, dict):
        return 1.0
    missing = [k for k in required if answer.get(k) is None or answer.get(k) == "" or answer.get(k) == {}]
    return len(missing) / len(required)


def _without_required(tool: dict) -> dict:
    """A copy of a tool spec with every 'required' list removed (missing fields are filled in afterwards)."""
    def strip(node):
        if isinstance(node, dict):
            return {k: strip(v) for k, v in node.items() if k != "required"}
        if isinstance(node, list):
            return [strip(x) for x in node]
        return node
    return strip(tool)


def _num(v) -> float | None:
    try:
        return float(str(v).strip()) if v not in (None, "") else None
    except ValueError:
        return None


def _duration(v) -> float | None:
    """'7.66s', '1m2.5s', '120ms', '2h0m0s' or plain seconds -> seconds."""
    if v in (None, ""):
        return None
    txt = str(v).strip()
    plain = _num(txt)
    if plain is not None:
        return plain
    total, found = 0.0, False
    for val, unit in re.findall(r"([\d.]+)(ms|h|m|s)", txt):
        found = True
        total += float(val) * {"ms": 0.001, "h": 3600, "m": 60, "s": 1}[unit]
    return total if found else None


def _retry_after(header: str | None, body: str) -> float:
    if header:
        try:
            return max(1.0, float(header))
        except ValueError:
            pass
    m = RETRY_IN_RE.search(body)
    if m:
        val, unit = float(m.group(1)), (m.group(2) or "s").lower()
        return max(1.0, val / 1000 if unit == "ms" else val * 60 if unit.startswith("m") and unit != "ms" else val)
    return 20.0


def estimate_tokens(messages: list, tools: list | None, max_tokens: int) -> tuple[int, int]:
    chars = len(json.dumps(messages, ensure_ascii=False)) + len(json.dumps(tools or []))
    est_in = int(chars / 3.5) + 50
    return est_in + min(max_tokens, 900), est_in


# --- the pool ---------------------------------------------------------------------------------------
class ModelPool:
    can_fetch_remotely = False
    image_types = ("image/png", "image/jpeg", "image/webp")

    def __init__(self, backends: list[Backend], web: LocalWebTools | None = None, max_attempts: int = 8):
        if not backends:
            raise ValueError("the model pool has no backends - add at least one API key (see .env.example)")
        self.backends = backends
        self.web = web
        self.max_attempts = max_attempts
        self.usage: dict[str, Usage] = {}
        self.inflight: dict[int, dict] = {}
        self.failures = 0
        self._seq = itertools.count(1)
        self._ids = itertools.count(1)
        self._freed: asyncio.Event | None = None
        for b in backends:
            b.on_free = self._slot_freed
        self.model = "pool(" + ", ".join(b.label for b in backends) + ")"

    # --- labels / status used by the rest of the app ----------------------------------------------------
    @property
    def label(self) -> str:
        return "the model pool"

    @property
    def supports_vision(self) -> bool:
        return any(b.p.vision and not b.disabled for b in self.backends)

    def active(self) -> list[Backend]:
        return [b for b in self.backends if not b.disabled]

    def total_usage(self) -> Usage:
        total = Usage()
        for u in self.usage.values():
            total.add(u)
        return total

    def status_line(self) -> str:
        now = time.monotonic()
        parts = []
        for b in self.backends:
            state = ("off: " + b.disabled[:40]) if b.disabled else ("done for today" if b.exhausted else
                     (f"cooling {b.cooldown_until - now:.0f}s" if b.cooldown_until > now else f"{b.inflight} busy"))
            parts.append(f"{b.name} {b.stats['calls']} calls/{b.stats['rate_limited']} 429s ({state})")
        return "pool: " + " | ".join(parts)

    def capacity_per_minute(self, tokens_per_call: int = 1600) -> float:
        return sum(b.per_minute_capacity(tokens_per_call) for b in self.active() if not b.exhausted)

    def estimate_minutes(self, calls: int, tokens_per_call: int = 1600) -> float:
        """Minutes to make `calls` model calls with the pool's combined free limits."""
        cap = self.capacity_per_minute(tokens_per_call)
        daily = sum(min(b.daily_calls_left(tokens_per_call), 10**9) for b in self.active() if not b.exhausted)
        if cap <= 0 or daily < calls:
            return float("inf")
        return calls / cap

    # --- picking a backend ------------------------------------------------------------------------------
    def _pick(self, est: int, est_in: int, vision: bool, avoid: set,
              continuation: bool = False) -> tuple[Backend | None, float, list[str]]:
        now = time.monotonic()
        best, best_key, reasons = None, None, []
        for rank, b in enumerate(self.backends):
            why = b.usable(est, est_in, vision)
            if not why and continuation and b.p.own_tool_history:
                why = "can't continue a conversation another model started"
            if why or b.name in avoid:
                reasons.append(f"{b.name}: {why or 'failed on this request'}")
                continue
            wait = b.wait_for(est, now)
            key = (round(wait, 1), rank, b.inflight / max(1, b.p.concurrency))
            if best_key is None or key < best_key:
                best, best_key = b, key
        return best, (best_key[0] if best_key else 0.0), reasons

    def _slot_freed(self) -> None:
        if self._freed is not None:
            self._freed.set()

    async def _wait_capacity(self, seconds: float) -> None:
        """Sleep until capacity is expected, or until any in-flight call finishes (whichever is first)."""
        if self._freed is None:
            self._freed = asyncio.Event()
        ev = self._freed
        try:
            await asyncio.wait_for(ev.wait(), seconds)
        except asyncio.TimeoutError:
            pass
        if ev.is_set():
            self._freed = asyncio.Event()  # fresh event for the next round of waiters

    async def _complete(self, agent: str, purpose: str, body: dict, vision: bool, continuation: bool = False,
                        avoid: set | None = None) -> dict:
        est, est_in = estimate_tokens(body["messages"], body.get("tools"), body.get("max_tokens", 1000))
        avoid = set(avoid or ())
        preferred_avoid = set(avoid)
        attempts = 0
        cid = next(self._seq)
        entry = {"agent": agent, "purpose": purpose, "started": time.monotonic(), "attempt": 1, "state": "queued"}
        self.inflight[cid] = entry
        try:
            while True:
                backend, wait, reasons = self._pick(est, est_in, vision, avoid, continuation)
                if backend is None:
                    if avoid:  # everything failed once on this request: forget and let cool-downs decide
                        avoid.clear()
                        if preferred_avoid:
                            preferred_avoid.clear()
                        continue
                    self.failures += 1
                    raise LLMError(f"{agent}: no model available for '{purpose}' - " + "; ".join(reasons))
                if wait > 0:
                    entry["state"] = "queued"
                    entry["purpose"] = f"{purpose} (waiting {wait:.0f}s for capacity)"
                    await self._wait_capacity(min(wait, 3.0))
                    continue
                attempts += 1
                if attempts > self.max_attempts:
                    self.failures += 1
                    raise LLMError(f"{agent}: '{purpose}' failed on {attempts - 1} attempts across the pool")
                rec = backend.reserve(est)
                entry.update(state="calling", started=time.monotonic(), attempt=attempts,
                             purpose=f"{purpose} via {backend.name}")
                started = time.monotonic()
                try:
                    data = await asyncio.wait_for(backend.chat(body), backend.timeout)
                except asyncio.TimeoutError:
                    backend.settle(rec, None)
                    backend.stats["errors"] += 1
                    backend.cooldown_until = time.monotonic() + 15
                    avoid.add(backend.name)
                    log.warning("[%s] %s: %s timed out after %.0fs - trying another model", agent, purpose,
                                backend.name, backend.timeout)
                    continue
                except RateLimited as exc:
                    backend.settle(rec, None)
                    backend.stats["rate_limited"] += 1
                    if exc.daily:
                        backend.exhausted = "daily quota used up (HTTP 429)"
                        log.warning("[%s] %s: daily quota used up - skipping it for the rest of the day",
                                    agent, backend.name)
                    else:
                        # a provider that keeps refusing gets longer and longer pauses (20s, 40s, 80s ... 5 min)
                        backend.consecutive_429 += 1
                        if backend.consecutive_429 >= 5 and backend.stats["calls"] == 0:
                            # refused every single request this run (e.g. a free plan that isn't activated):
                            # stop trying it so it doesn't keep costing time
                            backend.disabled = "refused every request with HTTP 429 - check that provider's plan"
                            log.warning("%s: %s", backend.name, backend.disabled)
                            continue
                        pause = min(300.0, exc.retry_after * 2 ** (backend.consecutive_429 - 1))
                        backend.cooldown_until = time.monotonic() + pause
                        log.info("[%s] %s: rate limited, cooling down %.0fs - using another model", agent,
                                 backend.name, pause)
                    continue
                except Fatal as exc:
                    backend.settle(rec, None)
                    backend.disabled = str(exc)[:120]
                    log.error("%s disabled: %s", backend.name, exc)
                    continue
                except BadRequest as exc:
                    backend.settle(rec, None)
                    backend.stats["errors"] += 1
                    if backend.extra_body and any(k.split("_")[0] in str(exc).lower() for k in backend.extra_body):
                        log.info("%s rejected optional settings %s; retrying without them",
                                 backend.name, list(backend.extra_body))
                        backend.extra_body = {}
                        continue
                    log.warning("[%s] %s: %s rejected the request (%s) - trying another model", agent, purpose,
                                backend.name, str(exc)[:160])
                    avoid.add(backend.name)
                    continue
                except Transient as exc:
                    backend.settle(rec, None)
                    backend.stats["errors"] += 1
                    backend.cooldown_until = time.monotonic() + 10
                    avoid.add(backend.name)
                    log.warning("[%s] %s: %s problem (%s) - trying another model", agent, purpose, backend.name, exc)
                    continue
                except Exception as exc:  # unexpected: never let one provider's odd reply crash an agent
                    backend.settle(rec, None)
                    backend.stats["errors"] += 1
                    backend.cooldown_until = time.monotonic() + 10
                    avoid.add(backend.name)
                    log.warning("[%s] %s: %s unexpected %s: %s - trying another model", agent, purpose,
                                backend.name, type(exc).__name__, str(exc)[:160])
                    continue
                took = time.monotonic() - started
                usage = data.get("usage") or {}
                tin, tout = usage.get("prompt_tokens", est_in) or 0, usage.get("completion_tokens", 0) or 0
                backend.settle(rec, tin + tout)
                backend.consecutive_429 = 0
                backend.latencies.append(took)
                backend.stats["calls"] += 1
                backend.stats["in_tokens"] += tin
                backend.stats["out_tokens"] += tout
                backend.stats["seconds"] += took
                u = self.usage.setdefault(agent, Usage())
                u.calls += 1
                u.input_tokens += tin
                u.output_tokens += tout
                choice = (data.get("choices") or [{}])[0]
                msg = choice.get("message") or {}
                calls = [c.get("function", {}).get("name") for c in msg.get("tool_calls") or []]
                log.info("[%s] %s ok: %s | %.1fs | %s in / %s out tokens%s", agent, backend.label, purpose, took,
                         f"{tin:,}", f"{tout:,}", f" | calls {', '.join(calls)}" if calls else "")
                msg["_finish"] = choice.get("finish_reason")
                msg["_backend"] = backend.name
                return msg
        finally:
            self.inflight.pop(cid, None)

    async def expand_models(self) -> None:
        """Free tiers limit each MODEL separately, so one key can serve several models at once. For providers
        that allow it, add their other tool-capable models as extra pool members (each with its own limits)."""
        for b in list(self.backends):
            if not (b.p.discover_more and b.auto_model) or b.disabled:
                continue
            try:
                ids = await asyncio.wait_for(b.list_models(), 20)
            except Exception as exc:  # noqa: BLE001 - discovery is a bonus, never a failure
                log.debug("could not list %s models: %s", b.provider, exc)
                continue
            have = {x.model for x in self.backends if x.provider == b.provider}
            picks = [m for m in ids if m not in have and any(w in m.lower() for w in EXTRA_MODEL_WANT)
                     and not any(w in m.lower() for w in EXTRA_MODEL_SKIP)]
            picks.sort(key=lambda m: next((i for i, w in enumerate(EXTRA_MODEL_WANT) if w in m.lower()), 99))
            added = [b.clone(m) for m in picks[:MAX_EXTRA_MODELS]]
            if added:
                at = self.backends.index(b) + 1
                self.backends[at:at] = added
                for x in added:
                    x.on_free = self._slot_freed
                log.info("  %s: also using %s (free limits are per model)", b.provider,
                         ", ".join(x.model for x in added))
        self.model = "pool(" + ", ".join(b.label for b in self.backends) + ")"

    async def preflight(self) -> float:
        """Ping every backend once; disable the ones that don't answer. Fails only if none work."""
        started = time.monotonic()
        await self.expand_models()

        async def ping(b: Backend):
            t0 = time.monotonic()
            try:
                if getattr(b, "is_extra", False):
                    # an added model must really call tools, or it's no use to the agents
                    data = await asyncio.wait_for(b.chat({
                        "messages": [{"role": "user", "content": "Call submit_ok with ok=true."}],
                        "tools": [{"type": "function", "function": {
                            "name": "submit_ok", "description": "Confirm.",
                            "parameters": {"type": "object", "properties": {"ok": {"type": "boolean"}}}}}],
                        "tool_choice": "required", "max_tokens": 64}), 60)
                    msg = ((data.get("choices") or [{}])[0].get("message") or {})
                    if not msg.get("tool_calls"):
                        return b, "doesn't call tools", 0
                else:
                    await asyncio.wait_for(b.chat({"messages": [{"role": "user", "content": "Reply with OK."}],
                                                   "max_tokens": 16}), 60)
                b.day_requests += 1  # (a one-word ping says nothing about real call latency, so it isn't recorded)
                return b, "", time.monotonic() - t0
            except BadRequest as exc:
                if b.extra_body:
                    b.extra_body = {}
                    return await ping(b)
                return b, str(exc)[:160], 0
            except RateLimited as exc:
                if exc.daily:
                    b.exhausted = "daily quota used up"
                    return b, "daily quota already used up today", 0
                b.cooldown_until = time.monotonic() + exc.retry_after
                return b, "", -1.0  # reachable, just rate-limited right now
            except Fatal as exc:
                if b.p.discover_free and b.auto_model and ("not found" in str(exc) or "unavailable" in str(exc)):
                    try:
                        new = await b.discover_free_model()
                    except Exception as dexc:  # noqa: BLE001
                        return b, f"{exc} (and listing free models failed: {dexc})"[:200], 0
                    if new:
                        log.info("  %-10s %s is no longer free - switching to %s", b.name, b.model, new)
                        b.model, b.extra_body, b.auto_model = new, {}, False
                        return await ping(b)
                    return b, "no free model with tool calling is offered right now", 0
                return b, (str(exc) or type(exc).__name__)[:160], 0
            except (Transient, asyncio.TimeoutError) as exc:
                msg = (str(exc) or type(exc).__name__)[:200]
                if re.match(r"HTTP 5\d\d", msg) or isinstance(exc, asyncio.TimeoutError):
                    # overloaded / temporarily down: keep it, just start with a cool-down
                    b.cooldown_until = time.monotonic() + 60
                    b.preflight_note = msg
                    return b, "", -2.0
                return b, msg, 0
            except Exception as exc:  # anything unexpected disables this backend only, never the whole check
                return b, f"{type(exc).__name__}: {str(exc)[:140]}", 0

        results = await asyncio.gather(*(ping(b) for b in self.backends))
        for b, err, took in results:
            if err and not b.exhausted:
                b.disabled = err
                log.warning("  %-10s %-28s NOT USED: %s", b.name, b.model, err)
            elif b.exhausted:
                log.warning("  %-10s %-28s skipped: %s", b.name, b.model, b.exhausted)
            elif took == -2.0:
                log.warning("  %-10s %-28s busy right now (%s) - kept in the pool, will retry in 60s",
                            b.name, b.model, getattr(b, "preflight_note", "")[:120])
            elif took < 0:
                log.info("  %-10s %-28s OK (reachable, rate-limited this second; cooling down) limits: %s",
                         b.name, b.model, _limits_text(b.p))
            else:
                log.info("  %-10s %-28s OK (%.1fs) limits: %s", b.name, b.model, took, _limits_text(b.p))
        if not [b for b in self.backends if not b.disabled and not b.exhausted]:
            raise LLMError("no model in the pool is usable - check the API keys in .env (details above)")
        log.info("pool capacity: about %.0f model calls per minute combined", self.capacity_per_minute())
        return time.monotonic() - started

    # --- the agent loop (same interface as GeminiLLM / OllamaLLM) --------------------------------------
    def _new_id(self) -> str:
        return f"c{next(self._ids):08d}"  # 9 alphanumeric chars: accepted by every provider (Mistral is strictest)

    @staticmethod
    def _user_content(prompt: str | list) -> tuple[Any, bool]:
        if isinstance(prompt, str):
            return prompt, False
        parts, vision = [], False
        for part in prompt:
            if part.get("type") == "text":
                parts.append({"type": "text", "text": part["text"]})
            elif part.get("type") == "image":
                vision = True
                parts.append({"type": "image_url",
                              "image_url": {"url": f"data:{part['mime_type']};base64,{part['data']}"}})
        return parts, vision

    @staticmethod
    def _spec(t: ClientTool) -> dict:
        return {"type": "function", "function": {"name": t.name, "description": t.description,
                                                 "parameters": t.input_schema}}

    async def run(self, *, agent: str, system: str, prompt: str | list, submit: ClientTool,
                  tools: list[ClientTool] | None = None, server_tools: list[dict] | None = None,
                  max_turns: int = 8, max_tokens: int = 1200, force_submit: bool = False,
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
        if self.web is not None and (want_search or want_fetch):
            tools += self.web.tools(budget, want_search, want_fetch)
        by_name = {t.name: t for t in tools}
        content, vision = self._user_content(prompt)
        messages: list[dict] = [
            {"role": "system", "content": system + f"\n\nWhen ready, call {submit.name} with your final answer."},
            {"role": "user", "content": content}]
        result = LLMResult(output={})

        def finish(output: Any) -> LLMResult:
            output, filled = fill_required(output if isinstance(output, dict) else {}, submit.input_schema)
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

        forced = force_submit or not tools
        for turn in range(1, max_turns + 1):
            result.turns = turn
            if forced or turn == max_turns:
                body = {"messages": messages, "tools": [self._spec(submit)], "tool_choice": "required",
                        "max_tokens": max_tokens, "temperature": 0.2}
            else:
                body = {"messages": messages, "tools": [self._spec(t) for t in tools] + [self._spec(submit)],
                        "tool_choice": "auto", "max_tokens": max_tokens, "temperature": 0.2}
            continuation = any(m.get("tool_calls") for m in messages)
            redo_avoid: set = set()
            for redo in range(4):
                msg = await self._complete(agent, f"{purpose} (turn {turn})", body, vision, continuation, redo_avoid)
                calls = []
                for c in msg.get("tool_calls") or []:
                    fn = c.get("function") or {}
                    args = fn.get("arguments")
                    if isinstance(args, str):
                        args = repair_json(args) or {}
                    calls.append((self._new_id(), fn.get("name"), args or {}, c.get("id")))
                submitted = next((c for c in calls if c[1] == submit.name), None)
                answer = submitted[2] if submitted else (repair_json(msg.get("content") or "") if not calls else None)
                gap = _missing_share(answer, submit.input_schema) if (submitted or forced) else 0.0
                cut_off = msg.get("_finish") == "length" and (not calls or gap > 0)
                if redo < 3 and (gap >= 0.5 or cut_off):
                    # a half-empty or cut-off answer isn't a verdict: ask a different model instead
                    log.warning("[%s] %s: %s's answer was %s - asking another model", agent, purpose,
                                msg.get("_backend"), "cut off" if cut_off else f"missing {gap:.0%} of the fields")
                    redo_avoid.add(msg.get("_backend"))
                    continue
                break
            if submitted is not None:
                result.tool_calls.append(submit.name)
                return finish(submitted[2])
            if not calls:
                parsed = repair_json(msg.get("content") or "")
                if parsed and any(k in parsed for k in submit.input_schema.get("required", [])):
                    return finish(parsed)  # answered with JSON text instead of a tool call
                if forced and turn >= 2:
                    return finish(parsed or {})
                messages.append({"role": "assistant", "content": msg.get("content") or ""})
                messages.append({"role": "user", "content": f"Now call {submit.name} with your final answer."})
                forced = True
                continue
            messages.append({"role": "assistant", "content": msg.get("content") or None, "tool_calls": [
                {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
                for cid, name, args, _ in calls]})
            for cid, name, args, _ in calls:
                result.tool_calls.append(name)
                tool = by_name.get(name)
                if tool is None or tool.handler is None:
                    out: Any = f"Unknown tool {name}"
                else:
                    try:
                        out = await tool.handler(args)
                    except Exception as exc:
                        out = f"Tool error: {type(exc).__name__}: {exc}"
                messages.append({"role": "tool", "tool_call_id": cid,
                                 "content": out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)})
        return finish({})


def _error_text(body: str) -> str:
    """One readable line from an error body: the provider's own message if it's JSON, else the text squashed."""
    try:
        data = json.loads(body)
        if isinstance(data, list) and data:
            data = data[0]
        err = data.get("error", data) if isinstance(data, dict) else data
        msg = err.get("message") if isinstance(err, dict) else err
        if msg:
            return " ".join(str(msg).split())[:300]
    except (ValueError, AttributeError):
        pass
    return " ".join(body.split())[:300]


def _limits_text(p: Preset) -> str:
    bits = [f"{p.rpm}/min" if p.rpm else "", f"{p.rpd}/day" if p.rpd else "",
            f"{p.tpm // 1000}K tok/min" if p.tpm else "", f"{p.tpd // 1000}K tok/day" if p.tpd else "",
            f"{p.concurrency} at once"]
    return ", ".join(b for b in bits if b) or "none"


def build_pool(order: list[str], env: dict, web: LocalWebTools | None = None, http=None,
               timeout: float = 120.0) -> ModelPool:
    """Create backends for the providers in `order` that have an API key (Ollama needs none)."""
    backends = []
    for name in order:
        if name not in PRESETS:
            raise ValueError(f"unknown provider '{name}' in ICP_POOL (choose from {', '.join(PRESETS)})")
        base = PRESETS[name]
        up = name.upper()
        key = env.get(base.key_env, "") if base.key_env else ""
        if base.needs_key and not key:
            continue
        if base.retired:
            log.warning("  %-10s skipped: %s", name, base.retired)
            continue

        def num(suffix: str, default: int) -> int:
            try:
                return int(env.get(f"ICP_{up}_{suffix}") or default)
            except ValueError:
                return default
        from dataclasses import replace
        model = env.get(f"ICP_{up}_MODEL") or base.model
        # copy every preset field, then apply the .env overrides (so new preset fields are never lost)
        p = replace(base, base_url=env.get(f"ICP_{up}_BASE_URL") or base.base_url, model=model,
                    rpm=num("RPM", base.rpm), rpd=num("RPD", base.rpd), tpm=num("TPM", base.tpm),
                    tpd=num("TPD", base.tpd), concurrency=num("CONCURRENCY", base.concurrency),
                    extra_body=dict(base.extra_body) if model == base.model else {})
        # a local GPU model answers far slower than a cloud API: give it more time, give clouds less
        b = Backend(name, p, key, http=http, timeout=max(timeout, 180.0) if name == "ollama" else timeout)
        b.auto_model = not env.get(f"ICP_{up}_MODEL")
        b.overridden = {k.lower() for k in ("RPM", "RPD", "TPM", "TPD") if env.get(f"ICP_{up}_{k}")}
        backends.append(b)
        # extra models of the same provider, each with its own free limits: ICP_GROQ_MODELS=llama-3.3-70b-versatile,...
        for extra in [m.strip() for m in (env.get(f"ICP_{up}_MODELS") or "").split(",") if m.strip()]:
            if extra != b.model:
                backends.append(b.clone(extra))
        if (env.get(f"ICP_{up}_DISCOVER") or "").lower() in ("0", "false", "no", "off"):
            b.p.discover_more = False
    return ModelPool(backends, web=web)
