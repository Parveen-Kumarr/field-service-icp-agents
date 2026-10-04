"""Tests for the model pool (several free providers used together) and fast mode."""
import asyncio
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fake_openai import Behaviour, FakeProviders, big_site  # noqa: E402
from fake_ollama import fake_web  # noqa: E402
from icp_agents import run_team, scraper  # noqa: E402
from icp_agents.llm import ClientTool, LLMError  # noqa: E402
from icp_agents.pool import PRESETS, Backend, Preset, build_pool  # noqa: E402
from icp_agents.providers import make_llm  # noqa: E402
from icp_agents.websearch import LocalWebTools  # noqa: E402
from test_agents import settings, site  # noqa: E402

KEYS = {"GROQ_API_KEY": "g", "CEREBRAS_API_KEY": "c", "MISTRAL_API_KEY": "m", "GITHUB_TOKEN": "h",
        "OPENROUTER_API_KEY": "o", "GEMINI_API_KEY": "x"}
SUBMIT = ClientTool("submit_research", "Submit.", {"type": "object", "properties": {
    "canonical_name": {"type": "string"}}, "required": ["canonical_name"]})


def pool(fake, order=("groq", "cerebras"), env=None, **kw):
    return build_pool(list(order), env or KEYS, http=fake.client(), **kw)


def run(p, prompt="Company: Stryker (seen at x)", **kw):
    return asyncio.run(p.run(agent="Researcher", system="s", prompt=prompt, submit=SUBMIT, force_submit=True, **kw))


# --- building the pool -------------------------------------------------------------------------
def test_only_providers_with_keys_are_used_and_overrides_apply():
    p = build_pool(["groq", "cerebras", "mistral", "ollama"], {"GROQ_API_KEY": "g", "ICP_GROQ_MODEL": "qwen/qwen3.8-27b",
                                                               "ICP_GROQ_RPM": "10"})
    assert [b.name for b in p.backends] == ["groq", "ollama"]  # ollama needs no key
    g = p.backends[0]
    assert g.model == "qwen/qwen3.8-27b" and g.p.rpm == 10 and g.p.tpm == 8000
    assert g.extra_body == {}  # model-specific settings dropped for an overridden model


# --- limits ---------------------------------------------------------------------------------------
def test_rpm_tpm_and_concurrency_limits():
    b = Backend("t", Preset("http://x", "", "m", rpm=2, tpm=1000, concurrency=5, needs_key=False))
    now = time.monotonic()
    assert b.wait_for(300, now) == 0
    b.reserve(300); b.reserve(300)
    assert b.wait_for(300, now) > 55  # third request in the same minute must wait
    c = Backend("t2", Preset("http://x", "", "m", tpm=1000, concurrency=5, needs_key=False))
    c.reserve(900)
    assert c.wait_for(300, now) > 55  # token budget for this minute is spent
    d = Backend("t3", Preset("http://x", "", "m", concurrency=1, needs_key=False))
    d.reserve(10)
    assert 0 < d.wait_for(10, now) <= 1  # busy
    e = Backend("t4", Preset("http://x", "", "m", rpd=1, needs_key=False))
    e.reserve(10)
    assert "daily request limit" in e.usable(10, 10, False)
    assert "input limit" in Backend("gh", PRESETS["github"]).usable(9000, 9000, False)


# --- routing, failover, recovery --------------------------------------------------------------------
def test_rate_limited_provider_cools_down_and_work_moves_on():
    fake = FakeProviders({"groq": Behaviour(rate_limit_after=0, rate_limit_times=1)})
    p = pool(fake)
    res = run(p)
    assert res.output["canonical_name"] == "Stryker"
    assert [n for n, _ in fake.requests] == ["groq", "cerebras"]
    assert p.backends[0].stats["rate_limited"] == 1 and p.backends[0].cooldown_until > time.monotonic()


def test_daily_quota_retires_provider_for_the_day():
    fake = FakeProviders({"groq": Behaviour(daily_quota_after=1)})
    p = pool(fake)
    run(p); run(p); run(p)
    assert p.backends[0].exhausted and [n for n, _ in fake.requests].count("groq") == 2  # 1 ok + the 429
    assert "done for today" in p.status_line()


def test_server_errors_fail_over_and_bad_key_disables():
    fake = FakeProviders({"groq": Behaviour(fail_5xx_times=5)})
    p = pool(fake)
    assert run(p).output["canonical_name"] == "Stryker"
    assert fake.requests[-1][0] == "cerebras"

    import httpx

    def refuse(request):
        return httpx.Response(401, json={"error": "invalid api key"}) if "groq" in request.url.host else \
            httpx.Response(200, json={"choices": [{"message": {"content": '{"canonical_name": "X"}'}}]})
    p2 = build_pool(["groq", "cerebras"], KEYS, http=httpx.AsyncClient(transport=httpx.MockTransport(refuse)))
    assert run(p2).output == {"canonical_name": "X"}  # JSON text answer accepted
    assert "key rejected" in p2.backends[0].disabled


def test_unsupported_optional_setting_is_dropped():
    fake = FakeProviders({"groq": Behaviour(reject_extra=True)})
    p = pool(fake, order=("groq",))
    assert run(p).output["canonical_name"] == "Stryker"
    assert "reasoning_effort" in fake.requests[0][1] and "reasoning_effort" not in fake.requests[1][1]


def test_all_providers_unavailable_gives_a_clear_error():
    fake = FakeProviders({"groq": Behaviour(daily_quota_after=0.5)})
    p = pool(fake, order=("groq",))
    p.backends[0].exhausted = "daily quota used up"
    try:
        run(p)
        raise AssertionError("expected LLMError")
    except LLMError as exc:
        assert "no model available" in str(exc) and "groq: daily quota used up" in str(exc)


def test_conversation_moves_between_providers_with_portable_tool_ids():
    """A multi-turn task: turn 1 on groq calls a tool; groq then rate-limits; turn 2 continues on cerebras."""
    fake = FakeProviders({"groq": Behaviour(rate_limit_after=1, rate_limit_times=5)})
    p = pool(fake)
    asked = []

    async def ask(args):
        asked.append(args)
        return "They service hospital beds."
    tool = ClientTool("ask_researcher", "ask", {"type": "object", "properties": {"question": {"type": "string"}}}, ask)
    verdict = ClientTool("submit_verdict", "Submit.", {"type": "object", "properties": {"summary": {"type": "string"}},
                                                       "required": ["summary"]})
    res = asyncio.run(p.run(agent="Analyst", system="s", prompt="Account: SAM Service Inc.\nNick Cribb",
                            submit=verdict, tools=[tool]))
    assert asked and res.output["summary"]
    names = [n for n, _ in fake.requests]
    assert names[0] == "groq" and names[-1] == "cerebras"
    history = fake.requests[-1][1]["messages"]
    ids = [c["id"] for m in history if m.get("tool_calls") for c in m["tool_calls"]]
    assert ids and all(re.fullmatch(r"[a-zA-Z0-9]{9}", i) for i in ids)  # provider ids replaced by portable ones
    assert history[-1]["role"] == "tool" and history[-1]["tool_call_id"] == ids[0]


def test_images_only_go_to_vision_models():
    fake = FakeProviders()
    p = build_pool(["groq", "gemini"], KEYS, http=fake.client())
    logos = ClientTool("submit_logos", "Submit.", {"type": "object", "properties": {"logos": {"type": "array"}},
                                                   "required": ["logos"]})
    asyncio.run(p.run(agent="Scout", system="s", submit=logos, force_submit=True,
                      prompt=[{"type": "text", "text": "Logo 0:"},
                              {"type": "image", "data": "AAAA", "mime_type": "image/png"}]))
    name, body = fake.requests[-1]
    assert name == "gemini"
    assert body["messages"][1]["content"][1] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}


# --- capacity planning -------------------------------------------------------------------------
def test_free_tier_capacity_plan_for_a_full_run():
    """Planning check with the published free-tier limits: can ~70 accounts finish in 5 minutes?"""
    p = build_pool(["groq", "cerebras", "mistral", "openrouter", "gemini"], KEYS)
    n = 70
    calls = int(-(-n // 25) + n * 0.65 * 2.85 + 1)
    minutes = p.estimate_minutes(calls)
    assert calls < 140 and minutes <= 5, (calls, minutes, p.capacity_per_minute())
    solo = build_pool(["groq"], KEYS)  # one free provider alone can't
    assert solo.estimate_minutes(calls) > 5


# --- fast mode, whole team ----------------------------------------------------------------------------
def test_fast_mode_full_site_on_the_pool(tmp_path):
    fake = FakeProviders({"groq": Behaviour(latency=0.01, rate_limit_after=20, rate_limit_times=3),
                          "cerebras": Behaviour(latency=0.01), "mistral": Behaviour(latency=0.01, json_text=True)})
    s = settings(tmp_path, max_accounts=0)
    s.llm_provider, s.fast, s.concurrency, s.search_provider, s.search_interval = "pool", True, 12, "duckduckgo", 0
    s.pool_order, s.include_logos = ["groq", "cerebras", "mistral"], False
    fast_limits = {f"ICP_{p}_{k}": v for p in ("GROQ", "CEREBRAS", "MISTRAL")
                   for k, v in (("RPM", "10000"), ("TPM", "100000000"), ("TPD", "100000000"), ("RPD", "100000"))}
    llm = make_llm(s, pool_http=fake.client(), web_client=fake_web(), env={**KEYS, **fast_limits})
    llm.web.cooldown = 0
    fetcher = scraper.LiveFetcher("ua", client=big_site(50))
    started = time.monotonic()
    rt, team = asyncio.run(run_team(s, llm, fetcher=fetcher, timeout=60))
    took = time.monotonic() - started
    coord, analyst = team["coordinator"], team["analyst"]
    assert coord.failed is None and coord.report and not coord.report.get("error")
    n = len(coord.expected)
    assert n >= 60  # 50 speaker companies + sponsor wall
    triaged = [v for v in analyst.verdicts.values() if v.get("triaged")]
    researched = team["researcher"].records
    assert len(triaged) >= 20 and len(researched) + len(triaged) == n  # non-buyers skipped research entirely
    assert all(coord.status[a]["stage"] == "done" for a in coord.expected)
    calls = sum(b.stats["calls"] for b in llm.backends)
    assert calls <= int(len(researched) * 3.2 + 10)  # ~2-3 calls per researched account in fast mode
    assert {b.name for b in llm.backends if b.stats["calls"]} == {"groq", "cerebras", "mistral"}  # load spread
    assert any("Estimated model work" in m.text for m in rt.transcript)
    assert any(m.subject == "triage_accounts" for m in rt.transcript)
    assert took < 30


def test_non_json_200_reply_disables_only_that_provider():
    """Seen on a real Windows run: one endpoint answered HTTP 200 with an empty/HTML body and --check crashed."""
    import httpx

    def handler(request):
        if "groq" in request.url.host:
            return httpx.Response(200, text="", headers={"content-type": "text/html"})
        if "mistral" in request.url.host:
            return httpx.Response(200, json={"error": {"message": "model not available"}})
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "OK"}}],
                                         "usage": {"prompt_tokens": 5, "completion_tokens": 1}})
    p = build_pool(["groq", "mistral", "cerebras"], KEYS, http=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    asyncio.run(p.preflight())
    groq, mistral, cerebras = p.backends
    assert "not JSON" in groq.disabled and "(empty body)" in groq.disabled
    assert "model not available" in mistral.disabled
    assert not cerebras.disabled


def test_retired_github_models_is_skipped():
    p = build_pool(["groq", "github"], KEYS)
    assert [b.name for b in p.backends] == ["groq"]


def test_openrouter_switches_to_a_current_free_model_when_default_is_withdrawn():
    """Seen on a real run: 'openai/gpt-oss-120b:free' answered 404 'This model is unavailable for free'."""
    import httpx
    asked = []

    def handler(request):
        if request.method == "GET" and request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [
                {"id": "some/chat-only:free", "supported_parameters": ["max_tokens"], "context_length": 999999},
                {"id": "acme/small:free", "supported_parameters": ["tools", "tool_choice"], "context_length": 32000},
                {"id": "qwen/qwen3-big:free", "supported_parameters": ["tools"], "context_length": 131072},
                {"id": "paid/model", "supported_parameters": ["tools"], "context_length": 200000}]})
        model = json.loads(request.content)["model"]
        asked.append(model)
        if model == "openai/gpt-oss-120b:free":
            return httpx.Response(404, json={"error": {"message": "This model is unavailable for free."}})
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "OK"}}]})
    p = build_pool(["openrouter"], KEYS, http=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    asyncio.run(p.preflight())
    b = p.backends[0]
    assert not b.disabled and b.model == "qwen/qwen3-big:free" and asked[-1] == "qwen/qwen3-big:free"
    # a model the user chose in .env is never swapped silently
    p2 = build_pool(["openrouter"], {**KEYS, "ICP_OPENROUTER_MODEL": "openai/gpt-oss-120b:free"},
                    http=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    try:
        asyncio.run(p2.preflight())
    except LLMError:
        pass
    assert p2.backends[0].disabled and p2.backends[0].model == "openai/gpt-oss-120b:free"


def test_busy_provider_at_check_time_stays_in_the_pool():
    """Seen on a real run: Gemini answered 503 'high demand' during the check and was dropped for the whole run."""
    import httpx

    def handler(request):
        if "googleapis" in request.url.host:
            return httpx.Response(503, text='[{\n  "error": {\n    "code": 503,\n    "message": "This model is '
                                             'currently experiencing high demand."\n  }\n}]')
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "OK"}}]})
    p = build_pool(["groq", "gemini"], KEYS, http=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    asyncio.run(p.preflight())
    gemini = p.backends[1]
    assert not gemini.disabled and gemini.cooldown_until > time.monotonic()
    assert "\n" not in gemini.preflight_note and "high demand" in gemini.preflight_note


# --- several models per provider, limits learned from the provider ------------------------------------
def test_other_models_of_a_provider_join_as_their_own_members():
    """Free limits are per model: one Groq key can run gpt-oss-120b AND llama-3.3-70b AND llama-4 at once."""
    fake = FakeProviders({"groq": Behaviour(models=[
        "openai/gpt-oss-120b", "llama-3.3-70b-versatile", "meta-llama/llama-4-scout-17b-16e-instruct",
        "qwen/qwen3-32b", "whisper-large-v3", "meta-llama/llama-guard-4-12b", "llama-3.1-8b-instant",
        "openai/gpt-oss-20b"], no_tools_models=("qwen/qwen3-32b",))})
    p = pool(fake, order=("groq",))
    asyncio.run(p.preflight())
    names = [b.model for b in p.backends if not b.disabled]
    assert names[0] == "openai/gpt-oss-120b"
    assert {"openai/gpt-oss-20b", "llama-3.3-70b-versatile", "meta-llama/llama-4-scout-17b-16e-instruct"} <= set(names)
    assert not any(w in " ".join(names) for w in ("whisper", "guard", "8b-instant"))
    assert len(p.backends) == 4  # the default + at most 3 extras
    extras = [b for b in p.backends if getattr(b, "is_extra", False)]
    assert all(b.provider == "groq" and b.name.startswith("groq/") for b in extras)
    assert p.capacity_per_minute() > 3 * build_pool(["groq"], KEYS).capacity_per_minute()


def test_explicit_model_list_and_no_tool_models_dropped():
    fake = FakeProviders({"cerebras": Behaviour(no_tools_models=("llama3.1-8b",))})
    p = build_pool(["cerebras"], {**KEYS, "ICP_CEREBRAS_MODELS": "qwen-3-235b,llama3.1-8b", "ICP_CEREBRAS_DISCOVER": "0"},
                   http=fake.client())
    asyncio.run(p.preflight())
    state = {b.model: b.disabled for b in p.backends}
    assert state["gpt-oss-120b"] == "" and state["qwen-3-235b"] == "" and "tools" in state["llama3.1-8b"]


def test_limits_are_read_from_provider_headers():
    fake = FakeProviders({"groq": Behaviour(headers={
        "x-ratelimit-limit-requests": "14400", "x-ratelimit-limit-tokens": "18000",
        "x-ratelimit-remaining-requests": "14399", "x-ratelimit-remaining-tokens": "900",
        "x-ratelimit-reset-tokens": "7.5s"})})
    p = pool(fake, order=("groq",), env={**KEYS, "ICP_GROQ_DISCOVER": "0"})
    run(p)
    g = p.backends[0]
    assert g.p.rpd == 14400 and g.p.tpm == 18000
    assert g.cooldown_until - time.monotonic() > 5  # nearly out of tokens this minute: pause until the reset
    q = build_pool(["groq"], {**KEYS, "ICP_GROQ_TPM": "5000"})  # a limit set in .env always wins
    q.backends[0].learn_limits({"x-ratelimit-limit-tokens": "18000"})
    assert q.backends[0].p.tpm == 5000


def test_strict_provider_gets_schema_without_required_and_half_empty_answers_are_redone():
    import copy
    fake = FakeProviders({"groq": Behaviour(drop_fields=("summary", "scores", "disqualifier", "confidence"))})
    p = pool(fake, env={**KEYS, "ICP_GROQ_DISCOVER": "0", "ICP_CEREBRAS_DISCOVER": "0"})
    verdict = ClientTool("submit_verdict", "Submit.", {"type": "object", "properties": {
        "scores": {"type": "object"}, "disqualifier": {"type": "string"}, "summary": {"type": "string"},
        "confidence": {"type": "string"}, "note_to_strategist": {"type": "string"}},
        "required": ["scores", "disqualifier", "summary", "confidence", "note_to_strategist"]})
    res = asyncio.run(p.run(agent="Analyst", system="s", prompt="Account: ABB\nx", submit=verdict, force_submit=True))
    sent_groq = next(b for n, b in fake.requests if n == "groq")
    assert "required" not in json.dumps(sent_groq["tools"])  # Groq validates 'required' and rejects the call
    assert [n for n, _ in fake.requests] == ["groq", "cerebras"]  # 4 of 5 fields missing -> asked cerebras
    assert res.output["summary"] == "ABB scored."


def test_gemini_never_continues_another_models_conversation_and_429s_back_off():
    fake = FakeProviders({"groq": Behaviour(rate_limit_after=1, rate_limit_times=9)})
    p = build_pool(["groq", "gemini"], KEYS, http=fake.client())
    asked = []

    async def ask(args):
        asked.append(args)
        return "hospital beds"
    tool = ClientTool("ask_researcher", "ask", {"type": "object", "properties": {"question": {"type": "string"}}}, ask)
    verdict = ClientTool("submit_verdict", "Submit.", {"type": "object", "properties": {"summary": {"type": "string"}},
                                                       "required": ["summary"]})
    try:
        asyncio.run(asyncio.wait_for(p.run(agent="Analyst", system="s", prompt="Account: SAM\nNick Cribb",
                                           submit=verdict, tools=[tool]), 4))
    except (asyncio.TimeoutError, LLMError):
        pass
    after_first = [n for n, b in fake.requests if any(m.get("tool_calls") for m in b["messages"])]
    assert asked and "gemini" not in after_first
    g = p.backends[0]
    assert g.consecutive_429 >= 2 and g.cooldown_until - time.monotonic() > 1.5  # 1s, then 2s, 4s ...
