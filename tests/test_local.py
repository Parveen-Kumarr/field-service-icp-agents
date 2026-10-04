"""Tests for the local-model option: Ollama client, local web tools, and a full team run on it."""
import asyncio
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fake_ollama import FakeOllama, fake_web  # noqa: E402
from icp_agents import run_team  # noqa: E402
from icp_agents.config import Settings  # noqa: E402
from icp_agents.llm import ClientTool, LLMError, describe_error, search_tool, url_tool  # noqa: E402
from icp_agents.local_llm import OllamaLLM  # noqa: E402
from icp_agents.providers import make_llm  # noqa: E402
from icp_agents.websearch import LocalWebTools, ToolBudget, parse_ddg_html, parse_searxng  # noqa: E402

FIX = ROOT / "tests" / "fixtures"
SUBMIT = ClientTool("submit_x", "Submit.", {"type": "object", "properties": {"answer": {"type": "string"}},
                                            "required": ["answer"]})


# --- settings / factory ----------------------------------------------------------------
def test_provider_defaults(monkeypatch):
    for k in ("ICP_LLM_PROVIDER", "ICP_LLM_MODEL", "ICP_CONCURRENCY", "ICP_LLM_TIMEOUT", "ICP_FAST"):
        monkeypatch.delenv(k, raising=False)
    s = Settings()
    assert (s.llm_provider, s.concurrency, s.llm_timeout, s.fast) == ("pool", 12, 45.0, True)  # default
    monkeypatch.setenv("ICP_LLM_PROVIDER", "ollama")
    s = Settings()
    assert (s.model, s.concurrency, s.llm_timeout, s.fast) == ("gemma4:12b", 2, 300.0, False)
    monkeypatch.setenv("ICP_LLM_PROVIDER", "gemini")
    s = Settings()
    assert (s.model, s.concurrency, s.llm_timeout) == ("gemini-3.8-flash", 4, 180.0)
    monkeypatch.setenv("ICP_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("ICP_LLM_MODEL", "qwen3:14b")
    assert Settings().model == "qwen3:14b"
    assert isinstance(make_llm(Settings()), OllamaLLM)


def test_gemini_daily_quota_is_not_retried():
    class Quota(Exception):
        status_code = 429
    msg, transient = describe_error(Quota("Rate limit exceeded for model gemini-3.8-flash (limit: 20 requests "
                                          "per day on Free Tier)"))
    assert not transient and "daily quota exhausted" in msg and "ICP_LLM_PROVIDER=ollama" in msg
    msg, transient = describe_error(Quota("Too many requests per minute"))
    assert transient


def test_gemini_strips_private_tool_markers():
    tool = search_tool(4)
    assert tool["_max_uses"] == 4
    sent = {}

    class C:
        async def create(self, **body):
            sent.update(body)
            from types import SimpleNamespace as NS
            return NS(id="i", status="completed", errors=None, usage=None,
                      steps=[{"type": "function_call", "id": "f", "name": "submit_x", "arguments": {"answer": "ok"}}])
    from types import SimpleNamespace as NS
    from icp_agents.llm import GeminiLLM
    asyncio.run(GeminiLLM("g", client=NS(aio=NS(interactions=C()))).run(
        agent="R", system="s", prompt="p", submit=SUBMIT, server_tools=[tool, url_tool()]))
    assert sent["tools"][0] == {"type": "google_search"}


# --- local web tools ------------------------------------------------------------------
def test_parse_duckduckgo_results():
    res = parse_ddg_html((FIX / "ddg_results.html").read_text())
    assert [r["url"] for r in res] == ["https://example.com/stryker-procare", "https://example.com/stryker-about"]
    assert "1,000+ field service engineers" in res[0]["snippet"]  # ads and DDG links are dropped


def test_parse_searxng():
    assert parse_searxng({"results": [{"title": "T", "url": "https://a.com", "content": "c"}, {"title": "x"}]}) == [
        {"title": "T", "url": "https://a.com", "snippet": "c"}]


def test_search_and_fetch_tools_are_capped_and_record_sources():
    async def go():
        web = LocalWebTools("ua", client=fake_web(), page_chars=60, provider="duckduckgo", min_interval=0)
        budget = ToolBudget(max_searches=1, max_fetches=1)
        search, fetch = web.tools(budget, True, True)
        r1 = await search.handler({"query": "Stryker field service"})
        r2 = await search.handler({"query": "again"})
        page = await fetch.handler({"url": r1["results"][0]["url"]})
        over = await fetch.handler({"url": "https://example.com/x"})
        return r1, r2, page, over, budget
    r1, r2, page, over, budget = asyncio.run(go())
    assert r1["results"][0]["url"] == "https://example.com/stryker"
    assert "limit reached" in r2["error"] and "limit reached" in over["error"]
    assert page["title"].startswith("stryker") and len(page["text"]) <= 60 and page["truncated"]
    assert [s["via"] for s in budget.sources] == ["fetch_page"]  # one entry per URL; reading it upgrades it
    assert budget.searches == 1 and budget.fetches == 1


def test_blocked_duckduckgo_gives_a_clear_error():
    async def go():
        web = LocalWebTools("ua", client=fake_web(ddg_blocked=True), provider="duckduckgo", min_interval=0, cooldown=0)
        [search] = web.tools(ToolBudget(), True, False)
        return await search.handler({"query": "x"})
    assert "rate limiting" in asyncio.run(go())["error"]


# --- Ollama client ---------------------------------------------------------------------
def test_ollama_preflight_explains_missing_model_and_stopped_server():
    fake = FakeOllama(installed=("qwen3:8b",))
    llm = OllamaLLM("gemma4:12b", http=fake.client())
    try:
        asyncio.run(llm.preflight())
        raise AssertionError("expected LLMError")
    except LLMError as exc:
        assert "ollama pull gemma4:12b" in str(exc) and "qwen3:8b" in str(exc)

    def refuse(request):
        raise httpx.ConnectError("connection refused", request=request)
    down = OllamaLLM("gemma4:12b", http=httpx.AsyncClient(base_url="http://localhost:11434",
                                                         transport=httpx.MockTransport(refuse)))
    try:
        asyncio.run(down.preflight())
        raise AssertionError("expected LLMError")
    except LLMError as exc:
        assert "start the Ollama app" in str(exc)

    ok = OllamaLLM("gemma4:12b", http=FakeOllama().client())
    assert asyncio.run(ok.preflight()) >= 0


def test_ollama_images_go_in_the_images_field():
    msg = OllamaLLM._user_message([{"type": "text", "text": "Logo 0:"},
                                   {"type": "image", "data": "AAAA", "mime_type": "image/png"},
                                   {"type": "image", "data": "BBBB", "mime_type": "image/webp"}], ("image/png",))
    assert msg == {"role": "user", "content": "Logo 0:\n[image 1]", "images": ["AAAA"]}


# --- the whole team on the local model ----------------------------------------------------
def test_full_team_run_on_local_model(tmp_path):
    from test_agents import settings, site
    fake = FakeOllama()
    s = settings(tmp_path)
    s.llm_provider, s.model, s.search_provider, s.search_interval = "ollama", "gemma4:12b", "duckduckgo", 0
    llm = make_llm(s, ollama_http=fake.client(), web_client=fake_web())
    llm.web.cooldown = 0
    rt, team = asyncio.run(run_team(s, llm, fetcher=site(), timeout=60))
    coord = team["coordinator"]
    assert coord.failed is None and coord.report and Path(coord.report["xlsx"]).exists()

    # the vendor briefing came from reading the vendor site locally with fetch_page
    assert team["analyst"].briefing["source"] == "live"
    # research used local web_search + fetch_page and recorded both as sources
    rec = team["researcher"].records["stryker"]
    assert rec["sources"][0] == {"url": "https://example.com/stryker", "title": "stryker | example",
                                 "via": "fetch_page"}  # found by web_search, then read in full
    assert rec["canonical_name"] == "Stryker"  # came from the forced structured answer
    # the same conversations happen: a question with a linked answer, and a challenge that revises a verdict
    q = next(m for m in rt.transcript if m.sender == "Analyst" and m.performative == "request")
    assert any(m.in_reply_to == q.id and "mechanical services contractor" in m.text for m in rt.transcript)
    assert team["analyst"].verdicts["tetra pak"]["tier"].startswith("B")
    assert team["analyst"].verdicts["stryker"]["total"] == 92
    # logos without alt text were read from images by the local model
    assert {"tetra pak", "omron"} <= set(coord.expected)
    # every forced answer was a structured request without tools; string arguments were parsed
    assert any("format" in r for r in fake.requests)
    u = llm.total_usage()
    assert u.calls == len(fake.requests) and u.web_searches > 0 and u.url_fetches > 0
