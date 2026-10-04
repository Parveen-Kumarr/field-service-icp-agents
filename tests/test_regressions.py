"""Regression tests for the failures seen in the first real local run (run_20260930_014930.log):

1. The run finished all accounts, then hung for 7.5 hours without writing a report
   (KeyError in the Coordinator's wrap-up left the run waiting forever).
2. The local model ran out of output tokens (6,000 tokens, 150 s) and cut its JSON off;
   the partial object had no 'canonical_name' / 'summary' -> KeyError in three agents.
3. DuckDuckGo refused 41 of 44 searches; the model kept issuing new searches for minutes.
4. The speakers page was built by JavaScript, so 0 speakers were found.
"""
import asyncio
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fake_gemini import FakeGemini  # noqa: E402
from fake_ollama import fake_web  # noqa: E402
from icp_agents import LLM, run_team  # noqa: E402
from icp_agents import scraper  # noqa: E402
from icp_agents.llm import ClientTool  # noqa: E402
from icp_agents.local_llm import OllamaLLM  # noqa: E402
from icp_agents.schema_tools import fill_required, repair_json  # noqa: E402
from icp_agents.websearch import LocalWebTools, ToolBudget  # noqa: E402
from test_agents import BASE, FIX, settings, site  # noqa: E402

VERDICT_LIKE = ClientTool("submit_research", "Submit.", {
    "type": "object",
    "properties": {"canonical_name": {"type": "string"}, "confidence": {"type": "string", "enum": ["high", "low"]},
                   "facts": {"type": "array", "items": {"type": "object"}}},
    "required": ["canonical_name", "confidence", "facts"]})


# --- 1. the report is never lost --------------------------------------------------------
def test_run_ends_and_keeps_partial_report_even_if_final_report_fails(tmp_path, monkeypatch):
    from icp_agents.agents.reporter import ReporterAgent

    def broken_write(self):
        raise RuntimeError("disk full")
    monkeypatch.setattr(ReporterAgent, "write", broken_write)
    rt, team = asyncio.run(run_team(settings(tmp_path, max_accounts=4), LLM("g", client=FakeGemini()),
                                    fetcher=site(), timeout=30))  # before the fix this waited forever
    coord = team["coordinator"]
    assert rt.finished.is_set() and "disk full" in coord.report["error"]
    partial = tmp_path / "FSN_West_ICP_partial.xlsx"
    assert partial.exists() and (tmp_path / "agent_conversation_partial.md").exists()


def test_wrap_up_survives_verdicts_with_missing_fields(tmp_path):
    async def go():
        from icp_agents import build_team
        rt, team = build_team(settings(tmp_path, max_accounts=2), LLM("g", client=FakeGemini()), fetcher=site())
        rt.start()
        team["analyst"].verdicts["x"] = {"account": "X", "tier": "C - Nurture"}  # no summary/total
        coord = team["coordinator"]
        coord.expected, coord.status = ["x"], {"x": {"stage": "done"}}
        coord.report = None
        await coord._check_done()
        await rt.stop()
        return coord, rt
    coord, rt = asyncio.run(go())
    assert rt.finished.is_set() and coord.report and not coord.report.get("error")


# --- 2. truncated / incomplete model answers -------------------------------------------------
def test_repair_truncated_json_keeps_complete_fields():
    cut = '{"canonical_name": "Acme Service AI", "confidence": "high", "facts": [{"claim": "a"}, {"claim": "very long and cu'
    assert repair_json(cut) == {"canonical_name": "Acme Service AI", "confidence": "high", "facts": [{"claim": "a"}]}
    assert repair_json('Sure! {"a": 1} Hope that helps') == {"a": 1}


def test_missing_and_invalid_fields_get_neutral_defaults():
    out, filled = fill_required({"confidence": "maybe"}, VERDICT_LIKE.input_schema)
    assert out == {"confidence": "low", "canonical_name": "not provided", "facts": []}
    assert set(filled) == {"canonical_name", "confidence", "facts"}


def _ollama(replies: list[dict]) -> tuple[OllamaLLM, list]:
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        r = replies.pop(0)
        return httpx.Response(200, json={"message": {"role": "assistant", "content": r["content"]},
                                         "done_reason": r.get("done_reason", "stop"),
                                         "prompt_eval_count": 100, "eval_count": 50})
    http = httpx.AsyncClient(base_url="http://localhost:11434", transport=httpx.MockTransport(handler))
    return OllamaLLM("gemma4:12b", http=http, max_output=2048), sent


def test_cut_off_answer_is_retried_shorter_then_repaired():
    cut = {"content": '{"canonical_name": "Baxter Planning", "facts": [{"claim": "one"}, {"claim": "tw',
           "done_reason": "length"}
    llm, sent = _ollama([cut, dict(cut)])
    res = asyncio.run(llm.run(agent="Researcher", system="s", prompt="p", submit=VERDICT_LIKE, force_submit=True,
                              max_tokens=6000))
    assert res.output == {"canonical_name": "Baxter Planning", "facts": [{"claim": "one"}], "confidence": "low"}
    assert len(sent) == 2 and "Be concise" in sent[1]["messages"][-1]["content"]
    assert all(b["options"]["num_predict"] == 2048 for b in sent)  # the 6,000-token ask was capped
    assert all(b["options"]["repeat_penalty"] > 1 for b in sent)


def test_cut_off_answer_recovered_by_the_shorter_retry():
    llm, sent = _ollama([{"content": '{"canonical_name": "X", "fa', "done_reason": "length"},
                         {"content": '{"canonical_name": "X", "confidence": "high", "facts": []}'}])
    res = asyncio.run(llm.run(agent="R", system="s", prompt="p", submit=VERDICT_LIKE, force_submit=True))
    assert res.output == {"canonical_name": "X", "confidence": "high", "facts": []}


# --- 3. refused searches ----------------------------------------------------------------------
def test_repeated_refusals_pause_search_and_tell_the_model_to_stop():
    async def go():
        web = LocalWebTools("ua", client=fake_web(ddg_blocked=True), provider="duckduckgo", min_interval=0,
                            cooldown=0, give_up_after=2)
        budget = ToolBudget(max_searches=4)
        [search] = web.tools(budget, True, False)
        first = await search.handler({"query": "Baxter Planning integrations"})
        second = await search.handler({"query": "Circuitry.ai website"})
        [other_task] = web.tools(ToolBudget(max_searches=4), True, False)  # another account's research
        third = await other_task.handler({"query": "GoFormz company"})
        return web, budget, first, second, third
    web, budget, first, second, third = asyncio.run(go())
    assert web.paused_for() > 0  # searching stops instead of hammering a blocked engine
    assert "do NOT call web_search again" in first["error"]
    assert "limit reached" in second["error"]  # this task gets no more searches
    assert budget.max_searches == budget.searches == 1
    assert "paused" in third["error"] and "do NOT call web_search again" in third["error"]
    assert web.stats["refused"] == 2 and web.stats["skipped"] == 1


def test_searches_are_spaced_out():
    async def go():
        web = LocalWebTools("ua", client=fake_web(), provider="duckduckgo", min_interval=0.2)
        t0 = asyncio.get_running_loop().time()
        await asyncio.gather(web.search("a field service"), web.search("b field service"), web.search("c"))
        return asyncio.get_running_loop().time() - t0
    assert asyncio.run(go()) >= 0.4


def test_ddgs_provider_maps_results():
    class FakeDDGS:
        def text(self, query, max_results, backend):
            assert backend == "auto"
            return [{"title": "Stryker ProCare", "href": "https://www.stryker.com/procare", "body": "Service."}]
    web = LocalWebTools("ua", provider="ddgs", ddgs_factory=FakeDDGS, min_interval=0)
    assert asyncio.run(web.search("Stryker")) == [
        {"title": "Stryker ProCare", "url": "https://www.stryker.com/procare", "snippet": "Service."}]


# --- 4. JavaScript-built speakers page ------------------------------------------------------------
def test_js_built_speakers_page_is_rendered_and_saved(tmp_path, monkeypatch):
    shell = "<html><body><div id='app'></div><script src='/app.js'></script></body></html>"
    pages = {"/": (FIX / "home.html").read_text(), "/speakers": shell, "/sponsors": (FIX / "sponsors.html").read_text()}

    def handler(request):
        if request.url.path.startswith("/media/logos/"):
            return httpx.Response(200, content=f"PNG:{request.url.path}".encode(), headers={"content-type": "image/png"})
        body = pages.get(request.url.path)
        return httpx.Response(200, text=body) if body else httpx.Response(405)

    async def fake_render(url, timeout_ms=45000):
        assert url == BASE + "speakers"
        return (FIX / "speakers.html").read_text()  # what the browser sees after JavaScript runs
    monkeypatch.setattr(scraper, "render_with_browser", fake_render)

    s = settings(tmp_path, max_accounts=3, render_js=True)
    fetcher = scraper.LiveFetcher("ua", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    rt, team = asyncio.run(run_team(s, LLM("g", client=FakeGemini()), fetcher=fetcher, timeout=30))
    coord = team["coordinator"]
    # named speakers at non-sponsor companies first (Aquant sponsors the event, so it comes after them)
    assert set(coord.expected) == {"ciena", "crown equipment", "stryker"}
    assert (tmp_path / "pages" / "speakers.html").read_text() == shell
    assert (tmp_path / "pages" / "speakers_rendered.html").exists()
    assert any("after rendering it in a headless browser" in m.text for m in rt.transcript)


def test_speakers_found_in_embedded_page_data():
    data = ('<script>self.__next_f.push([1,"{\\"speakers\\":[{\\"name\\":\\"Sara Smith\\",\\"jobTitle\\":\\"Service AI '
            'Program Owner\\",\\"company\\":\\"Waters Corporation\\"}]}"])</script>')
    html = data.replace('\\"', '"')
    got = scraper.parse_speakers(html)
    assert got == [{"name": "Sara Smith", "title": "Service AI Program Owner", "company": "Waters Corporation"}]
