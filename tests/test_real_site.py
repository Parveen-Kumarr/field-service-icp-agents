"""Regression tests from the real Field Service Next / Service Next West site (pages saved on 1 Oct 2026).

What the real run showed:
- The event had rolled over to the 2027 edition: /speakers had no names yet, only
  "Want to see who spoke previously? View the list here" -> /speakers/2026. The Scout skipped
  archive links, so it found 0 speakers.
- Navigation links point at a sibling subdomain (servicenextwest.wbresearch.com).
- The headless browser got "403 Forbidden".
- Home-page logos had alt="img" and src with a trailing space ('logos_0024_GE.jpg '), which broke
  the image-type check, so names came from file names: 'Logos 0024 Ge'. The organiser's own logo
  ('wbrevent') was listed as a company.
"""
import asyncio
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fake_ollama import fake_web  # noqa: E402
from fake_openai import FakeProviders  # noqa: E402
from icp_agents import run_team, scraper  # noqa: E402
from icp_agents.agents.scout import _name_from_filename  # noqa: E402
from icp_agents.providers import make_llm  # noqa: E402
from test_agents import settings  # noqa: E402
from fake_openai import Behaviour  # noqa: E402,F401

REAL = ROOT / "tests" / "fixtures" / "real"
BASE = "https://fieldserviceusa.wbresearch.com/"
SPEAKERS_2026 = "<html><body><h1>2026 Speakers</h1>" + "".join(
    f'<div class="speaker-card"><img alt="{n}" src="/h/{i}.jpg"><h4>{n}</h4><p>{t}</p><p>{c}</p></div>'
    for i, (n, t, c) in enumerate([
        ("Scott Day", "COO", "Crane 1 Services"),
        ("David Mueller", "Vice President Global Service", "Hach"),
        ("Greg Friesen", "VP & General Manager Global Services", "Ciena"),
        ("Fabio Raffone", "VP Service Operation Americas", "Tetra Pak"),
        ("Sara Smith", "Service AI Program Owner", "Waters Corporation")])) + "</body></html>"


def real_site():
    pages = {"/": (REAL / "home.html").read_text(), "/speakers": (REAL / "speakers.html").read_text(),
             "/sponsors": (REAL / "sponsors.html").read_text(), "/speakers/2026": SPEAKERS_2026,
             "/landing/attendee-list-email": "<html><body><h1>Get the attendee list</h1><form></form></body></html>"}

    def handler(request):
        if request.url.path.startswith(("/UploadedFiles", "/eco")) or "iqpc" in request.url.host:
            return httpx.Response(200, content=b"GIF89a", headers={"content-type": "image/gif"})
        body = pages.get(request.url.path.rstrip("/") or "/")
        return httpx.Response(200, text=body) if body else httpx.Response(404)
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_previous_edition_link_and_sibling_subdomain_navigation():
    html = (REAL / "speakers.html").read_text()
    assert scraper.previous_speakers_link(html, BASE + "speakers") == "https://servicenextwest.wbresearch.com/speakers/2026"
    found = scraper.discover_pages((REAL / "home.html").read_text(), BASE)
    assert found["speakers"] == "https://servicenextwest.wbresearch.com/speakers"
    assert found["sponsors"].endswith("/sponsors")


def test_logo_file_names_and_blocked_pages():
    assert _name_from_filename("/UploadedFiles/EventPage/315401/images/logos_0024_GE.jpg ") == "GE"
    assert _name_from_filename("/x/logos_0023_honeywell.svg.jpg") == "Honeywell"
    assert _name_from_filename("/x/logos_0019_johnson-controls.jpg") == "Johnson Controls"
    assert _name_from_filename("/p/GoFormz_yNK2EOTeeFiyVsyblaUNHaGus5prceobyOCAdLxX.png") == "GoFormz"
    assert _name_from_filename("/p/OjT3g29qdRGmvAyL0HiASiNox6X1MzRWw9a97P02.png") == ""
    assert scraper.ORGANISER_NAMES.match("Wbrevent")
    assert scraper.looks_blocked("<html><head><title>403 Forbidden</title></head><body></body></html>")
    assert not scraper.looks_blocked((REAL / "speakers.html").read_text())
    groups = scraper.logo_groups((REAL / "home.html").read_text(), BASE)
    assert all(not l["src"].endswith(" ") for logos in groups.values() for l in logos)


def test_full_run_on_the_real_pages(tmp_path):
    fake = FakeProviders()
    s = settings(tmp_path, max_accounts=0)
    s.event_url, s.render_js = BASE, False
    s.llm_provider, s.fast, s.concurrency, s.search_provider, s.search_interval = "pool", True, 8, "duckduckgo", 0
    s.pool_order = ["groq", "cerebras"]  # no vision model: home logos are named from their files
    llm = make_llm(s, pool_http=fake.client(), web_client=fake_web(),
                   env={"GROQ_API_KEY": "g", "CEREBRAS_API_KEY": "c", "ICP_GROQ_TPM": "10000000",
                        "ICP_GROQ_RPM": "10000", "ICP_CEREBRAS_RPM": "10000", "ICP_CEREBRAS_TPM": "10000000"})
    llm.web.cooldown = 0
    rt, team = asyncio.run(run_team(s, llm, fetcher=scraper.LiveFetcher("ua", client=real_site()), timeout=60))
    scout, coord = team["scout"], team["coordinator"]
    assert coord.failed is None and coord.report and not coord.report.get("error")
    accts = scout.accounts
    speakers = [a for a in accts.values() if a["contacts"]]
    assert {a["company"] for a in speakers} >= {"Crane 1 Services", "Hach", "Ciena", "Tetra Pak", "Waters Corporation"}
    assert all("Speaker (2026 edition)" in a["roles"] for a in speakers)
    names = {a["company"] for a in accts.values()}
    assert {"GE", "Honeywell", "Johnson Controls", "Siemens"} <= names
    assert not any(n.lower().startswith("logos") or "wbr" in n.lower() for n in names)
    assert not any("Attendee list" in a["roles"] for a in accts.values())  # the attendee page is only a form
    sponsors = [a for a in accts.values() if "Sponsor / Exhibitor" in a["roles"]]
    assert sponsors and not any("Attendee logo (home page)" in a["roles"] for a in sponsors)
    texts = " ".join(m.text for m in rt.transcript)
    assert "who spoke previously" in texts and "2026" in texts
    assert any(p.role == "speakers_2026" for p in scout.pages)


def test_scores_in_any_shape_and_short_search_queries():
    from icp_agents.agents.analyst import compute
    from icp_agents.agents.researcher import _search_query
    total, tier, clean = compute('{"industry": {"points": 25, "why": "x"}, "assets": 18, "scale": "12/15"}', None)
    assert total == 55 and clean["assets"]["points"] == 18 and clean["scale"]["points"] == 12
    assert compute("garbage", None)[0] == 0
    q = _search_query("Coherent Corporation", "Any recent (last 6-12 months) news on Coherent Corporation's field "
                      "service, customer service org, or AI/digital service initiatives - e.g., a service transformation")
    assert q.startswith("Coherent Corporation ") and len(q.split()) <= 9 and "recent" not in q


def test_searches_run_in_parallel_and_no_results_is_not_a_refusal():
    import time as _t
    from icp_agents.websearch import LocalWebTools

    class SlowDDGS:
        def text(self, query, max_results, backend):
            _t.sleep(0.3)
            if "nothing" in query:
                raise RuntimeError("No results found.")
            return [{"title": query, "href": "https://x.example/" + query.replace(" ", "-"), "body": "b"}]
    web = LocalWebTools("ua", provider="ddgs", ddgs_factory=SlowDDGS, min_interval=0.05, parallel=3, cooldown=5)

    async def go():
        t0 = asyncio.get_running_loop().time()
        res = await asyncio.gather(web.search("a"), web.search("b"), web.search("c"), web.search("nothing here"))
        return asyncio.get_running_loop().time() - t0, res
    took, res = asyncio.run(go())
    assert took < 0.9  # 4 searches of 0.3 s, 3 at a time - not 1.2 s+ in a row, and no 5 s cool-down
    assert res[3] == [] and web.stats["refused"] == 0


# --- from the first full real run (1 Oct 2026): 119 accounts, 24 minutes, then the report crashed ----------
def test_answers_are_bent_into_the_schema_shape():
    """A model returned facts as plain strings; the report crashed on f.get('claim')."""
    from icp_agents.agents.researcher import RESEARCH_SCHEMA
    from icp_agents.schema_tools import fill_required
    out, _ = fill_required({"canonical_name": "X", "facts": ["Runs 500 technicians", {"claim": "ok"}],
                            "asset_complexity": "4", "open_questions": "What FSM do they use?"}, RESEARCH_SCHEMA)
    assert out["facts"] == [{"claim": "Runs 500 technicians"}, {"claim": "ok"}]
    assert out["asset_complexity"] == 4 and out["open_questions"] == ["What FSM do they use?"]


def test_micro_batcher_shares_calls_and_falls_back_per_item():
    from icp_agents.batching import MicroBatcher
    calls = []

    async def run_batch(items):
        calls.append([k for k, _ in items])
        return {k: v * 10 for k, v in items if k != "skip"}  # the model "forgot" one item

    async def go():
        b = MicroBatcher(run_batch, max_size=3, window=0.05)
        return await asyncio.gather(*(b.submit(k, v) for k, v in [("a", 1), ("b", 2), ("skip", 3), ("c", 4)]))
    res = asyncio.run(go())
    assert res == [10, 20, None, None]  # 'skip' left out -> caller does it alone; 'c' alone -> its own call
    assert calls == [["a", "b", "skip"]]


def test_full_real_conference_in_few_model_calls(tmp_path):
    """All 119 accounts from the real 2026 speaker list + sponsors + logos. Without batching this took 348
    model calls (the real run: ~530 calls / 24 min at ~23 calls/min free capacity)."""
    pages = {"/": (REAL / "home.html").read_text(), "/speakers": (REAL / "speakers.html").read_text(),
             "/sponsors": (REAL / "sponsors.html").read_text(), "/speakers/2026": (REAL / "speakers_2026.html").read_text(),
             "/landing/attendee-list-email": "<html><body><form></form></body></html>"}

    def handler(request):
        if "iqpc" in request.url.host or request.url.path.startswith("/UploadedFiles"):
            return httpx.Response(200, content=b"GIF89a", headers={"content-type": "image/gif"})
        body = pages.get(request.url.path.rstrip("/") or "/")
        return httpx.Response(200, text=body) if body else httpx.Response(404)
    fake = FakeProviders()
    s = settings(tmp_path, max_accounts=0)
    s.event_url, s.render_js, s.heartbeat = BASE, False, 0
    s.llm_provider, s.fast, s.concurrency, s.search_provider, s.search_interval = "pool", True, 12, "duckduckgo", 0
    s.pool_order, s.batch_size = ["groq", "cerebras"], 4
    env = {"GROQ_API_KEY": "g", "CEREBRAS_API_KEY": "c", "ICP_GROQ_DISCOVER": "0", "ICP_CEREBRAS_DISCOVER": "0",
           "ICP_GROQ_TPM": "10000000", "ICP_GROQ_RPM": "10000", "ICP_CEREBRAS_RPM": "10000", "ICP_CEREBRAS_TPM": "10000000"}
    llm = make_llm(s, pool_http=fake.client(), web_client=fake_web(), env=env)
    llm.web.cooldown = 0
    fetcher = scraper.LiveFetcher("ua", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    rt, team = asyncio.run(run_team(s, llm, fetcher=fetcher, timeout=120))
    coord = team["coordinator"]
    assert coord.failed is None and coord.report and not coord.report.get("error")
    assert len(coord.expected) >= 110 and all(coord.status[a]["stage"] == "done" for a in coord.expected)
    calls = sum(b.stats["calls"] for b in llm.backends)
    assert calls <= 130, calls  # ~4 accounts per research/scoring call
    names = [[t["function"]["name"] for t in b.get("tools") or []] for _, b in fake.requests]
    assert sum("submit_research_batch" in n for n in names) >= 20
    # every account still got its own message from the Researcher to the Analyst
    assert sum(1 for m in rt.transcript if m.subject == "validate_account") == len(team["researcher"].records)


def test_partial_shared_answers_fall_back_instead_of_wrong_scores(tmp_path):
    """Real run 3: ABB came out 'Not ICP 16' from a shared verdict that scored only 'assets'; 11 accounts
    errored because a shared research result left out required fields."""
    from types import SimpleNamespace
    from icp_agents.agents.analyst import AnalystAgent

    class OneShot:
        async def run(self, **kw):
            return SimpleNamespace(output={"verdicts": [
                {"id": "abb", "scores": {"assets": {"points": 16, "why": "x"}}, "summary": "partial"},
                {"id": "hach", "scores": {d: {"points": 10, "why": "ok"} for d in
                                          ("industry", "assets", "scale", "persona", "signals")},
                 "disqualifier": "none", "summary": "full", "confidence": "high", "note_to_strategist": "n"}]})
    agent = AnalystAgent.__new__(AnalystAgent)
    agent.llm, agent.settings, agent.briefing, agent.name = OneShot(), settings(tmp_path), None, "Analyst"
    agent._system = lambda: "s"
    out = asyncio.run(agent._score_batch([("abb", ({"canonical_name": "ABB"}, {"company": "ABB"})),
                                          ("hach", ({"canonical_name": "Hach"}, {"company": "Hach"}))]))
    assert "abb" not in out and out["hach"]["summary"] == "full"  # ABB gets its own call
