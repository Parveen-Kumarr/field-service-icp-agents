"""Tests for the "never silently stuck" guarantees: timeouts, retries, polling,
readable errors, the heartbeat, the log file, and the preflight check."""
import asyncio
import logging
import sys
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from icp_agents import LLM, run_team  # noqa: E402
from icp_agents import log as logs  # noqa: E402
from icp_agents.llm import ClientTool, LLMError, describe_error  # noqa: E402
from icp_agents.monitor import status_lines  # noqa: E402
from icp_agents.runtime import Runtime  # noqa: E402
from icp_agents.agents.base import BaseAgent  # noqa: E402

SUBMIT = ClientTool("submit_x", "submit", {"type": "object", "properties": {"answer": {"type": "string"}}})


def done(answer="ok", iid="int_ok"):
    return NS(id=iid, status="requires_action", errors=None, usage={"total_input_tokens": 10, "total_output_tokens": 2},
              steps=[{"type": "function_call", "id": "fc_1", "name": "submit_x", "arguments": {"answer": answer}}])


class ScriptedClient:
    """create() plays a script: a number = hang that many seconds, an exception = raise it, else return it."""

    def __init__(self, script, polls=None):
        self.script, self.polls, self.creates, self.gets = list(script), list(polls or []), 0, 0
        self.aio = NS(interactions=self)

    async def create(self, **body):
        self.creates += 1
        item = self.script.pop(0)
        if isinstance(item, (int, float)):
            await asyncio.sleep(item)
        if isinstance(item, Exception):
            raise item
        return item

    async def get(self, id):
        self.gets += 1
        return self.polls.pop(0)


class RateLimited(Exception):
    status_code = 429


def test_hung_call_times_out_and_is_retried(caplog):
    client = ScriptedClient([30, done("second try")])
    llm = LLM("gemini-test", client=client, timeout=0.2, max_retries=2)
    logging.getLogger("icp").propagate = True  # let pytest capture our warnings
    with caplog.at_level(logging.WARNING, logger="icp"):
        res = asyncio.run(llm.run(agent="Researcher", system="s", prompt="p", submit=SUBMIT, purpose="research Stryker"))
    assert res.output == {"answer": "second try"} and client.creates == 2
    assert any("no response within" in r.getMessage() and "Retry 1/2" in r.getMessage() for r in caplog.records)
    assert llm.inflight == {}  # nothing left registered as in flight


def test_in_progress_interaction_is_polled_until_done():
    client = ScriptedClient([NS(id="int_1", status="in_progress", steps=[], usage=None, errors=None)],
                            polls=[NS(id="int_1", status="in_progress", steps=[], usage=None, errors=None), done()])
    llm = LLM("gemini-test", client=client, timeout=5, poll_interval=0.01)
    res = asyncio.run(llm.run(agent="Scout", system="s", prompt="p", submit=SUBMIT))
    assert res.output == {"answer": "ok"} and client.gets == 2


def test_rate_limit_retries_then_fails_with_a_readable_reason():
    client = ScriptedClient([RateLimited("Resource exhausted")] * 3)
    llm = LLM("gemini-test", client=client, timeout=5, max_retries=2)
    import icp_agents.llm as mod
    orig = mod.asyncio.sleep

    async def fast_sleep(_):  # skip the backoff wait in tests
        await orig(0)
    mod.asyncio.sleep = fast_sleep
    try:
        asyncio.run(llm.run(agent="Analyst", system="s", prompt="p", submit=SUBMIT))
        raise AssertionError("expected LLMError")
    except LLMError as exc:
        assert "quota / rate limit" in str(exc) and client.creates == 3 and llm.failures == 1
    finally:
        mod.asyncio.sleep = orig


def test_error_descriptions_point_at_the_fix():
    class NotFound(Exception):
        status_code = 404
    msg, transient = describe_error(NotFound("models/gemini-x is not found"))
    assert "model not found" in msg and "ICP_LLM_MODEL" in msg and not transient
    msg, transient = describe_error(type("ConnectError", (Exception,), {})("reset"))
    assert transient and "network" in msg


def test_heartbeat_shows_calls_in_flight_and_agent_activity():
    async def scenario():
        rt = Runtime()
        agent = BaseAgent(rt)
        agent.name = "Researcher"
        rt.agents = {"Researcher": agent}
        client = ScriptedClient([5])
        llm = LLM("gemini-test", client=client, timeout=10)
        with agent.doing("researching Stryker"):
            task = asyncio.create_task(llm.run(agent="Researcher", system="s", prompt="p", submit=SUBMIT,
                                               purpose="research Stryker"))
            await asyncio.sleep(0.05)
            lines = status_lines(rt, llm, slow_after=0.01)
            task.cancel()
        return lines

    lines = asyncio.run(scenario())
    assert "1 in flight" in lines[0]
    assert any(l.strip().startswith("model <- Researcher: research Stryker (turn 1)") and "SLOW" in l for l in lines)
    assert any("Researcher: researching Stryker" in l for l in lines)
    assert all(l.isascii() for l in lines)  # safe on Windows consoles


def test_coordinator_failure_stops_the_run_instead_of_hanging(tmp_path):
    from test_agents import settings, site
    from fake_gemini import FakeGemini
    from icp_agents.agents.coordinator import CoordinatorAgent

    async def broken_start(self):
        raise RuntimeError("boom")
    orig = CoordinatorAgent.start
    CoordinatorAgent.start = broken_start
    try:
        rt, team = asyncio.run(run_team(settings(tmp_path), LLM("g", client=FakeGemini()), fetcher=site(), timeout=10))
    finally:
        CoordinatorAgent.start = orig
    assert "boom" in team["coordinator"].failed


def test_no_vendor_url_uses_generic_profile_without_a_gemini_call(tmp_path):
    from test_agents import settings, site
    from fake_gemini import FakeGemini
    client = FakeGemini()
    rt, team = asyncio.run(run_team(settings(tmp_path, vendor_url="", vendor_name="", max_accounts=2),
                                    LLM("g", client=client), fetcher=site(), timeout=30))
    brief = next(m for m in rt.transcript if m.subject == "vendor_briefing")
    assert "No vendor website is configured" in brief.text
    assert not any("submit_briefing" in str(r["tools"]) for r in client.interactions.requests)
    assert team["coordinator"].report and "our client" in rt.transcript[0].text


def test_log_file_records_calls_messages_and_fetches(tmp_path):
    from test_agents import settings, site
    from fake_gemini import FakeGemini
    path = logs.setup_logging("WARNING", tmp_path / "logs" / "run.log")
    try:
        asyncio.run(run_team(settings(tmp_path, max_accounts=2, heartbeat=0.002), LLM("g", client=FakeGemini()),
                             fetcher=site(), timeout=30))
    finally:
        for h in list(logging.getLogger("icp").handlers):
            h.close()
        logging.getLogger("icp").handlers.clear()
    text = path.read_text()
    assert "Gemini ok: research" in text  # every Gemini call with its purpose and timing
    assert "GET https://fieldserviceusa.wbresearch.com/speakers -> HTTP 200" in text  # every page fetch
    assert "Coordinator -> Scout [request] scrape_event" in text  # every agent message
    assert "icp.heartbeat" in text and "messages | accounts" in text  # periodic status reports


def test_preflight_reports_success_and_failure():
    ok = LLM("g", client=ScriptedClient([NS(id="i", status="completed", steps=[], usage=None, errors=None)]))
    assert asyncio.run(ok.preflight()) >= 0

    class BadKey(Exception):
        status_code = 401
    bad = LLM("g", client=ScriptedClient([BadKey("API key not valid")]))
    try:
        asyncio.run(bad.preflight())
        raise AssertionError("expected LLMError")
    except LLMError as exc:
        assert "API key rejected" in str(exc)
