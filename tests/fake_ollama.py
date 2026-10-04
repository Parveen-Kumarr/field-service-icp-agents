"""A simulated Ollama server and web, used only by the tests.

`FakeOllama` answers POST /api/chat and GET /api/tags in Ollama's JSON format and
plays each agent's role (identified by its submit_* tool, or, for a structured
request, by the submit name in the final instruction). The Researcher really
uses the local web_search and fetch_page tools against `fake_web()`, which
serves DuckDuckGo-style result pages and company pages. The real OllamaLLM,
local web tools, runtime and agents run unchanged against it.
"""
from __future__ import annotations

import base64
import json
import re
from urllib.parse import parse_qs, quote

import httpx

from fake_gemini import KNOWN, LOGO_NAMES, SCORES, _key


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def fake_web(ddg_blocked: bool = False) -> httpx.AsyncClient:
    """DuckDuckGo HTML results for any query + a page for every result URL + the vendor site."""
    def handler(request: httpx.Request) -> httpx.Response:
        host, path = request.url.host, request.url.path
        if host in ("html.duckduckgo.com", "lite.duckduckgo.com"):
            if ddg_blocked:
                return httpx.Response(202, text="<html>anomaly</html>")
            q = parse_qs(request.content.decode()).get("q", [""])[0]
            slug = _slug(q.split(" field service")[0])
            html = (f'<div class="result"><h2><a class="result__a" href="//duckduckgo.com/l/?uddg='
                    f'{quote("https://example.com/" + slug, safe="")}">{q} - official site</a></h2>'
                    f'<a class="result__snippet">Everything about {q}.</a></div>')
            return httpx.Response(200, text=html, headers={"content-type": "text/html"})
        if host == "example.com":
            return httpx.Response(200, headers={"content-type": "text/html; charset=utf-8"},
                                  text=f"<html><head><title>{path[1:]} | example</title></head><body>"
                                       f"<h1>{path[1:]}</h1><p>Company page for {path[1:]}: installed base, "
                                       f"field service organisation and products.</p></body></html>")
        if host == "acme-service-ai.example":
            return httpx.Response(200, headers={"content-type": "text/html"},
                                  text="<html><title>Acme Service AI</title><body>AI agents for field service "
                                       "teams: guided troubleshooting, escalation prediction.</body></html>")
        return httpx.Response(404)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class FakeOllama:
    def __init__(self, model: str = "gemma4:12b", installed: tuple[str, ...] = ("gemma4:12b",),
                 string_args: bool = True):
        self.model = model
        self.installed = installed
        self.string_args = string_args  # some models return arguments as a JSON string
        self.requests: list[dict] = []

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url="http://localhost:11434", transport=httpx.MockTransport(self._handle))

    # --- HTTP -------------------------------------------------------------------------
    def _handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": n} for n in self.installed]})
        if request.url.path == "/api/chat":
            body = json.loads(request.content)
            self.requests.append(body)
            if body["model"] not in self.installed:
                return httpx.Response(404, json={"error": f"model '{body['model']}' not found"})
            msg = self._reply(body)
            return httpx.Response(200, json={"model": body["model"], "message": {"role": "assistant", **msg},
                                             "done": True, "done_reason": "stop",
                                             "prompt_eval_count": 800, "eval_count": 120})
        return httpx.Response(404)

    def _call(self, name: str, args: dict) -> dict:
        return {"content": "", "tool_calls": [{"function": {
            "name": name, "arguments": json.dumps(args) if self.string_args else args}}]}

    def _json(self, obj: dict) -> dict:
        return {"content": json.dumps(obj)}

    def _reply(self, body: dict) -> dict:
        msgs = body["messages"]
        user = next(m for m in msgs if m["role"] == "user")
        prompt = user["content"]
        tool_msgs = [m for m in msgs if m["role"] == "tool"]
        structured = "format" in body
        if structured:
            submit = re.search(r"final answer for (submit_\w+)", msgs[-1]["content"]).group(1)
            assert "tools" not in body  # forced answers go out without tools
        elif "Reply with the single word OK" in prompt:
            return {"content": "OK"}
        else:
            submit = [t["function"]["name"] for t in body["tools"] if t["function"]["name"].startswith("submit_")][-1]
        return getattr(self, "_" + submit)(body, prompt, tool_msgs, structured, user)

    # --- per agent ----------------------------------------------------------------------
    def _submit_briefing(self, body, prompt, tools, structured, user):
        if not tools:
            url = re.search(r"https?://\S+", prompt).group(0).rstrip(".,)")
            return self._call("fetch_page", {"url": url})
        page = json.loads(tools[-1]["content"])
        assert "AI agents for field service" in page["text"]
        return self._call("submit_briefing", {
            "offering": "AI agents for field service teams.", "agents": ["Guided troubleshooting", "Escalation prediction"],
            "target_verticals": ["medical devices", "telecom"], "icp_signals": ["complex installed base"],
            "message_to_team": "I've read the vendor's site live on this machine."})

    def _submit_research(self, body, prompt, tools, structured, user):
        company = prompt.split("Company as listed: ")[1].split(" (variants")[0]
        k = _key(company)
        data = next(v for kk, v in KNOWN.items() if kk in k or k in kk)
        if structured:
            assert len(tools) == 2  # searched, then read a page, before answering
            out = {**data, "facts": [{"claim": f"{data['canonical_name']} services {data['what_they_service']}",
                                      "source_url": f"https://example.com/{_slug(company)}"}],
                   "note_to_analyst": f"{data['canonical_name']} looks like a {data['relationship_to_vendor']}."}
            out.setdefault("open_questions", [])
            return self._json(out)
        if not tools:
            return self._call("web_search", {"query": f"{company} field service"})
        if len(tools) == 1:
            results = json.loads(tools[0]["content"])["results"]
            return self._call("fetch_page", {"url": results[0]["url"]})
        return {"content": f"{company} is a company I researched."}  # prose: the client must force the JSON

    def _submit_answer(self, body, prompt, tools, structured, user):
        if not tools:
            return self._call("web_search", {"query": "answer field service"})
        ans = ("SAM Service is a regional mechanical services contractor - HVAC and plant equipment, small scale."
               if "SAM Service" in prompt else "Recent news: they announced a service digitisation program this quarter.")
        return self._call("submit_answer", {"answer": ans, "confidence": "medium",
                                            "facts": [{"claim": ans, "source_url": "https://example.com/answer"}]})

    def _submit_verdict(self, body, prompt, tools, structured, user):
        name = prompt.split("Account: ")[1].split("\n")[0]
        if "ask the Researcher" in prompt and not tools:
            return self._call("ask_researcher", {"question": f"What equipment does {name} actually support?"})
        (i, a, s, p, g), dq = SCORES[name]
        return self._call("submit_verdict", {
            "scores": {"industry": {"points": i, "why": "v"}, "assets": {"points": a, "why": "a"},
                       "scale": {"points": s, "why": "s"}, "persona": {"points": p, "why": "p"},
                       "signals": {"points": g, "why": "g"}},
            "disqualifier": dq, "best_contact": "", "summary": f"{name} scored on the evidence.",
            "confidence": "medium", "note_to_strategist": "Lead with escalation cost."})

    def _submit_play(self, body, prompt, tools, structured, user):
        name = prompt.split("Account: ")[1].split(" - ")[0]
        if not tools:
            return self._call("ask_researcher", {"question": f"Any recent news at {name}?"})
        if name == "Tetra Pak" and len(tools) == 1:
            return self._call("challenge_analyst", {"argument": "No named Tetra Pak contact is at the event, so "
                                                                "persona can't justify an A."})
        return self._call("submit_play", {
            "lead_with": ["Guided troubleshooting"], "pain_hypothesis": "Escalations.", "why_now": "New program.",
            "openers": [{"contact": "Service leader", "channel": "email", "message": f"Hi - {name}..."}],
            "next_step": "15-minute walkthrough", "note_to_coordinator": "Ready."})

    def _submit_response(self, body, prompt, tools, structured, user):
        assert structured
        return self._json({"answer": "Fair point - no named contact.", "revise": True, "revised_scores": {
            "industry": {"points": 24, "why": "v"}, "assets": {"points": 20, "why": "a"},
            "scale": {"points": 15, "why": "s"}, "persona": {"points": 0, "why": "none"},
            "signals": {"points": 3, "why": "g"}}})

    def _submit_logos(self, body, prompt, tools, structured, user):
        assert structured and user.get("images")
        logos, idx = [], None
        for line in prompt.splitlines():
            if line.startswith("Logo "):
                idx = int(line.split()[1].rstrip(":"))
            m = re.match(r"\[image (\d+)\]", line)
            if m:
                marker = base64.b64decode(user["images"][int(m.group(1)) - 1]).decode()
                logos.append({"index": idx, "company": next((v for kk, v in LOGO_NAMES.items() if kk in marker), "")})
        return self._json({"logos": logos})

    def _submit_debrief(self, body, prompt, tools, structured, user):
        return self._json({"headline": "Medtech leads.", "top_accounts": [{"account": "Stryker", "why": "scale"}],
                           "patterns": ["AI programs active."], "recommended_actions": ["Follow up."]})
