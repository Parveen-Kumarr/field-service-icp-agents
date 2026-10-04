"""Simulated OpenAI-compatible LLM providers for the model-pool tests.

One MockTransport serves every provider host (api.groq.com, api.cerebras.ai, ...). Each
provider can be given a latency, a 429 after N calls (with Retry-After), a daily-quota
429, server errors, or a habit of answering with JSON text instead of a tool call. The
fake plays each agent's role from the submit_* tool it is offered, in OpenAI format.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field

import httpx

from fake_gemini import KNOWN, SCORES, _key

HOSTS = {"api.groq.com": "groq", "api.cerebras.ai": "cerebras", "api.mistral.ai": "mistral",
         "models.github.ai": "github", "openrouter.ai": "openrouter",
         "generativelanguage.googleapis.com": "gemini", "localhost": "ollama"}
SOFTWARE = ("software", "consulting", "analytics", "cloud", " ai")


@dataclass
class Behaviour:
    latency: float = 0.0
    rate_limit_after: int = 0          # 429 (per-minute) once this many calls were served...
    rate_limit_times: int = 0          # ...this many times
    daily_quota_after: int = 0         # 429 "per day" after this many calls
    fail_5xx_times: int = 0
    json_text: bool = False            # answer with JSON in content instead of a tool call
    reject_extra: bool = False         # 400 if reasoning_effort is sent
    models: list = field(default_factory=list)   # what GET /models lists
    headers: dict = field(default_factory=dict)  # rate-limit headers sent with every answer
    no_tools_models: tuple = ()        # models that answer in text even when a tool call is required
    reject_required_missing: bool = False  # like Groq: 400 if a tool schema's required fields would be missing
    drop_fields: tuple = ()            # leave these fields out of submit answers (a sloppy model)


@dataclass
class FakeProviders:
    behaviours: dict = field(default_factory=dict)
    requests: list = field(default_factory=list)
    served: dict = field(default_factory=dict)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self._handle))

    async def _handle(self, request: httpx.Request) -> httpx.Response:
        name = HOSTS.get(request.url.host, request.url.host)
        b = self.behaviours.setdefault(name, Behaviour())
        if request.method == "GET":
            return httpx.Response(200, json={"object": "list", "data": [{"id": m, "active": True} for m in b.models]})
        body = json.loads(request.content)
        self.requests.append((name, body))
        n = self.served.get(name, 0)
        if b.latency:
            await asyncio.sleep(b.latency)
        if b.reject_extra and "reasoning_effort" in body:
            return httpx.Response(400, json={"error": {"message": "unknown parameter reasoning_effort"}})
        if b.daily_quota_after and n >= b.daily_quota_after:
            return httpx.Response(429, json={"error": {"message": "Rate limit reached: requests per day (RPD)"}})
        if b.rate_limit_times > 0 and n >= b.rate_limit_after:
            b.rate_limit_times -= 1
            return httpx.Response(429, headers={"retry-after": "1"},
                                  json={"error": {"message": "Rate limit reached for requests per minute"}})
        if b.fail_5xx_times > 0:
            b.fail_5xx_times -= 1
            return httpx.Response(503, text="overloaded")
        self.served[name] = n + 1
        if body.get("model") in b.no_tools_models:
            return httpx.Response(200, headers=b.headers, json={"choices": [{"index": 0, "finish_reason": "stop",
                                  "message": {"role": "assistant", "content": "OK"}}]})
        msg = self._reply(body)
        if b.drop_fields and msg.get("tool_calls"):
            fn = msg["tool_calls"][0]["function"]
            args = json.loads(fn["arguments"])
            fn["arguments"] = json.dumps({k: v for k, v in args.items() if k not in b.drop_fields})
        if b.json_text and msg.get("tool_calls"):
            fn = msg["tool_calls"][0]["function"]
            if fn["name"].startswith("submit_"):
                msg = {"role": "assistant", "content": "Here you go:\n" + fn["arguments"]}
        return httpx.Response(200, headers=b.headers, json={
            "id": "x", "object": "chat.completion", "model": body["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", **msg}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 600, "completion_tokens": 150, "total_tokens": 750}})

    # --- roles --------------------------------------------------------------------------------------
    @staticmethod
    def _call(name: str, args: dict) -> dict:
        return {"content": None, "tool_calls": [{"id": "call_" + "x" * 20, "type": "function",
                                                 "function": {"name": name, "arguments": json.dumps(args)}}]}

    def _reply(self, body: dict) -> dict:
        msgs = body["messages"]
        user = next(m for m in msgs if m["role"] == "user")
        prompt = user["content"] if isinstance(user["content"], str) else " ".join(
            p.get("text", "") for p in user["content"] if p.get("type") == "text")
        tools = [t["function"]["name"] for t in body.get("tools") or []]
        if not tools:
            return {"content": "OK"}
        submit = [t for t in tools if t.startswith("submit_")][-1]
        results = [m for m in msgs if m["role"] == "tool"]
        return getattr(self, "_" + submit)(prompt, results, body, user)

    def _submit_triage(self, prompt, results, body, user):
        out = []
        for m in re.finditer(r"id=(.+?) \| ([^|\n]+)", prompt):
            aid, name = m.group(1), m.group(2).strip()
            low = name.lower()
            if "acme service ai" in low:
                d = "self"
            elif low in ("aquant", "neuron7") or low.endswith(" ai"):
                d = "competitor"
            elif low in ("ifs",) or any(w in low for w in ("software", "cloud", "analytics")):
                d = "vendor_only"
            elif "consulting" in low:
                d = "consultant"
            else:
                d = "research"
            out.append({"id": aid, "decision": d, "reason": f"{name} judged {d}"})
        return self._call("submit_triage", {"accounts": out})

    def _submit_research(self, prompt, results, body, user):
        company = re.search(r"Company: (.+?) \(seen", prompt).group(1)
        data = next((v for kk, v in KNOWN.items() if kk in _key(company) or _key(company) in kk), None) or dict(
            canonical_name=company, industry="industrial_equipment", asset_complexity=4, scale="large",
            relationship_to_vendor="prospect", confidence="medium", what_they_service="industrial equipment")
        out = {**data, "facts": [{"claim": f"{company} services equipment", "source_url": "https://example.com/x"}],
               "note_to_analyst": f"{company} looks like a {data['relationship_to_vendor']}.", "open_questions": []}
        return self._call("submit_research", out)

    def _blocks(self, prompt, field="Company"):
        return re.findall(rf"### id=(.+?)\n{field}: (.+?)(?: \(seen|\n)", prompt)

    def _submit_research_batch(self, prompt, results, body, user):
        out = []
        for aid, company in self._blocks(prompt):
            one = json.loads(self._submit_research(f"Company: {company} (seen", results, body, user)
                             ["tool_calls"][0]["function"]["arguments"])
            out.append({"id": aid, **one})
        return self._call("submit_research_batch", {"results": out})

    def _submit_verdict_batch(self, prompt, results, body, user):
        out = []
        for aid, name in self._blocks(prompt, "Account"):
            (i, a, s_, p, g), dq = SCORES.get(name, ((27, 16, 12, 16, 8), "none"))
            out.append({"id": aid, "scores": {
                "industry": {"points": i, "why": "v"}, "assets": {"points": a, "why": "a"},
                "scale": {"points": s_, "why": "s"}, "persona": {"points": p, "why": "p"},
                "signals": {"points": g, "why": "g"}}, "disqualifier": dq, "best_contact": "",
                "summary": f"{name} scored.", "confidence": "medium", "note_to_strategist": "Lead with escalations."})
        return self._call("submit_verdict_batch", {"verdicts": out})

    def _submit_play_batch(self, prompt, results, body, user):
        out = [{"id": aid, "lead_with": ["Guided troubleshooting"], "pain_hypothesis": "Escalations.",
                "why_now": "to be confirmed", "openers": [{"contact": "Service leader", "channel": "email",
                                                          "message": "Hi..."}],
                "next_step": "Walkthrough", "note_to_coordinator": "Ready."}
               for aid, _ in self._blocks(prompt, "Account")]
        return self._call("submit_play_batch", {"plays": out})

    def _submit_answer(self, prompt, results, body, user):
        return self._call("submit_answer", {"answer": "They announced a service program this quarter.",
                                            "confidence": "medium"})

    def _submit_verdict(self, prompt, results, body, user):
        name = prompt.split("Account: ")[1].split("\n")[0]
        if "Nick Cribb" in prompt and not results and "ask_researcher" in [t["function"]["name"] for t in body["tools"]]:
            return self._call("ask_researcher", {"question": f"What does {name} actually service?"})
        (i, a, s, p, g), dq = SCORES.get(name, ((27, 16, 12, 16, 8), "none"))
        return self._call("submit_verdict", {
            "scores": {"industry": {"points": i, "why": "v"}, "assets": {"points": a, "why": "a"},
                       "scale": {"points": s, "why": "s"}, "persona": {"points": p, "why": "p"},
                       "signals": {"points": g, "why": "g"}},
            "disqualifier": dq, "best_contact": "", "summary": f"{name} scored.", "confidence": "medium",
            "note_to_strategist": "Lead with escalation cost."})

    def _submit_play(self, prompt, results, body, user):
        return self._call("submit_play", {
            "lead_with": ["Guided troubleshooting"], "pain_hypothesis": "Escalations.", "why_now": "New program.",
            "openers": [{"contact": "Service leader", "channel": "email", "message": "Hi..."}],
            "next_step": "Walkthrough", "note_to_coordinator": "Ready."})

    def _submit_response(self, prompt, results, body, user):
        return self._call("submit_response", {"answer": "Keeping it.", "revise": False})

    def _submit_briefing(self, prompt, results, body, user):
        return self._call("submit_briefing", {"offering": "AI agents for field service.", "target_verticals": ["medtech"],
                                              "icp_signals": ["complex assets"], "message_to_team": "Briefed."})

    def _submit_debrief(self, prompt, results, body, user):
        return self._call("submit_debrief", {"headline": "Done.", "top_accounts": [], "patterns": [],
                                             "recommended_actions": []})

    def _submit_extraction(self, prompt, results, body, user):
        return self._call("submit_extraction", {"speakers": []})

    def _submit_ok(self, prompt, results, body, user):
        return self._call("submit_ok", {"ok": True})

    def _submit_logos(self, prompt, results, body, user):
        return self._call("submit_logos", {"logos": []})


def big_site(n_speakers: int = 50) -> httpx.AsyncClient:
    """A conference site with n speakers (from varied companies) and a sponsor wall, like the real one."""
    kinds = ["Medical Systems", "Robotics", "Energy Services", "Water Technologies", "Instruments", "Software",
             "Consulting", "Cloud Analytics", "Equipment Corp", "Telecom Networks"]
    cards = []
    for i in range(n_speakers):
        company = f"Company{i:02d} {kinds[i % len(kinds)]}"
        cards.append(f'<img alt="Person{i:02d} Speaker, VP Global Service at {company}">')
    speakers = "<html><body><h1>Our Speakers</h1>" + "".join(cards) + "</body></html>"
    sponsors = ("<html><body><h2>Sponsors</h2>" + "".join(
        f'<img src="/s/{n}.png" alt="{n} Logo">' for n in
        ["Aquant", "Neuron7", "IFS", "Acme Service AI", "Blue Consulting", "Omega Software", "Dell Technologies",
         "Panasonic Corporation", "Zeta Cloud Analytics", "Northwind Equipment", "Contoso Medical",
         "Fabrikam Robotics", "Tailspin Energy", "Wide World Water", "Litware Instruments"]) + "</body></html>")
    home = ('<html><body><nav><a href="/speakers">Speakers</a><a href="/sponsors">Our Sponsors</a></nav>'
            "<h1>Welcome</h1></body></html>")
    pages = {"/": home, "/speakers": speakers, "/sponsors": sponsors}

    def handler(request):
        body = pages.get(request.url.path)
        return httpx.Response(200, text=body) if body else httpx.Response(404)
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))
