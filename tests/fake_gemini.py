"""A stand-in for the Gemini Interactions API used only by the tests.

It answers like Gemini would for each agent (identified by its submit_* tool):
Google Search and URL-context steps with url_citation annotations, a prose
answer that the loop must nudge into a submit, function calls that make agents
ask each other questions, a challenge, and vision on logo images. Follow-ups
use previous_interaction_id, as the real stateful API does. The real runtime,
tool loop and agents run unchanged against it.
"""
from __future__ import annotations

import base64
import itertools
from types import SimpleNamespace as NS

_ids = itertools.count(1)

KNOWN = {
    "stryker": dict(canonical_name="Stryker", industry="medical_devices", asset_complexity=5, scale="enterprise",
                    relationship_to_vendor="prospect", confidence="high", what_they_service="Surgical and hospital equipment (ProCare service)"),
    "ciena": dict(canonical_name="Ciena", industry="telecom", asset_complexity=5, scale="large",
                  relationship_to_vendor="prospect", confidence="high", what_they_service="Optical networking platforms"),
    "crown equipment": dict(canonical_name="Crown Equipment", industry="industrial_equipment", asset_complexity=4,
                            scale="large", relationship_to_vendor="prospect", confidence="medium",
                            what_they_service="Forklifts and InfoLink telematics"),
    "aquant": dict(canonical_name="Aquant", industry="software_vendor", asset_complexity=0, scale="mid",
                   relationship_to_vendor="competitor", confidence="high", what_they_service="AI for service (software)"),
    "sam service": dict(canonical_name="SAM Service Inc.", industry="other", asset_complexity=2, scale="unknown",
                        relationship_to_vendor="unknown", confidence="low", what_they_service="unclear",
                        open_questions=["What equipment does SAM Service support?"]),
    "acme service ai": dict(canonical_name="Acme Service AI", industry="software_vendor", asset_complexity=0,
                            scale="small", relationship_to_vendor="self", confidence="high",
                            what_they_service="AI agents for field service (the vendor itself)"),
    "dell technologies": dict(canonical_name="Dell Technologies", industry="data_center_hardware", asset_complexity=4,
                              scale="enterprise", relationship_to_vendor="partner_or_vendor", confidence="high",
                              what_they_service="Servers, storage, PCs with a large field service org"),
    "ifs": dict(canonical_name="IFS", industry="software_vendor", asset_complexity=0, scale="large",
                relationship_to_vendor="partner_or_vendor", confidence="high", what_they_service="FSM software"),
    "tetra pak": dict(canonical_name="Tetra Pak", industry="food_beverage_equipment", asset_complexity=5,
                      scale="enterprise", relationship_to_vendor="prospect", confidence="high",
                      what_they_service="Food processing and packaging lines"),
    "omron": dict(canonical_name="Omron", industry="robotics_automation", asset_complexity=5, scale="enterprise",
                  relationship_to_vendor="prospect", confidence="high", what_they_service="Industrial automation"),
}
SCORES = {  # industry, assets, scale, persona, signals ; disqualifier
    "Stryker": ((30, 20, 15, 17, 10), "none"), "Ciena": ((30, 20, 12, 18, 6), "none"),
    "Crown Equipment": ((27, 16, 12, 16, 8), "none"), "Aquant": ((0, 0, 8, 2, 0), "competitor"),
    "SAM Service Inc.": ((10, 6, 4, 12, 3), "none"), "Acme Service AI": ((0, 0, 4, 0, 0), "self"),
    "Dell Technologies": ((27, 16, 15, 5, 8), "vendor_only"), "IFS": ((0, 0, 12, 0, 0), "vendor_only"),
    "Tetra Pak": ((24, 20, 15, 12, 6), "none"), "Omron": ((27, 20, 15, 5, 6), "none"),
}
LOGO_NAMES = {"tetra-pak": "Tetra Pak", "a8f3c1e9b2d4": "Omron", "stryker": "Stryker"}


def call(name, args):
    return {"type": "function_call", "id": f"fc_{next(_ids)}", "name": name, "arguments": args}


def search(query, url, title="Result"):
    cid = f"gs_{next(_ids)}"
    return [{"type": "google_search_call", "id": cid, "arguments": {"queries": [query]}},
            {"type": "google_search_result", "call_id": cid, "result": [{"search_suggestions": "<div/>"}]},
            {"type": "model_output", "content": [{"type": "text", "text": f"Found {title}.", "annotations": [
                {"type": "url_citation", "url": url, "title": title, "start_index": 0, "end_index": 5}]}]}]


def read_url(url):
    cid = f"uc_{next(_ids)}"
    return [{"type": "url_context_call", "id": cid, "arguments": {"urls": [url]}},
            {"type": "url_context_result", "call_id": cid, "result": [{"url": url, "status": "success"}]}]


def _text(parts) -> str:
    return " ".join(p.get("text", "") for p in parts if isinstance(p, dict)) if isinstance(parts, list) else str(parts)


def _key(name: str) -> str:
    from icp_agents.scraper import normalize_company
    return normalize_company(name)


class FakeInteractions:
    def __init__(self):
        self.requests: list[dict] = []
        self.state: dict[str, dict] = {}

    async def create(self, **body):
        self.requests.append(body)
        prev = body.get("previous_interaction_id")
        if prev:
            st = self.state[prev]
            new = body["input"]
            st["results"] += [x for x in new if x.get("type") == "function_result"]
            st["nudged"] = st["nudged"] or any(x.get("type") == "text" for x in new)
        else:
            submit = [t["name"] for t in body["tools"] if t.get("name", "").startswith("submit_")][-1]
            st = {"submit": submit, "input": body["input"], "prompt": _text(body["input"]), "results": [],
                  "nudged": False, "forced": []}
        st["forced"].append(((body.get("generation_config") or {}).get("tool_choice") or {}).get("allowed_tools"))
        steps, grounding = getattr(self, "_" + st["submit"])(st)
        iid = f"int_{next(_ids)}"
        self.state[iid] = st
        status = "requires_action" if any(s["type"] == "function_call" for s in steps) else "completed"
        return NS(id=iid, status=status, steps=steps, errors=None,
                  usage={"total_input_tokens": 900, "total_output_tokens": 250, "total_thought_tokens": 100,
                         "grounding_tool_count": [{"type": "google_search", "count": grounding}] if grounding else []})

    # --- per agent ----------------------------------------------------------------
    def _submit_briefing(self, st):
        assert "https://acme-service-ai.example/" in st["prompt"]  # the configured vendor URL
        return [*read_url("https://acme-service-ai.example/"), call("submit_briefing", {
            "offering": "Physical AI agents that capture expert technicians' judgment for field service.",
            "agents": ["Guided troubleshooting", "Expert knowledge capture", "Device log analysis"], "target_verticals": ["medical devices", "telecom", "energy"],
            "icp_signals": ["complex installed base", "costly escalations"], "competitors": ["Aquant", "Neuron7"],
            "message_to_team": "I've read the vendor's site live."})], 0

    def _submit_research(self, st):
        company = st["prompt"].split("Company as listed: ")[1].split(" (variants")[0]
        k = _key(company)
        data = next((v for kk, v in KNOWN.items() if kk in k or k in kk), None)
        assert data, f"unexpected company {company}"
        if not st["nudged"]:  # first reply: searches, then answers in prose without submitting
            return search(f"{company} field service", f"https://example.com/{k.replace(' ', '-')}", company), 1
        assert st["forced"][-1]["tools"] == ["submit_research"]  # the loop forced the submit
        out = {**data, "facts": [{"claim": f"{data['canonical_name']} services {data['what_they_service']}",
                                  "source_url": f"https://example.com/{k.replace(' ', '-')}"}],
               "note_to_analyst": f"{data['canonical_name']} looks like a {data['relationship_to_vendor']}."}
        out.setdefault("open_questions", [])
        return [call("submit_research", out)], 0

    def _submit_answer(self, st):
        ans = ("SAM Service is a regional mechanical services contractor - HVAC and plant equipment, small scale."
               if "SAM Service" in st["prompt"] else "Recent news: they announced a service digitisation program this quarter.")
        return [*search("answer", "https://example.com/answer"),
                call("submit_answer", {"answer": ans, "confidence": "medium",
                                       "facts": [{"claim": ans, "source_url": "https://example.com/answer"}]})], 1

    def _submit_verdict(self, st):
        name = st["prompt"].split("Account: ")[1].split("\n")[0]
        if "ask the Researcher" in st["prompt"] and not st["results"]:
            return [call("ask_researcher", {"question": f"What equipment does {name} actually support, and how big is it?"})], 0
        (i, a, s, p, g), dq = SCORES[name]
        return [call("submit_verdict", {
            "scores": {"industry": {"points": i, "why": "vertical"}, "assets": {"points": a, "why": "assets"},
                       "scale": {"points": s, "why": "scale"}, "persona": {"points": p, "why": "persona"},
                       "signals": {"points": g, "why": "signals"}},
            "disqualifier": dq, "best_contact": "", "summary": f"{name} scored on the evidence.",
            "confidence": "medium", "note_to_strategist": "Lead with escalation cost."})], 0

    def _submit_play(self, st):
        name = st["prompt"].split("Account: ")[1].split(" - ")[0]
        if not st["results"]:
            return [call("ask_researcher", {"question": f"Any recent news at {name} I can open with?"})], 0
        if name == "Tetra Pak" and len(st["results"]) == 1:
            return [call("challenge_analyst", {"argument": "No named contact from Tetra Pak is at the event "
                                                           "(logo only), so persona can't justify an A."})], 0
        return [call("submit_play", {
            "lead_with": ["Guided troubleshooting", "Device log analysis"], "pain_hypothesis": "Escalations on complex equipment.",
            "why_now": "Service digitisation program announced.", "talk_track": ["one", "two"],
            "openers": [{"contact": "Clint Olsen" if name == "Stryker" else "Service leader", "channel": "email",
                         "message": f"Hi - noticed {name}'s service program..."}],
            "next_step": "15-minute tour", "risks": "none", "note_to_coordinator": "Ready for review."})], 0

    def _submit_response(self, st):
        return [call("submit_response", {
            "answer": "Fair point - with no named contact the persona score was too generous.", "revise": True,
            "revised_scores": {"industry": {"points": 24, "why": "v"}, "assets": {"points": 20, "why": "a"},
                               "scale": {"points": 15, "why": "s"}, "persona": {"points": 0, "why": "none named"},
                               "signals": {"points": 3, "why": "g"}}})], 0

    def _submit_logos(self, st):
        logos, idx = [], None
        for part in st["input"]:
            if part["type"] == "text" and part["text"].startswith("Logo "):
                idx = int(part["text"].split()[1].rstrip(":"))
            elif part["type"] == "image":
                assert part["mime_type"] == "image/png"
                marker = base64.b64decode(part["data"]).decode()
                logos.append({"index": idx, "company": next((v for k, v in LOGO_NAMES.items() if k in marker), "")})
        return [call("submit_logos", {"logos": logos})], 0

    def _submit_extraction(self, st):
        assert "https://fieldserviceusa.wbresearch.com/speakers" in st["prompt"]  # URL context needs the URL in the prompt
        return [*read_url("https://fieldserviceusa.wbresearch.com/speakers"),
                call("submit_extraction", {"speakers": [
                    {"name": "Clint Olsen", "title": "Sr. Director", "company": "Stryker"},
                    {"name": "Greg Friesen", "title": "VP", "company": "Ciena"}], "sponsors": ["Aquant"]})], 0

    def _submit_debrief(self, st):
        return [call("submit_debrief", {"headline": "Medtech and telecom lead the list.",
                                        "top_accounts": [{"account": "Stryker", "why": "ProCare scale"}],
                                        "patterns": ["Service AI programs are active."],
                                        "recommended_actions": ["Book Stryker follow-up."]})], 0


class FakeGemini:
    """Mimics google.genai.Client: client.aio.interactions.create(...)."""

    def __init__(self):
        self.interactions = FakeInteractions()
        self.aio = NS(interactions=self.interactions)
