"""Agent 3 - ICP Analyst: decides how well each account fits the vendor's ICP.

The model scores the Researcher's brief against the ICP rubric, one justified
score per dimension. When the evidence is thin it asks the Researcher a
specific question and waits for the answer before scoring. The total and tier
are computed in code from the per-dimension points, so the arithmetic is
never left to the model. It also defends or revises its verdict when the
Strategist challenges it.
"""
from __future__ import annotations

import asyncio
import json
import re
from types import SimpleNamespace

from ..batching import MicroBatcher, norm_id
from ..vendor_profile import DISQUALIFIER_LABELS, RUBRIC, rubric_text, tier_for
from ..schema_tools import fill_required, repair_json
from ..llm import ClientTool
from ..messages import HANDOFF, AgentMessage
from .base import BaseAgent

VENDOR_PROSPECT_CUTOFF = 70

_dim = {"type": "object", "properties": {"points": {"type": "integer"}, "why": {"type": "string"}},
        "required": ["points", "why"]}
VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {"type": "object", "properties": {k: _dim for k in RUBRIC}, "required": list(RUBRIC)},
        "disqualifier": {"type": "string", "enum": ["none", *DISQUALIFIER_LABELS]},
        "best_contact": {"type": "string", "description": "Name of the contact to engage first, or '' if none"},
        "summary": {"type": "string", "description": "One or two sentences: why this account does or doesn't fit"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "note_to_strategist": {"type": "string", "description": "What the Strategist should lean on or watch out for"},
    },
    "required": ["scores", "disqualifier", "summary", "confidence", "note_to_strategist"],
}
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "description": "Your reply to the colleague, in your own words"},
        "revise": {"type": "boolean", "description": "True only if their point changes your scores"},
        "revised_scores": {"type": "object", "properties": {k: _dim for k in RUBRIC}},
    },
    "required": ["answer", "revise"],
}
QUESTION_TOOL = {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}
TRIAGE_SCHEMA = {
    "type": "object",
    "properties": {"accounts": {"type": "array", "items": {"type": "object", "properties": {
        "id": {"type": "string"},
        "decision": {"type": "string", "enum": ["research", "competitor", "vendor_only", "consultant", "self"]},
        "reason": {"type": "string", "description": "Under 15 words"}}, "required": ["id", "decision", "reason"]}}},
    "required": ["accounts"],
}
# Generic hints for triage: companies that sell software/services TO field-service teams, or advise them.
NON_BUYER_HINTS = ("Salesforce, ServiceNow, SAP, Oracle, Microsoft, IFS, PTC, ServiceMax, Zinier, ServiceTitan, Praxedo, "
                   "GoFormz, FieldAware, Aquant, Neuron7, Circuitry.ai, Dezide, Bolt Data, Appify, Gomocha, Syncron, "
                   "Baxter Planning, Markt Pilot, AblyPro, Help Lightning, TeamViewer, Geotab, Lytx, Samsara, Deloitte, "
                   "Accenture, KPMG, PwC, EY, RSM, Capgemini, Cognizant, Infosys, TCS, Wipro")
TRIAGE_BATCH = 25


def _points(val) -> tuple[int, str]:
    """Points and reason from whatever shape a model used: {"points": 18, "why": ..}, 18, "18", "18/20"."""
    if isinstance(val, dict):
        raw, why = val.get("points", val.get("score", 0)), str(val.get("why") or val.get("reason") or "")
    else:
        raw, why = val, ""
    if isinstance(raw, str):
        m = re.search(r"-?\d+(?:\.\d+)?", raw)
        raw = float(m.group()) if m else 0
    try:
        return int(round(float(raw or 0))), why
    except (TypeError, ValueError):
        return 0, why


def compute(scores, disqualifier: str | None) -> tuple[int, str, dict]:
    if isinstance(scores, str):  # some models send the object as a JSON string
        scores = repair_json(scores) or {}
    if not isinstance(scores, dict):
        scores = {}
    clean = {}
    for dim, (mx, _) in RUBRIC.items():
        pts, why = _points(scores.get(dim))
        clean[dim] = {"points": max(0, min(mx, pts)), "why": why}
    total = sum(d["points"] for d in clean.values())
    dq = None if disqualifier in (None, "", "none") else str(disqualifier)
    if dq == "vendor_only" and total >= VENDOR_PROSPECT_CUTOFF:
        return total, tier_for(total, None) + " (also a vendor/sponsor)", clean
    return total, tier_for(total, dq), clean


class AnalystAgent(BaseAgent):
    name = "Analyst"
    role = "Scores each account against the vendor's ICP and defends or revises the verdict"

    def __init__(self, runtime, llm=None, settings=None):
        super().__init__(runtime, llm, settings)
        self.verdicts: dict[str, dict] = {}
        self.research: dict[str, dict] = {}
        self._batcher: MicroBatcher | None = None

    async def handle(self, msg: AgentMessage) -> None:
        if msg.subject == "triage_accounts":
            with self.doing(f"triaging {len(msg.data.get('accounts', []))} accounts"):
                await self.triage(msg)
        elif msg.subject == "validate_account":
            with self.doing(f"scoring {msg.data.get('canonical_name', msg.thread)}"):
                await self.validate(msg)
        elif msg.performative == "request":
            with self.doing(f"responding to {msg.sender} about {msg.thread}"):
                await self.respond(msg)

    def _system(self) -> str:
        return (f"You are the ICP Analyst on a sales-intelligence team qualifying prospects for {self.vendor()}. "
                "You decide how well a company fits the vendor's ideal customer profile, using only the evidence you "
                "have. Be decisive and specific; every score needs a one-sentence reason tied to evidence.\n\n"
                "What the vendor does:\n"
                + self.briefing_text() + "\n\nICP rubric (points per dimension):\n" + rubric_text()
                + "\n\nDisqualifiers: competitor (sells AI for service), self (the vendor itself), consultant (advisors, not "
                  "buyers), vendor_only (sells software/services to field service teams and has no large field "
                  "service operation of its own). A vendor that also runs a big field service org is 'none'.")

    async def triage(self, msg: AgentMessage) -> None:
        """Fast mode: classify every account in a few batched calls; disqualify clear non-buyers without research."""
        accounts = msg.data["accounts"]
        batches = [accounts[i:i + TRIAGE_BATCH] for i in range(0, len(accounts), TRIAGE_BATCH)]

        async def one(batch: list[dict]) -> list[dict]:
            lines = "\n".join(
                f"- id={a['account_id']} | {a['company']} | seen as {', '.join(a.get('roles', []))}"
                + (f" | people: {'; '.join((c['name'] + ' (' + c.get('title', '') + ')') for c in a['contacts'][:3])}"
                   if a.get("contacts") else "") for a in batch)
            res = await self.llm.run(
                agent=self.name, force_submit=True, max_tokens=200 + 45 * len(batch),
                purpose=f"triage {len(batch)} accounts",
                system=(f"You are the ICP Analyst qualifying prospects for {self.vendor()}, a vendor of AI agents for "
                        "field service on complex physical equipment. Triage conference companies from their name and "
                        "context. Choose 'research' for any company that installs, services or operates physical "
                        "equipment, and whenever you are unsure. Only disqualify companies that clearly sell software "
                        "or services to field-service teams (competitor if they sell AI for service, else "
                        "vendor_only), advisory/consulting firms (consultant), or the vendor itself (self). A company "
                        "that sells software but also runs a large field-service operation of its own is 'research'. "
                        f"Typical non-buyers: {NON_BUYER_HINTS}."
                        + (f" The vendor itself is {self.settings.vendor_name}." if self.settings and
                           self.settings.vendor_name else "")),
                prompt=f"Triage these accounts (return every id exactly once):\n{lines}",
                submit=ClientTool("submit_triage", "Submit one decision per account.", TRIAGE_SCHEMA))
            return res.output.get("accounts") or []

        outcomes = await asyncio.gather(*(one(b) for b in batches), return_exceptions=True)
        results = [r if isinstance(r, list) else [] for r in outcomes]  # a failed batch: its accounts get researched
        for r in outcomes:
            if isinstance(r, Exception):
                self.log.warning("one triage batch failed (%s) - its accounts will be researched", r)
        decisions = {str(d.get("id")): d for batch in results for d in batch if isinstance(d, dict) and d.get("id")}
        research, skipped = [], []
        for a in accounts:
            d = decisions.get(a["account_id"]) or {}
            d.setdefault("decision", "research")  # a model that left the decision out: research it, don't crash
            d.setdefault("reason", "not triaged")
            if d["decision"] == "research" or d["decision"] not in DISQUALIFIER_LABELS:
                research.append(a["account_id"])
                continue
            total, tier, scores = compute({}, d["decision"])
            self.verdicts[a["account_id"]] = {
                "account": a["company"], "account_id": a["account_id"], "tier": tier, "total": total,
                "scores": scores, "disqualifier": d["decision"], "summary": f"Triaged: {d.get('reason', '')}",
                "confidence": "medium", "questions_asked": [], "triaged": True}
            skipped.append((a, tier, d.get("reason", "")))
        await self.reply(msg, f"Triage done: {len(research)} accounts are worth researching, {len(skipped)} are "
                              "clearly not buyers: " + "; ".join(f"{a['company']} ({t.split(' - ')[0]})"
                                                                 for a, t, _ in skipped[:12])
                         + (" ..." if len(skipped) > 12 else "") + ".", {"research": research})
        for a, tier, reason in skipped:
            await self.say("Coordinator", "account_complete",
                           f"Triage verdict on {a['company']}: {tier} - {reason}. No research needed.",
                           {"account_id": a["account_id"], "tier": tier, "total": 0}, thread=a["account_id"])

    async def _score_batch(self, items: list) -> dict:
        """One model call that scores several accounts (each from its own research brief)."""
        keep = ("what_they_service", "industry", "asset_complexity", "scale", "size_evidence", "service_organisation",
                "fsm_stack_signals", "ai_or_digital_initiatives", "regulated", "relationship_to_vendor", "confidence",
                "note_to_analyst")
        blocks = []
        for key, (rec, acct) in items:
            brief = {k: rec.get(k) for k in keep if rec.get(k) not in (None, "", [], "not provided")}
            brief["facts"] = [f.get("claim") if isinstance(f, dict) else f for f in (rec.get("facts") or [])][:3]
            people = "; ".join(f"{c['name']} ({c.get('title', '')})" for c in acct.get("contacts", [])[:4]) or "none named"
            blocks.append(f"### id={key}\nAccount: {rec.get('canonical_name') or acct['company']}\n"
                          f"Seen as: {', '.join(acct.get('roles', []))}\nContacts: {people}\n"
                          f"Brief: {json.dumps(brief, ensure_ascii=False)[:2500]}")
        item = {**VERDICT_SCHEMA, "properties": {"id": {"type": "string"}, **VERDICT_SCHEMA["properties"]},
                "required": ["id", *VERDICT_SCHEMA["required"]]}
        schema = {"type": "object", "properties": {"verdicts": {"type": "array", "items": item}},
                  "required": ["verdicts"]}
        res = await self.llm.run(
            agent=self.name, max_tokens=250 + 420 * len(items), force_submit=True,
            purpose=f"score {len(items)} accounts together",
            system=self._system() + "\nYou get several accounts at once: score each one on its own evidence and "
                   "return one verdict per id, using the id exactly as given. Keep every 'why' under 15 words and "
                   "each summary to one sentence.",
            prompt="\n\n".join(blocks) + "\n\nSubmit one verdict per account.",
            submit=ClientTool("submit_verdict_batch", "Submit a verdict for every account.", schema))
        by_id = {norm_id(v.get("id")): v for v in res.output.get("verdicts") or [] if isinstance(v, dict)}
        out = {}
        for key, _ in items:
            v = by_id.get(norm_id(key))
            if not v:
                continue
            scores = v.get("scores")
            if isinstance(scores, str):
                scores = repair_json(scores) or {}
            # a verdict must score every dimension; a partial one gets its own call instead of a wrong total
            if not isinstance(scores, dict) or any(scores.get(dim) in (None, "", {}) for dim in RUBRIC):
                continue
            v.pop("id", None)
            out[key] = fill_required({**v, "scores": scores}, VERDICT_SCHEMA)[0]
        return out

    async def validate(self, msg: AgentMessage) -> None:
        rec = msg.data
        acct = rec["account"]
        acct_id = acct["account_id"]
        self.research[acct_id] = rec
        asked: list[dict] = []

        async def ask_researcher(args: dict) -> str:
            reply = await self.ask("Researcher", "question", args["question"], {"account_id": acct_id}, thread=acct_id)
            if reply is None:
                return "The Researcher didn't answer in time; proceed with the evidence you have."
            asked.append({"q": args["question"], "a": reply.text})
            return json.dumps({"answer": reply.text, "facts": reply.data.get("facts", []),
                               "confidence": reply.data.get("confidence")}, ensure_ascii=False)

        evidence = {k: v for k, v in rec.items() if k not in ("account", "sources")}
        fast = bool(getattr(self.settings, "fast", False))
        must_ask = (rec.get("confidence") == "low" and not rec.get("sources")) if fast else (
            rec.get("confidence") == "low" or bool(rec.get("open_questions")))
        shared = None
        if fast and not must_ask and self.batch_size() > 1:  # no question needed: score together with others
            if self._batcher is None:
                self._batcher = MicroBatcher(self._score_batch, self.batch_size(), 3.0, "Analyst")
            shared = await self._batcher.submit(acct_id, (rec, acct))
        res = SimpleNamespace(output=shared) if shared else await self.llm.run(
            agent=self.name, max_tokens=1600 if fast else 4000, max_turns=4 if fast else 14,
            system=self._system() + ("\nKeep every 'why' under 15 words and the summary to one sentence."
                                     if fast else ""),
            purpose=f"score {rec.get('canonical_name', acct['company'])}",
            tools=[ClientTool("ask_researcher",
                              "Ask the Researcher a specific question about this company and wait for the answer. "
                              "Use it when a fact you need for scoring is missing or uncertain. Max 2 questions.",
                              QUESTION_TOOL, ask_researcher)],
            prompt=(f"Account: {rec.get('canonical_name', acct['company'])}\n"
                    f"Seen at the conference as: {', '.join(acct.get('roles', []))}\n"
                    f"Contacts at the event ({len(acct.get('contacts', []))}): "
                    + json.dumps(acct.get("contacts", []), ensure_ascii=False)
                    + f"\n\nResearcher's brief:\n{json.dumps(evidence, ensure_ascii=False)[:9000]}\n\n"
                    + ("The research is low-confidence or has open questions: ask the Researcher about the most "
                       "important gap before you score.\n" if must_ask else "")
                    + "Score the account with submit_verdict."),
            submit=ClientTool("submit_verdict", "Submit your ICP verdict.", VERDICT_SCHEMA))
        out = res.output
        total, tier, scores = compute(out.get("scores", {}), out.get("disqualifier"))
        if total == 0 and not any(v["why"] for v in scores.values()) and out.get("disqualifier") in (None, "", "none"):
            tier = "Needs review - no usable score"  # every model's answer was incomplete: don't call it Not ICP
        if out.get("summary") in (None, "", "not provided"):  # the model left it out: say it plainly from the scores
            strong = [k for k, v in scores.items() if v["points"] >= 0.7 * RUBRIC[k][0]]
            out["summary"] = (f"Strongest on {', '.join(strong)}." if strong else "No dimension scored strongly.")
        if out.get("note_to_strategist") in (None, "", "not provided"):
            out["note_to_strategist"] = rec.get("note_to_analyst") or "Lead with the account's service model."
        verdict = {**out, "scores": scores, "total": total, "tier": tier, "questions_asked": asked,
                   "account_id": acct_id, "account": rec.get("canonical_name", acct["company"])}
        self.verdicts[acct_id] = verdict

        breakdown = ", ".join(f"{k} {v['points']}/{RUBRIC[k][0]}" for k, v in scores.items())
        text = (f"Verdict on {verdict['account']}: {tier} ({total}/100; {breakdown}). {out.get('summary', '')}"
                + (f" I checked {len(asked)} point(s) with the Researcher first." if asked else ""))
        if tier.startswith(("A - ", "B - ")):
            await self.say("Strategist", "build_play",
                           f"{text} Over to you: {out.get('note_to_strategist', '')}"
                           + (f" Start with {out['best_contact']}." if out.get("best_contact") else ""),
                           {"verdict": verdict, "research": rec}, performative=HANDOFF, thread=acct_id)
        else:
            await self.say("Coordinator", "account_complete", f"{text} No outreach play needed.",
                           {"account_id": acct_id, "tier": tier, "total": total}, thread=acct_id)

    async def respond(self, msg: AgentMessage) -> None:
        acct_id = msg.data.get("account_id") or msg.thread
        v = self.verdicts.get(acct_id)
        if not v:
            await self.reply(msg, "I haven't scored that account yet.")
            return
        challenge = bool(msg.data.get("challenge"))
        res = await self.llm.run(
            agent=self.name, system=self._system(), max_tokens=2500, force_submit=True,
            purpose=f"respond to {msg.sender} on {v['account']}",
            prompt=(f"{msg.sender} {'challenges' if challenge else 'asks about'} your verdict on {v['account']}:\n"
                    f"\"{msg.text}\"\n\nYour verdict: {json.dumps({k: v[k] for k in ('scores', 'total', 'tier', 'summary', 'disqualifier')}, ensure_ascii=False)}\n"
                    f"Research: {json.dumps({k: x for k, x in self.research.get(acct_id, {}).items() if k not in ('account', 'sources')}, ensure_ascii=False)[:6000]}\n\n"
                    "Answer them directly. Revise your scores only if their point is supported by evidence."),
            submit=ClientTool("submit_response", "Submit your reply.", RESPONSE_SCHEMA))
        out = res.output
        if out.get("revise") and out.get("revised_scores"):
            old = v["tier"]
            total, tier, scores = compute(out["revised_scores"], v.get("disqualifier"))
            v.update(scores=scores, total=total, tier=tier, revised_after=msg.sender, revision_reason=msg.text)
            await self.reply(msg, f"{out['answer']} I've revised {v['account']} from {old} to {tier} ({total}/100).",
                             {"revised": True, "tier": tier, "total": total})
            await self.say("Coordinator", "verdict_revised",
                           f"After {msg.sender}'s challenge I revised {v['account']} from {old} to {tier} ({total}/100).",
                           {"account_id": acct_id, "tier": tier, "total": total}, thread=acct_id)
        else:
            await self.reply(msg, f"{out['answer']} I'm keeping {v['tier']} ({v['total']}/100).",
                             {"revised": False, "tier": v["tier"], "total": v["total"]})
