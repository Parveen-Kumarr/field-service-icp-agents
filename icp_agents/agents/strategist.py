"""Agent 4 - Strategist: turns an A/B verdict into a first conversation.

Before writing, it asks the Researcher for something current and specific to
open with, and can question or challenge the Analyst's verdict. It then picks
which of the vendor's capabilities to lead with and drafts openers per contact. It
drafts only; a person reviews and sends everything.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from ..batching import MicroBatcher, norm_id
from ..schema_tools import fill_required
from ..vendor_profile import CAPABILITIES
from ..llm import ClientTool
from ..messages import HANDOFF, AgentMessage
from .base import BaseAgent

PLAY_SCHEMA = {
    "type": "object",
    "properties": {
        "lead_with": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 4,
                      "description": "The vendor capabilities or products to lead with"},
        "pain_hypothesis": {"type": "string"},
        "why_now": {"type": "string", "description": "The timely hook, with its source if you have one"},
        "talk_track": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
        "openers": {"type": "array", "items": {"type": "object", "properties": {
            "contact": {"type": "string"}, "channel": {"type": "string", "enum": ["email", "linkedin", "in-person"]},
            "message": {"type": "string", "description": "Under 90 words, specific, no hype"}},
            "required": ["contact", "message"]}},
        "next_step": {"type": "string"},
        "risks": {"type": "string"},
        "note_to_coordinator": {"type": "string"},
    },
    "required": ["lead_with", "pain_hypothesis", "openers", "next_step", "note_to_coordinator"],
}
ASK_SCHEMA = {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}
CHALLENGE_SCHEMA = {"type": "object", "properties": {"argument": {"type": "string"}}, "required": ["argument"]}


class StrategistAgent(BaseAgent):
    name = "Strategist"
    role = "Plans the first conversation for strong-fit accounts and drafts openers (never sends)"

    def __init__(self, runtime, llm=None, settings=None):
        super().__init__(runtime, llm, settings)
        self.plays: dict[str, dict] = {}
        self._batcher: MicroBatcher | None = None
        self._talks = 0

    def _system(self) -> str:
        return (f"You are the Engagement Strategist on a sales-intelligence team working for {self.vendor()}. "
                "You plan the first "
                "conversation with a service leader met at a field service conference. You write like a peer who "
                "understands field service, not like a marketer: specific, short, useful. Never invent facts; "
                "only use what the team found. Never cite a source (press release, article, announcement) that "
                "isn't in the research you were given; if there's no specific trigger, say why_now is to be "
                "confirmed and base the opener on the account's service model instead. You draft; a human sends."
                "\n\nWhat the vendor does:\n" + self.briefing_text()
                + f"\n\nCapabilities you can lead with: {', '.join(self.capabilities())}.")

    async def _play_batch(self, items: list) -> dict:
        """One model call that drafts plays for several accounts."""
        blocks = []
        for key, (verdict, rec, contacts) in items:
            facts = [f.get("claim") if isinstance(f, dict) else f for f in (rec.get("facts") or [])][:3]
            blocks.append(
                f"### id={key}\nAccount: {verdict['account']} - {verdict['tier']} ({verdict['total']}/100)\n"
                f"Analyst: {verdict.get('summary', '')} Note: {verdict.get('note_to_strategist', '')}\n"
                f"Contacts: {json.dumps(contacts, ensure_ascii=False) if contacts else 'none named - suggest who to find'}\n"
                f"Services: {rec.get('what_they_service', '')}. Service org: {rec.get('service_organisation', '')}. "
                f"Digital/AI: {', '.join(rec.get('ai_or_digital_initiatives') or []) or 'unknown'}. Facts: {facts}")
        item = {**PLAY_SCHEMA, "properties": {"id": {"type": "string"}, **PLAY_SCHEMA["properties"]},
                "required": ["id", *PLAY_SCHEMA["required"]]}
        schema = {"type": "object", "properties": {"plays": {"type": "array", "items": item}}, "required": ["plays"]}
        res = await self.llm.run(
            agent=self.name, max_tokens=250 + 520 * len(items), force_submit=True,
            purpose=f"plan first conversations for {len(items)} accounts together",
            system=self._system() + "\nYou get several accounts at once: write one play per id (id exactly as "
                   "given), one opener per named contact (or one for the role to find), openers under 70 words.",
            prompt="\n\n".join(blocks) + "\n\nSubmit one play per account.",
            submit=ClientTool("submit_play_batch", "Submit a play for every account.", schema))
        by_id = {norm_id(p.get("id")): p for p in res.output.get("plays") or [] if isinstance(p, dict)}
        out = {}
        for key, _ in items:
            p = by_id.get(norm_id(key))
            if p and p.get("openers") and p.get("lead_with"):
                p.pop("id", None)
                out[key] = fill_required(p, PLAY_SCHEMA)[0]
        return out

    async def handle(self, msg: AgentMessage) -> None:
        if msg.subject == "build_play":
            with self.doing(f"planning the first conversation with {msg.data['verdict']['account']}"):
                await self.build(msg)

    def capabilities(self) -> list[str]:
        b = self.briefing or {}
        return (b.get("agents") or CAPABILITIES) if b.get("source") == "live" else CAPABILITIES

    async def build(self, msg: AgentMessage) -> None:
        verdict, rec = msg.data["verdict"], msg.data["research"]
        acct = rec["account"]
        acct_id = acct["account_id"]
        dialogue: list[dict] = []

        async def ask_researcher(args: dict) -> str:
            reply = await self.ask("Researcher", "question", args["question"], {"account_id": acct_id}, thread=acct_id)
            if reply is None:
                return "No answer in time; continue without it."
            dialogue.append({"with": "Researcher", "q": args["question"], "a": reply.text})
            return json.dumps({"answer": reply.text, "facts": reply.data.get("facts", [])}, ensure_ascii=False)

        async def ask_analyst(args: dict) -> str:
            reply = await self.ask("Analyst", "question", args["question"], {"account_id": acct_id}, thread=acct_id)
            if reply is None:
                return "No answer in time; continue without it."
            dialogue.append({"with": "Analyst", "q": args["question"], "a": reply.text})
            return reply.text

        async def challenge_analyst(args: dict) -> str:
            reply = await self.challenge("Analyst", "challenge", args["argument"], {"account_id": acct_id},
                                         thread=acct_id)
            if reply is None:
                return "No answer in time; continue without it."
            dialogue.append({"with": "Analyst", "challenge": args["argument"], "a": reply.text})
            if reply.data.get("revised"):
                verdict.update(tier=reply.data["tier"], total=reply.data["total"])
            return json.dumps({"answer": reply.text, "tier": reply.data.get("tier")}, ensure_ascii=False)

        contacts = acct.get("contacts", [])
        fast = bool(getattr(self.settings, "fast", False))
        # fast mode: the top accounts get the full conversation (a hook from the Researcher, a challenge if the
        # verdict looks off); the others are drafted together with other accounts in one shared model call
        # at most `talk_limit` such conversations per run (each costs 2-3 model calls)
        top = (str(verdict.get("tier", "")).startswith("A - ") and (verdict.get("total") or 0) >= 85
               and self._talks < int(getattr(self.settings, "talk_limit", 8) or 0))
        if fast and top:
            self._talks += 1
        shared = None
        if fast and not top and self.batch_size() > 1:
            if self._batcher is None:
                self._batcher = MicroBatcher(self._play_batch, max(2, self.batch_size() - 1), 3.0, "Strategist")
            shared = await self._batcher.submit(acct_id, (verdict, rec, contacts))
        instructions = (
            "Use what the research already gives you. Only if it contains nothing specific to open with, ask the "
            "Researcher one question. If the verdict looks clearly wrong, challenge the Analyst. Keep every field "
            "short: openers under 70 words. Then submit_play, with one opener per named contact (or one for the role "
            "to find)." if fast else
            "First ask the Researcher for one current, specific hook for this account. If anything in the "
            "verdict looks off, question or challenge the Analyst. Then submit_play, with one opener per "
            "named contact (or one for the role to find).")
        res = SimpleNamespace(output=shared, sources=[]) if shared else await self.llm.run(
            agent=self.name, max_tokens=1000 if fast else 5000, max_turns=5 if fast else 14,
            tools=[
                ClientTool("ask_researcher", "Ask the Researcher for a specific, current fact (recent news, service "
                           "initiative, launch, acquisition, hiring) to open the conversation with.", ASK_SCHEMA,
                           ask_researcher),
                ClientTool("ask_analyst", "Ask the ICP Analyst to explain part of the verdict.", ASK_SCHEMA, ask_analyst),
                ClientTool("challenge_analyst", "Disagree with the verdict, with your argument, if the evidence "
                           "suggests the fit is weaker or stronger than scored. The Analyst may revise it.",
                           CHALLENGE_SCHEMA, challenge_analyst),
            ],
            purpose=f"plan the first conversation with {verdict['account']}",
            system=self._system(),
            prompt=(f"Account: {verdict['account']} - {verdict['tier']} ({verdict['total']}/100)\n"
                    f"Analyst's summary: {verdict['summary']}\nAnalyst's note to you: {verdict['note_to_strategist']}\n"
                    f"Contacts met at the event: {json.dumps(contacts, ensure_ascii=False) if contacts else 'none named - suggest who to find'}\n"
                    f"Research: {json.dumps({k: v for k, v in rec.items() if k not in ('account', 'sources')}, ensure_ascii=False)[:7000]}\n\n"
                    + instructions),
            submit=ClientTool("submit_play", "Submit the engagement play.", PLAY_SCHEMA))
        play = {**fill_required(res.output, PLAY_SCHEMA)[0], "dialogue": dialogue, "sources": res.sources}
        self.plays[acct_id] = play
        await self.say("Coordinator", "account_complete",
                       f"Play ready for {verdict['account']} ({verdict['tier']}): lead with "
                       f"{' + '.join(play['lead_with'])}. Pain: {play['pain_hypothesis'].rstrip('.')}. "
                       f"Why now: {(play.get('why_now') or 'n/a').rstrip('.')}. Next step: {play['next_step'].rstrip('.')}. "
                       f"{len(play['openers'])} opener draft(s) waiting for human review. {play['note_to_coordinator']}",
                       {"account_id": acct_id, "tier": verdict["tier"], "total": verdict["total"]},
                       performative=HANDOFF, thread=acct_id)
