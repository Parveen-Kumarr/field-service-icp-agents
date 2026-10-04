"""Agent 2 - Researcher: finds out, live, who each company is.

- On start-up it reads the vendor's website (ICP_VENDOR_URL) and briefs the whole
  team, so the ICP is grounded in what the vendor says today.
- For every account the Scout hands over, it researches the web (Gemini: Google
  Search + URL context; local model: our web_search + fetch_page tools), cites sources, and hands a written brief to the ICP Analyst.
- It answers follow-up questions from the Analyst and Strategist, with
  new searches when its first research doesn't cover the question.
"""
from __future__ import annotations

import asyncio
import json
import re

from types import SimpleNamespace

from ..batching import MicroBatcher, norm_id
from ..schema_tools import fill_required
from ..vendor_profile import INDUSTRIES, VENDOR_BASELINE
from ..llm import ClientTool, search_tool, url_tool
from ..messages import BROADCAST, HANDOFF, AgentMessage
from .base import BaseAgent

RESEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "canonical_name": {"type": "string"},
        "website": {"type": "string"},
        "what_they_service": {"type": "string", "description": "Physical equipment/assets they install, maintain or support"},
        "industry": {"type": "string", "enum": INDUSTRIES},
        "asset_complexity": {"type": "integer", "minimum": 0, "maximum": 5},
        "scale": {"type": "string", "enum": ["enterprise", "large", "mid", "small", "unknown"]},
        "size_evidence": {"type": "string", "description": "Revenue / employees / installed base, with year"},
        "service_organisation": {"type": "string", "description": "What is known about their field service / support org"},
        "fsm_stack_signals": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
        "ai_or_digital_initiatives": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
        "regulated": {"type": "boolean"},
        "relationship_to_vendor": {"type": "string",
                                    "enum": ["prospect", "competitor", "partner_or_vendor", "consultant", "self", "unknown"]},
        "facts": {"type": "array", "items": {"type": "object", "properties": {
            "claim": {"type": "string"}, "source_url": {"type": "string"}}, "required": ["claim"]}, "maxItems": 8},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "open_questions": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
        "note_to_analyst": {"type": "string", "description": "2-4 sentences to the ICP Analyst: what matters for fit and what is uncertain"},
    },
    "required": ["canonical_name", "what_they_service", "industry", "asset_complexity", "scale",
                 "relationship_to_vendor", "confidence", "note_to_analyst"],
}
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "description": "Direct answer written to the colleague who asked"},
        "facts": {"type": "array", "items": {"type": "object", "properties": {
            "claim": {"type": "string"}, "source_url": {"type": "string"}}, "required": ["claim"]}, "maxItems": 8},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["answer", "confidence"],
}
BRIEFING_SCHEMA = {
    "type": "object",
    "properties": {
        "offering": {"type": "string"}, "agents": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
        "target_verticals": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
        "buyer_personas": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
        "value_claims": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
        "integrations": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
        "competitors": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
        "icp_signals": {"type": "array", "items": {"type": "string"}, "maxItems": 6,
                        "description": "What makes a company a strong fit, in your words"},
        "message_to_team": {"type": "string"},
    },
    "required": ["offering", "target_verticals", "icp_signals", "message_to_team"],
}


STOPWORDS = set("""a an the and or of in on for to with by at from is are was were be been any do does did has have had
what which who whom whose when where why how this that these those their its it they them there as e.g. eg such
recent last past months month year years specific news about like e g i.e public publicly current currently""".split())


def _search_query(company: str, question: str, words: int = 7) -> str:
    """A short web query from a colleague's long question: the company plus the question's key words."""
    low_company = company.lower()
    keep = []
    for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9&+/-]*", question):
        lw = w.lower()
        if lw in STOPWORDS or lw in low_company or len(lw) < 3 or lw in (k.lower() for k in keep):
            continue
        keep.append(w)
        if len(keep) >= words:
            break
    return f"{company} " + " ".join(keep)


def _missing_core(record: dict, required: list) -> int:
    return sum(1 for k in required if record.get(k) in (None, "", "not provided"))


class ResearcherAgent(BaseAgent):
    name = "Researcher"
    role = "Researches each company live on the web and answers colleagues' questions with sources"

    def __init__(self, runtime, llm=None, settings=None):
        super().__init__(runtime, llm, settings)
        self.records: dict[str, dict] = {}
        self._research_gate = asyncio.Semaphore(settings.concurrency if settings else 4)
        # questions never queue behind bulk research
        self._question_gate = asyncio.Semaphore(6 if settings and getattr(settings, "fast", False) else 3)
        self.briefing: dict | None = None
        self._batcher: MicroBatcher | None = None
        self.briefed = asyncio.Event()

    async def handle(self, msg: AgentMessage) -> None:
        if msg.subject == "brief_team_on_vendor":
            await self.brief_team(msg)
        elif msg.subject == "research_account":
            company = msg.data.get("company", msg.thread)
            with self.doing(f"waiting for the vendor briefing before researching {company}"):
                try:  # research is grounded in the briefing; don't start before it (max 3 min)
                    await asyncio.wait_for(self.briefed.wait(), 180)
                except asyncio.TimeoutError:
                    self.log.warning("no vendor briefing after 3 min; researching %s with the generic profile", company)
            with self.doing(f"queued for a research slot: {company}"):
                await self._research_gate.acquire()
            try:
                with self.doing(f"researching {company}"):
                    await self.research(msg)
            finally:
                self._research_gate.release()
        elif msg.performative == "request":
            async with self._question_gate:
                with self.doing(f"answering {msg.sender}'s question ({msg.thread})"):
                    await self.answer(msg)

    # --- briefing -----------------------------------------------------------------
    async def brief_team(self, msg: AgentMessage) -> None:
        url = self.settings.vendor_url
        if not url:
            self.briefing = {"offering": VENDOR_BASELINE, "target_verticals": [], "icp_signals": [],
                             "message_to_team": "No vendor website is configured (ICP_VENDOR_URL), so we'll work "
                                                "from the built-in profile of a field-service AI vendor.",
                             "source": "baseline"}
        else:
            try:
                with self.doing(f"reading the vendor site {url} to brief the team"):
                    res = await self.llm.run(
                        agent=self.name, max_tokens=4000, purpose="read the vendor site and brief the team",
                        server_tools=[url_tool()],
                        system="You are the research lead on a sales-intelligence team. Read the vendor's live "
                               "website and brief your colleagues precisely. Report only what the site says.",
                        prompt=f"Read {url} live (and its product or agents page if linked). Brief the team on what "
                               f"this vendor sells, to whom, its named products or agents, value claims, "
                               f"integrations, competitors, and what makes a company a strong fit as its customer.",
                        submit=ClientTool("submit_briefing", "Submit the team briefing.", BRIEFING_SCHEMA))
                self.briefing = {**res.output, "source": "live", "sources": res.sources}
            except Exception as exc:
                self.log.warning("could not read the vendor site live: %s", exc)
                self.briefing = {"offering": VENDOR_BASELINE, "target_verticals": [], "icp_signals": [],
                                 "message_to_team": f"I couldn't read the vendor site live ({type(exc).__name__}); "
                                                    "use the built-in profile.", "source": "baseline"}
        self.briefed.set()
        b = self.briefing
        self.log.info("vendor briefing ready (%s)", b["source"])
        await self.say(BROADCAST, "vendor_briefing",
                       f"{b['message_to_team']} In short: {b['offering'][:400]} "
                       f"Target verticals: {', '.join(b.get('target_verticals', [])[:12]) or 'see profile'}. "
                       f"Strong-fit signals: {'; '.join(b.get('icp_signals', [])[:6]) or 'see profile'}.", b)
        await self.reply(msg, "Briefing sent to the whole team.", {"source": b["source"]})

    # --- research -----------------------------------------------------------------
    def _fast(self) -> bool:
        return bool(getattr(self.settings, "fast", False)) and getattr(self.llm, "web", None) is not None

    async def _quick_search(self, query: str) -> list[dict]:
        """One search, run by code (not a model tool loop). Returns [] if search is unavailable."""
        web = getattr(self.llm, "web", None)
        if web is None:
            return []
        try:
            return (await web.search(query))[:5]
        except Exception as exc:
            self.log.info("quick search for %r unavailable (%s) - using the model's own knowledge", query, exc)
            return []

    def _fast_system(self) -> str:
        return (f"You are the Researcher on a sales-intelligence team qualifying prospects for {self.vendor()}, a "
                "vendor of AI agents for field service on complex physical equipment. Use the search results and "
                "your own knowledge of well-known companies. Be brief: short phrases, at most 3 facts (cite the "
                "result URL when a fact comes from one), say 'unknown' rather than guess numbers.")

    async def _research_batch(self, items: list) -> dict:
        """One model call that researches several companies (each with its own search results)."""
        blocks = []
        for key, (a, hits) in items:
            people = "; ".join(f"{c['name']} ({c.get('title', '')})" for c in a.get("contacts", [])[:3]) or "none named"
            snips = "\n".join(f"  [{i + 1}] {h['title']} - {h['snippet'][:220]} ({h['url']})"
                              for i, h in enumerate(hits[:4])) or "  (no search results - use what you know)"
            blocks.append(f"### id={key}\nCompany: {a['company']} (seen as {', '.join(a.get('roles', []))})\n"
                          f"People: {people}\nSearch results:\n{snips}")
        item_schema = {**RESEARCH_SCHEMA, "properties": {"id": {"type": "string"}, **RESEARCH_SCHEMA["properties"]},
                       "required": ["id", *RESEARCH_SCHEMA["required"]]}
        schema = {"type": "object", "properties": {"results": {"type": "array", "items": item_schema}},
                  "required": ["results"]}
        res = await self.llm.run(
            agent=self.name, max_tokens=200 + 450 * len(items), force_submit=True,
            purpose=f"research {len(items)} companies together",
            system=self._fast_system() + " You get several companies at once: research each one separately and "
                   "return one result per id, using the id exactly as given.",
            prompt="\n\n".join(blocks) + "\n\nSubmit one research result per company.",
            submit=ClientTool("submit_research_batch", "Submit research for every company.", schema))
        by_id = {norm_id(r.get("id")): r for r in res.output.get("results") or [] if isinstance(r, dict)}
        out = {}
        for key, (a, hits) in items:
            r = by_id.get(norm_id(key))
            if not r or _missing_core(r, RESEARCH_SCHEMA["required"]) > 2:
                continue  # left out or too thin: this account gets its own call instead
            r.pop("id", None)
            out[key] = fill_required(r, RESEARCH_SCHEMA)[0]  # every field present, in the right shape
        return out

    async def research_fast(self, msg: AgentMessage):
        """Fast mode: one search + one compact model call (shared with other accounts when several are in
        flight). Returns an object with .output, .sources and .searches."""
        a = msg.data
        contacts = "\n".join(f"- {c['name']}, {c.get('title','')}" for c in a.get("contacts", [])) or "- none named"
        with self.doing(f"searching the web for {a['company']}"):
            hits = await self._quick_search(f"{a['company']} company products field service")
        sources = [{"url": h["url"], "title": h["title"], "via": "web_search"} for h in hits]
        if self.batch_size() > 1:
            if self._batcher is None:
                # searches finish a few seconds apart, so wait up to 5 s to fill a batch
                self._batcher = MicroBatcher(self._research_batch, self.batch_size(), 5.0, "Researcher")
            with self.doing(f"researching {a['company']} (shared model call)"):
                shared = await self._batcher.submit(a["account_id"], (a, hits))
            if shared:
                return SimpleNamespace(output=shared, sources=sources, searches=1 if hits else 0)
        snippets = "\n".join(f"[{i + 1}] {h['title']} - {h['snippet']} ({h['url']})" for i, h in enumerate(hits)) \
            or "(no search results - use what you know and mark unknowns)"
        res = await self.llm.run(
            agent=self.name, max_tokens=900, force_submit=True, purpose=f"research {a['company']}",
            system=self._fast_system(),
            prompt=(f"Company: {a['company']} (seen at Field Service Next West as {', '.join(a.get('roles', []))})\n"
                    f"Contacts:\n{contacts}\n\nSearch results:\n{snippets}\n\nSubmit your research."),
            submit=ClientTool("submit_research", "Submit your research on the company.", RESEARCH_SCHEMA))
        res.sources = [{"url": h["url"], "title": h["title"], "via": "web_search"} for h in hits]
        res.searches = 1 if hits else 0
        return res

    async def research(self, msg: AgentMessage) -> None:
        a = msg.data
        if self._fast():
            res = await self.research_fast(msg)
            record = {**res.output, "account": a, "sources": res.sources, "searches": res.searches}
            await self._hand_to_analyst(a, record, res)
            return
        contacts = "\n".join(f"- {c['name']}, {c.get('title','')}" for c in a.get("contacts", [])) or "- none named"
        res = await self.llm.run(
            agent=self.name, max_tokens=6000, purpose=f"research {a['company']}",
            server_tools=[search_tool(self.settings.searches_per_account), url_tool()],
            system=(f"You are the Researcher on a sales-intelligence team qualifying prospects for {self.vendor()}, "
                    "a vendor of AI agents for field service on complex physical equipment. Research companies on "
                    "the live web, prefer the company's own site and reputable sources, cite a source URL for every "
                    "fact, and say plainly when something is unknown. Do not guess sizes or numbers. Use at most "
                    f"{self.settings.searches_per_account} web searches and read at most 3 pages."
                    "\n\nWhat the vendor does:\n" + self.briefing_text()),
            prompt=(f"Research this company from the Field Service Next West 2026 conference.\n"
                    f"Company as listed: {a['company']} (variants: {', '.join(a.get('name_variants', []))})\n"
                    f"How it appeared: {', '.join(a.get('roles', []))}\nContacts:\n{contacts}\n\n"
                    "Find: what equipment they install/service, industry, size, their field service organisation, "
                    "FSM/CRM stack signals (ServiceNow, SAP, Salesforce/ServiceMax, IFS, Oracle...), any service AI or "
                    "digital programs, whether they are regulated, and whether they are a prospect, competitor, "
                    "partner/vendor, consultant, or the vendor itself. Then submit your research."),
            submit=ClientTool("submit_research", "Submit your research on the company.", RESEARCH_SCHEMA))
        record = {**res.output, "account": a, "sources": res.sources, "searches": res.searches}
        await self._hand_to_analyst(a, record, res)

    async def _hand_to_analyst(self, a: dict, record: dict, res) -> None:
        record = fill_required(record, RESEARCH_SCHEMA)[0]  # never hand over a record with missing fields
        if not record.get("canonical_name") or record["canonical_name"] == "not provided":
            record["canonical_name"] = a["company"]  # the name as seen on the conference site
        self.records[a["account_id"]] = record
        n_src = len(res.sources)
        await self.say(
            "Analyst", "validate_account",
            f"{record['canonical_name']}: {record['note_to_analyst']} "
            f"(What they service: {record['what_they_service']}. Industry: {record['industry']}, scale: "
            f"{record['scale']}, relationship: {record['relationship_to_vendor']}. Confidence {record['confidence']}; "
            f"{res.searches} searches, {n_src} sources.)"
            + (f" Open questions: {'; '.join(record.get('open_questions') or [])}." if record.get("open_questions") else ""),
            record, performative=HANDOFF, thread=a["account_id"])

    # --- follow-up questions ----------------------------------------------------------
    async def answer(self, msg: AgentMessage) -> None:
        acct_id = msg.data.get("account_id") or msg.thread
        known = self.records.get(acct_id, {})
        known_view = {k: v for k, v in known.items() if k not in ("account", "sources")}
        fast = self._fast()
        company = known.get("canonical_name") or (known.get("account") or {}).get("company") or acct_id
        hits, fresh = [], ""
        if fast:  # one quick search on the question itself, run by code, so the answer has something new in it
            with self.doing(f"searching the web to answer {msg.sender} about {company}"):
                hits = await self._quick_search(_search_query(company, msg.text))
            fresh = "\n".join(f"[{i + 1}] {h['title']} - {h['snippet']} ({h['url']})" for i, h in enumerate(hits))
        res = await self.llm.run(
            agent=self.name, max_tokens=1000 if fast else 3000, force_submit=fast,
            purpose=f"answer {msg.sender}'s question on {acct_id}",
            server_tools=[] if fast else [search_tool(2)],
            system=("You are the Researcher on a sales-intelligence team. A colleague has asked you a question. "
                    "Use what you already found" + (" and the fresh search results" if fast else
                    ", search again (at most 2 searches) if it doesn't answer the question") +
                    ", cite sources (only URLs that appear in what you were given - never invent a source, date or "
                    "press release), and answer directly and briefly. If the evidence doesn't answer it, say what "
                    "you do know that is closest, and what remains unknown - never just 'I don't know'."),
            prompt=(f"{msg.sender} asks about {company}:\n\"{msg.text}\"\n\n"
                    f"What you found earlier:\n{json.dumps(known_view, ensure_ascii=False)[:6000]}"
                    + (f"\n\nFresh search results:\n{fresh}" if fresh else "")),
            submit=ClientTool("submit_answer", "Submit your answer.", ANSWER_SCHEMA))
        if hits:
            res.sources = [{"url": h["url"], "title": h["title"], "via": "web_search"} for h in hits]
            res.searches = 1
        facts = res.output.get("facts") or []
        if known and facts:
            known.setdefault("facts", []).extend(facts)
            known.setdefault("sources", []).extend(res.sources)
        answer = (res.output.get("answer") or "").strip()
        if not answer or answer == "not provided":
            answer = ("I couldn't find anything specific on that in public sources; treat it as unknown. "
                      + (f"Closest known facts: {known.get('note_to_analyst', '')}" if known.get("note_to_analyst") else ""))
            res.output["answer"] = answer
        await self.reply(msg, answer, {**res.output, "sources": res.sources, "searches": res.searches})
