"""Agent 0 - Coordinator: runs the team.

Kicks off the vendor briefing and the live scrape in parallel, tracks
every account through research -> verdict -> play, posts progress, handles
agent errors, and when every account is done writes a debrief (model) and
asks the Reporter for the deliverables.
"""
from __future__ import annotations

import asyncio
import json

from ..llm import ClientTool
from ..messages import BROADCAST, AgentMessage
from .base import BaseAgent

DEBRIEF_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "top_accounts": {"type": "array", "items": {"type": "object", "properties": {
            "account": {"type": "string"}, "why": {"type": "string"}}, "required": ["account", "why"]}},
        "patterns": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
        "recommended_actions": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
    },
    "required": ["headline", "top_accounts", "patterns", "recommended_actions"],
}


CHECKPOINT_EVERY = 10.0  # seconds between partial-report saves


class CoordinatorAgent(BaseAgent):
    name = "Coordinator"
    role = "Plans the run, tracks each account through the team, and closes with a debrief"

    def __init__(self, runtime, llm=None, settings=None):
        super().__init__(runtime, llm, settings)
        self.expected: list[str] | None = None
        self.status: dict[str, dict] = {}
        self.debrief: dict | None = None
        self.report: dict | None = None
        self.failed: str | None = None
        self.scrape_info: dict = {}
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        s = self.settings
        await self.say(BROADCAST, "kickoff",
                       f"Team, we're qualifying the companies at Field Service Next West ({s.event_url}) for "
                       f"{self.vendor()}. Plan: Researcher, "
                       + (f"read {s.vendor_url} live and brief everyone. " if s.vendor_url else
                          "brief everyone on the vendor profile. ")
                       + "Scout, collect the speaker, sponsor "
                       "and attendee lists live from the site - no saved data. Researcher researches each account on "
                       "the web; Analyst scores it against the ICP and asks the Researcher when evidence is thin; "
                       "Strategist plans the first conversation for A and B accounts. I'll track progress."
                       + (f" This run is capped at {s.max_accounts} accounts." if s.max_accounts else ""))
        briefing = asyncio.create_task(self.ask("Researcher", "brief_team_on_vendor",
                                                "Please brief the team on the vendor before research starts.",
                                                timeout=240))
        reply = await self.ask("Scout", "scrape_event",
                               "Collect the live company list from the conference site and hand each account to the "
                               "Researcher as soon as it's ready. Tell me what you found.",
                               {"max_accounts": s.max_accounts}, timeout=900)
        await briefing
        if reply is None or reply.data.get("error"):
            self.failed = reply.data.get("error") if reply else "Scout did not respond within 15 minutes"
            self.log.error("run stopped: %s", self.failed)
            await self.say(BROADCAST, "abort", f"Stopping the run: {self.failed}")
            self.runtime.finished.set()
            return
        self.scrape_info = reply.data
        async with self._lock:
            self.expected = reply.data["accounts"]
            for a in self.expected:
                self.status.setdefault(a, {"stage": "research"})
        eta = ""
        if hasattr(self.llm, "estimate_minutes"):
            n = len(self.expected)
            prospects = n * 0.65 if s.fast else n
            batch = max(1, int(getattr(s, "batch_size", 1) or 1)) if s.fast else 1
            if s.fast and batch > 1:
                # research + scoring shared `batch` accounts per call; plays shared, except the top quarter
                # (A >= 85), which get their own conversation (~2.5 calls each with a question)
                calls = int(-(-n // 25) + 2 * -(-prospects // batch) + -(-prospects * 0.75 // max(1, batch - 1))
                            + prospects * 0.25 * 2.5 + 2)
                tokens_per_call = 3200
            else:
                calls = int(-(-n // 25) + prospects * (2.85 if s.fast else 5) + 1)
                tokens_per_call = 1600
            minutes = self.llm.estimate_minutes(calls, tokens_per_call)
            eta = (f" Estimated model work: about {calls} calls, roughly {minutes:.1f} minutes with the pool's current "
                   f"free capacity (~{self.llm.capacity_per_minute():.0f} calls/min)." if minutes != float("inf") else
                   f" Warning: about {calls} model calls are needed but the pool's remaining daily quota looks too "
                   "small - add another provider key or expect some accounts to fail.")
            self.log.info("ETA:%s", eta)
        await self.say(BROADCAST, "plan_confirmed",
                       f"Scout found {reply.data['total_found']} companies ({reply.data['speakers']} speakers). "
                       f"{len(self.expected)} accounts are in the pipeline; research runs "
                       f"{s.concurrency} at a time" + (" in fast mode (triage first, compact research)" if s.fast
                                                        else "") + "." + eta)
        await self._check_done()

    async def handle(self, msg: AgentMessage) -> None:
        if msg.subject in ("account_complete", "error"):
            acct = msg.data.get("account_id") or msg.thread
            async with self._lock:
                if acct:
                    self.status[acct] = {"stage": "done" if msg.subject == "account_complete" else "error",
                                         "tier": msg.data.get("tier"), "total": msg.data.get("total"),
                                         "error": msg.data.get("error")}
            await self._progress()
            await self._checkpoint()
            await self._check_done()
        elif msg.subject == "verdict_revised":
            async with self._lock:
                st = self.status.setdefault(msg.data["account_id"], {})
                st.update(tier=msg.data["tier"], total=msg.data["total"])

    def _done_ids(self) -> list[str]:
        return [a for a, st in self.status.items() if st.get("stage") in ("done", "error")]

    async def _progress(self) -> None:
        if not self.expected:
            return
        done = self._done_ids()
        if len(done) % 5 and len(done) != len(self.expected):
            return
        tiers: dict[str, int] = {}
        for a in done:
            tier = self.status[a].get("tier") or ""
            key = f"{tier[:1]}-tier" if tier[:1] in ("A", "B", "C") and tier[1:3] == " -" else (
                "other" if tier else "errors")
            tiers[key] = tiers.get(key, 0) + 1
        await self.say(BROADCAST, "progress",
                       f"Progress: {len(done)}/{len(self.expected)} accounts finished. So far: "
                       + ", ".join(f"{k} {v}" for k, v in sorted(tiers.items())) + ".")

    async def _check_done(self) -> None:
        async with self._lock:
            if self.expected is None or self.runtime.finished.is_set() or self.report is not None:
                return
            if any(self.status.get(a, {}).get("stage") not in ("done", "error") for a in self.expected):
                return
            self.report = {}  # claim the wrap-up so it runs once
        await self._wrap_up()

    async def _checkpoint(self) -> None:
        """Save a partial workbook after every finished account, so a crash or Ctrl+C never loses results."""
        reporter = self.runtime.agents.get("Reporter")
        if reporter is None:
            return
        # At most one save every few seconds: writing a workbook per account would slow a fast run
        # (60 triaged accounts can finish within a second of each other). The final report follows anyway.
        now = asyncio.get_running_loop().time()
        if now - getattr(self, "_last_checkpoint", -1e9) < CHECKPOINT_EVERY:
            return
        self._last_checkpoint = now
        try:
            path = await asyncio.to_thread(reporter.write_partial)
            self.log.debug("partial report saved: %s", path)
        except Exception as exc:  # a failed checkpoint must never stop the run
            self.log.warning("could not save the partial report: %s", exc)

    async def _wrap_up(self) -> None:
        try:
            analyst = self.runtime.agents["Analyst"]
            verdicts = [{"account": v.get("account", k), "tier": v.get("tier", ""), "total": v.get("total"),
                         "summary": v.get("summary", "")} for k, v in analyst.verdicts.items()]
            with self.doing("writing the debrief"):
                res = await self.llm.run(
                    agent=self.name, force_submit=True, max_tokens=1500, purpose="write the debrief",
                    system=f"You are the Coordinator of a sales-intelligence team working for {self.vendor()}. "
                           "Write a crisp debrief for the vendor's leadership from the team's verdicts. Be "
                           "specific; no filler.",
                    prompt=("Verdicts from this run (sorted by score):\n"
                            + json.dumps(sorted(verdicts, key=lambda v: -(v.get("total") or 0)),
                                         ensure_ascii=False)[:20000]
                            + "\n\ntop_accounts must be buyers only (A/B/C tiers, best first) - never vendors, "
                              "competitors, consultants or triaged non-buyers. Mention non-buyers only in patterns."),
                    submit=ClientTool("submit_debrief", "Submit the debrief.", DEBRIEF_SCHEMA))
            d = res.output
            buyers = {(v["account"] or "").lower() for v in verdicts if str(v.get("tier", ""))[:2] in ("A ", "B ", "C ")}
            top = [t for t in (d.get("top_accounts") or []) if (t.get("account") or "").lower() in buyers] \
                if buyers else []
            self.debrief = {"headline": d.get("headline", ""), "top_accounts": top,
                            "patterns": d.get("patterns") or [], "recommended_actions": d.get("recommended_actions") or []}
            await self.say(BROADCAST, "debrief",
                           f"{self.debrief['headline']} Top accounts: "
                           + "; ".join(f"{t.get('account', '?')} ({t.get('why', '')})"
                                       for t in self.debrief["top_accounts"][:5])
                           + ". Patterns: " + " ".join(self.debrief["patterns"][:3]), self.debrief)
        except Exception as exc:
            self.log.warning("debrief failed (%s); writing the report without it", exc)
            self.debrief = {"headline": f"Debrief unavailable ({type(exc).__name__})", "top_accounts": [],
                            "patterns": [], "recommended_actions": []}
        try:
            reply = await self.ask("Reporter", "write_report", "Everything is in. Please write the spreadsheet, the "
                                   "conversation transcript and the JSON results.", timeout=300)
            self.report = reply.data if reply and not reply.data.get("error") else {
                "error": (reply.data.get("error") if reply else "Reporter did not respond")}
            if self.report.get("error"):
                self.log.error("the final report was not written: %s", self.report["error"])
        except Exception as exc:
            self.report = {"error": f"{type(exc).__name__}: {exc}"}
            self.log.error("the final report was not written: %s", exc)
        finally:
            self.runtime.finished.set()  # whatever happened above, the run ends here instead of hanging
