"""Agent 1 - Scout: collects the conference company list LIVE.

1. Fetches the event home page and discovers the speakers / sponsors /
   attendee pages from the site's own navigation.
2. Scrapes speakers (structured parse; if the layout isn't recognised, the model
   reads the live page text), sponsor logos and the home page's attendee logo
   strip (logos without text are read by the model's vision).
3. If the site can't be reached from this machine and the model is Gemini, Gemini
   reads it live from Google's servers (URL context). A local model has no such
   channel. There is no saved-data fallback: with no
   live data the run stops.
4. Streams each account to the Researcher as soon as it's ready.
"""
from __future__ import annotations

import asyncio
import base64
import json
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse

from .. import scraper
from ..llm import ClientTool, image_part, text_part, url_tool
from ..messages import HANDOFF, AgentMessage
from .base import BaseAgent


class ScrapeError(RuntimeError):
    pass


SPEAKER_SCHEMA = {
    "type": "object",
    "properties": {
        "speakers": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"}, "title": {"type": "string"}, "company": {"type": "string"}},
            "required": ["name", "company"]}},
        "sponsors": {"type": "array", "items": {"type": "string"},
                     "description": "Sponsor/exhibitor company names, if the page lists them"},
        "pages_read": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "string", "description": "Anything the team should know about the source"},
    },
    "required": ["speakers"],
}
LOGO_SCHEMA = {
    "type": "object",
    "properties": {"logos": {"type": "array", "items": {"type": "object", "properties": {
        "index": {"type": "integer"},
        "company": {"type": "string", "description": "Company name, or empty string if unreadable"}},
        "required": ["index", "company"]}}},
    "required": ["logos"],
}


KNOWN_SHORT = {"ge": "GE", "hp": "HP", "ibm": "IBM", "jnj": "Johnson & Johnson", "3m": "3M", "abb": "ABB",
               "phillips": "Philips", "whirlpol": "Whirlpool", "schneider": "Schneider Electric"}


def _name_from_filename(src: str) -> str:
    """Best-effort company name from a logo file name, e.g. 'logos_0024_GE.jpg' -> 'GE',
    'logos_0023_honeywell.svg.jpg' -> 'Honeywell'. Returns '' for hashes and meaningless names."""
    stem = Path(urlparse(src.strip()).path).name
    stem = re.sub(r"(?i)(\.(svg|png|jpe?g|gif|webp))+$", "", stem)
    stem = re.sub(r"(?i)\b(logos?|img|image|\d{2,}x\d{2,}|v\d+|final|new|copy|color|colour|grey|gray|white|"
                  r"black|rgb|cmyk|web|small|large|transparent)\b", " ", re.sub(r"[-_.]+", " ", stem))
    stem = re.sub(r"\b\d+\b", " ", stem)
    # drop upload hashes like 'yNK2EOTeeFiyVsybla...' (long, with digits or random capitals)
    words = [w for w in stem.split() if not (len(w) >= 12 and (re.search(r"\d", w) or
                                                               sum(c.isupper() for c in w) >= 4))]
    if not words or not re.search(r"[A-Za-z]{2,}", " ".join(words)):
        return ""
    joined = " ".join(words)
    if joined.lower() in KNOWN_SHORT:
        return KNOWN_SHORT[joined.lower()]
    return " ".join(w if (w.isupper() and len(w) <= 4) or any(c.isupper() for c in w[1:]) else w.capitalize()
                    for w in words)


class ScoutAgent(BaseAgent):
    name = "Scout"
    role = "Collects the live speaker, sponsor and attendee-logo lists from the conference site"

    def __init__(self, runtime, llm=None, settings=None, fetcher=None):
        super().__init__(runtime, llm, settings)
        self.fetcher = fetcher
        self.pages: list[scraper.Page] = []
        self.accounts: dict[str, dict] = {}
        self.capture_path: Path | None = None

    async def handle(self, msg: AgentMessage) -> None:
        if msg.subject == "scrape_event":
            with self.doing("collecting the company list from the conference site"):
                await self.scrape(msg)

    # --- helpers --------------------------------------------------------------
    async def _fetch(self, url: str, role: str) -> scraper.Page:
        with self.doing(f"fetching the {role} page {url}"):
            page = await self.fetcher.fetch(url, role)
        self.pages.append(page)
        self._save_html(role, page.html)
        return page

    def _save_html(self, name: str, html: str) -> None:
        """Keep a copy of every page as the site sent it (output/pages/), to diagnose parsing problems."""
        if not html:
            return
        try:
            folder = Path(self.settings.output_dir) / "pages"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"{name}.html").write_text(html, encoding="utf-8")
        except OSError as exc:
            self.log.debug("could not save %s page: %s", name, exc)

    def _add(self, company: str, role: str, source: str, contact: dict | None = None) -> None:
        company = " ".join(company.split()).strip(" .,")
        if len(company) < 2 or scraper.ORGANISER_NAMES.match(company):
            return
        key = scraper.normalize_company(company) or company.lower()
        acct = self.accounts.setdefault(key, {"account_id": key, "company": company, "name_variants": [],
                                              "roles": [], "sources": [], "contacts": []})
        if company not in acct["name_variants"]:
            acct["name_variants"].append(company)
        if role not in acct["roles"]:
            acct["roles"].append(role)
        if source not in acct["sources"]:
            acct["sources"].append(source)
        if contact and all(c["name"].lower() != contact["name"].lower() for c in acct["contacts"]):
            acct["contacts"].append(contact)

    async def _llm_extract_from_text(self, url: str, text: str) -> list[dict]:
        """Ask the model to list the speakers, a few thousand characters at a time (keeps each answer short)."""
        chunks = [text[i:i + 6000] for i in range(0, len(text), 6000)] or [""]
        self.log.info("speaker layout not recognised on %s - asking %s to read the page text (%d chars, %d part(s))",
                      url, self.llm.label, len(text), len(chunks))
        found: list[dict] = []
        for n, chunk in enumerate(chunks, 1):
            res = await self.llm.run(
                agent=self.name, force_submit=True, max_tokens=4000,
                purpose=f"read speakers from the page text (part {n}/{len(chunks)})",
                system="You extract structured data from conference web pages. Only include people and companies "
                       "that actually appear in the text. Never invent anyone. If there are no speakers, return an "
                       "empty list.",
                prompt=f"This is part {n} of the live text of {url}. List every speaker with name, job title and "
                       f"company.\n\n{chunk}",
                submit=ClientTool("submit_extraction", "Submit the speakers found in the text.", SPEAKER_SCHEMA))
            found += [x for x in res.output.get("speakers", []) if x.get("name") and x.get("company")]
        return found

    async def _llm_fetch_live(self) -> dict:
        """Second live channel: Gemini reads the pages itself from Google's servers (URL context)."""
        base = self.settings.event_url
        urls = [base, urljoin(base, "speakers"), urljoin(base, "sponsors")]
        res = await self.llm.run(
            agent=self.name, max_tokens=8000, purpose="read the conference site via URL context",
            server_tools=[url_tool()],
            system="You are a web research agent. Fetch the pages, then report exactly what is on them. "
                   "Never invent people or companies.",
            prompt=("Fetch these live conference pages and extract every speaker (name, title, company) and every "
                    "sponsor/exhibitor company you can see. Follow the site's navigation if a URL has moved.\n"
                    + "\n".join(urls)),
            submit=ClientTool("submit_extraction", "Submit what the live pages list.", SPEAKER_SCHEMA))
        res.output["_sources"] = res.sources
        return res.output

    async def _download_image(self, src: str) -> tuple[str, str] | None:
        """Fetch a logo image live and return (base64, mime type) for the model."""
        try:
            r = await self.fetcher.client.get(src)
        except Exception:
            return None
        mime = (r.headers.get("content-type") or "").split(";")[0].strip().lower()
        allowed = getattr(self.llm, "image_types", None)
        if r.status_code != 200 or not mime.startswith("image/") or mime == "image/svg+xml" or (
                allowed and mime not in allowed):
            return None
        return base64.b64encode(r.content).decode(), mime

    async def _read_logos(self, logos: list[dict], heading: str) -> list[str]:
        """Names for logos: alt text first, then the model's vision, then the file name."""
        # a readable file name ('logos_0024_GE.jpg') is as good as the image: no model call needed
        names = [l["name"] or _name_from_filename(l["src"]) for l in logos]
        unnamed = [(i, l) for i, l in enumerate(logos) if not names[i]]
        if unnamed and self.llm and self.settings.include_logos and not getattr(self.llm, "supports_vision", True):
            self.log.info("'%s': %d logos without text, but no model with image input is available - "
                          "using their file names", heading, len(unnamed))
        elif unnamed and self.llm and self.settings.include_logos:
            batches = [unnamed[k:k + 12] for k in range(0, len(unnamed), 12)]
            self.log.info("'%s': %d logos without text - reading them with %s vision in %d batch(es)",
                          heading, len(unnamed), self.llm.label, len(batches))
            for bn, chunk in enumerate(batches, 1):
                batch = [(i, l) for i, l in chunk
                         if l["src"].strip().lower().split("?")[0].endswith(scraper.VISION_OK)]
                if not batch:
                    continue
                content: list = [text_part(f"These are company logos from the conference web page section "
                                           f"'{heading}'. Name the company in each logo. Use an empty string "
                                           f"if a logo is unreadable or is not a company.")]
                with self.doing(f"downloading {len(batch)} logo images (batch {bn}/{len(batches)}, '{heading}')"):
                    images = await asyncio.gather(*[self._download_image(l["src"]) for _, l in batch])
                for (i, _), img in zip(batch, images):
                    if img:
                        content += [text_part(f"Logo {i}:"), image_part(*img)]
                if len(content) == 1:
                    continue
                try:
                    with self.doing(f"reading logos with the model (batch {bn}/{len(batches)}, '{heading}')"):
                        res = await self.llm.run(
                            agent=self.name, prompt=content, force_submit=True, max_tokens=2000,
                            purpose=f"read logos batch {bn}/{len(batches)}",
                            system="You read company logos precisely.",
                            submit=ClientTool("submit_logos", "Company per logo index.", LOGO_SCHEMA))
                    for item in res.output.get("logos", []):
                        if item.get("company") and 0 <= item["index"] < len(names):
                            names[item["index"]] = item["company"]
                except Exception as exc:
                    await self.say("Coordinator", "scrape_progress", f"Logo reading failed for a batch under "
                                   f"'{heading}' ({exc}); I'll fall back to file names there.")
        for i, l in enumerate(logos):
            if not names[i]:
                names[i] = _name_from_filename(l["src"])
        return [n for n in names if n]

    # --- main -----------------------------------------------------------------
    async def scrape(self, msg: AgentMessage) -> None:
        s = self.settings
        limit = msg.data.get("max_accounts") or 0
        own_fetcher = self.fetcher is None
        if own_fetcher:
            self.fetcher = scraper.LiveFetcher(s.user_agent, s.http_timeout)
        try:
            await self._scrape_local(s)
            if not self.accounts and s.scrape_mode == "live" and self.llm and not getattr(
                    self.llm, "can_fetch_remotely", False):
                self.log.error("the site can't be reached from this machine, and a local model has no cloud "
                               "channel to read it - check your internet/VPN, or use ICP_LLM_PROVIDER=gemini")
            if not self.accounts and s.scrape_mode == "live" and self.llm and getattr(
                    self.llm, "can_fetch_remotely", False):
                await self.say("Coordinator", "scrape_progress",
                               "I couldn't get usable data from this machine's connection to the site, so I'm asking "
                               "Gemini to read the conference pages live from Google's servers instead.")
                out = await self._llm_fetch_live()
                for sp in out.get("speakers", []):
                    self._add(sp["company"], "Speaker", "speakers page (live via Gemini URL context)",
                              {"name": sp["name"], "title": sp.get("title", "")})
                for sp in out.get("sponsors", []) if s.include_sponsors else []:
                    self._add(sp, "Sponsor / Exhibitor", "sponsors page (live via Gemini URL context)")
                for src in out.get("_sources", []):
                    self.pages.append(scraper.Page(role="gemini_url_context", url=src["url"], status=200,
                                                   channel="gemini-url_context",
                                                   fetched_at=src.get("retrieved_at") or scraper.now_iso()))
        finally:
            if own_fetcher:
                await self.fetcher.close()

        if not self.accounts:
            failures = "; ".join(f"{p.role}: {p.error or p.status}" for p in self.pages) or "no pages reached"
            raise ScrapeError(f"No live data could be collected ({failures}). Stopping - this run never uses saved data.")

        self._write_capture()
        ordered = sorted(self.accounts.values(), key=self._priority)
        if limit:
            ordered = ordered[:limit]
        speakers = sum(len(a["contacts"]) for a in self.accounts.values())
        await self.reply(msg,
            f"Live collection finished. I read {sum(p.ok for p in self.pages)} pages and found {speakers} speakers "
            f"across {len(self.accounts)} companies. I'm handing {len(ordered)} accounts to the Researcher now"
            f"{f' (capped at {limit} for this run)' if limit else ''}. Capture with page hashes and fetch times is "
            f"saved at {self.capture_path}.",
            {"accounts": [a["account_id"] for a in ordered], "total_found": len(self.accounts),
             "speakers": speakers, "pages": [p.record() for p in self.pages], "capture": str(self.capture_path)})

        if getattr(s, "fast", False) and len(ordered) > 1:
            # Fast mode: the Analyst triages every account first; only plausible buyers are researched.
            reply = await self.ask("Analyst", "triage_accounts",
                                   f"Before research starts, please triage these {len(ordered)} accounts: which are "
                                   "clearly not buyers (software vendors, consultants, competitors, the vendor itself) "
                                   "and which are worth researching?",
                                   {"accounts": ordered}, timeout=600)
            keep = set(reply.data.get("research", [])) if reply and not reply.data.get("error") else None
            if keep is not None:
                ordered = [a for a in ordered if a["account_id"] in keep]
        for a in ordered:
            await self._handoff(a)
            await asyncio.sleep(0)  # let the Researcher start while we keep handing off

    async def _scrape_local(self, s) -> None:
        home = await self._fetch(s.event_url, "home")
        if not home.ok:
            await self.say("Coordinator", "scrape_progress",
                           f"The conference home page didn't load from this machine ({home.error or home.status}).",
                           {"page": home.record()})
            return
        pages = scraper.discover_pages(home.html, s.event_url)
        pages.setdefault("speakers", urljoin(s.event_url, "speakers"))
        if s.include_sponsors:
            pages.setdefault("sponsors", urljoin(s.event_url, "sponsors"))
        await self.say("Coordinator", "scrape_progress",
                       f"Home page fetched live at {home.fetched_at} (HTTP {home.status}, {home.bytes:,} bytes, "
                       f"sha256 {home.sha256}). From the site's own navigation I found: "
                       + ", ".join(f"{k} -> {v}" for k, v in pages.items()) + ". Fetching them now.",
                       {"page": home.record(), "discovered": pages})

        roles = [r for r in ("speakers", "sponsors", "attendees") if r in pages]
        fetched = await asyncio.gather(*[self._fetch(pages[r], r) for r in roles])
        by_role = dict(zip(roles, fetched))

        # Speakers: parse the page as sent -> previous edition's list if this one isn't published yet ->
        # render it in a browser -> ask the model
        sp = by_role.get("speakers")
        speakers: list[dict] = []
        how, edition_note, speaker_role = "", "", "Speaker"
        if sp and sp.ok:
            speakers = scraper.parse_speakers(sp.html)
            how = "parsing the page as sent"
            html_for_text = sp.html
            if len(speakers) < 3:
                prev_url = scraper.previous_speakers_link(sp.html, sp.url)
                if prev_url and prev_url.rstrip("/") != sp.url.rstrip("/"):
                    year = (re.search(r"(20\d\d)", prev_url) or [None, "previous"])[1]
                    await self.say("Coordinator", "scrape_progress",
                                   f"The speakers page has no names yet - the new edition's speaker list isn't "
                                   f"published. The page links to who spoke previously ({prev_url}), so I'm reading "
                                   f"the {year} speaker list instead: the best live evidence of who attends.",
                                   {"page": sp.record(), "previous": prev_url})
                    prev = await self._fetch(prev_url, f"speakers_{year}")
                    if prev.ok:
                        sp, html_for_text = prev, prev.html
                        speakers = scraper.parse_speakers(prev.html)
                        how = f"parsing the {year} speaker list as sent"
                        edition_note = f" (the {year} edition - the new edition's list isn't published yet)"
                        speaker_role = f"Speaker ({year} edition)"
            if len(speakers) < 3:
                text_len = len(scraper.visible_text(html_for_text))
                self.log.info("speakers page: %d speakers parsed, %s chars of readable text in %s KB of HTML%s",
                              len(speakers), f"{text_len:,}", f"{len(html_for_text) / 1024:,.0f}",
                              " - the list is probably built by JavaScript" if text_len < 1500 else "")
                if s.render_js and text_len < 1500:
                    with self.doing("rendering the speakers page in a headless browser"):
                        rendered = await scraper.render_with_browser(sp.url)
                    if rendered:
                        sp.rendered_js = True
                        self._save_html(f"{sp.role}_rendered", rendered)
                        html_for_text = rendered
                        speakers = scraper.parse_speakers(rendered) or speakers
                        how = "parsing the page after rendering it in a headless browser"
            if len(speakers) < 3 and self.llm:
                with self.doing("reading the speakers page with the model"):
                    speakers = await self._llm_extract_from_text(sp.url, scraper.llm_page_text(html_for_text))
                how = f"{self.llm.label} reading the page text"
        for p in speakers:
            self._add(p["company"], speaker_role, f"speakers page{edition_note} ({sp.url})",
                      {"name": p["name"], "title": p.get("title", "")})
        if sp:
            companies = len({scraper.normalize_company(p["company"]) for p in speakers})
            hint = "" if speakers else (" Nothing usable on it. If the list is built by JavaScript, install a "
                                        f"browser for rendering ({scraper.PLAYWRIGHT_HINT}). The raw page is saved in "
                                        "output/pages/ for inspection.")
            await self.say("Coordinator", "scrape_progress",
                           (f"Speakers page{edition_note}: HTTP {sp.status}, fetched {sp.fetched_at}. Extracted "
                            f"{len(speakers)} speakers from {companies} companies by {how}. "
                            + (f"Examples: {', '.join(p['name'] + ' (' + p['company'] + ')' for p in speakers[:3])}."
                               if speakers else hint))
                           if sp.ok else f"Speakers page failed: {sp.error}.", {"page": sp.record()})

        # Sponsors (logos) and the attendee list
        for role in ("sponsors", "attendees"):
            pg = by_role.get(role)
            if not pg:
                continue
            if not pg.ok:
                await self.say("Coordinator", "scrape_progress",
                               f"The {role} page returned {pg.error or pg.status} - it looks gated or blocked, so I'll "
                               f"rely on the other pages for that list.", {"page": pg.record()})
                continue
            groups = scraper.pick_logo_sections(scraper.logo_groups(pg.html, pg.url), min_logos=2)
            if role == "attendees":  # every page repeats the sponsor slider; only attendee sections count here
                groups = {h: l for h, l in groups.items() if not re.search(r"sponsor|partner|exhibitor", h, re.I)}
            if not groups and s.render_js and role == "sponsors":
                with self.doing(f"rendering the {role} page in a headless browser"):
                    html = await scraper.render_with_browser(pg.url)
                if html:
                    pg.rendered_js = True
                    groups = scraper.pick_logo_sections(scraper.logo_groups(html, pg.url), min_logos=2)
            if role == "attendees" and not groups:
                await self.say("Coordinator", "scrape_progress",
                               f"The attendee page ({pg.url}) shows no company list - it's a request form (the full "
                               "list is gated), so I'll rely on the speakers, sponsors and logo strips.",
                               {"page": pg.record()})
                continue
            label = "Sponsor / Exhibitor" if role == "sponsors" else "Attendee list"
            count = 0
            for heading, logos in groups.items():
                for name in await self._read_logos(logos, heading):
                    self._add(name, label, f"{role} page logos ({pg.url})")
                    count += 1
            await self.say("Coordinator", "scrape_progress",
                           f"{role.title()} page: HTTP {pg.status}{' (rendered with a browser)' if pg.rendered_js else ''}; "
                           f"read {count} company logos from sections {list(groups) or 'none found'}.",
                           {"page": pg.record()})

        # Attendee logo strip on the home page ("Leading the way..." etc.)
        if s.include_logos:
            groups = scraper.pick_logo_sections(scraper.logo_groups(home.html, s.event_url))
            count = 0
            for heading, logos in groups.items():
                sponsorish = re.search(r"sponsor|partner|exhibitor", heading, re.I)
                if sponsorish and s.include_sponsors and by_role.get("sponsors") and by_role["sponsors"].ok:
                    continue  # the same sponsor slider as on the sponsors page, already read there
                label = "Sponsor / Exhibitor" if sponsorish else "Attendee logo (home page)"
                for name in await self._read_logos(logos, heading):
                    self._add(name, label, f"home page logo strip '{heading}'")
                    count += 1
            if groups:
                await self.say("Coordinator", "scrape_progress",
                               f"Home page logo strips {list(groups)}: identified {count} attending companies.")

    SERVICE_TITLE = re.compile(r"service|support|field|customer|aftermarket|operations|maintenance|technical", re.I)
    SENIOR_TITLE = re.compile(r"\b(chief|c[eo]o|vp|vice president|svp|evp|head|director|general manager|gm)\b", re.I)

    def _priority(self, a: dict) -> tuple:
        """Order for handing off (and for --max-accounts): named speakers at companies that aren't sponsors
        first - sponsors are mostly vendors selling to this audience - ranked by how senior and service-
        focused the speaker is; then sponsors that also speak; then logo-only companies."""
        titles = " ".join(c.get("title", "") for c in a["contacts"])
        sponsor = "Sponsor / Exhibitor" in a["roles"]
        score = (2 if self.SERVICE_TITLE.search(titles) else 0) + (1 if self.SENIOR_TITLE.search(titles) else 0)
        return (not a["contacts"], sponsor, -score, a["company"].lower())

    def _write_capture(self) -> None:
        out = Path(self.settings.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        stamp = scraper.now_iso().replace(":", "").replace("-", "")
        self.capture_path = out / f"live_capture_{stamp}.json"
        self.capture_path.write_text(json.dumps({
            "event_url": self.settings.event_url, "captured_at": scraper.now_iso(),
            "pages": [p.record() for p in self.pages], "accounts": list(self.accounts.values()),
        }, indent=2, ensure_ascii=False), encoding="utf-8")

    async def _handoff(self, a: dict) -> None:
        people = "; ".join(f"{c['name']} ({c['title']})" if c.get("title") else c["name"] for c in a["contacts"])
        variants = f" The site lists it as {' / '.join(a['name_variants'])}, which I've merged." \
            if len(a["name_variants"]) > 1 else ""
        text = (f"New account for research: {a['company']}. Seen live as {', '.join(a['roles'])}."
                f"{variants} "
                + (f"Named contacts: {people}. " if people else "No named person - it came from a logo/sponsor list. ")
                + "Please establish what equipment they install or service, how big their field service "
                  "organisation is, and anything that makes them a fit or a non-fit.")
        await self.say("Researcher", "research_account", text, a, performative=HANDOFF, thread=a["account_id"])
