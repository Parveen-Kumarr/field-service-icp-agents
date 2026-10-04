"""Live web access for the Scout: fetch pages, discover links, parse speakers, find logos.

Nothing here reads saved data. Every page is fetched at run time and recorded
with its HTTP status, byte size, SHA-256 and fetch time as proof of liveness.
"""
from __future__ import annotations

import hashlib
import html as htmllib
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from . import log as logs

log = logs.get("scraper")

SPEAKER_ALT = re.compile(r"^\s*(?P<name>[^,]{3,60}?),\s*(?P<title>.+?),?\s+at\s+(?P<company>[^,]{2,80}?)\s*$")
PAGE_KEYWORDS = {
    "speakers": ("speakers", "our speakers", "speaker faculty"),
    "agenda": ("view full agenda", "agenda"),
    "sponsors": ("our sponsors", "sponsors & exhibitors", "sponsors and exhibitors", "exhibitors"),
    "attendees": ("attendee list", "attendees", "who attends", "companies attending"),
}
SKIP_LINK_WORDS = ("opportunit", "become a sponsor", "download", "register", "brochure", "contact")
LOGO_SECTION_WORDS = ("leading the way", "attendee", "attending", "who attends", "sponsor", "exhibitor",
                      "partner", "companies", "organizations")
IGNORE_IMG_WORDS = ("facebook", "twitter", "linkedin", "instagram", "youtube", "icon", "arrow", "sprite",
                    "event-logo", "site-logo", "banner", "headshot", "avatar", "hero", "background")
VISION_OK = (".png", ".jpg", ".jpeg", ".gif", ".webp")
# Organisers' own branding that shows up in logo strips; never an attending company.
ORGANISER_NAMES = re.compile(r"^(wbr|wbr ?events?|wbrevent|iqpc|worldwide business research)$", re.I)
PREVIOUS_SPEAKERS_TEXT = ("spoke previously", "previous speakers", "past speakers", "view the list",
                          "last year's speakers", "speakers from")


def same_site(a: str, b: str) -> bool:
    """True if two URLs belong to the same site, allowing sibling subdomains (event.x.com / other.x.com).

    Conference sites often move between subdomains when an event is rebranded, e.g.
    fieldserviceusa.wbresearch.com -> servicenextwest.wbresearch.com.
    """
    ha, hb = urlparse(a).netloc.lower(), urlparse(b).netloc.lower()
    if ha == hb:
        return True
    return ".".join(ha.split(".")[-2:]) == ".".join(hb.split(".")[-2:]) and ha.count(".") >= 1


def looks_blocked(html: str) -> str:
    """Recognise an access-denied / bot-check page (returned with HTTP 200 by some browsers' fetches)."""
    if not html:
        return "empty page"
    head = html[:3000].lower()
    for marker in ("403 forbidden", "access denied", "request unsuccessful", "attention required",
                   "verify you are human", "just a moment...", "captcha"):
        if marker in head and len(html) < 20_000:
            return marker
    return ""


def previous_speakers_link(html: str, base_url: str) -> str:
    """The 'see who spoke previously' link a site shows before the new speaker list is published."""
    soup = BeautifulSoup(html, "html.parser")
    best = ""
    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a["href"]).split("#")[0]
        if not same_site(href, base_url):
            continue
        text = " ".join(a.get_text(" ").split()).lower()
        context = " ".join(a.parent.get_text(" ").split()).lower() if a.parent else text
        year_archive = re.search(r"/speakers?/20\d\d/?$", href)
        if year_archive or any(w in context for w in PREVIOUS_SPEAKERS_TEXT) and "speak" in (href + context):
            if year_archive:
                return href
            best = best or href
    return best


@dataclass
class Page:
    role: str
    url: str
    status: int | None = None
    channel: str = "local-http"
    fetched_at: str = ""
    bytes: int = 0
    sha256: str = ""
    html: str = field(default="", repr=False)
    error: str = ""
    rendered_js: bool = False

    @property
    def ok(self) -> bool:
        return bool(self.status and 200 <= self.status < 300 and self.html)

    def record(self) -> dict:
        d = asdict(self)
        d.pop("html")
        return d


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class LiveFetcher:
    def __init__(self, user_agent: str, timeout: float = 25.0, client=None):
        import httpx

        self.client = client or httpx.AsyncClient(
            headers={"User-Agent": user_agent, "Accept": "text/html,application/xhtml+xml"},
            timeout=timeout, follow_redirects=True)

    async def fetch(self, url: str, role: str) -> Page:
        page = Page(role=role, url=url, fetched_at=now_iso())
        started = time.monotonic()
        log.info("GET %s (%s page) ...", url, role)
        try:
            r = await self.client.get(url)
            page.status = r.status_code
            page.html = r.text if r.status_code < 400 else ""
            page.bytes = len(r.content)
            page.sha256 = hashlib.sha256(r.content).hexdigest()[:16]
            if r.status_code >= 400:
                page.error = f"HTTP {r.status_code}"
        except Exception as exc:
            page.error = f"{type(exc).__name__}: {exc}"[:200]
        took = time.monotonic() - started
        if page.error:
            log.warning("GET %s failed after %.1fs: %s", url, took, page.error)
        else:
            log.info("GET %s -> HTTP %s, %s KB in %.1fs", url, page.status, f"{page.bytes / 1024:,.0f}", took)
        return page

    async def close(self) -> None:
        await self.client.aclose()


BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/126.0 Safari/537.36")
PLAYWRIGHT_HINT = ("pip install playwright && python -m playwright install chromium")


async def render_with_browser(url: str, timeout_ms: int = 45000) -> str | None:
    """Render a JavaScript-built page in headless Chromium (Playwright), scrolling to trigger lazy loading."""
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        log.warning("%s looks JavaScript-built but Playwright isn't installed, so it can't be rendered. "
                    "To read such pages: %s", url, PLAYWRIGHT_HINT)
        return None
    started = time.monotonic()
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page(user_agent=BROWSER_UA, locale="en-US")
            await page.goto(url, wait_until="networkidle", timeout=timeout_ms)
            for _ in range(8):  # lazy-loaded speaker grids appear as you scroll
                await page.mouse.wheel(0, 4000)
                await page.wait_for_timeout(400)
            await page.wait_for_load_state("networkidle", timeout=timeout_ms)
            html = await page.content()
            await browser.close()
        blocked = looks_blocked(html)
        if blocked:
            log.warning("the site refused the headless browser for %s (%s) - ignoring the rendered page", url, blocked)
            return None
        log.info("rendered %s in a headless browser (%.1fs, %s KB)", url, time.monotonic() - started,
                 f"{len(html) / 1024:,.0f}")
        return html
    except Exception as exc:
        hint = f" - run: {PLAYWRIGHT_HINT}" if "Executable doesn't exist" in str(exc) else ""
        log.warning("could not render %s in a browser: %s%s", url, str(exc).splitlines()[0][:200], hint)
        return None


# --- parsing -------------------------------------------------------------------
def discover_pages(html: str, base_url: str) -> dict[str, str]:
    """Find the speakers / agenda / sponsors / attendee pages from the site's own navigation."""
    soup = BeautifulSoup(html, "html.parser")
    found: dict[str, str] = {}
    for a in soup.find_all("a", href=True):
        text = " ".join(a.get_text(" ").split()).lower()
        href = urljoin(base_url, a["href"])
        if not same_site(href, base_url) or not text or any(w in text for w in SKIP_LINK_WORDS):
            continue
        if re.search(r"/20\d\d/?$", href):  # previous years' archives
            continue
        for role, words in PAGE_KEYWORDS.items():
            if role not in found and any(text == w or text.startswith(w) for w in words):
                found[role] = href.split("#")[0]
    return found


PERSON = re.compile(r"^[A-Z][\w'.\-]+(?: [A-Z][\w'.\-]*){1,3}$")
JSON_SPEAKER = re.compile(
    r'"(?:name|fullName|speakerName)"\s*:\s*"(?P<name>[^"]{3,60})"[^{}]{0,400}?'
    r'"(?:title|jobTitle|position|job_title)"\s*:\s*"(?P<title>[^"]{2,120})"[^{}]{0,400}?'
    r'"(?:company|companyName|organization|organisation|employer)"\s*:\s*"(?P<company>[^"]{2,80})"', re.S)
QUOTED_ALT = re.compile(r'["\'>](?P<name>[A-Z][^,"\'<>]{2,58}),\s*(?P<title>[^"\'<>]{2,120}?),?\s+at\s+'
                        r'(?P<company>[^,"\'<>]{2,80}?)\s*["\'<]')


def _add_speaker(out: list, seen: set, name: str, title: str, company: str) -> None:
    name, title, company = (" ".join(x.split()).strip(" ,") for x in (name, title, company))
    if not (name and company) or not PERSON.match(name) or company.lower().endswith(" logo"):
        return
    key = (name.lower(), company.lower())
    if key not in seen:
        seen.add(key)
        out.append({"name": name, "title": title, "company": company})


def parse_speakers(html: str) -> list[dict]:
    """Find speakers in a conference page, trying several layouts in turn:

    1. image attributes 'Name, Title at Company' (alt / title / data-alt / aria-label), incl. inside <noscript>
    2. speaker cards: an element whose class mentions 'speaker' with name / title / company lines
    3. the raw HTML, including data embedded in scripts: quoted 'Name, Title at Company' strings and
       JSON objects with name + title + company keys
    """
    soup = BeautifulSoup(html, "html.parser")
    out: list[dict] = []
    seen: set = set()
    for img in soup.find_all(["img", "source", "div", "a"]):
        for attr in ("alt", "title", "data-alt", "aria-label", "data-title"):
            val = " ".join((img.get(attr) or "").split())
            m = SPEAKER_ALT.match(val) if val and not val.lower().endswith(" logo") else None
            if m:
                _add_speaker(out, seen, m["name"], m["title"], m["company"])
    for noscript in soup.find_all("noscript"):  # some sites hide the static markup here
        inner = BeautifulSoup(noscript.decode_contents(), "html.parser")
        for img in inner.find_all("img"):
            val = " ".join((img.get("alt") or "").split())
            m = SPEAKER_ALT.match(val) if val else None
            if m:
                _add_speaker(out, seen, m["name"], m["title"], m["company"])
    if len(out) < 3:
        for card in soup.find_all(class_=re.compile(r"speaker", re.I)):
            lines = [" ".join(t.split()) for t in card.stripped_strings]
            lines = [x for x in lines if x and len(x) < 140 and not x.lower().startswith(("view", "read more", "bio"))]
            if len(lines) >= 3 and PERSON.match(lines[0]):
                _add_speaker(out, seen, lines[0], lines[1], lines[2])
    if len(out) < 3:
        raw = htmllib.unescape(html)
        for m in JSON_SPEAKER.finditer(raw):
            _add_speaker(out, seen, m["name"], m["title"], m["company"])
        for m in QUOTED_ALT.finditer(raw):
            _add_speaker(out, seen, m["name"], m["title"], m["company"])
    return out


def llm_page_text(html: str, limit: int = 60_000) -> str:
    """Readable text for a model; if stripping layout leaves almost nothing, keep everything."""
    text = visible_text(html, limit)
    if len(text) >= 1500:
        return text
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "svg"]):
        tag.decompose()
    alts = [" ".join((i.get("alt") or "").split()) for i in soup.find_all("img") if i.get("alt")]
    full = "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())
    return (full + ("\n\nImage descriptions:\n" + "\n".join(alts) if alts else ""))[:limit]


def visible_text(html: str, limit: int = 80_000) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "header", "footer", "nav"]):
        tag.decompose()
    text = "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())
    return text[:limit]


def _name_from_img(img, src: str) -> str:
    for attr in ("alt", "title", "aria-label"):
        v = " ".join((img.get(attr) or "").split())
        if v and v.lower() not in ("img", "image", "logo"):
            return re.sub(r"\s+logo$", "", v, flags=re.I).strip()
    return ""


def logo_groups(html: str, base_url: str) -> dict[str, list[dict]]:
    """Group logo images by the section heading they sit under.

    Returns {heading: [{"name": alt-derived name or "", "src": absolute url}]}.
    """
    soup = BeautifulSoup(html, "html.parser")
    heading = "(top of page)"
    groups: dict[str, list[dict]] = {}
    for el in soup.find_all(["h1", "h2", "h3", "h4", "img"]):
        if el.name != "img":
            heading = " ".join(el.get_text(" ").split())[:120] or heading
            continue
        src = (el.get("src") or el.get("data-src") or el.get("data-lazy-src") or "").strip()
        if not src or src.startswith("data:"):
            continue
        src = urljoin(base_url, src)
        # judge by file path + alt + class, never the host (the event serves its own images)
        low = (urlparse(src).path + " " + (el.get("alt") or "") + " " + " ".join(el.get("class") or [])).lower()
        if any(w in low for w in IGNORE_IMG_WORDS):
            continue
        name = _name_from_img(el, src)
        if name and SPEAKER_ALT.match(name):  # a person's headshot, not a logo
            continue
        groups.setdefault(heading, []).append({"name": name, "src": src})
    return groups


def pick_logo_sections(groups: dict[str, list[dict]], min_logos: int = 3) -> dict[str, list[dict]]:
    return {h: logos for h, logos in groups.items()
            if len(logos) >= min_logos and any(w in h.lower() for w in LOGO_SECTION_WORDS)}


def normalize_company(name: str) -> str:
    words = re.sub(r"[^a-z0-9& ]+", " ", name.lower()).split()
    drop = {"inc", "corp", "corporation", "co", "ltd", "llc", "group", "the", "company", "plc", "gmbh", "ag", "sa"}
    return " ".join(w for w in words if w not in drop)
