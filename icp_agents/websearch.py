"""Web research tools for a local model: web_search and fetch_page.

Gemini searches and reads pages on Google's servers. A local model can't, so
these tools do it from this machine and hand the model the results as text:

- web_search: the ddgs package (default; rotates across Bing, Brave, DuckDuckGo,
  Google, Mojeek...), DuckDuckGo's HTML page directly, or a SearXNG instance you
  run yourself (ICP_SEARCH_PROVIDER=searxng, ICP_SEARXNG_URL). Searches are paced.
- fetch_page: downloads a page and returns its title and readable text,
  trimmed to ICP_PAGE_CHARS so it fits the model's context.

Each tool is hard-capped per task and records every URL it used as a source.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from urllib.parse import parse_qs, unquote, urlparse

from bs4 import BeautifulSoup

from . import log as logs
from .scraper import visible_text

log = logs.get("web")

DDG_HTML = "https://html.duckduckgo.com/html/"
DDG_LITE = "https://lite.duckduckgo.com/lite/"


class SearchError(RuntimeError):
    pass


def _ddg_target(href: str) -> str:
    """DuckDuckGo wraps result links as //duckduckgo.com/l/?uddg=<real url>."""
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        return unquote(target)
    return href


def parse_ddg_html(html: str, limit: int = 6) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    out, seen = [], set()
    for a in soup.select("a.result__a, a.result-link"):
        url = _ddg_target(a.get("href", ""))
        if not url.startswith("http") or "duckduckgo.com" in urlparse(url).netloc or url in seen:
            continue
        seen.add(url)
        container = a.find_parent(class_="result") or a.find_parent("tr") or a.parent
        snippet_el = container.select_one(".result__snippet, .result-snippet") if container else None
        if snippet_el is None and a.find_parent("tr"):  # lite layout: snippet is in a following row
            nxt = a.find_parent("tr").find_next_sibling("tr")
            snippet_el = nxt.select_one(".result-snippet") if nxt else None
        out.append({"title": " ".join(a.get_text(" ").split()), "url": url,
                    "snippet": " ".join(snippet_el.get_text(" ").split()) if snippet_el else ""})
        if len(out) >= limit:
            break
    return out


def parse_searxng(data: dict, limit: int = 6) -> list[dict]:
    return [{"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
            for r in (data.get("results") or [])[:limit] if r.get("url")]


@dataclass
class ToolBudget:
    """Per-task counters so each agent task can be capped and its sources recorded."""
    max_searches: int = 4
    max_fetches: int = 3
    searches: int = 0
    fetches: int = 0
    failed_searches: int = 0
    sources: list[dict] = field(default_factory=list)

    def add_source(self, url: str, title: str, via: str) -> None:
        if not url:
            return
        for s in self.sources:
            if s["url"] == url:
                if via == "fetch_page":  # a page actually read outranks a search hit
                    s["via"], s["title"] = via, title or s["title"]
                return
        self.sources.append({"url": url, "title": title or "", "via": via})


def parse_ddgs(items: list[dict], limit: int = 6) -> list[dict]:
    """Results from the ddgs package: {'title', 'href', 'body'}."""
    return [{"title": r.get("title", ""), "url": r.get("href") or r.get("url", ""),
             "snippet": r.get("body") or r.get("snippet", "")}
            for r in items[:limit] if (r.get("href") or r.get("url"))]


class LocalWebTools:
    """Search + page reading from this machine, paced so search engines don't block us.

    - Up to `parallel` searches run at once; their START times are spaced `min_interval` seconds apart
      (ddgs rotates across several engines, so a few concurrent queries are fine).
    - "No results" is an answer, not a refusal: it returns [] without any cool-down.
    - A refused search cools down (without blocking other searches) and retries once.
    - After `give_up_after` refusals in a row, search is paused for `pause_for` seconds and the model is told
      to stop searching and answer from what it has, instead of burning minutes on more queries.
    """

    def __init__(self, user_agent: str, timeout: float = 25.0, provider: str = "ddgs",
                 searxng_url: str = "", page_chars: int = 8000, max_results: int = 6, client=None,
                 min_interval: float = 2.0, cooldown: float = 20.0, give_up_after: int = 3, pause_for: float = 60.0,
                 ddgs_factory=None, parallel: int = 3):
        import httpx

        self.provider = provider
        self.searxng_url = searxng_url.rstrip("/")
        self.page_chars = page_chars
        self.max_results = max_results
        self.min_interval, self.cooldown = min_interval, cooldown
        self.give_up_after, self.pause_for = give_up_after, pause_for
        self._lock = asyncio.Lock()   # guards the spacing of search start times only
        self._slots = asyncio.Semaphore(max(1, parallel))
        self._last_search = 0.0
        self._refused_in_row = 0
        self._paused_until = 0.0
        self.stats = {"ok": 0, "refused": 0, "skipped": 0}
        self._ddgs_factory = ddgs_factory
        self.client = client or httpx.AsyncClient(
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                   "(KHTML, like Gecko) Chrome/126.0 Safari/537.36 " + user_agent,
                     "Accept-Language": "en-US,en;q=0.9"},
            timeout=timeout, follow_redirects=True)
        if provider == "searxng" and not self.searxng_url:
            raise ValueError("ICP_SEARCH_PROVIDER=searxng needs ICP_SEARXNG_URL")
        if provider == "ddgs" and ddgs_factory is None:
            try:
                from ddgs import DDGS  # multi-engine: rotates across Bing, Brave, DuckDuckGo, Google, Mojeek, ...
                self._ddgs_factory = lambda: DDGS(timeout=int(timeout))
            except ImportError:
                log.warning("the 'ddgs' package is not installed (pip install ddgs) - falling back to DuckDuckGo only")
                self.provider = "duckduckgo"

    # --- search ------------------------------------------------------------------------
    async def _search_once(self, query: str) -> list[dict]:
        if self.provider == "searxng":
            r = await self.client.get(f"{self.searxng_url}/search", params={"q": query, "format": "json"})
            if r.status_code != 200:
                raise SearchError(f"SearXNG returned HTTP {r.status_code} (is format=json enabled in its settings?)")
            return parse_searxng(r.json(), self.max_results)
        if self.provider == "ddgs":
            def run():
                return self._ddgs_factory().text(query, max_results=self.max_results, backend="auto")
            try:
                return parse_ddgs(await asyncio.to_thread(run) or [], self.max_results)
            except Exception as exc:
                if "no results" in str(exc).lower():
                    return []  # the engines answered: nothing matches this query
                raise SearchError(f"all search engines refused or failed ({type(exc).__name__}: {str(exc)[:120]})")
        results, status = [], None
        for endpoint in (DDG_HTML, DDG_LITE):
            r = await self.client.post(endpoint, data={"q": query})
            status = r.status_code
            if r.status_code == 200:
                results = parse_ddg_html(r.text, self.max_results)
                if results:
                    break
            log.debug("DuckDuckGo %s returned HTTP %s with %d results", endpoint, r.status_code, len(results))
        if not results and status != 200:
            raise SearchError(f"DuckDuckGo refused the search (HTTP {status}) - it is rate limiting this machine")
        return results

    def paused_for(self) -> float:
        return max(0.0, self._paused_until - time.monotonic())

    async def search(self, query: str) -> list[dict]:
        if self.paused_for() > 0:
            self.stats["skipped"] += 1
            raise SearchError(f"web search is paused for {self.paused_for():.0f}s after repeated refusals")
        async with self._slots:  # a few searches at a time across all agents
            for attempt in (1, 2):
                async with self._lock:  # space out the start times
                    wait = self._last_search + self.min_interval - time.monotonic()
                    if wait > 0:
                        await asyncio.sleep(wait)
                    self._last_search = time.monotonic()
                started = time.monotonic()
                try:
                    results = await self._search_once(query)
                except SearchError as exc:
                    self.stats["refused"] += 1
                    self._refused_in_row += 1
                    if self._refused_in_row >= self.give_up_after:
                        self._paused_until = time.monotonic() + self.pause_for
                        log.warning("search refused %d times in a row - pausing web search for %.0fs "
                                    "(agents will answer from what they have). Consider ICP_SEARCH_PROVIDER=searxng",
                                    self._refused_in_row, self.pause_for)
                        raise
                    if attempt == 2:
                        raise
                    log.warning("search %r refused (%s) - cooling down %.0fs and retrying once", query, exc, self.cooldown)
                    await asyncio.sleep(self.cooldown)
                    continue
                self._refused_in_row = 0
                self.stats["ok"] += 1
                log.info("search %r -> %d results in %.1fs (%s)", query, len(results), time.monotonic() - started,
                         self.provider)
                return results
        raise SearchError("unreachable")

    # --- fetch -----------------------------------------------------------------------
    async def fetch(self, url: str) -> dict:
        if not url.startswith(("http://", "https://")):
            raise ValueError("only http(s) URLs can be fetched")
        started = time.monotonic()
        r = await self.client.get(url)
        ctype = (r.headers.get("content-type") or "").lower()
        if r.status_code >= 400:
            log.warning("fetch %s -> HTTP %s", url, r.status_code)
            return {"url": str(r.url), "status": r.status_code, "error": f"HTTP {r.status_code}"}
        if "html" not in ctype and "text" not in ctype:
            return {"url": str(r.url), "status": r.status_code, "error": f"not a web page ({ctype or 'unknown type'})"}
        soup = BeautifulSoup(r.text, "html.parser")
        title = " ".join((soup.title.get_text(" ") if soup.title else "").split())
        text = visible_text(r.text, limit=self.page_chars)
        log.info("fetch %s -> HTTP %s, %d chars in %.1fs", url, r.status_code, len(text), time.monotonic() - started)
        return {"url": str(r.url), "status": r.status_code, "title": title, "text": text,
                "truncated": len(text) >= self.page_chars}

    async def close(self) -> None:
        await self.client.aclose()

    # --- as model tools ------------------------------------------------------------------
    def tools(self, budget: ToolBudget, want_search: bool, want_fetch: bool):
        from .llm import ClientTool

        async def web_search(args: dict) -> dict:
            if budget.searches >= budget.max_searches:
                return {"error": f"search limit reached ({budget.max_searches}); answer with what you have"}
            budget.searches += 1
            query = str(args.get("query", "")).strip()
            try:
                results = await self.search(query)
            except Exception as exc:
                log.warning("search %r failed: %s", query, exc)
                budget.failed_searches += 1
                if self.paused_for() > 0 or budget.failed_searches >= 2:
                    budget.max_searches = budget.searches  # no more searches in this task
                    return {"error": f"{exc}. Web search is unavailable right now: do NOT call web_search again. "
                                     "Use fetch_page on the company's own website if you know it, otherwise answer "
                                     "with what you know and mark unknown facts as unknown (confidence low)."}
                return {"error": str(exc)}
            for res in results:
                budget.add_source(res["url"], res["title"], "web_search")
            return {"query": query, "results": results}

        async def fetch_page(args: dict) -> dict:
            if budget.fetches >= budget.max_fetches:
                return {"error": f"page limit reached ({budget.max_fetches}); answer with what you have"}
            budget.fetches += 1
            url = str(args.get("url", "")).strip()
            try:
                page = await self.fetch(url)
            except Exception as exc:
                log.warning("fetch %s failed: %s", url, exc)
                return {"url": url, "error": f"{type(exc).__name__}: {exc}"}
            if not page.get("error"):
                budget.add_source(page["url"], page.get("title", ""), "fetch_page")
            return page

        out = []
        if want_search:
            out.append(ClientTool("web_search", "Search the web. Returns titles, URLs and snippets. Use short, "
                                  "specific queries (company name + topic).",
                                  {"type": "object", "properties": {"query": {"type": "string"}},
                                   "required": ["query"]}, web_search))
        if want_fetch:
            out.append(ClientTool("fetch_page", "Read a web page (from a search result or a URL you were given). "
                                  "Returns its title and text.",
                                  {"type": "object", "properties": {"url": {"type": "string"}},
                                   "required": ["url"]}, fetch_page))
        return out
