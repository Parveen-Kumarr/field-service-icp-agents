"""Run the field-service ICP agent team live.

    python run.py                      # full live run
    python run.py --max-accounts 10    # quicker demo on the first 10 accounts
    python run.py --check              # only test the model(s) and the site, then exit
    python run.py --log-level DEBUG    # show every model call, tool call and message on the console
    python run.py --provider gemini    # use Gemini instead of the local model for this run
    python run.py --provider pool      # several free cloud providers together (fastest; set their keys)

Model: ICP_LLM_PROVIDER in .env - "pool" (default: free cloud providers used together), "ollama"
(local Gemma 4 via Ollama) or "gemini" (needs GEMINI_API_KEY). Every run scrapes the conference site live; there
is no saved-data mode. A detailed log of every run is written to output/logs/.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from icp_agents import Settings, run_team
from icp_agents import log as logs
from icp_agents.llm import LLMError
from icp_agents.providers import make_llm
from icp_agents.scraper import LiveFetcher
from icp_agents.ui import ConsoleUI

log = logs.get("run")


def parse_args(s: Settings) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--event-url", default=s.event_url)
    ap.add_argument("--vendor-name", default=s.vendor_name, help="company whose ICP you qualify against (optional)")
    ap.add_argument("--vendor-url", default=s.vendor_url, help="its website; read live to brief the team (optional)")
    ap.add_argument("--max-accounts", type=int, default=s.max_accounts, help="0 = all accounts found")
    ap.add_argument("--concurrency", type=int, default=None, help="accounts researched in parallel")
    ap.add_argument("--searches", type=int, default=s.searches_per_account, help="Google searches per account (guideline)")
    ap.add_argument("--scrape", choices=["live", "local"], default=s.scrape_mode,
                    help="live: local HTTP, then Gemini URL context if blocked; local: local HTTP only")
    ap.add_argument("--no-sponsors", action="store_true")
    ap.add_argument("--no-logos", action="store_true", help="skip reading logo images with Gemini vision")
    ap.add_argument("--provider", choices=["pool", "ollama", "gemini"], default=s.llm_provider,
                    help="which model the agents think with (default from ICP_LLM_PROVIDER)")
    ap.add_argument("--model", default=None, help="model name (default: gemma4:12b for ollama, gemini-3.8-flash for gemini)")
    ap.add_argument("--llm-timeout", type=float, default=None, help="seconds before a model call is retried")
    ap.add_argument("--heartbeat", type=float, default=s.heartbeat, help="seconds between status reports (0 = off)")
    ap.add_argument("--log-level", default=s.log_level, choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                    help="console log level (the log file always has DEBUG)")
    ap.add_argument("--fast", dest="fast", action="store_true", default=None,
                    help="triage first, one search + compact answers per account (default for --provider pool)")
    ap.add_argument("--no-fast", dest="fast", action="store_false", help="full research mode")
    ap.add_argument("--check", action="store_true", help="test the model and the site, then exit")
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--out", default=s.output_dir)
    ap.add_argument("--plain", action="store_true", help="plain console output (no colours)")
    return ap.parse_args()


async def preflight(s: Settings, llm) -> bool:
    ok = True
    what = {"gemini": "Gemini", "pool": "the model pool"}.get(s.llm_provider, f"Ollama ({s.ollama_url})")
    log.info("Preflight 1/2: checking %s ...", what if s.llm_provider == "pool" else f"{what} with model {s.model}")
    try:
        took = await llm.preflight()
        log.info("Preflight 1/2: %s OK (%.1fs)%s", what, took,
                 "" if s.llm_provider == "pool" else f" - model {s.model} answers")
    except LLMError as exc:
        log.error("Preflight 1/2: the model is not usable: %s", exc)
        ok = False
    log.info("Preflight 2/2: fetching %s ...", s.event_url)
    fetcher = LiveFetcher(s.user_agent, s.http_timeout)
    try:
        page = await fetcher.fetch(s.event_url, "home")
    finally:
        await fetcher.close()
    if page.ok:
        log.info("Preflight 2/2: conference site reachable (HTTP %s)", page.status)
    elif s.scrape_mode == "live" and s.llm_provider == "gemini":  # only Gemini can read the site remotely
        log.warning("Preflight 2/2: site not reachable from this machine (%s); the Scout will ask Gemini to read "
                    "it via URL context instead", page.error or page.status)
    else:
        log.error("Preflight 2/2: site not reachable from this machine (%s)%s", page.error or page.status,
                  "" if s.llm_provider == "gemini" else " - the Scout can only read what this machine can reach")
        ok = False
    return ok


def main() -> int:
    s = Settings()
    args = parse_args(s)
    s.event_url, s.vendor_name, s.vendor_url = args.event_url, args.vendor_name, args.vendor_url
    s.max_accounts, s.concurrency, s.searches_per_account = args.max_accounts, args.concurrency, args.searches
    s.scrape_mode, s.output_dir = args.scrape, args.out
    if args.provider != s.llm_provider:
        s.llm_provider, s.model, s.concurrency, s.llm_timeout = args.provider, "", 0, 0
    if args.model:
        s.model = args.model
    if args.concurrency:
        s.concurrency = args.concurrency
    if args.llm_timeout:
        s.llm_timeout = args.llm_timeout
    if args.fast is not None:
        s.fast = args.fast
    elif args.provider != Settings().llm_provider and not os.environ.get("ICP_FAST"):
        s.fast = None  # switching provider on the command line: use that provider's default
    s.resolve()
    s.include_sponsors, s.include_logos = not args.no_sponsors, not args.no_logos
    s.heartbeat, s.log_level = args.heartbeat, args.log_level

    log_path = logs.setup_logging(s.log_level, Path(s.output_dir) / "logs" /
                                  f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    log.info("Log file: %s", log_path)

    if s.llm_provider == "pool":
        from icp_agents.pool import PRESETS
        have = [n for n in s.pool_order if n in PRESETS and not PRESETS[n].retired
                and (not PRESETS[n].needs_key or os.environ.get(PRESETS[n].key_env))]
        if not have:
            log.error("ICP_LLM_PROVIDER=pool but no provider key is set. Add at least one of: %s (see .env.example)",
                      ", ".join(PRESETS[n].key_env for n in s.pool_order
                                if n in PRESETS and PRESETS[n].needs_key and not PRESETS[n].retired))
            return 2
        log.info("Model pool providers (in order): %s", ", ".join(have))
        missing = [PRESETS[n].key_env for n in s.pool_order
                   if n in PRESETS and PRESETS[n].needs_key and not PRESETS[n].retired and n not in have]
        cloud = [n for n in have if PRESETS[n].needs_key]
        if len(cloud) < 3 and missing:
            log.warning("Only %d cloud provider key(s) set - the run works but will be slower than the ~5 minute "
                        "target for a full conference. Add more free keys for speed: %s", len(cloud), ", ".join(missing))
    if s.llm_provider == "gemini" and not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        log.error("GEMINI_API_KEY is not set. Add it to .env (see .env.example). "
                  "Get a key at https://aistudio.google.com/apikey")
        return 2

    log.info("Settings: provider=%s | model=%s | event=%s | vendor=%s | max_accounts=%s | concurrency=%d | "
             "scrape=%s | model timeout=%.0fs | heartbeat=%.0fs%s", s.llm_provider, s.model, s.event_url,
             s.vendor_name or s.vendor_url or "(generic profile)", s.max_accounts or "all", s.concurrency,
             s.scrape_mode, s.llm_timeout, s.heartbeat,
             (f" | search={s.search_provider} | num_ctx={s.ollama_num_ctx} | think={s.ollama_think}"
              if s.llm_provider == "ollama" else "") + (" | FAST mode" if s.fast else ""))

    llm = make_llm(s)

    async def go():
        # preflight and the team share one event loop (and so one Gemini connection pool)
        if not args.skip_preflight or args.check:
            ok = await preflight(s, llm)
            if args.check or not ok:
                if not ok:
                    log.error("Preflight failed - fix the problem above and run again (details in %s)", log_path)
                return ok, None
        return True, await run_team(s, llm, on_message=ConsoleUI(plain=args.plain))

    started = time.time()
    try:
        ok, result = asyncio.run(go())
    except KeyboardInterrupt:
        log.warning("Interrupted. What happened up to now is in %s", log_path)
        return 130
    if result is None:
        return 0 if ok else 1
    rt, team = result
    coord = team["coordinator"]
    if coord.failed:
        log.error("Run stopped: %s (details in %s)", coord.failed, log_path)
        return 1

    u = llm.total_usage()
    stats = rt.stats()
    print("=" * 100)
    print(f"Finished in {time.time() - started:,.0f}s | {stats['messages']} agent messages, "
          f"{stats['questions_answered']} questions answered between agents")
    print(f"{s.llm_provider} ({s.model}): {u.calls} calls, {u.input_tokens:,} in / {u.output_tokens:,} out / "
          f"{u.thought_tokens:,} thinking tokens, {u.web_searches} web searches, {u.url_fetches} pages read")
    for k in ("xlsx", "transcript", "json"):
        if coord.report and coord.report.get(k):
            print(f"{k:>10}: {coord.report[k]}")
    if hasattr(llm, "backends"):
        for b in llm.backends:
            st = b.stats
            print(f"  {b.name:<10} {b.model:<28} {st['calls']:>4} calls  {st['in_tokens'] + st['out_tokens']:>8,} tokens  "
                  f"{st['rate_limited']:>3} rate-limited  {b.disabled or b.exhausted or ''}")
    print(f"{'log':>10}: {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
