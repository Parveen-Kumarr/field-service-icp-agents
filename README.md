# Field-Service ICP Agent Team

A team of six AI agents that qualifies the companies at **Field Service Next West**, **live**, against the ideal customer profile (ICP) of a vendor of AI agents for field service. The agents do four things:

- scrape the conference site at run time
- research every company on the web
- score each one against the ICP
- plan the first conversation for strong fits

While they work, they ask each other questions and wait for the answers. You watch the conversation in your terminal, alongside a heartbeat that shows exactly what each agent is waiting on.

```
                 ┌──────────────── Coordinator ────────────────┐
                 │  plans, tracks every account, debriefs      │
                 ▼                                             │
  Scout ──handoff──► Researcher ──handoff──► Analyst ──handoff──► Strategist ──► Coordinator ──► Reporter
  (live scrape)      (web research)  ◄─ask/answer─┘  ◄─ask/challenge─┘
                         ▲─────────────── ask/answer ──────────────┘
```

The agents can think with one of three options:

- the **model pool** (default): several **free** cloud models used together (Groq, Cerebras, Mistral, OpenRouter, Gemini, plus your local Ollama as a backup). Each is kept inside its own free-tier limits and work goes to whichever has capacity soonest. With 4-5 free keys, a **full conference run takes about 5 minutes**.
- a **local model on your own GPU** via [Ollama](https://ollama.com) (Gemma 4, no API key, no quota, slower), or
- **Google Gemini** alone (API key; the free tier allows only about 20 calls a day, too few for a run).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                # Windows: copy .env.example .env
```

**Model pool (default, `ICP_LLM_PROVIDER=pool`):** create free keys and put them in `.env`. None needs a card, and each takes about a minute:

| Provider | `.env` key | Sign up | Free limits used by default |
|---|---|---|---|
| Groq | `GROQ_API_KEY` | https://console.groq.com/keys | 30 req/min, 1,000/day, 8K tokens/min |
| Cerebras | `CEREBRAS_API_KEY` | https://cloud.cerebras.ai | 5 req/min (can't exceed), 30K tokens/min, 1M tokens/day |
| Mistral | `MISTRAL_API_KEY` | https://console.mistral.ai/api-keys | free "Experiment" plan (opts you in to data training) |
| OpenRouter | `OPENROUTER_API_KEY` | https://openrouter.ai/keys | 20 req/min, 50/day on `:free` models; if the default free model is withdrawn, the pool picks a current free one with tool calling |
| Gemini | `GEMINI_API_KEY` | https://aistudio.google.com/apikey | ~20/day; the pool's only cloud model that reads logo images |

**One key can serve several models.** Groq and Cerebras count free limits per model, not per key. At start-up the pool asks them which models your key can use and adds up to 3 more tool-capable ones (e.g. Llama 3.3 70B, Llama 4, Qwen 3), each with its own limits. It then reads each model's real limits from the provider's replies and pauses a model just before it runs out. You can list models yourself with `ICP_GROQ_MODELS=...`, or turn discovery off with `ICP_GROQ_DISCOVER=false`.

**Speed depends on how many keys you add.** Their limits add up: one key is enough for a `--max-accounts 10` demo, while 3-5 keys run the full conference in about 5 minutes. (GitHub Models was retired by GitHub on 30 July 2026, so it is no longer an option.) If Ollama is running, it joins as overflow capacity and to read logos. `python run.py --check` pings every provider and tells you which ones answer.

**Local model (`ICP_LLM_PROVIDER=ollama`):**

1. Install Ollama from https://ollama.com/download and start it. On Windows it runs in the system tray.
2. Download the model once: `ollama pull gemma4:12b` (7.6 GB). It fits a 12 GB GPU.
3. Install the browser used to render JavaScript-built pages (the speakers list is one), once: `python -m playwright install chromium`
4. That's it. No key is needed. Research runs from your machine through the built-in `web_search` (several search engines via `ddgs`) and `fetch_page` tools.

Also install the browser once for every option: `python -m playwright install chromium`.

**Gemini alone:** set `ICP_LLM_PROVIDER=gemini` and `GEMINI_API_KEY` in `.env`, or pass `--provider gemini` for one run.

Optionally set `ICP_VENDOR_NAME` and `ICP_VENDOR_URL` in `.env`. The Researcher will then read that vendor's site live and brief the team. Without them, the team uses a built-in profile of a field-service AI vendor.

## Run

```bash
python run.py --check            # quick check: model reachable and loaded, conference site reachable
python run.py --max-accounts 10  # quick demo on 10 accounts
python run.py                    # full live run: every company found on the site
python run.py --log-level DEBUG  # also show every tool call and message detail on the console
python run.py --provider gemini  # use Gemini for this run
python run.py --provider ollama  # use only the local model for this run
python run.py --no-fast          # full research mode (several searches per account, longer reasoning)
python -m pytest -q              # tests: no API key or network needed
```

Every run starts with the same preflight check and stops straight away, with the reason, if the model can't be used. For example: Ollama isn't running, the model isn't downloaded, a bad Gemini key, or an exhausted daily quota.

## How long a run takes

### Model pool + fast mode (default)

Fast mode (`ICP_FAST`, on by default with the pool) spends model calls only where they matter:

1. **Triage.** The Analyst screens every account in batches of 25, using one call per batch. Competitors, software vendors, consultants and the vendor itself are settled there and skip research entirely (usually a third or more of a conference).
2. **One search per account.** The Researcher runs one web search in code and makes one compact model call over the top results.
3. **Shared model calls.** When several accounts are in flight, the Researcher researches about 4 at once in one model call, the Analyst scores 4 at once, and the Strategist drafts plays for 3 at once. Every account still gets its own message to the next agent. Free tiers cap requests per minute (Cerebras allows 5 per model), so this does about 3 times more work per request. A simulated full run on the real 2026 speaker list needs about 105 calls instead of 348. `ICP_BATCH=1` turns this off.
4. **Compact scoring and plays.** The Analyst asks follow-up questions only when the evidence is really thin, and the Strategist does the same.

A full site (~70 accounts) needs about 130 model calls. The start-of-run message estimates the time from the pool's live capacity:

| Keys set | Combined capacity | Full run (~130 calls) |
|---|---|---|
| Groq only | ~5 calls/min (its 8K tokens/min is the bottleneck) | not enough: its 200K tokens/day covers ~125 calls. Fine for a 10-account demo |
| Groq + Cerebras + Mistral | ~40 calls/min | ~4-6 min |
| + OpenRouter + Gemini + Ollama running | ~50-60 calls/min | **~3-5 min** |

These are planning figures from the published free limits. Real runs add network and search time (up to `ICP_SEARCH_PARALLEL` web searches run at once, their starts spaced `ICP_SEARCH_INTERVAL` seconds apart). Scraping the site takes ~30 s of that.

### Local model (full research mode)

Measured on the first real local run (RTX 4080 Laptop 12 GB, `gemma4:12b`, 2 accounts in parallel), before the fixes in this version:

| | Time |
|---|---|
| Preflight + scraping the site | ~35 seconds |
| 10 accounts, research to verdicts | 51 minutes (~5 minutes per account) |
| Model calls | 109 calls, ~96 minutes of GPU time, Researcher ~49 s and Analyst ~81 s per call |

Most of that time was waste the fixes remove:
- Blocked searches made the Researcher re-search repeatedly (86 calls for 10 accounts).
- Answers ran on for thousands of tokens (one call wrote 6,000 tokens in 150 s).

Expect roughly 2.5-4 minutes per account now, which is still slower than Gemini. A full run of the site (about 60-75 accounts once the speakers are read) is a few hours locally. Start with `--max-accounts 10`, and read the elapsed time and "accounts done" in the heartbeat.

## Choosing a model

| | Local (Ollama, Gemma 4) | Gemini |
|---|---|---|
| Cost / limits | Free, no quota | Free tier ~20 calls/day; paid tier for real runs |
| Web research | `web_search` + `fetch_page` on your machine (hard-capped per task) | Google Search + URL context on Google's servers |
| Site blocks your machine | Run stops (the local model can only read what you can reach) | Gemini reads the site instead |
| Speed | 2 accounts at a time; each step takes seconds to tens of seconds on a laptop GPU | Faster, 4+ accounts at a time |
| Quality | Good; a bit less thorough research | Strongest |

Other local models work too. Set `ICP_LLM_MODEL`:
- `gemma4:26b` or `gemma4:31b` (about 19-20 GB, for a 24 GB GPU)
- `qwen3:14b` (text only, so add `--no-logos`)

With thinking on (`ICP_OLLAMA_THINK=true`), judgement is better but runs are slower.

## Watching a run

The console interleaves three things:

- **Agent messages**, as they're sent. Answers show which question they reply to (`re #38`).
- **Log lines**: every page fetch with its HTTP status and timing, every model call with what it was for and how long it took, every local web search and page read, and every retry with its reason.
- **A heartbeat** every 15 seconds (`ICP_HEARTBEAT`). It shows elapsed time, accounts done, every model call in flight, and what each agent is doing. Anything waiting over 2 minutes is marked `SLOW`.

```
18:10:05 INFO  icp.heartbeat   pool: groq 41 calls/0 429s (3 busy) | cerebras 12 calls/1 429s (cooling 12s) | mistral 30 calls/0 429s (2 busy)
18:10:05 INFO  icp.heartbeat 4s elapsed | 16 messages | accounts 0/4 done | model: 2 in flight, 1 queued, 0 failed
18:10:05 INFO  icp.heartbeat   model <- Researcher: research Ciena (turn 2) - 14s
18:10:05 INFO  icp.heartbeat   Researcher: queued for a research slot: SAM Service Inc - 1s
18:10:06 INFO  icp.web    search 'Ciena field service' -> 6 results in 0.8s
18:10:19 INFO  icp.ollama [Researcher] gemma4:12b ok: research Ciena (turn 2) | 13.2s | 3,410 in / 96 out tokens | calls fetch_page
```

With the pool, the heartbeat also shows each provider's load against its limits. A provider that answers "429 rate limited" cools down and the call moves to another provider at once. A provider whose daily quota runs out is retired for the day, and a rejected key disables that provider.

A model call that doesn't answer within `ICP_LLM_TIMEOUT` (45s per cloud model in the pool, 180s for Ollama in the pool, 300s local, 180s Gemini) is retried up to 4 times, and the log says why each time. A run never waits silently.

Every run writes a full DEBUG log to `output/logs/run_<time>.log`. It records every model call, tool call, web search, retry, page fetch, agent message and heartbeat. If a run misbehaves, that file tells you where.

## The agents

| Agent | Does | Talks to |
|---|---|---|
| **Coordinator** | Kicks off the briefing and scrape in parallel, tracks each account, posts progress, handles errors, writes the debrief | everyone |
| **Scout** | Finds the speakers, sponsors and attendee pages from the site's own navigation; scrapes them live; reads logos with Gemini vision; if the site blocks this machine, Gemini reads it live (URL context) instead | hands accounts to Researcher |
| **Researcher** | Briefs the team on the vendor; researches each company with Google Search and URL context and cites sources; answers colleagues' questions | Analyst, Strategist |
| **Analyst** | Scores each account on 5 ICP dimensions with reasons; asks the Researcher when evidence is thin; defends or revises verdicts when challenged | Researcher, Strategist |
| **Strategist** | For A/B accounts: asks the Researcher for a current hook, can question or challenge the Analyst, picks which capabilities to lead with, drafts openers (never sends) | Researcher, Analyst |
| **Reporter** | Writes the workbook, transcript and JSON | Coordinator |

## Outputs (`output/`)

- `FSN_West_ICP_<time>.xlsx`, with these sheets:
  - **Summary**: run facts and Coordinator debrief
  - **Accounts (ICP)**: scores per dimension with reasoning
  - **Contacts & Openers**
  - **Evidence**: claims with source URLs
  - **Agent Conversation**: every message in full
  - **Pages Scraped (live)**: HTTP status, fetch time, SHA-256
  - **Usage**
  - **Model Pool** (pool runs): calls, tokens, rate limits hit and errors per provider
  - **ICP Rubric**
- `agent_conversation_<time>.md`: the readable transcript
- `icp_results_<time>.json`: everything, for a CRM or another agent
- `live_capture_<time>.json`: what the Scout scraped, with proof of when and how
- `logs/run_<time>.log`: the full DEBUG log of the run
- `FSN_West_ICP_partial.xlsx` + `agent_conversation_partial.md`: rewritten as accounts finish (at most every 10 s), so results survive a crash or Ctrl+C
- `pages/`: every conference page exactly as the site sent it (and as rendered by the browser), for checking the scraping

## Settings (`.env`)

| Setting | Default | Meaning |
|---|---|---|
| `ICP_LLM_PROVIDER` | `pool` | `pool` (free cloud models together), `ollama` (local) or `gemini` |
| `GROQ_API_KEY`, `CEREBRAS_API_KEY`, `MISTRAL_API_KEY`, `OPENROUTER_API_KEY`, `GEMINI_API_KEY` | blank | Pool providers; only those with a key are used |
| `ICP_POOL` | `groq,cerebras,mistral,openrouter,gemini,ollama` | Pool members in order of preference |
| `ICP_FAST` | on for pool | Triage + one search + compact answers per account; `false` = full research |
| `ICP_BATCH` | `4` | Accounts per shared model call in fast mode (1 = off) |
| `ICP_TALKS` | `8` | Fast mode: top accounts that get a full one-to-one Strategist conversation |
| `ICP_<NAME>_MODEL` / `_RPM` / `_RPD` / `_TPM` / `_TPD` / `_CONCURRENCY` / `_BASE_URL` | published free limits | Per-provider overrides, e.g. `ICP_GROQ_MODEL`, or raise limits on a paid tier |
| `ICP_LLM_MODEL` | provider default | `gemma4:12b` (ollama) / `gemini-3.8-flash` (gemini) |
| `ICP_OLLAMA_URL` | `http://localhost:11434` | Where Ollama listens |
| `ICP_OLLAMA_NUM_CTX` | `16384` | Local context window; lower it if the GPU runs out of memory |
| `ICP_OLLAMA_THINK` | `false` | Local thinking mode: better judgement, slower |
| `ICP_SEARCH_PROVIDER` | `ddgs` | Local web search: `ddgs` (multi-engine, no key), `duckduckgo`, or `searxng` (+ `ICP_SEARXNG_URL`) |
| `ICP_SEARCH_INTERVAL` | `1` | Seconds between search starts |
| `ICP_SEARCH_PARALLEL` | `3` | Web searches at once (shared by all agents) |
| `ICP_OLLAMA_MAX_OUTPUT` | `2048` | Most tokens one local reply may generate |
| `ICP_PAGE_CHARS` | `8000` | Characters of each web page given to the local model |
| `ICP_EVENT_URL` | Field Service Next West | Conference site |
| `ICP_VENDOR_NAME` / `ICP_VENDOR_URL` | blank | The vendor whose ICP you qualify against; blank = built-in generic profile |
| `ICP_SCRAPE_MODE` | `live` | `live` (this machine, then Gemini URL context) or `local` (this machine only) |
| `ICP_MAX_ACCOUNTS` | `0` (all) | Cap for demos |
| `ICP_CONCURRENCY` | pool 12 / ollama 2 / gemini 4 | Accounts researched in parallel |
| `ICP_SEARCHES_PER_ACCOUNT` | `4` | Web searches per account (hard cap locally; a guideline for Gemini) |
| `ICP_LLM_TIMEOUT` | pool 45 / ollama 300 / gemini 180 | Seconds before a model call counts as stuck and is retried |
| `ICP_HEARTBEAT` | `15` | Seconds between status reports (0 = off) |
| `ICP_LOG_LEVEL` | `INFO` | Console detail; the log file always has DEBUG |
| `ICP_RENDER_JS` | `auto` | Use Playwright for script-built pages if installed |

Cost guide: each account uses about 3-6 Gemini calls plus Google Search grounding, billed at Google's current rates. The run prints tokens, thinking tokens and search counts, and the workbook's Usage tab breaks them down by agent. Start with `--max-accounts 10`.

## If a run is slow or stops

- **"daily quota used up (HTTP 429)":** a model's free daily allowance is spent. Groq allows about 200K tokens a day per model, and a full run uses about 80K per Groq model, so 2-3 full runs a day use it up. The pool carries on with the other models, more slowly. Groq's allowance resets over the next 24 hours.
- **"Only 1 cloud provider key(s) set":** the run works, just slower. Add more free keys (see Setup).
- **"no model in the pool is usable":** every key failed its check. The lines above it say why for each provider (bad key, no network, model not available on your plan). Set a different model with `ICP_<NAME>_MODEL` if one isn't offered to you.
- **The heartbeat shows "waiting Ns for capacity":** every provider is at its per-minute limit, and the pool waits for the next free slot. Add keys or accept the wait. If your account has higher limits, raise `ICP_<NAME>_RPM` / `_TPM`.
- **"can't reach Ollama":** start the Ollama app (or run `ollama serve`).
- **"model is not downloaded":** run the `ollama pull ...` command it prints.
- **The heartbeat shows local calls taking minutes:** lower `ICP_OLLAMA_NUM_CTX` to 8192, keep `ICP_CONCURRENCY` at 1-2, and close other GPU-heavy apps. The first call also waits while the model loads.
- **"search refused ... pausing web search":** the search engines are rate-limiting you. The agents then stop searching and answer from what they have (confidence marked low). Raise `ICP_SEARCH_INTERVAL` (e.g. 3) or set `ICP_SEARCH_PARALLEL=1`, or run SearXNG and set `ICP_SEARCH_PROVIDER=searxng`.
- **"The speakers page has no names yet":** the site has rolled over to its next edition before announcing speakers (in October 2026 this site became "Service Next West 2027"). The Scout follows the page's own "who spoke previously" link, reads that list instead, and marks those accounts `Speaker (2026 edition)`.
- **"0 speakers" / "the list is probably built by JavaScript":** install the browser with `python -m playwright install chromium`. The raw page is saved in `output/pages/speakers.html`.
- **"the model left out ... filled with neutral defaults" or "answer was cut off":** the local model's answer was incomplete. It is repaired automatically; if it happens often, lower `ICP_OLLAMA_NUM_CTX` or use a larger model.
- **Gemini "daily quota exhausted":** the run stops immediately rather than retrying. Wait for the reset, enable billing, or switch to `ICP_LLM_PROVIDER=ollama`.
- **Gemini "HTTP 429" (per minute):** lower `--concurrency`; these are retried automatically.

Docs: `docs/DESIGN.md` (plan, architecture, decisions) and `docs/COMMUNITY_AND_VIDEO.md` (deliverables 2 and 3).
