# Field-Service ICP Agent Team: Design Document

The goal: collect the Field Service Next West company list from the conference website, have agents validate which companies fit the ICP of a vendor of AI agents for field service, make the agents talk to each other, and take it as far as it usefully goes. The vendor is configurable (`ICP_VENDOR_NAME` / `ICP_VENDOR_URL`); no company is named in the code.

---

## 1. Plan

| Phase | Output |
|---|---|
| Understand the vendor | What it sells, to whom, why buyers pay (section 2). Re-read live from `ICP_VENDOR_URL` when set |
| Understand the source | Site structure: public speakers page, sponsor wall, logo strip, gated attendee list |
| Define | ICP rubric, six agent roles, message protocol |
| Design | Async multi-agent runtime with real request/reply; Gemini function calling per agent |
| Develop | Runtime, tool loop, live scraper, six agents, reporter, live console |
| Observe | Console + file logging, heartbeat of in-flight work, per-call timeouts and retries, preflight check (section 6) |
| Models | Three interchangeable options: a pool of free cloud models (default), local Gemma 4 via Ollama, or Gemini (section 5) |
| Speed | Fast mode + model pool: a full conference in ~5 minutes on free tiers (section 5.1) |
| Verify | 68 tests, incl. a full team run on each model option and a 65-account run on the pool |

### What changed from v1, and why

v1 read a saved copy of the pages and a hand-written company list, and its agents exchanged one-word status events. It finished in under a second, which showed it was a lookup, not an agent system. v2 fixes this at the root:

| v1 | v2 |
|---|---|
| Snapshot of the site used by default | Live scrape every run; no saved-data mode; stops if no live data |
| Hand-written company facts | Researcher researches each company live with web search/fetch and cites sources |
| Vendor profile hard-coded | Researcher reads the vendor's site live at start-up and briefs the team |
| Deterministic scoring rules | Analyst (Gemini) scores each dimension with a reason; code only sums and tiers |
| Messages like `lead.enriched` | Messages are full sentences plus data; agents ask, wait, answer and challenge |
| Sequential, synchronous | Agents run concurrently with their own inboxes |

## 2. The vendor (baseline understanding)

- **Product:** AI agents for field service that capture how the best technicians diagnose and give that knowledge to every technician: guided troubleshooting, expert-knowledge capture, dispatch, escalation prediction, parts forecasting, log analysis, PII redaction, and more.
- **Deployment:** runs on existing ServiceNow, SAP FSM, Salesforce/ServiceMax or a CMMS without data cleanup.
- **Value:** fewer expensive expert escalations, higher first-time fix, parts readiness, and keeping knowledge when senior technicians retire.
- **Verticals:** medical devices, telecom, energy, lab instruments, industrial, data centers, robotics, water, food equipment, defense, mobility.
- **Competitors:** other vendors selling AI software for service teams (the Researcher identifies them per account).

**ICP:** an OEM or operator servicing *complex physical assets* in the field, *at scale*, where *downtime is expensive*, reached through a *service leader*.

This generic profile lives in `vendor_profile.py`. When `ICP_VENDOR_URL` is set, the Researcher reads that site live on every run and all agents reason from its briefing instead, including the vendor's own product names.

## 3. Define

### 3.1 Rubric (100 points, scored by the Analyst with a reason per dimension)

| Dimension | Max | What earns points |
|---|---|---|
| Industry | 30 | Vertical central to the vendor |
| Assets | 20 | Complex equipment; costly wrong or repeat fixes; telemetry/logs; regulated devices |
| Scale | 15 | Installed base and field service org size |
| Persona | 20 | Named contact owns service/support/field ops/service AI, and is senior |
| Signals | 15 | Service AI program, workforce pressure, FSM stack, several attendees, regulation |

Tiers are computed in code from the summed points: **A** ≥ 75, **B** 55-74, **C** 35-54, otherwise Not ICP.

Disqualifiers are competitor, self (the vendor itself), consultant, and vendor-only. A vendor that runs a big service org itself, such as Dell, stays a prospect at ≥ 70.

### 3.2 Message protocol

Every message carries these fields:

- `sender`, `recipient` (an agent or the whole team)
- `performative`: inform, handoff, request, reply, or challenge
- `subject`
- `text`: a full sentence written for the colleague
- `data`: structured payload
- `in_reply_to`
- `thread`: the account id, which ties every message about one account together

`ask()` sends a request and suspends the asking agent until the reply arrives (or a timeout passes). The other agent keeps working meanwhile.

## 4. Architecture

```
run.py ─► Runtime (asyncio) ── inbox per agent, router, pending-reply futures, transcript, live console
             │
             ├─ Coordinator ── kickoff ─► ask Researcher: brief_team_on_vendor    (parallel)
             │                          ─► ask Scout: scrape_event
             ├─ Scout ──── httpx (live) ─► discover pages from nav ─► parse speakers / logo walls
             │              │  layout not recognised ─► Gemini reads live page text
             │              │  logos without text   ─► Gemini vision (image downloaded live)
             │              │  site blocks this machine ─► Gemini URL context (live, Google servers)
             │              └─ handoff per account ─► Researcher        (streams as found)
             ├─ Researcher ── Gemini + Google Search + URL context ─► submit_research (facts + source URLs)
             │              └─ handoff ─► Analyst ;  answers questions (separate gate, never queued behind research)
             ├─ Analyst ──── Gemini + ask_researcher ─► submit_verdict ─► code sums + tiers
             │              ├─ A/B ─► handoff to Strategist ;  C/Not ─► Coordinator
             │              └─ answers / revises on challenge ─► tells Coordinator
             ├─ Strategist ─ Gemini + ask_researcher + ask_analyst + challenge_analyst ─► submit_play
             ├─ Coordinator ─ tracks all accounts ─► Gemini debrief ─► ask Reporter
             ├─ Reporter ─── xlsx + transcript.md + json
             └─ Heartbeat ── every 15s: in-flight Gemini calls + each agent's current activity
```

### Key decisions

| Decision | Why |
|---|---|
| Asyncio actors with inboxes | Agents genuinely work at the same time: research continues while the Analyst waits on an answer |
| `ask()` = request + future | A question really blocks the asker's reasoning until the answer arrives; the answer is fed back into its Gemini loop as a function result |
| Each agent's structured output is a `submit_*` function | Reliable JSON output while still letting Gemini search the web and ask colleagues first; when needed the loop forces it with `tool_choice: any` |
| Stateful Gemini Interactions | Follow-ups send only new function results plus `previous_interaction_id`; Google keeps the history and thought signatures, so nothing is re-serialised by hand |
| Code computes totals and tiers | The model judges, code does the arithmetic, so there are no inconsistent totals |
| Live-only data, with proof | Every page is recorded with HTTP status, time and SHA-256; no data means no run |
| Two live channels | Corporate networks and bot filters block scrapers; Gemini URL context reads the same live page from Google's servers |
| Discovery from the site's own nav | Survives URL changes year to year; the gated attendee list is reported, never faked |
| Separate gate for questions | Prevents deadlock where the Analyst waits on a Researcher that is busy with bulk research |
| Drafts only | A human reviews and sends every opener |
| Nothing waits silently | Every Gemini call has a timeout, logged retries and a plain-English error; `in_progress` interactions are polled; a Coordinator crash ends the run with its reason |

## 5. Model options

The agents never talk to a model API directly. Each calls `llm.run(...)` with:
- a task
- its own tools (e.g. `ask_researcher`)
- "server tools" (web search, page reading)
- a `submit_*` schema

Three clients implement that interface, and `ICP_LLM_PROVIDER` picks one: `pool` (default, section 5.1), `ollama` or `gemini`.

| | `ollama` - `local_llm.py` | `gemini` - `llm.py` |
|---|---|---|
| Model | Gemma 4 (`gemma4:12b`, fits a 12 GB GPU): native tool calling + vision | `gemini-3.8-flash` |
| Web search / page reading | `websearch.py` tools run on this machine: DuckDuckGo or SearXNG search, page fetch with text extraction; hard-capped per task | Google Search + URL context on Google's servers |
| Forcing the structured answer | No `tool_choice` in Ollama, so the final request drops tools and constrains the reply to the submit schema (`format`) | `tool_choice: any` on the submit function |
| Conversation state | Stateless: the full conversation is sent each turn | Stateful: `previous_interaction_id` |
| Logo reading | Images in the message's `images` field (PNG/JPEG) | Image parts |
| Preflight | Ollama running? model pulled? then a one-word call (loads it onto the GPU) | One-word call; bad key / model / quota reported in plain English |
| Quota | None | Free tier ~20 calls/day; an exhausted daily quota stops the run at once, no retries |

Why Gemma 4 locally: it is the local model family with native function calling *and* vision *and* a thinking mode. All three are used here: agents ask each other questions through tools, the Scout reads logos, and scoring benefits from reasoning. The 12B size fits a 12 GB laptop GPU entirely in VRAM.

### 5.1 The model pool and fast mode (`pool.py`)

**Goal:** qualify a whole conference (~60-75 accounts) in about 5 minutes without paying. No single free tier allows that: Groq allows 8K tokens a minute, Cerebras 5 requests a minute, and Gemini about 20 requests a day. Their limits add up, though, so the pool uses them together.

**How it works:**

- **One interface.** Most free providers offer an OpenAI-compatible `/chat/completions` endpoint, so one `Backend` class drives Groq, Cerebras, Mistral, OpenRouter, Gemini (its OpenAI endpoint) and Ollama. (GitHub Models was supported until GitHub retired it on 30 July 2026; it is now skipped with a message.) OpenRouter's free line-up changes often, so when its default free model is withdrawn the preflight lists `/models` and switches to a current `:free` model that supports tool calling. Only providers with a key in `.env` join; Ollama needs none.
- **Limits known in advance.** Each backend carries its published free-tier limits (requests/tokens per minute and per day, concurrency, max input). A local limiter tracks a sliding one-minute window plus daily counters, so the pool avoids a 429 rather than causing one. Every limit can be overridden per provider in `.env`.
- **Several models per key.** Free tiers limit each *model* separately. For Groq and Cerebras the preflight lists the key's models (`GET /models`) and adds up to 3 tool-capable ones as extra members. Each is verified with a real tool call, and each has its own limiter. `ICP_<NAME>_MODELS` adds models explicitly.
- **Limits learned live.** Groq and Cerebras report each model's daily request and per-minute token limits, and what's left, in reply headers. The pool adopts those limits (unless `.env` overrides them). When fewer than ~1,500 tokens are left this minute, it pauses that model until the reset instead of collecting a 429.
- **Routing.** Each call goes to the backend that can take it soonest, then by preference order, then by load. Calls that must wait sleep until the expected free slot or until any call finishes, whichever comes first.
- **Failure handling.** Each kind of failure has its own response:
  - a 429 with Retry-After cools that backend down, and the call moves on at once;
  - a daily-quota 429 retires the backend for the day;
  - a 401/403 disables it;
  - a 5xx or timeout makes the call avoid that backend and try another;
  - a 400 that names an optional setting (e.g. `reasoning_effort`) drops that setting and retries;
  - repeated 429s back off exponentially (up to 5 minutes);
  - an answer missing more than half its required fields, or cut off, is redone on a different model instead of being filled with defaults.
- **Provider quirks handled.** Groq rejects tool calls that miss a `required` field, so it gets the schema without `required` (fields are filled afterwards). Gemini 3 can only continue tool conversations it started, so it only takes first turns. Cloud calls time out after 45 s, local ones after 180 s.
- **Portable conversations.** A multi-turn task can move between providers mid-conversation, because tool-call ids are rewritten to a neutral 9-character form every provider accepts. Images go only to vision-capable backends (Gemini, Ollama).
- **Web tools.** Research uses the local `web_search` / `fetch_page` tools (the same as the Ollama option), so any model can do it.
- **Forced structured answer.** The final turn offers only the `submit_*` function with `tool_choice: "required"`. A model that answers in JSON text instead is accepted, and missing fields are filled with neutral defaults.

**Fast mode (`ICP_FAST`, on by default with the pool)** cuts the calls per account, which matters more than raw model speed:

| Step | Full mode | Fast mode |
|---|---|---|
| Screening | none | Analyst triages all accounts in batches of 25 (1 call per batch). Competitors, vendors, consultants and self skip research |
| Research | model-driven loop, up to 4 searches, 3-8 calls | one search run in code + one compact call |
| Scoring | may ask the Researcher several questions | asks only if evidence is truly thin |
| Play | may question and challenge | questions only when nothing specific is known |
| Calls for ~70 accounts | ~350 | ~130 |

**Micro-batching (`batching.py`).** When several accounts are in flight, the Researcher, Analyst and Strategist collect the tasks that arrive within 0.6 s (up to `ICP_BATCH`, default 4) and answer them in one model call ("research these 4 companies", each with its own search results). Each account still gets its own message to the next agent. If the model leaves an item out, or the batch fails, that item falls back to a normal single call. The top accounts (A-tier, 85 or more) keep a one-to-one Strategist conversation with a hook question and possible challenge. A simulated run of the real 119-account list needs about 105 calls instead of 348.

Agents still talk to each other in fast mode: the Scout asks the Analyst to triage and waits for the answer; the Analyst still asks the Researcher when evidence is thin; challenges still happen. They only skip conversations with nothing to add.

At the start of a run the Coordinator estimates the model work from the account count and the pool's live capacity, and posts it (or warns when the remaining daily quota looks too small). The heartbeat adds a per-provider line (calls, 429s, busy or cooling), and the workbook gets a Model Pool sheet.

**Capacity with published free limits** (planning figures, checked by a test):

| Keys | Calls/min | ~130 calls |
|---|---|---|
| Groq alone | ~5 | not enough: 200K tokens/day ≈ 125 calls |
| Groq + Cerebras + Mistral | ~40 | ~3-4 min + network/search time |
| + OpenRouter + Gemini + Ollama | ~50-60 | ~2-3 min + network/search time (a real `--check` measured 52 calls/min) |

## 6. Development

```
icp_agents/
  runtime.py         Runtime: routing, ask/reply futures, transcript, shutdown
  messages.py        AgentMessage + performatives
  llm.py             Gemini Interactions loop: Google Search/URL context, function tools, forced submit, citations, usage, retries
  local_llm.py       Ollama loop (Gemma 4): same interface; local tools; structured-output submit; images; retries
  websearch.py       Local web_search (DuckDuckGo / SearXNG) and fetch_page tools with per-task caps
  batching.py        Micro-batching: several accounts per model call, per-account messages unchanged
  pool.py            Model pool: free OpenAI-compatible providers, per-provider limits, routing, failover, ETA
  schema_tools.py    Fills missing required fields and repairs cut-off JSON from any model
  providers.py       Builds the model client from ICP_LLM_PROVIDER
  scraper.py         Live fetch, nav discovery, speaker parser, logo sections, JS rendering (optional)
  vendor_profile.py  Generic vendor profile, capability list, rubric, tiers
  config.py          Settings from .env
  log.py             Console + file logging
  monitor.py         Heartbeat: status of in-flight calls and agent activity
  ui.py              Live console of the conversation
  agents/            base, coordinator, scout, researcher, analyst, strategist, reporter
run.py               CLI
tests/               fake_gemini.py (simulated Gemini Interactions API), fixtures/ (HTML shaped like the real site),
                     test_agents.py, test_sdk_contract.py (real google-genai SDK against mock HTTP),
                     test_observability.py (timeouts, retries, polling, heartbeat, log file, preflight),
                     test_local.py + fake_ollama.py (Ollama client, local web tools, full team run on a local model),
                     test_regressions.py (failures from the first real run), test_pool.py + fake_openai.py (simulated
                     OpenAI-compatible providers with latency, 429s, daily quotas, 5xx, bad keys)
```

## 7. Observability

A long run must never look frozen. Four mechanisms make it visible:

- **Logging (`log.py`).** The console shows INFO; `output/logs/run_<time>.log` always keeps DEBUG. The log records:
  - every page fetch, with HTTP status and time
  - every Gemini call, with its purpose, duration, tokens and searches
  - every retry, with the reason
  - every tool call and every agent message
- **Heartbeat (`monitor.py`).** Every `ICP_HEARTBEAT` seconds it reports:
  - elapsed time, messages, and accounts done
  - each Gemini call in flight: which agent, for what purpose, how long so far
  - what each agent is doing: researching, queued for a slot, waiting on a colleague's answer
  - `SLOW` against anything waiting over 2 minutes
- **Timeouts and retries (`llm.py`).** Each call gets `ICP_LLM_TIMEOUT` seconds.
  - Timeouts, rate limits (429), server errors and network errors are retried with backoff, up to 4 times.
  - Errors are translated into what to do, e.g. "model not found - set ICP_LLM_MODEL".
  - `in_progress` / `queued` interactions are polled until they finish.
- **Preflight (`run.py`).** A one-word Gemini call plus a fetch of the site runs before the team starts, so a bad key, wrong model or exhausted quota stops the run in seconds, not minutes. `python run.py --check` runs only this.

## 8. Verification

- `python -m pytest -q`: 68 passed in about 8 seconds. The tests cover:
  - the speaker parser and nav discovery (the "Sponsorship Opportunities" and 2024 archive links are excluded)
  - logo section detection
  - a full team run: briefing, question/answer pairs linked by id, a challenge that revises a verdict, disqualifiers, and workbook contents
  - force-live stop when the site is unreachable
  - Gemini URL-context fallback
  - the real `google-genai` SDK round-trip: Google Search / URL-context steps and `url_citation` sources, stateful follow-ups with `previous_interaction_id`, forcing the submit function, image input, and failed interactions
  - the local option: DuckDuckGo/SearXNG parsing, capped search/fetch tools, Ollama preflight hints (not running, model not pulled), images, and a full team run on a simulated Ollama server where the Researcher really uses the local tools
  - Gemini's exhausted daily quota is reported without retries
  - regressions from the first real local run: the end-of-run hang (report now always written, plus a partial report after each account), cut-off JSON repaired and missing fields filled, refused searches pausing search instead of looping, and a JavaScript-built speakers page rendered in a headless browser
  - the model pool: only keyed providers join, and `.env` overrides apply; RPM/TPM/concurrency/daily limits; a 429 cools a provider and work moves on; a daily quota retires it; 5xx failover; a bad key disables it; an unsupported setting is dropped; a clear error when nothing is available; a conversation moving between providers with portable tool-call ids; images only to vision models; the free-tier capacity plan (70 accounts ≤ 5 min with five keys, not with one); and a fast-mode run of a 65-account site across three providers (non-buyers triaged out, load spread, ~2-3 calls per researched account)
  - a hung call timing out and being retried, `in_progress` polling, rate-limit retries ending in a readable error, the heartbeat showing in-flight work, a Coordinator crash stopping the run, the generic profile when no vendor URL is set, and the log file recording calls, fetches and messages
- Bugs the tests caught and fixed:
  - Logos were dropped because the event hosts images on its own domain.
  - The vendor's own "self" tier was routed as an A-tier account.
  - A console error could stall message delivery.
  - A package variable hid the logging module, which would have crashed `run.py` at start-up.
  - httpx's `ConnectError` wasn't recognised as a retryable network error.
  - The first run on the real site (1 Oct 2026) found 0 speakers. The event had rolled over to "Service Next West 2027": `/speakers` had no names yet, only a "who spoke previously" link to `/speakers/2026`, which the Scout skipped as an archive. Other problems in the same run: navigation moved to a sibling subdomain, and the headless browser got a 403 page, which was then parsed as if it were content. Home-page logos had `alt="img"` and a trailing space in `src`, so they were named from their files ("Logos 0024 Ge"). The organiser's own logo was listed as a company, and a 503 from Gemini during the check removed it for the whole run. The fixes: the Scout follows the previous-edition link and labels those speakers by year; subdomains of the same site count as one site; blocked pages are recognised and ignored; logo file names are cleaned ("GE", "Honeywell"); organiser logos are dropped; busy providers stay in the pool with a cool-down. `test_real_site.py` replays the real saved pages.
  - Second real run (1 Oct 2026, 160 s for 10 accounts): with the cap, sponsors that also speak (mostly vendors) went first, and Agilent got 0/100 because a local answer was cut off and accepted. Groq rejected answers with a missing field, Gemini rejected a conversation another model started, a free OpenRouter model hung for 90 s, and a question to the Researcher got "I don't have that information". The fixes: named speakers at non-sponsor companies go first, ranked by service and seniority of title; half-empty or cut-off answers are redone elsewhere; the Groq, Gemini and timeout fixes above; fast-mode answers run one search on the question first.
  - Third real run (1 Oct 2026, 10 accounts in 229 s): 7 A-tier, 1 C, 1 error, and Diebold Nixdorf wrongly at 0/100. Web search turned out to be the bottleneck: one lock was held through each search, the spacing and even the 20 s cool-down, and "No results found" counted as a refusal, so about 130 searches in a full run would have waited 12+ minutes. The error was a verdict whose scores came back as a string. The fixes: up to 3 searches at once with only their start times spaced; "no results" returns an empty list; question searches use a short keyword query; scores are accepted in any shape (object, number, "18/20", JSON string); answers get more room and never come back empty; up to 3 redos on other models, then "Needs review" rather than a false 0; logos with readable file names skip image reading.
  - First full real run (1 Oct 2026): 119 accounts, 36 A / 28 B / 16 C, but 24 minutes, then no report. Triage crashed on one item missing its `decision`, so all 119 accounts were researched. A model returned facts as plain strings, which crashed the partial and final reports. The heartbeat showed 20 to 60 calls queued with 1 to 4 in flight, because the free limits really bind: Groq allows 8K tokens per minute per model and Cerebras 5 requests per minute per model, about 25 calls a minute in all. The fixes: answers are bent into the schema's shape before any agent sees them (`schema_tools.conform`); triage tolerates incomplete items, and one failed batch no longer discards the others; providers that refuse every request are dropped; and micro-batching (below) cuts calls by about 3.3x.
  - Second full real run (1 Oct 2026): all 119 accounts and the report written, in 15.5 min (from 24) and 304 calls (from about 530). But batching mostly paired accounts (24 shared research calls for 2, one for 3), because searches finish a few seconds apart and the batch window was 0.6 s. Shared answers missing fields crashed 11 accounts (KeyError), and a shared verdict that scored only one dimension gave ABB "Not ICP 16". A web-search pause of 300 s left later accounts without sources, which triggered extra questions. Groq's two gpt-oss models hit their daily token quota (about 200K/day) after several runs that day. The fixes: wait up to 5 s (research) and 3 s (scoring, plays) to fill a batch; every shared item is completed against its schema, and incomplete ones (e.g. not every dimension scored) fall back to their own call; one-to-one Strategist conversations are capped (`ICP_TALKS`, default 8); the search pause is 60 s.
  - Pool/fast mode: the partial report was rewritten after every account; on a fast run 60 workbook saves stalled the whole team (18 s to 1.4 s on the test site once throttled to every 10 s). Queued calls polled every 0.5 s and now wake as soon as a slot frees.
  - First real local run: the program hung for 7.5 hours after finishing (a crash in the wrap-up never ended the run), local answers ran past the output limit and lost required fields, blocked searches caused minutes of re-searching, and the JavaScript-built speakers page yielded 0 speakers. All four are fixed and covered by `test_regressions.py`.
- **Not verifiable in the build sandbox:** calls to the real Gemini API and the other pool providers (no keys, network blocked) and the real conference site (blocked). Free-tier limits change, so the presets are defaults you can override. The first live run on a machine with a key is the final check.

## 9. Roadmap

1. Attendee list: if the organiser can export the gated list (CSV), add it through the Scout's `_add()`.
2. CRM agent: listens for completed plays and upserts into Salesforce or HubSpot.
3. Multi-event: run across AAMI eXchange, Field Service Medical and Palm Springs; flag accounts appearing everywhere.
4. Meeting-prep agent: a one-page brief per A-tier account before the event.
5. Feedback loop: sellers mark fit or no-fit; the Analyst's rubric guidance is tuned from outcomes.
