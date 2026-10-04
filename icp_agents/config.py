"""Runtime settings, read from the environment / .env file."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

PROVIDERS = ("pool", "ollama", "gemini")
DEFAULT_MODEL = {"pool": "pool", "ollama": "gemma4:12b", "gemini": "gemini-3.8-flash"}
# how many accounts are researched at once (a local GPU runs few requests at once; the pool runs many)
DEFAULT_CONCURRENCY = {"pool": 12, "ollama": 2, "gemini": 4}
DEFAULT_LLM_TIMEOUT = {"pool": 45.0, "ollama": 300.0, "gemini": 180.0}


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _int(name: str, default: int) -> int:
    try:
        return int(_env(name) or default)
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(_env(name) or default)
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    v = _env(name).lower()
    return default if not v else v in ("1", "true", "yes", "on")


@dataclass
class Settings:
    event_url: str = field(default_factory=lambda: _env("ICP_EVENT_URL", "https://fieldserviceusa.wbresearch.com/"))
    # The company whose ICP prospects are qualified against. With a URL, the Researcher reads its site live
    # and briefs the team; without one, the generic profile in vendor_profile.py is used.
    vendor_name: str = field(default_factory=lambda: _env("ICP_VENDOR_NAME"))
    vendor_url: str = field(default_factory=lambda: _env("ICP_VENDOR_URL"))

    # Which model the agents think with:
    #   "pool"   - several free cloud providers used together, each within its own limits (fastest)
    #   "ollama" - local model on your GPU (no limits, slower)
    #   "gemini" - Google Gemini API only
    llm_provider: str = field(default_factory=lambda: _env("ICP_LLM_PROVIDER", "pool").lower())
    # For "pool": which providers to use, in order of preference (only those with an API key are used).
    pool_order: list = field(default_factory=lambda: [x.strip().lower() for x in _env(
        "ICP_POOL", "groq,cerebras,mistral,openrouter,gemini,ollama").split(",") if x.strip()])
    # Fast mode: triage all accounts in one pass, one search + one compact call per researched account.
    # Blank = on for "pool", off otherwise.
    fast: bool | None = field(default_factory=lambda: None if not _env("ICP_FAST") else _bool("ICP_FAST", False))
    batch_size: int = field(default_factory=lambda: _int("ICP_BATCH", 4))  # accounts per model call in fast mode (1 = off)
    talk_limit: int = field(default_factory=lambda: _int("ICP_TALKS", 8))  # fast mode: one-to-one Strategist conversations
    model: str = field(default_factory=lambda: _env("ICP_LLM_MODEL"))          # blank = provider default
    ollama_url: str = field(default_factory=lambda: _env("ICP_OLLAMA_URL", "http://localhost:11434"))
    ollama_num_ctx: int = field(default_factory=lambda: _int("ICP_OLLAMA_NUM_CTX", 16384))
    ollama_think: bool = field(default_factory=lambda: _bool("ICP_OLLAMA_THINK", False))
    ollama_max_output: int = field(default_factory=lambda: _int("ICP_OLLAMA_MAX_OUTPUT", 2048))  # tokens per reply

    # Web research for the local model (Gemini uses Google Search / URL context on Google's side instead).
    search_provider: str = field(default_factory=lambda: _env("ICP_SEARCH_PROVIDER", "ddgs").lower())
    search_interval: float = field(default_factory=lambda: _float("ICP_SEARCH_INTERVAL", 1.0))  # seconds between search starts
    search_parallel: int = field(default_factory=lambda: _int("ICP_SEARCH_PARALLEL", 3))  # searches at once
    searxng_url: str = field(default_factory=lambda: _env("ICP_SEARXNG_URL"))
    page_chars: int = field(default_factory=lambda: _int("ICP_PAGE_CHARS", 8000))  # page text given to the model

    # Scraping: "live" = local HTTP, then the model's own web access if blocked, else stop. "local" = local only.
    scrape_mode: str = field(default_factory=lambda: _env("ICP_SCRAPE_MODE", "live"))
    concurrency: int = field(default_factory=lambda: _int("ICP_CONCURRENCY", 0))    # 0 = provider default
    max_accounts: int = field(default_factory=lambda: _int("ICP_MAX_ACCOUNTS", 0))  # 0 = no limit
    searches_per_account: int = field(default_factory=lambda: _int("ICP_SEARCHES_PER_ACCOUNT", 4))
    include_sponsors: bool = True
    include_logos: bool = True
    render_js: bool = field(default_factory=lambda: _env("ICP_RENDER_JS", "auto") != "off")
    output_dir: str = field(default_factory=lambda: _env("ICP_OUTPUT_DIR", "output"))
    llm_timeout: float = field(default_factory=lambda: _float("ICP_LLM_TIMEOUT", 0))   # 0 = provider default
    heartbeat: float = field(default_factory=lambda: _float("ICP_HEARTBEAT", 15))
    log_level: str = field(default_factory=lambda: _env("ICP_LOG_LEVEL", "INFO"))
    http_timeout: float = 25.0
    user_agent: str = "Mozilla/5.0 (compatible; FieldServiceICPResearch/2.2)"

    def __post_init__(self) -> None:
        self.resolve()

    def resolve(self) -> "Settings":
        """Fill provider-dependent defaults (call again after changing llm_provider)."""
        if self.llm_provider not in PROVIDERS:
            raise ValueError(f"ICP_LLM_PROVIDER must be one of {PROVIDERS}, not '{self.llm_provider}'")
        if not self.model or self.model in DEFAULT_MODEL.values():
            self.model = DEFAULT_MODEL[self.llm_provider]
        if not self.concurrency:
            self.concurrency = DEFAULT_CONCURRENCY[self.llm_provider]
        if not self.llm_timeout:
            self.llm_timeout = DEFAULT_LLM_TIMEOUT[self.llm_provider]
        if self.fast is None:
            self.fast = self.llm_provider == "pool"
        return self
