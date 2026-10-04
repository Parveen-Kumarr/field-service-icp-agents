"""Pick the model the agents think with, from settings (ICP_LLM_PROVIDER)."""
from __future__ import annotations

from .config import Settings


def make_llm(settings: Settings, *, gemini_client=None, ollama_http=None, web_client=None, pool_http=None,
             env: dict | None = None):
    if settings.llm_provider == "pool":
        import os
        from .pool import build_pool
        from .websearch import LocalWebTools
        web = LocalWebTools(settings.user_agent, settings.http_timeout, provider=settings.search_provider,
                            searxng_url=settings.searxng_url, page_chars=settings.page_chars, client=web_client,
                            min_interval=settings.search_interval,
                            parallel=getattr(settings, "search_parallel", 3))
        return build_pool(settings.pool_order, dict(os.environ) if env is None else env, web=web, http=pool_http,
                          timeout=settings.llm_timeout)
    if settings.llm_provider == "gemini":
        from .llm import GeminiLLM
        return GeminiLLM(model=settings.model, client=gemini_client, timeout=settings.llm_timeout,
                         max_parallel=max(4, settings.concurrency + 2))
    from .local_llm import OllamaLLM
    from .websearch import LocalWebTools
    web = LocalWebTools(settings.user_agent, settings.http_timeout, provider=settings.search_provider,
                        searxng_url=settings.searxng_url, page_chars=settings.page_chars, client=web_client,
                        min_interval=settings.search_interval,
                        parallel=getattr(settings, "search_parallel", 3))
    return OllamaLLM(model=settings.model, base_url=settings.ollama_url, web=web, num_ctx=settings.ollama_num_ctx,
                     think=settings.ollama_think, timeout=settings.llm_timeout,
                     max_parallel=max(1, settings.concurrency), http=ollama_http,
                     max_output=settings.ollama_max_output)
