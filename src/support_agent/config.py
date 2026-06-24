"""Centralized, typed configuration.

Why a config module at all (vs. reading os.environ everywhere)?
- One place to see every knob the system has.
- Typed + validated at startup, so a missing key fails fast and loud,
  not deep inside a request handler in production.
- Secrets come from the environment / .env — never hardcoded.

This is a production habit worth forming on day one.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration for the support agent.

    Values are read from environment variables prefixed with SUPPORT_AGENT_,
    falling back to the defaults below. ANTHROPIC_API_KEY is read without a
    prefix because that's the name the Anthropic SDK itself expects.
    """

    model_config = SettingsConfigDict(
        env_prefix="SUPPORT_AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # The SDK reads ANTHROPIC_API_KEY from the environment on its own, but we
    # surface it here so the app can fail fast with a clear message if it's missing.
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")

    # Two models, two jobs (we'll lean on this hard in Phase 7 — cost/latency):
    #   primary  → hard reasoning, the customer-facing answer
    #   fast     → cheap, fast helpers: classification, routing, summaries
    primary_model: str = "claude-opus-4-8"
    fast_model: str = "claude-haiku-4-5"

    # A sane default cap. Streaming responses can go higher (see the docs).
    max_tokens: int = 1024

    # ---- Phase 1: RAG ---------------------------------------------------- #
    # Voyage AI is Anthropic's recommended embedding/reranking provider. It needs
    # its own key (free tier at voyageai.com). If absent, the RAG code falls back
    # to a local toy embedder so the pipeline still RUNS — but quality will be poor.
    voyage_api_key: str = Field(default="", alias="VOYAGE_API_KEY")

    # Model names can change; if you hit a "model not found" error, check
    # https://docs.voyageai.com for the current lineup and override via .env.
    embedding_model: str = "voyage-3.5"
    rerank_model: str = "rerank-2.5"

    # Where the knowledge base markdown lives (relative to repo root).
    knowledge_base_dir: str = "data/knowledge_base"

    # Chunking: max characters per chunk and overlap between adjacent chunks.
    chunk_max_chars: int = 1000
    chunk_overlap_chars: int = 150

    # Retrieval: how many chunks to pull from the vector store, and — when
    # reranking is on — how many to keep after the reranker re-scores them.
    retrieve_top_n: int = 20   # first-stage recall (embedding search)
    final_top_k: int = 4       # what we actually feed the model after reranking

    # ---- Phase 2: context & memory -------------------------------------- #
    # Where per-customer long-term memory is persisted (JSON files for now).
    memory_dir: str = "data/memory"

    # Token budget for the verbatim conversation history. When the running
    # history is estimated to exceed this, we summarize the oldest turns to
    # keep the prompt (and its cost) bounded. This is the core lever of Phase 2.
    context_budget_tokens: int = 1200

    # How many of the most recent messages we always keep VERBATIM (never
    # summarized). 4 ≈ the last two user/assistant exchanges.
    keep_recent_messages: int = 4

    # ---- Phase 7: cost & latency ---------------------------------------- #
    # Route easy turns to the fast/cheap model and hard turns to the strong one.
    # The single biggest cost lever (Haiku is ~5x cheaper than Opus). When False,
    # every turn uses the primary model (the pre-Phase-7 behavior / the baseline).
    routing_enabled: bool = True

    # ---- Phase 8: reliability ------------------------------------------- #
    # Bound how long we wait on the model API, and how many times the SDK retries
    # transient errors (429/5xx/timeouts) with its own backoff before giving up.
    request_timeout_s: float = 30.0
    max_retries: int = 3

    # If the chosen model keeps failing (transiently) after the SDK's retries, fall
    # back to this model so an Opus outage degrades to a Haiku answer, not an error.
    fallback_model: str = "claude-haiku-4-5"

    # Circuit breaker for the model API: after this many consecutive transient
    # failures, stop trying for `breaker_reset_seconds` and fail fast (degrade).
    breaker_failure_threshold: int = 5
    breaker_reset_seconds: float = 30.0

    # ---- Phase 9: security ---------------------------------------------- #
    # Per-customer rate limit (token bucket): burst up to `capacity`, sustained
    # `refill_per_sec`. Protects against abuse and runaway cost.
    rate_limit_capacity: int = 10
    rate_limit_refill_per_sec: float = 1.0


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance.

    lru_cache makes this a lightweight singleton: read/validate the environment
    once, reuse everywhere. Tests can call get_settings.cache_clear() to reset.
    """
    return Settings()
