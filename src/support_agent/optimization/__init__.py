"""Cost & latency optimization (Phase 7).

    router.py   ModelRouter — pick the cheapest model that can do the job for each turn.

Other Phase 7 levers live where they naturally belong and are reused, not reinvented here:
prompt caching (LLMClient.cache_system, applied in the agent loop), context trimming
(config.final_top_k), and streaming (LLMClient.stream). See docs/phase-7-cost-latency.md.
"""
