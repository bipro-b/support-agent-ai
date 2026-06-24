"""Reliability (Phase 8).

Every external call can fail: the model API rate-limits (429), overloads (529), times out;
the vector store, session store, and tools all have bad days. Reliability is the deliberate
answer to "what happens when each of these fails?" — so a dependency hiccup produces a
sensible response instead of a stack trace.

    retry.py            retry transient failures with exponential backoff + jitter
    circuit_breaker.py  stop hammering a dependency that's down; fail fast; probe recovery
    resilient.py        model fallback + graceful degradation for the generate step

These map onto the failure domains we named in Phase 4 §7. Read docs/phase-8-reliability.md.
"""
