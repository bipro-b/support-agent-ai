"""Observability (Phase 6).

Evals (Phase 5) tell you quality dropped; observability tells you WHY. It answers the
questions you actually ask during a production incident: what did we retrieve? what prompt
did we send? which tools ran? what did each step cost and how long did it take?

The classic three pillars, all keyed by a trace id so you can correlate them:
    tracing.py   spans — a timed, nested tree of one request's steps (the core abstraction)
    logging.py   structured (JSON) logs with the trace id attached
    metrics.py   aggregates across requests — p50/p95 latency, cost, token throughput

We build a small tracer ourselves (modeled on OpenTelemetry) so the mechanics are visible;
in production you'd export to OTel/Jaeger or an LLM-specific tool (LangSmith, Langfuse).
Read docs/phase-6-observability.md alongside these files.
"""
