"""Phase 6 — See inside the agent: traces, structured logs, and metrics.

Runs a few turns through the engine with observability wired in, and shows all three
pillars:
  - the TRACE tree for each turn (what ran, nested, with per-step timing + cost)
  - the STRUCTURED LOG line emitted per turn (JSON, with trace_id)
  - the METRICS rolled up across all turns (p50/p95 latency, total cost)

Run (needs ANTHROPIC_API_KEY; VOYAGE_API_KEY recommended):

    python examples/phase6_observability.py
"""

from __future__ import annotations

from support_agent.observability.logging import StructuredLogger
from support_agent.observability.metrics import MetricsAggregator
from support_agent.observability.tracing import render_trace
from support_agent.service.engine import SupportEngine

TURNS = [
    "What's your return window?",                       # RAG-only: retrieve -> answer
    "Can you check the status of my order 1234?",       # tool: lookup_order
    "Start a return for the damaged headphones on 1234.",  # sensitive tool path
]


def main() -> None:
    metrics = MetricsAggregator()
    logger = StructuredLogger()  # prints JSON log lines to stdout
    engine = SupportEngine(logger=logger, metrics=metrics)

    for i, message in enumerate(TURNS, start=1):
        print(f"\n{'=' * 72}\nTurn {i}: {message}")
        result = engine.handle_turn(
            customer_id="cust_rahim", session_id="obs_demo", message=message
        )
        print(f"Answer: {result.answer}\n")
        print("Trace (what actually happened, with timings + cost):")
        print(render_trace(result.trace))  # the span tree for this turn
        # The structured log line for this turn was just printed by the logger above
        # (the {"event": "turn_complete", ...} JSON).

    print(f"\n{'=' * 72}")
    print(metrics.render())
    print("\nThe trace answers 'why was THIS turn slow/expensive?'; the metrics answer")
    print("'how is the system doing overall?'. Together they make prod debuggable.")


if __name__ == "__main__":
    main()
