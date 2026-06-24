"""Phase 7 — Measure the cost/latency win from model routing.

Runs the same mix of turns twice — routing OFF (every turn on Opus, the baseline) and
routing ON (easy turns to Haiku, hard turns to Opus) — and compares cost and latency using
the Phase 6 metrics. This is "measure, optimize, measure again" made concrete.

IMPORTANT: a cheaper number is only a win if quality held. After this, run the Phase 5 gate
(`pytest tests/test_eval_quality.py`) to confirm routing didn't drop the pass rate. Cheaper
AND still-correct is the goal; cheaper alone is easy and worthless.

Run (needs ANTHROPIC_API_KEY; VOYAGE_API_KEY recommended):

    python examples/phase7_cost_latency.py
"""

from __future__ import annotations

from support_agent.observability.metrics import MetricsAggregator
from support_agent.service.engine import SupportEngine

# A realistic mix: mostly easy FAQs (should route to Haiku) + a couple that need tools/Opus.
TURNS = [
    "What are your business hours?",
    "How long does standard shipping take?",
    "Do you ship internationally?",
    "What payment methods do you accept?",
    "How do I reset my password?",
    "Can you check the status of my order 1234?",          # needs lookup_order -> Opus
    "Start a return for the damaged headphones on order 1234.",  # action -> Opus
]


def run_suite(engine: SupportEngine, *, routing: bool) -> tuple[MetricsAggregator, list[str]]:
    engine.router.enabled = routing
    metrics = MetricsAggregator()
    engine.metrics = metrics
    models_used: list[str] = []
    for i, message in enumerate(TURNS):
        # Fresh, isolated customer + session per turn so the two runs are comparable.
        tag = f"{'on' if routing else 'off'}_{i}"
        result = engine.handle_turn(customer_id=tag, session_id=tag, message=message)
        models_used.append(result.usage.model)
    return metrics, models_used


def main() -> None:
    engine = SupportEngine()  # router enabled by default; we toggle it per run

    print("Baseline: routing OFF (every turn on the strong model)...")
    base_metrics, _ = run_suite(engine, routing=False)

    print("Optimized: routing ON (easy turns -> fast model)...")
    routed_metrics, models = run_suite(engine, routing=True)

    print(f"\n{'=' * 72}\nPer-turn model chosen by the router:")
    for message, model in zip(TURNS, models):
        print(f"  {model:<20} {message}")

    base, routed = base_metrics.snapshot(), routed_metrics.snapshot()
    print(f"\n{'=' * 72}\nBASELINE (all Opus):")
    print("  " + base_metrics.render().replace("\n", "\n  "))
    print("\nROUTED:")
    print("  " + routed_metrics.render().replace("\n", "\n  "))

    if base["cost_usd_total"] > 0:
        saved = 1 - routed["cost_usd_total"] / base["cost_usd_total"]
        print(f"\nCost change from routing: {saved:+.0%} "
              f"(${base['cost_usd_total']} -> ${routed['cost_usd_total']})")
    print("\nNow run `pytest tests/test_eval_quality.py` to confirm quality held. A cost")
    print("drop only counts if the eval pass rate didn't fall with it.")


if __name__ == "__main__":
    main()
