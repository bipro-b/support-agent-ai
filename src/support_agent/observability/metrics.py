"""Metrics: aggregates ACROSS requests.

A trace explains one request; metrics describe the fleet. The questions metrics answer:
"what's our p95 latency?", "what are we spending per turn?", "did latency spike after the
deploy?". You watch these on dashboards and alert on them.

Two things worth internalizing for an interview:

- **Percentiles, not averages.** The mean hides the users having a bad time. p50 is the
  typical experience; p95 is the slow tail that drives complaints. LLM latency is
  high-variance (a turn with three tool calls is far slower than a one-shot answer), so the
  tail matters a lot. Always report p50 AND p95.
- **Cost is a first-class metric for LLM apps.** Unlike a normal web service, each request
  has a direct, variable dollar cost. Track it like you track latency — it's how you catch a
  prompt change that quietly tripled the bill.
"""

from __future__ import annotations


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    lo = int(k)
    if lo + 1 < len(s):
        return s[lo] + (k - lo) * (s[lo + 1] - s[lo])
    return s[lo]


class MetricsAggregator:
    """Ingests per-turn summaries (from summarize_trace) and reports aggregates."""

    def __init__(self) -> None:
        self._turns: list[dict] = []

    def record(self, summary: dict) -> None:
        self._turns.append(summary)

    def snapshot(self) -> dict:
        n = len(self._turns)
        if n == 0:
            return {"n_turns": 0}
        latencies = [t["latency_ms"] for t in self._turns]
        costs = [t["cost_usd"] for t in self._turns]
        return {
            "n_turns": n,
            "latency_ms_p50": round(_percentile(latencies, 0.50), 1),
            "latency_ms_p95": round(_percentile(latencies, 0.95), 1),
            "latency_ms_max": round(max(latencies), 1),
            "cost_usd_total": round(sum(costs), 6),
            "cost_usd_avg": round(sum(costs) / n, 6),
            "tokens_total": sum(t["input_tokens"] + t["output_tokens"] for t in self._turns),
            "llm_calls_avg": round(sum(t["llm_calls"] for t in self._turns) / n, 2),
            "tool_calls_avg": round(sum(t["tool_calls"] for t in self._turns) / n, 2),
        }

    def render(self) -> str:
        snap = self.snapshot()
        if snap["n_turns"] == 0:
            return "No turns recorded."
        return (
            f"Metrics over {snap['n_turns']} turns:\n"
            f"  latency: p50={snap['latency_ms_p50']}ms  p95={snap['latency_ms_p95']}ms  "
            f"max={snap['latency_ms_max']}ms\n"
            f"  cost:    total=${snap['cost_usd_total']}  avg=${snap['cost_usd_avg']}/turn\n"
            f"  tokens:  total={snap['tokens_total']}\n"
            f"  calls:   {snap['llm_calls_avg']} llm/turn  {snap['tool_calls_avg']} tools/turn"
        )
