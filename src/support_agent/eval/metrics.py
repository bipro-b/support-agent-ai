"""Aggregate per-case results into a scorecard.

Two views that answer two different questions:
  - per-dimension pass rate -> "is our tool use solid but our groundedness weak?"
    (tells you WHERE to improve)
  - overall case pass rate   -> the single number a CI gate thresholds on
    (tells you WHETHER to ship)

A regression gate compares `overall_pass_rate` against a threshold; if a change drops it
below the bar, CI fails. That's how "don't silently regress quality" becomes mechanical.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .runner import CaseResult


@dataclass
class Scorecard:
    overall_pass_rate: float
    n_cases: int
    n_passed: int
    per_dimension: dict[str, tuple[int, int]]  # dimension -> (passed, total)

    def render(self) -> str:
        lines = [
            f"Overall: {self.n_passed}/{self.n_cases} cases passed "
            f"({self.overall_pass_rate:.0%})",
            "Per dimension:",
        ]
        for dim, (passed, total) in sorted(self.per_dimension.items()):
            rate = passed / total if total else 0.0
            lines.append(f"  {dim:<16} {passed}/{total} ({rate:.0%})")
        return "\n".join(lines)


def score(results: list[CaseResult]) -> Scorecard:
    per_dim: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [passed, total]
    n_passed = 0
    for case in results:
        if case.passed:
            n_passed += 1
        for check in case.checks:
            per_dim[check.dimension][1] += 1
            if check.passed:
                per_dim[check.dimension][0] += 1

    n = len(results)
    return Scorecard(
        overall_pass_rate=(n_passed / n if n else 0.0),
        n_cases=n,
        n_passed=n_passed,
        per_dimension={k: (v[0], v[1]) for k, v in per_dim.items()},
    )
