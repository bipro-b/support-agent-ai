"""Run the agent over the eval cases and apply the right check to each.

The runner is decoupled from HOW you produce an answer: you pass a `respond_fn` that maps
a question to an `AgentResponse` (answer + tools called + sources seen). The example wires
in the real engine; tests wire in a fake. That seam is what lets us verify the eval harness
itself without spending money on the model.

Per case we run only the checks its labels call for (see EvalCase.dimensions):
  - tool_use     DETERMINISTIC: were the expected tools actually called?
  - refusal      judge: did it correctly abstain?
  - answer_quality / groundedness  judge.
A case PASSES only if every applicable check passes — partial credit hides regressions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .cases import EvalCase
from .judge import LLMJudge


@dataclass
class AgentResponse:
    answer: str
    tools_called: list[str] = field(default_factory=list)
    context: str = ""


@dataclass
class CheckResult:
    dimension: str
    passed: bool
    detail: str


@dataclass
class CaseResult:
    case_id: str
    question: str
    checks: list[CheckResult]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks) and bool(self.checks)


def _check_tool_use(case: EvalCase, resp: AgentResponse) -> CheckResult:
    expected = set(case.expected_tools)
    called = set(resp.tools_called)
    passed = expected.issubset(called)
    return CheckResult(
        "tool_use",
        passed,
        f"expected {sorted(expected)} ⊆ called {sorted(called)}",
    )


def run_evals(
    cases: list[EvalCase],
    respond_fn,
    judge: LLMJudge,
) -> list[CaseResult]:
    results: list[CaseResult] = []
    for case in cases:
        resp: AgentResponse = respond_fn(case.question)
        checks: list[CheckResult] = []

        if case.should_refuse:
            v = judge.refusal(case.question, resp.answer)
            checks.append(CheckResult("refusal", v.passed, v.reason))
        else:
            if case.reference:
                v = judge.answer_quality(case.question, resp.answer, case.reference)
                checks.append(CheckResult("answer_quality", v.passed, f"score={v.score} — {v.reason}"))
            if case.check_grounded:
                v = judge.groundedness(resp.answer, resp.context)
                checks.append(CheckResult("groundedness", v.passed, f"score={v.score} — {v.reason}"))

        if case.expected_tools:
            checks.append(_check_tool_use(case, resp))

        results.append(CaseResult(case.id, case.question, checks))
    return results
