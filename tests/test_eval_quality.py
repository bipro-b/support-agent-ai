"""Phase 5 — the regression GATE.

This is the eval turned into a pass/fail test so quality can't silently regress. In CI you
run it (on a schedule or pre-release, since it costs model calls) and it FAILS the build if
the overall pass rate drops below the threshold.

It skips automatically when no API key is present, so the rest of the test suite still runs
for free in ordinary CI. Run it deliberately:

    pytest tests/test_eval_quality.py -v

Tune THRESHOLD to your real baseline: run the eval a few times, see where a healthy build
lands, and set the gate a little below that so normal LLM non-determinism doesn't flap it.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

CASES_PATH = Path(__file__).resolve().parent / "eval" / "answer_quality.json"
THRESHOLD = 0.80  # require >= 80% of cases to pass

pytestmark = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="needs ANTHROPIC_API_KEY (eval calls the model)",
)


def test_answer_quality_above_threshold() -> None:
    from support_agent.eval.cases import load_cases
    from support_agent.eval.judge import LLMJudge
    from support_agent.eval.metrics import score
    from support_agent.eval.runner import AgentResponse, run_evals
    from support_agent.service.engine import SupportEngine

    cases = load_cases(CASES_PATH)
    engine = SupportEngine()
    judge = LLMJudge(engine.llm)

    counter = {"n": 0}

    def respond(question: str) -> AgentResponse:
        counter["n"] += 1
        tag = f"evaltest_{counter['n']}"
        r = engine.handle_turn(customer_id=tag, session_id=tag, message=question)
        return AgentResponse(r.answer, r.tools_called or [], r.context or "")

    card = score(run_evals(cases, respond, judge))
    print("\n" + card.render())
    assert card.overall_pass_rate >= THRESHOLD, (
        f"Quality regressed: {card.overall_pass_rate:.0%} < {THRESHOLD:.0%}\n"
        + card.render()
    )
