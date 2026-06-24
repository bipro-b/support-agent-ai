"""Phase 5 — Run the end-to-end eval suite and print a scorecard.

This runs the REAL agent over the labeled cases and grades each with deterministic checks
(tool use) and the LLM judge (quality, groundedness, refusal). It costs model calls — evals
are not free, which is exactly why they're a deliberate, periodic activity, not something
you run on every keystroke.

Run (needs ANTHROPIC_API_KEY; VOYAGE_API_KEY strongly recommended for good retrieval):

    python examples/phase5_eval.py

Each case uses a fresh, isolated customer + session so cases don't contaminate each other
through long-term memory. The engine (and its KB index) is built once and shared.
"""

from __future__ import annotations

from pathlib import Path

from support_agent.eval.cases import load_cases
from support_agent.eval.judge import LLMJudge
from support_agent.eval.metrics import score
from support_agent.eval.runner import AgentResponse, run_evals
from support_agent.service.engine import SupportEngine

CASES_PATH = Path(__file__).resolve().parents[1] / "tests" / "eval" / "answer_quality.json"


def make_responder(engine: SupportEngine):
    counter = {"n": 0}

    def respond(question: str) -> AgentResponse:
        counter["n"] += 1
        # Isolate every case: unique customer + session => no memory cross-contamination.
        tag = f"eval_{counter['n']}"
        result = engine.handle_turn(customer_id=tag, session_id=tag, message=question)
        return AgentResponse(
            answer=result.answer,
            tools_called=result.tools_called or [],
            context=result.context or "",
        )

    return respond


def main() -> None:
    cases = load_cases(CASES_PATH)
    engine = SupportEngine()                 # built once; indexes the KB once
    judge = LLMJudge(engine.llm)             # judge with the strong model

    print(f"Running {len(cases)} eval cases...\n")
    results = run_evals(cases, make_responder(engine), judge)

    for r in results:
        mark = "PASS" if r.passed else "FAIL"
        print(f"[{mark}] {r.case_id}: {r.question}")
        for c in r.checks:
            cmark = "ok " if c.passed else "XX "
            print(f"     {cmark}{c.dimension}: {c.detail}")

    print(f"\n{'=' * 70}")
    print(score(results).render())
    print("\nUse this scorecard as your baseline. Change a prompt or model, rerun, and")
    print("compare — that's how you optimize (Phase 7) WITHOUT silently regressing.")


if __name__ == "__main__":
    main()
