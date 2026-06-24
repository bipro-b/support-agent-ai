"""End-to-end evaluation (Phase 5).

Phase 1 measured RETRIEVAL quality (did we fetch the right chunk?). This package measures
the quality of the agent's actual BEHAVIOR — the thing the customer experiences:

    cases.py    EvalCase + the labeled dataset loader
    judge.py    LLMJudge — an LLM scoring outputs against a rubric (for subjective quality)
    runner.py   run the agent over cases, apply checks, collect results
    metrics.py  aggregate into a scorecard (pass rate per dimension)

Three dimensions, two kinds of check:
  - tool-use correctness  -> DETERMINISTIC (compare expected vs actual tool calls)
  - refusal / abstention  -> model-graded (did it correctly decline to guess?)
  - answer quality        -> model-graded (correct & helpful vs reference)
  - groundedness          -> model-graded (every claim supported by the sources?)

Prefer deterministic checks where possible; reach for the LLM judge only for the
subjective parts. Read docs/phase-5-evals.md alongside these files.
"""
