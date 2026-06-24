"""LLM-as-judge: using a model to score outputs no exact-match check can.

Most of what makes a support answer "good" — correctness, helpfulness, faithfulness to the
sources — is free text you can't grade with `==`. The practical answer is to have a strong
model grade the output against a clear rubric. This is **LLM-as-judge**, and it's standard
in production eval pipelines.

It is powerful but not free of pitfalls — know them, because interviewers ask:
  - **Use a strong model to judge.** A weak judge gives weak signal; we judge with the
    primary model, not the fast one.
  - **The rubric is everything.** Vague rubrics ("is it good?") give noisy scores. Each
    rubric below defines what each score means and what to penalize.
  - **Known biases:** LLM judges can favor longer answers, their own style, or the first
    option in a pairwise test. Mitigate with tight rubrics and, in production, calibration
    against human labels on a sample.
  - **Non-determinism:** the same answer can score slightly differently across runs. Treat
    scores as a signal over a dataset, not a precise per-item truth; run periodically and
    watch the aggregate trend.
  - **Constrain the output.** We demand a tiny JSON object so the verdict is parseable.
    (Structured outputs are the production-robust way to guarantee this shape.)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..llm import LLMClient, user


@dataclass
class Verdict:
    passed: bool
    score: int          # 1-5 (or 0/1 mapped to 1/5 for boolean rubrics)
    reason: str


def _parse_json(text: str) -> dict:
    """Pull the first {...} object out of the model's reply, tolerantly."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


_QUALITY_SYSTEM = """You are a strict evaluator of customer-support answers.
Given a QUESTION, the agent's ANSWER, and REFERENCE key facts a correct answer must
reflect, rate the answer's correctness and helpfulness.
Scoring (1-5): 5 = fully correct and helpful, consistent with the reference facts;
3 = partially correct or missing key information; 1 = incorrect or unhelpful.
Penalize factual errors heavily. Respond with ONLY JSON: {"score": <1-5>, "reason": "<brief>"}."""

_GROUNDED_SYSTEM = """You evaluate whether an ANSWER is fully supported by the provided SOURCES
(i.e. free of hallucination).
Scoring (1-5): 5 = every factual claim is supported by the sources; 3 = mostly supported
with minor unsupported additions; 1 = contains clear claims not present in the sources.
An answer that correctly states it lacks the information is fully grounded (score 5).
Respond with ONLY JSON: {"score": <1-5>, "reason": "<brief>"}."""

_REFUSAL_SYSTEM = """You judge whether an ANSWER appropriately ABSTAINS on an out-of-scope
question: it must NOT fabricate an answer, and should say it doesn't have that information
and/or offer a human handoff.
Respond with ONLY JSON: {"abstained": true|false, "reason": "<brief>"}."""


class LLMJudge:
    def __init__(self, llm: LLMClient, *, pass_threshold: int = 4) -> None:
        self._llm = llm
        self._pass_threshold = pass_threshold

    def _score(self, system: str, payload: str) -> Verdict:
        out = self._llm.complete([user(payload)], system=system, max_tokens=250)
        data = _parse_json(out.text)
        score = int(data.get("score", 0) or 0)
        reason = str(data.get("reason", "no reason parsed"))
        return Verdict(passed=score >= self._pass_threshold, score=score, reason=reason)

    def answer_quality(self, question: str, answer: str, reference: str) -> Verdict:
        payload = f"QUESTION:\n{question}\n\nANSWER:\n{answer}\n\nREFERENCE:\n{reference}"
        return self._score(_QUALITY_SYSTEM, payload)

    def groundedness(self, answer: str, sources: str) -> Verdict:
        payload = f"SOURCES:\n{sources}\n\nANSWER:\n{answer}"
        return self._score(_GROUNDED_SYSTEM, payload)

    def refusal(self, question: str, answer: str) -> Verdict:
        payload = f"QUESTION:\n{question}\n\nANSWER:\n{answer}"
        out = self._llm.complete([user(payload)], system=_REFUSAL_SYSTEM, max_tokens=200)
        data = _parse_json(out.text)
        abstained = bool(data.get("abstained", False))
        reason = str(data.get("reason", "no reason parsed"))
        return Verdict(passed=abstained, score=5 if abstained else 1, reason=reason)
