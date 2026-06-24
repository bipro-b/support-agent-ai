"""Model routing: the single biggest cost lever.

Opus costs ~5x what Haiku does. Most support turns are easy ("what's your return window?")
and a cheap model answers them perfectly; a few are hard (multi-step reasoning, account
actions) and need the strong model. Sending *every* turn to Opus is like taking a taxi to
the mailbox. **Routing** classifies each turn and sends it to the cheapest model that can do
the job.

The catch — and the reason Phase 5 (evals) came first: routing trades quality for cost, so
you must *measure* that the cheap model still passes your evals on the turns you route to it.
Route too aggressively and quality drops; the eval gate is what tells you you went too far.

Two classifiers here:

- **LLM classifier (default):** a single cheap fast-model call labels the turn SIMPLE or
  COMPLEX. Costs a little, but tiny next to an avoided Opus call. We bias toward COMPLEX on
  any uncertainty — a wrong "SIMPLE" hurts quality, a wrong "COMPLEX" only costs money.
- **Heuristic (free):** rules over the text (order numbers, action verbs, length). Zero
  latency/cost, less accurate. Good when the classifier's own latency matters.

A meta-point: the router itself must be cheap. A router that costs as much as what it saves
is pointless — so we classify with the fast model, never the one we're trying to avoid.
"""

from __future__ import annotations

import re

from ..config import Settings
from ..llm import LLMClient, user

_CLASSIFIER_SYSTEM = """Classify a customer-support message by difficulty.
Reply with ONE word:
- SIMPLE: a general question answerable from policy/FAQ knowledge (hours, fees, how-to).
- COMPLEX: needs account-specific actions or tools (look up an order, start a return/refund,
  cancel), references a specific order, or requires multi-step reasoning.
Reply with only SIMPLE or COMPLEX."""

# Cheap signals that a turn is COMPLEX (used by the heuristic classifier).
_COMPLEX_PATTERNS = re.compile(
    r"\b(order|return|refund|cancel|track|tracking|exchange|human|agent|#?\d{3,})\b",
    re.IGNORECASE,
)


class ModelRouter:
    def __init__(
        self,
        llm: LLMClient,
        settings: Settings,
        *,
        enabled: bool | None = None,
        use_classifier: bool = True,
    ) -> None:
        self._llm = llm
        self._settings = settings
        self.enabled = settings.routing_enabled if enabled is None else enabled
        self.use_classifier = use_classifier

    def choose(self, question: str) -> str:
        """Return the model id to use for this turn."""
        if not self.enabled:
            return self._settings.primary_model  # routing off -> always the strong model

        is_complex = (
            self._classify_llm(question)
            if self.use_classifier
            else self._classify_heuristic(question)
        )
        return self._settings.primary_model if is_complex else self._settings.fast_model

    # -- classifiers: return True if COMPLEX -------------------------------- #
    def _classify_llm(self, question: str) -> bool:
        out = self._llm.complete(
            [user(question)],
            system=_CLASSIFIER_SYSTEM,
            model=self._settings.fast_model,  # never classify with the model we're avoiding
            max_tokens=5,
        )
        # Bias toward COMPLEX (the safe-for-quality default) unless it clearly said SIMPLE.
        return "SIMPLE" not in out.text.strip().upper()

    def _classify_heuristic(self, question: str) -> bool:
        if _COMPLEX_PATTERNS.search(question):
            return True
        return len(question) > 200  # long messages tend to be involved
