"""Model fallback + graceful degradation for the generate step.

This ties the primitives together for the one call that matters most — the model. The policy:

  1. Try the chosen model. (The SDK already retried transient errors internally.)
  2. If it STILL fails transiently, fall back to a second model — an Opus overload becomes a
     Haiku answer, not an error. (A weaker answer beats no answer.)
  3. If a PERMANENT error occurs (bad request, auth), don't bother falling back — it'll fail
     the same way. Stop and degrade.
  4. If everything fails, return a **graceful degraded answer** — a calm "try again / I'll get
     a human" — never a 500 or a stack trace to the customer.

A circuit breaker gates the whole thing: if the model API has been failing, we fail fast to
the degraded answer instead of making every user wait through retries.

`is_transient` is injected (the engine passes the Anthropic-aware classifier; tests pass
their own), keeping this module free of a hard SDK dependency and easy to test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..llm import Usage

DEGRADED_ANSWER = (
    "I'm sorry — I'm having trouble completing that right now. Please try again in a "
    "moment, or I can connect you with a human agent."
)


@dataclass
class GenerateOutcome:
    answer: str
    usage: Usage
    steps: list[str]
    tools_called: list[str]
    model: str
    degraded: bool


def is_transient_anthropic(exc: Exception) -> bool:
    """True if an Anthropic error is worth retrying / falling back on (vs permanent)."""
    import anthropic

    transient = tuple(
        t
        for t in (
            getattr(anthropic, "RateLimitError", None),
            getattr(anthropic, "APITimeoutError", None),
            getattr(anthropic, "APIConnectionError", None),
            getattr(anthropic, "InternalServerError", None),
            getattr(anthropic, "OverloadedError", None),
        )
        if t is not None
    )
    return isinstance(exc, transient)


def resilient_generate(
    run_fn: Callable[[str], tuple],
    *,
    models: list[str],
    is_transient: Callable[[Exception], bool],
    breaker=None,
    on_event: Callable[[str, dict], None] | None = None,
) -> GenerateOutcome:
    """Run `run_fn(model) -> (answer, usage, steps, tools_called)` with fallback + degrade.

    Returns a GenerateOutcome. `degraded=True` means every attempt failed and the canned
    answer is being returned. Never raises — the customer always gets *something* civil.
    """
    def emit(event: str, **fields) -> None:
        if on_event is not None:
            on_event(event, fields)

    for model in models:
        if breaker is not None and not breaker.allow():
            emit("circuit_open", model=model)
            break  # fail fast — don't even try
        try:
            answer, usage, steps, tools_called = run_fn(model)
            if breaker is not None:
                breaker.record_success()
            return GenerateOutcome(answer, usage, steps, tools_called, model, degraded=False)
        except Exception as exc:  # noqa: BLE001 — we deliberately never leak an exception
            transient = is_transient(exc)
            if breaker is not None and transient:
                breaker.record_failure()
            emit("generate_failure", model=model, transient=transient, error=type(exc).__name__)
            if not transient:
                break  # permanent error — a different model won't help
            # transient: fall through to the next model

    # Everything failed (or the circuit was open) — degrade gracefully.
    return GenerateOutcome(
        answer=DEGRADED_ANSWER,
        usage=Usage(model=models[0] if models else "none", input_tokens=0, output_tokens=0),
        steps=["degraded: all model attempts failed"],
        tools_called=[],
        model=models[0] if models else "none",
        degraded=True,
    )
