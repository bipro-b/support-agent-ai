"""Phase 8 — Reliability primitives under injected faults.

You can't summon a real model outage on demand, so we INJECT faults to show each mechanism
working. This example needs no API key — it exercises the reliability primitives directly,
which is exactly how you'd unit-test them.

    python examples/phase8_reliability.py

How these wire into the live system (see engine.py): the SDK retries transient model errors;
`resilient_generate` adds model fallback + graceful degradation gated by the circuit breaker;
retrieval/session failures are caught and degraded; `start_return` is idempotent.
"""

from __future__ import annotations

from support_agent.agent.tools import RETURNS_LOG, start_return
from support_agent.llm import Usage
from support_agent.reliability.circuit_breaker import CircuitBreaker
from support_agent.reliability.resilient import resilient_generate
from support_agent.reliability.retry import retry_call


class TransientError(Exception):
    """Stands in for a 429/503/timeout — worth retrying."""


def section(t: str) -> None:
    print(f"\n{'=' * 70}\n{t}\n{'=' * 70}")


def demo_retry() -> None:
    section("1. Retry with exponential backoff (transient failure recovers)")
    attempts = {"n": 0}

    def flaky_store_read():
        attempts["n"] += 1
        if attempts["n"] < 3:
            print(f"   attempt {attempts['n']}: TransientError (will retry)")
            raise TransientError("temporary blip")
        print(f"   attempt {attempts['n']}: success")
        return "data"

    slept: list[float] = []
    result = retry_call(
        flaky_store_read,
        max_attempts=5,
        retryable=lambda e: isinstance(e, TransientError),
        sleep=slept.append,        # capture instead of really sleeping
        rng=lambda: 0.0,           # deterministic backoff for the demo
    )
    print(f"   -> got '{result}' after backoff waits {[round(s, 2) for s in slept]}s")
    print("   (real waits grow 0.5,1,2,... with jitter, so clients don't retry in lockstep)")


def demo_circuit_breaker() -> None:
    section("2. Circuit breaker (stop hammering a dependency that's down)")
    clock = {"t": 0.0}
    cb = CircuitBreaker(failure_threshold=3, reset_seconds=10, clock=lambda: clock["t"])

    print(f"   start: state={cb.state}, allow={cb.allow()}")
    for i in range(3):
        cb.record_failure()
        print(f"   failure {i + 1}: state={cb.state}")
    print(f"   while OPEN: allow={cb.allow()}  <- requests fail FAST, no waiting")
    clock["t"] = 11
    print(f"   after cooldown: allow={cb.allow()} (probe), state={cb.state}")
    cb.record_success()
    print(f"   probe succeeded: state={cb.state}  <- recovered")


def demo_fallback_and_degradation() -> None:
    section("3. Model fallback, then graceful degradation")
    is_transient = lambda e: isinstance(e, TransientError)

    def primary_down(model: str):
        if model == "claude-opus-4-8":
            raise TransientError("Opus overloaded (529)")
        return ("Here's your answer (from the fallback model).",
                Usage(model, 60, 15), ["agent: final answer"], [])

    out = resilient_generate(primary_down,
                             models=["claude-opus-4-8", "claude-haiku-4-5"],
                             is_transient=is_transient)
    print(f"   primary failed -> answered on {out.model}, degraded={out.degraded}")

    def everything_down(model: str):
        raise TransientError("whole API down")

    out2 = resilient_generate(everything_down,
                              models=["claude-opus-4-8", "claude-haiku-4-5"],
                              is_transient=is_transient)
    print(f"   all models failed -> degraded={out2.degraded}")
    print(f"   customer sees: \"{out2.answer}\"")
    print("   (a calm, useful message — never a 500 or a stack trace)")


def demo_idempotency() -> None:
    section("4. Idempotency (a retried side effect doesn't happen twice)")
    RETURNS_LOG.clear()
    print("   " + start_return("1234", "Wireless headphones", "damaged"))
    print("   --- the request is retried (network blip) ---")
    print("   " + start_return("1234", "Wireless headphones", "damaged"))
    print(f"   returns actually created: {len(RETURNS_LOG)}  <- exactly one, not two")


def main() -> None:
    demo_retry()
    demo_circuit_breaker()
    demo_fallback_and_degradation()
    demo_idempotency()
    section("Done")
    print("These primitives are wired into SupportEngine.handle_turn so a dependency")
    print("failure produces a graceful answer instead of taking down the turn.")


if __name__ == "__main__":
    main()
