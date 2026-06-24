"""Circuit breaker: stop hammering a dependency that's down.

Retries help with a brief blip. But if a dependency is *genuinely down*, retrying every
request just piles latency on every user (each one waits through the full retry sequence
before failing) and keeps load on the struggling service so it can't recover. A **circuit
breaker** detects sustained failure and "trips": for a cooldown window it fails fast —
immediately, without even trying — then cautiously probes whether the dependency recovered.

Three states (named after an electrical breaker):

    CLOSED     normal. Calls go through. Count consecutive failures.
       │  (failures reach threshold)
       ▼
    OPEN       tripped. Reject immediately (fail fast) for `reset_seconds`.
       │  (cooldown elapsed)
       ▼
    HALF_OPEN  probe. Let ONE call through.
       ├─ success → CLOSED (recovered)
       └─ failure → OPEN (still down; wait again)

The win: when a dependency is down, users get an instant graceful response instead of a slow
timeout, and the dependency gets breathing room to recover. `clock` is injectable for tests.
"""

from __future__ import annotations

import time
from typing import Callable


class CircuitBreaker:
    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        reset_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._threshold = failure_threshold
        self._reset_seconds = reset_seconds
        self._clock = clock
        self._failures = 0
        self._state = "closed"
        self._opened_at = 0.0

    @property
    def state(self) -> str:
        return self._state

    def allow(self) -> bool:
        """Return True if a call may proceed right now."""
        if self._state == "open":
            if self._clock() - self._opened_at >= self._reset_seconds:
                self._state = "half_open"  # cooldown over — allow one probe
                return True
            return False  # still tripped — fail fast
        return True  # closed or half_open

    def record_success(self) -> None:
        self._failures = 0
        self._state = "closed"

    def record_failure(self) -> None:
        self._failures += 1
        # A failed probe re-opens immediately; otherwise trip once we hit the threshold.
        if self._state == "half_open" or self._failures >= self._threshold:
            self._state = "open"
            self._opened_at = self._clock()
