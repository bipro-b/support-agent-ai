"""Retry transient failures with exponential backoff and jitter.

A **transient** failure is one that might succeed if you just try again: a rate limit, a
500, a dropped connection, a timeout. The fix is to wait and retry. Three details separate a
correct retry from a harmful one:

- **Exponential backoff.** Wait longer after each failure (0.5s, 1s, 2s, ...). Retrying
  instantly hammers a struggling dependency and makes the outage worse — a "retry storm".
- **Jitter.** Randomize the delay a little. Without it, a thousand clients that failed at the
  same instant all retry at the same instant, creating synchronized waves of load. Jitter
  spreads them out.
- **Only retry what's retryable.** A 400 (bad request) or 401 (bad auth) will fail the same
  way forever — retrying wastes time and money. Classify, and retry only transient errors.

Note on the model specifically: the Anthropic SDK already retries 429/5xx/timeouts with
backoff internally (we configure `max_retries`). This helper is for OUR dependencies — a
session store, a tool's backend API — where we own the retry policy.

`sleep` and `rng` are injectable so tests run instantly and deterministically.
"""

from __future__ import annotations

import random
import time
from typing import Callable, TypeVar

T = TypeVar("T")


def retry_call(
    fn: Callable[[], T],
    *,
    max_attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    retryable: Callable[[Exception], bool] = lambda _e: True,
    sleep: Callable[[float], None] = time.sleep,
    rng: Callable[[], float] = random.random,
) -> T:
    """Call `fn`, retrying transient failures up to `max_attempts` times total."""
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:
            attempt += 1
            # Give up if we're out of attempts or the error isn't worth retrying.
            if attempt >= max_attempts or not retryable(exc):
                raise
            # Exponential backoff: base * 2^(attempt-1), capped.
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            # Jitter: scale to 50%-100% of the delay so clients don't retry in lockstep.
            delay *= 0.5 + 0.5 * rng()
            sleep(delay)
