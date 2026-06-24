"""Rate limiting: stop one principal from flooding the endpoint.

An LLM endpoint is unusually attractive to abuse because each request costs real money — an
attacker (or a buggy client) hammering `/chat` runs up your bill and starves real users. A
**token bucket** is the standard throttle: each principal has a bucket that holds up to
`capacity` tokens and refills at `refill_per_sec`. Each request spends one token; an empty
bucket means "rejected" (the API returns 429). This allows short bursts (up to capacity) while
bounding the sustained rate — the shape real traffic actually has.

`clock` is injectable so tests can advance time without sleeping.

Note: this in-memory limiter is per-process. Behind multiple replicas you'd use a shared store
(Redis) so the limit is enforced across the fleet — same idea as the session store in Phase 4.
"""

from __future__ import annotations

import time
from typing import Callable


class TokenBucket:
    def __init__(
        self, capacity: int, refill_per_sec: float, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._capacity = float(capacity)
        self._refill = refill_per_sec
        self._clock = clock
        self._tokens = float(capacity)
        self._last = clock()

    def allow(self, cost: float = 1.0) -> bool:
        now = self._clock()
        # Refill based on elapsed time, capped at capacity.
        self._tokens = min(self._capacity, self._tokens + (now - self._last) * self._refill)
        self._last = now
        if self._tokens >= cost:
            self._tokens -= cost
            return True
        return False


class RateLimiter:
    """One token bucket per key (e.g. per customer_id or IP)."""

    def __init__(
        self, capacity: int, refill_per_sec: float, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._capacity = capacity
        self._refill = refill_per_sec
        self._clock = clock
        self._buckets: dict[str, TokenBucket] = {}

    def allow(self, key: str) -> bool:
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = TokenBucket(self._capacity, self._refill, self._clock)
            self._buckets[key] = bucket
        return bucket.allow()
