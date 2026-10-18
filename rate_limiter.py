"""
Token-bucket rate limiter for the ResearchAgent — sits in api.py right
before agent.ask() is called, protecting your Groq quota.

Two independent buckets:
  - per-session: stops one browser tab from spamming requests
  - global: caps total throughput across every session combined — this is
    the one actually protecting your Groq rate limit / usage budget

In-memory, single-process. Fine for a solo-maintained deployment on a
single Render/Fly instance. If you ever scale to multiple backend
instances, swap the dicts for Redis (same pattern as the Upstash limiter
on the Next.js side).
"""

import threading
import time


class TokenBucket:
    def __init__(self, capacity: float, refill_per_sec: float):
        self.capacity = capacity
        self.refill_per_sec = refill_per_sec
        self.tokens = capacity
        self.last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill_locked(self):
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_sec)
        self.last_refill = now

    def try_consume(self, amount: float = 1.0) -> bool:
        with self._lock:
            self._refill_locked()
            if self.tokens >= amount:
                self.tokens -= amount
                return True
            return False

    def give_back(self, amount: float = 1.0):
        with self._lock:
            self.tokens = min(self.capacity, self.tokens + amount)

    def retry_after(self, amount: float = 1.0) -> float:
        with self._lock:
            self._refill_locked()
            deficit = amount - self.tokens
            return max(0.0, deficit / self.refill_per_sec)


class RateLimiter:
    """
    Defaults: each session gets a burst of 5 requests, refilling to ~1
    every 12s (5/min sustained). Global ceiling: burst of 20, refilling to
    ~1 every 3s (20/min sustained) across everyone.

    Tune these against your actual Groq plan limits — these are
    conservative starting points for a free-tier / low-traffic deployment.
    """

    def __init__(
        self,
        session_capacity: float = 5,
        session_refill_per_sec: float = 1 / 12,
        global_capacity: float = 20,
        global_refill_per_sec: float = 1 / 3,
    ):
        self._session_buckets: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()
        self._session_capacity = session_capacity
        self._session_refill = session_refill_per_sec
        self._global_bucket = TokenBucket(global_capacity, global_refill_per_sec)

    def _session_bucket(self, session_id: str) -> TokenBucket:
        with self._lock:
            bucket = self._session_buckets.get(session_id)
            if bucket is None:
                bucket = TokenBucket(self._session_capacity, self._session_refill)
                self._session_buckets[session_id] = bucket
            return bucket

    def check(self, session_id: str) -> tuple[bool, float]:
        """Returns (allowed, retry_after_seconds)."""
        if not self._global_bucket.try_consume():
            return False, self._global_bucket.retry_after()

        session_bucket = self._session_bucket(session_id)
        if not session_bucket.try_consume():
            self._global_bucket.give_back()  # don't burn the global token on a session-level reject
            return False, session_bucket.retry_after()

        return True, 0.0


# One shared instance, imported by api.py
limiter = RateLimiter()