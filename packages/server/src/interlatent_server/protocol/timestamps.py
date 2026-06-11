"""Monotonic control-timestamp helpers.

DRTC's LWW merge needs a clock that:
  - never goes backwards under any circumstance (clock skew, restarts,
    re-entrant calls)
  - is comparable across messages from the same client

We use a per-client monotonic counter seeded from time.monotonic_ns().
This is intentionally not a wall clock: timestamps only need to be
totally-ordered within one session.
"""

from __future__ import annotations

import threading
import time


class ControlClock:
    """Monotonic, per-client control-timestamp source.

    Returns strictly-increasing uint64 values. Safe for concurrent use.
    """

    __slots__ = ("_lock", "_last")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last = time.monotonic_ns()

    def tick(self) -> int:
        with self._lock:
            now = time.monotonic_ns()
            if now <= self._last:
                now = self._last + 1
            self._last = now
            return now
