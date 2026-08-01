"""Shared teleop arrival telemetry.

The QUIC channel (`quic_channel.QuicTeleopChannel`) tracks the inter-arrival
gaps of target frames — the node-observable half of teleop latency (relay→node
jitter and any in-browser-solver stalls). :class:`ArrivalTracker` owns the
accounting and hands the caller a summary dict once per window; the caller
formats and logs it.
"""
from __future__ import annotations

import time
from typing import Optional


class ArrivalTracker:
    """Rolling inter-arrival gap accounting for teleop frames.

    Call :meth:`note` once per accepted frame. It returns ``None`` most ticks
    and a summary dict once ``window_s`` has elapsed (then resets the window):
    ``{elapsed, n, rate_hz, gap_mean_ms, gap_max_ms, seq_span}``. ``seq_span``
    vs ``n`` ≈ frames the relay/dedupe collapsed or lost en route (so
    ``span > n`` is normal under duplication). Single-threaded: call from one
    thread (each channel's receive/supervisor thread). Pure + unit-tested.
    """

    def __init__(self, window_s: float) -> None:
        self._window_s = window_s
        self._count = 0
        self._last_ns = 0
        self._gap_sum_ms = 0.0
        self._gap_max_ms = 0.0
        self._seq_first = 0
        self._seq_last = 0
        self._window_started = time.monotonic()

    def note(self, frame) -> Optional[dict]:
        now_ns = frame.received_at_ns
        if self._count == 0:
            self._seq_first = frame.seq
        else:
            gap_ms = (now_ns - self._last_ns) / 1e6
            self._gap_sum_ms += gap_ms
            if gap_ms > self._gap_max_ms:
                self._gap_max_ms = gap_ms
        self._last_ns = now_ns
        self._seq_last = frame.seq
        self._count += 1

        now = time.monotonic()
        elapsed = now - self._window_started
        if elapsed < self._window_s:
            return None
        n = self._count
        gaps = n - 1
        summary = {
            "elapsed": elapsed,
            "n": n,
            "rate_hz": n / elapsed if elapsed > 0 else 0.0,
            "gap_mean_ms": (self._gap_sum_ms / gaps) if gaps > 0 else 0.0,
            "gap_max_ms": self._gap_max_ms,
            "seq_span": self._seq_last - self._seq_first,
        }
        self._count = 0
        self._gap_sum_ms = 0.0
        self._gap_max_ms = 0.0
        self._window_started = now
        return summary


__all__ = ["ArrivalTracker"]
