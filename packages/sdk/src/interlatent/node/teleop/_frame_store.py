"""Thread-safe latest-frame + sticky-estop store, shared by both channels.

The staleness rule (a held target older than :data:`_FRAME_STALE_MS` is treated
as absent so a frozen browser can't keep the arm moving) and the sticky operator
e-stop latch (set at decode, cleared only by :meth:`consume_estop`, surviving
staleness and reconnects) are **safety-critical and must be byte-identical
across transports** (ADR 0016). :class:`LatestFrameStore` is the single
implementation; both `TeleopChannel` (WS) and `QuicTeleopChannel` (QUIC) inherit
it, so the two paths can never drift.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

from .frame import TeleopFrame

# How stale a held target may be before it is treated as absent. At 30 Hz the
# producer fires every ~33 ms; 250 ms covers a handful of frames of jitter
# without letting a stuck target keep the arm moving after the browser has
# actually gone silent.
_FRAME_STALE_MS = 250


class LatestFrameStore:
    """Holds the most recent teleop frame and a sticky e-stop flag.

    Thread-safe: :meth:`latest_frame` / :meth:`consume_estop` may be read from
    the control-loop thread while the receive/supervisor thread stores frames.
    Subclasses store via :meth:`_store_frame` / :meth:`_latch_estop` and drop
    the held frame on disconnect via :meth:`_drop_frame`.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: Optional[TeleopFrame] = None
        # Sticky operator e-stop: latched at decode time, cleared only by
        # consume_estop(). Deliberately survives frame staleness and channel
        # reconnects — a panic press must never be droppable (ADR 0016).
        self._estop_seen = False

    def latest_frame(self) -> Optional[TeleopFrame]:
        """Return the most recent non-stale frame, or None.

        Stale frames (> :data:`_FRAME_STALE_MS` old) are treated as absent, so a
        frozen browser can't keep the arm moving — the control loop sees None
        and falls back to policy mode automatically.
        """
        with self._lock:
            frame = self._latest
        if frame is None:
            return None
        if (time.monotonic_ns() - frame.received_at_ns) / 1e6 > _FRAME_STALE_MS:
            return None
        return frame

    def consume_estop(self) -> bool:
        """Return-and-clear the sticky operator e-stop flag.

        Latched at decode time (not subject to the staleness rule or the
        disconnect frame-drop), so an ``estop:true`` frame that arrives during a
        loop stall or right before a reconnect still reaches the control loop's
        next tick exactly once. The caller owns what "handle" means (latch the
        SafetyGate; forward a hardware latch where one exists). See ADR 0016.
        """
        with self._lock:
            seen, self._estop_seen = self._estop_seen, False
        return seen

    def _store_frame(self, frame: TeleopFrame) -> None:
        with self._lock:
            self._latest = frame

    def _latch_estop(self) -> None:
        with self._lock:
            self._estop_seen = True

    def _drop_frame(self) -> None:
        # On disconnect, drop the last known frame so a stale "engaged" target
        # doesn't keep driving the arm across the gap.
        with self._lock:
            self._latest = None


__all__ = ["LatestFrameStore"]
