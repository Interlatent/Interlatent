"""LWW (Last-Write-Wins) action schedule.

The DRTC paper models the client's action queue as a CRDT: each
action is keyed by (action_step) and carries a control_timestamp.
When two messages arrive for the same step, the one with the larger
control_timestamp wins. The semilattice join is therefore associative
+ commutative + idempotent, which is what lets the network drop,
duplicate, or reorder messages without breaking trajectory continuity.

Concretely we keep a dict {action_step -> (control_timestamp, vector)}.
Pop returns the next-due action in step order. Old entries are
trimmed once the executor has consumed past them.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

import numpy as np


@dataclass
class ScheduledAction:
    action_step: int
    control_timestamp: int
    vector: np.ndarray


class ActionSchedule:
    def __init__(self) -> None:
        self._data: dict[int, ScheduledAction] = {}
        self._cursor: int = 0  # next step to execute
        self._lock = threading.Lock()

    # --- merge ---------------------------------------------------------

    def merge(self, actions: Iterable[ScheduledAction]) -> int:
        """Apply LWW for each incoming action. Returns count actually
        installed (i.e. with a strictly newer control_timestamp)."""
        installed = 0
        with self._lock:
            for a in actions:
                if a.action_step < self._cursor:
                    continue  # already executed; ignore
                cur = self._data.get(a.action_step)
                if cur is None or a.control_timestamp > cur.control_timestamp:
                    self._data[a.action_step] = a
                    installed += 1
        return installed

    # --- consume -------------------------------------------------------

    def pop_next(self) -> Optional[ScheduledAction]:
        """Pop and return the single action at the executor cursor,
        advancing the cursor by one.

        Returns None if the cursor step is not scheduled (cold start,
        or the queue drained faster than chunks arrived). The cursor is
        NOT advanced in that case — the rollout simply pauses until the
        next chunk fills the gap, so no actions are ever skipped.

        This one-action-per-tick contract is what makes the robot move
        smoothly: every step of every chunk is executed exactly once.
        """
        with self._lock:
            a = self._data.pop(self._cursor, None)
            if a is not None:
                self._cursor += 1
            return a

    def pop_due(self, up_to_step: int) -> list[ScheduledAction]:
        """Return all scheduled actions with step <= up_to_step, in
        order, and advance the executor cursor past them.

        Retained for introspection/tests. The control loop uses
        ``pop_next()`` — popping the whole backlog at once and only
        executing the last action drops the rest of the chunk."""
        out: list[ScheduledAction] = []
        with self._lock:
            steps = sorted(s for s in self._data if s <= up_to_step)
            for s in steps:
                out.append(self._data.pop(s))
            if out:
                self._cursor = max(self._cursor, out[-1].action_step + 1)
        return out

    # --- introspection -------------------------------------------------

    def queue_depth(self) -> int:
        with self._lock:
            return len(self._data)

    def next_action_step(self) -> int:
        """First step the server should produce on the next request —
        the current executor cursor (REPLACE / overlap mode).

        Do NOT append (anchor past the end of the queue). A policy
        chunk is "given what I observe NOW, do these N actions"; if it
        is queued behind a deep backlog it executes 1-2 s later from a
        stale robot state, so consecutive stale plans fight each other
        and the robot twitches in place instead of moving. Anchoring at
        the cursor means each chunk overwrites the queue with FRESH
        predictions (LWW: newer control_timestamp wins; already-executed
        steps are dropped by `merge`).

        Sustainability: replace mode nets ~(chunk_size - latency_steps)
        fresh actions per inference, so it needs chunk_size to comfortably
        exceed the latency — hence the chunk_size=50 default. At ~650 ms
        inference (~20 steps) that leaves ~30 steps of runway per cycle.
        Faster inference widens the margin; RTC in-painting (when
        re-enabled) smooths the overwrite seam."""
        with self._lock:
            return self._cursor

    def scheduled_spans(self) -> list[tuple[int, int]]:
        """Compact run-length representation of {scheduled steps}.

        We send this up with each Observation so the server can pull
        in-painting context for exactly the range we already have.
        """
        with self._lock:
            if not self._data:
                return []
            steps = sorted(self._data)
        spans: list[tuple[int, int]] = []
        run_lo = run_hi = steps[0]
        for s in steps[1:]:
            if s == run_hi + 1:
                run_hi = s
            else:
                spans.append((run_lo, run_hi))
                run_lo = run_hi = s
        spans.append((run_lo, run_hi))
        return spans

    def cursor(self) -> int:
        with self._lock:
            return self._cursor

    def flush(self) -> int:
        """Drop every queued action without advancing the cursor.

        The cursor stays where it is so the next Infer chunk anchors at
        the same step the policy was about to execute. Returns the
        number of dropped entries (for telemetry).
        """
        with self._lock:
            n = len(self._data)
            self._data.clear()
            return n

    def __iter__(self) -> Iterator[ScheduledAction]:
        with self._lock:
            return iter(sorted(self._data.values(), key=lambda a: a.action_step))
