"""Server-side cache of recent raw action chunks per session.

Why this exists:
    RTC in-painting needs the previous chunk's raw actions to stitch
    the next chunk onto the existing trajectory. We hold them in a
    bounded per-session deque so the next ``Infer`` can grab the
    unexecuted tail of the prior chunk.

The DRTC server runs as a single long-lived process on a persistent
GPU box (Prime Intellect, or any Linux host with a GPU), so the
buffer lives in-process — no external KV store needed. The
:class:`ChunkBuffer` protocol stays in place in case we ever want to
swap in Redis for a multi-process deployment, but there is no second
implementation today.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Protocol

import numpy as np


@dataclass
class StoredChunk:
    """One chunk of raw actions the server produced.

    `start_step` + index in `actions` gives the global action_step.
    Stored alongside the control_timestamp from the originating
    request so we can resolve LWW ordering on the server too if
    needed.
    """

    start_step: int
    control_timestamp: int
    actions: np.ndarray            # shape (chunk_size, action_dim), float32
    created_at: float              # time.time() for TTL bookkeeping


class ChunkBuffer(Protocol):
    """Per-session ring of recent StoredChunks, most-recent-last."""

    def append(self, session_id: str, chunk: StoredChunk) -> None: ...
    def recent(self, session_id: str, n: int = 4) -> list[StoredChunk]: ...
    def lookup_steps(
        self, session_id: str, start_step: int, end_step: int
    ) -> Optional[np.ndarray]:
        """Return actions covering [start_step, end_step] inclusive,
        or None if any step is missing. Used to reconstruct the
        client's recent schedule for RTC in-painting from the spans
        the client uploaded."""
        ...

    def drop(self, session_id: str) -> None: ...


class InMemoryChunkBuffer:
    """The DRTC server's only chunk buffer.

    A persistent-process deployment means a session's chunks always
    live in the same process for their entire lifetime, so this is
    enough — no shared KV store needed.
    """

    def __init__(self, max_chunks_per_session: int = 8) -> None:
        self._max = max_chunks_per_session
        self._data: dict[str, Deque[StoredChunk]] = {}

    def append(self, session_id: str, chunk: StoredChunk) -> None:
        q = self._data.setdefault(session_id, deque(maxlen=self._max))
        q.append(chunk)

    def recent(self, session_id: str, n: int = 4) -> list[StoredChunk]:
        q = self._data.get(session_id)
        if not q:
            return []
        return list(q)[-n:]

    def lookup_steps(
        self, session_id: str, start_step: int, end_step: int
    ) -> Optional[np.ndarray]:
        chunks = self.recent(session_id, n=self._max)
        return _stitch(chunks, start_step, end_step)

    def drop(self, session_id: str) -> None:
        self._data.pop(session_id, None)


def _stitch(
    chunks: list[StoredChunk], start_step: int, end_step: int
) -> Optional[np.ndarray]:
    """Concatenate stored actions covering [start_step, end_step].

    Walks chunks in order, slicing the overlap with the requested
    range. Returns None if any step in the range is uncovered — the
    server can then fall back to running inference without in-painting
    context rather than producing a stitched trajectory off bad data.
    """
    if end_step < start_step:
        return None
    needed = end_step - start_step + 1
    out_dim = None
    pieces: list[tuple[int, np.ndarray]] = []  # (offset_in_output, actions)
    for c in chunks:
        c_end = c.start_step + len(c.actions) - 1
        lo = max(start_step, c.start_step)
        hi = min(end_step, c_end)
        if lo > hi:
            continue
        sl = c.actions[lo - c.start_step : hi - c.start_step + 1]
        pieces.append((lo - start_step, sl))
        out_dim = sl.shape[1]
    if out_dim is None:
        return None
    out = np.full((needed, out_dim), np.nan, dtype=np.float32)
    for offset, sl in pieces:
        out[offset : offset + len(sl)] = sl
    if np.isnan(out).any():
        return None
    return out


def gc_inmemory(buf: InMemoryChunkBuffer, max_age_s: float = 3600.0) -> None:
    """Drop sessions whose last chunk is older than ``max_age_s``.

    Best-effort GC for sessions that never received a CloseSession
    (Pi crash, network drop) — keeps the buffer from leaking memory
    over the lifetime of the long-running server process.
    """
    now = time.time()
    dead = [
        s for s, q in buf._data.items()  # type: ignore[attr-defined]
        if q and (now - q[-1].created_at) > max_age_s
    ]
    for s in dead:
        buf.drop(s)
