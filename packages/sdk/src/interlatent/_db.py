"""CollectionDB — LatentDB facade with write buffering for SDK collection."""
from __future__ import annotations

import time

from ._schema import ActivationEvent
from ._storage import CollectionSQLiteBackend


class CollectionDB:
    """High-level write-buffered interface for activation collection."""

    def __init__(self, path: str, *, batch_size: int = 400) -> None:
        self._store = CollectionSQLiteBackend(path)
        self._batch_size = batch_size
        self._buffer: list[ActivationEvent] = []
        self._last_flush = time.perf_counter()

    def write_event(self, event: ActivationEvent) -> None:
        """Buffer an event, flushing when batch_size is reached."""
        self._buffer.append(event)
        if len(self._buffer) >= self._batch_size:
            self._flush_buffer()

    def update_step_contexts(self, *, contexts: dict[int, dict]) -> None:
        if not contexts:
            return
        self._flush_buffer()
        self._store.update_step_contexts(contexts=contexts)

    def flush(self) -> None:
        self._flush_buffer()
        self._store.flush()

    def close(self) -> None:
        self._flush_buffer()
        self._store.close()

    def _flush_buffer(self) -> None:
        if not self._buffer:
            return
        self._store.write_events(self._buffer)
        self._buffer.clear()
        self._last_flush = time.perf_counter()

    def __enter__(self) -> "CollectionDB":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
