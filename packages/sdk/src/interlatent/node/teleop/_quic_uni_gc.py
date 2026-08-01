"""Pure aioquic send-only-uni-stream leak GC (ADR 0020).

aioquic never collects a locally-opened send-only uni stream: its sweep needs
``QuicStream.is_finished`` (receiver AND sender), and a send-only stream's
receiver half is hardcoded un-finished, so every per-frame video stream would
otherwise linger in ``_quic._streams`` forever and the per-packet build cost
grows with session age (field signature: ADR 0020). :class:`UniStreamGC` parks
every uni stream this connection opens and discards each once its send side
acks — mirroring the relay's own per-connection discard.

Kept aioquic-free so it unit-tests without the dependency: the protocol
(:class:`~._quic_client._WTClientProtocol`) injects the two aioquic-touching
callables (``is_finished`` and ``discard``) and sweeps this on every
``transmit()``. A finished stream discards on the next sweep; a TTL-RESET
stream parks until its RESET is acked (``is_finished`` true) then discards once.
"""
from __future__ import annotations

from typing import Callable


class UniStreamGC:
    _MAX_PENDING = 256

    def __init__(
        self,
        is_finished: Callable[[int], bool],
        discard: Callable[[int], None],
    ) -> None:
        self._is_finished = is_finished
        self._discard = discard
        self._pending: "set[int]" = set()

    def add(self, sid: int) -> None:
        self._pending.add(sid)

    def sweep(self) -> None:
        if not self._pending:
            return
        for sid in [s for s in self._pending if self._is_finished(s)]:
            self._discard(sid)
            self._pending.discard(sid)
        if len(self._pending) > self._MAX_PENDING:
            # Runaway guard on a dying connection where nothing ever acks.
            self._pending = set(list(self._pending)[:self._MAX_PENDING])

    def clear(self) -> None:
        self._pending.clear()

    def pending_count(self) -> int:
        return len(self._pending)


__all__ = ["UniStreamGC"]
