# Copied from Interlatent-Main teleop-quic-relay/relay_core.py @ d5b7a1c0
# (2026-07-22). Upstream is the monorepo copy; sync fixes both ways.
"""Transport-agnostic pairing + routing core for the QUIC teleop relay.

The datagram forwarding rule is deliberately dumb: whatever a browser sends is
forwarded verbatim to the paired node and vice-versa. Deduplication (the
browser duplicates each target datagram 2–3×) and staleness live at the
endpoints, not here — the relay only rendezvouses the two sides by session id
and copies bytes. This mirrors the WS relay's ``_Pair``/``_attach``/``_detach``
(engine ``teleop_relay.py``), minus the WS-specific bits, so the logic is
unit-testable without aioquic.

The same dumbness applies to the video tee's WebTransport **unidirectional
streams** (one short-lived stream per JPEG frame, node→browser):
:class:`StreamForwardTable` maps each incoming uni stream to a lazily-opened
outgoing uni stream on the paired connection and bounds the copying (stream
count + per-stream bytes) so a misbehaving sender can't grow relay memory.
The relay never parses stream payloads.

A "peer" is any opaque handle the transport layer hands us (an aioquic
WebTransport session id in ``server.py``); this module never touches the wire.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

VALID_ROLES = ("browser", "node")

# The synthetic frame the WS relay sends the node when the browser drops, so
# the control loop falls back to policy/hold. Kept byte-identical here.
BROWSER_CLOSED_FRAME = b'{"engaged": false, "reason": "browser_closed"}'

P = TypeVar("P")


def parse_teleop_path(path: str) -> Optional[tuple[str, str]]:
    """``/teleop/<role>/<session_id>[?query]`` -> ``(role, session_id)`` or None.

    Query string (``?token=…``) is stripped by the caller; this accepts either.
    """
    p = path.split("?", 1)[0].strip("/")
    parts = p.split("/")
    if len(parts) != 3 or parts[0] != "teleop":
        return None
    role, session_id = parts[1], parts[2]
    if role not in VALID_ROLES or not session_id:
        return None
    return role, session_id


@dataclass
class _Pair(Generic[P]):
    browser: Optional[P] = None
    node: Optional[P] = None


class Registry(Generic[P]):
    """Session-id → {browser, node} peer pairing. Thread-safe.

    ``attach`` returns any prior peer on the same side that was superseded
    (the transport should close it, code 4000). ``peer_of`` gives the forward
    target for a received datagram. ``detach`` returns the surviving peer (to
    notify on browser close) and drops empty pairs.
    """

    def __init__(self) -> None:
        self._pairs: dict[str, _Pair[P]] = {}
        self._lock = threading.Lock()

    def attach(self, session_id: str, role: str, peer: P) -> Optional[P]:
        with self._lock:
            pair = self._pairs.setdefault(session_id, _Pair())
            prior = getattr(pair, role)
            setattr(pair, role, peer)
            return prior if prior is not None else None

    def peer_of(self, session_id: str, role: str) -> Optional[P]:
        other = "node" if role == "browser" else "browser"
        with self._lock:
            pair = self._pairs.get(session_id)
            return getattr(pair, other) if pair else None

    def detach(self, session_id: str, role: str, peer: P) -> Optional[P]:
        """Clear ``peer`` from its side (only if still current) and return the
        other side's peer. Drops the pair when both sides are gone."""
        with self._lock:
            pair = self._pairs.get(session_id)
            if pair is None:
                return None
            if getattr(pair, role) is peer:
                setattr(pair, role, None)
            other = "node" if role == "browser" else "browser"
            survivor = getattr(pair, other)
            if pair.browser is None and pair.node is None:
                self._pairs.pop(session_id, None)
            return survivor

    def session_count(self) -> int:
        with self._lock:
            return len(self._pairs)


# Per-connection bounds for forwarded uni streams (video tee). The node sends
# ~10 Hz × cams ≤ ~15 KB frames with its own in-flight cap of 6, so these are
# generous headroom, not tuning knobs: hitting them means a misbehaving peer.
MAX_FORWARDS_PER_CONN = 8
MAX_FORWARD_STREAM_BYTES = 262144
MAX_DROPPED_TRACKED = 1024

R = TypeVar("R")


class StreamForwardTable(Generic[R]):
    """Incoming-uni-stream → outgoing-handle map for one QUIC connection.

    ``out_ref`` is opaque to this module (``server.py`` stores
    ``(dest_peer, out_stream_id)``). Not locked: aioquic delivers one
    connection's events on a single loop thread. ``dropped`` remembers
    streams we refused/killed so their remaining chunks are discarded
    without re-deciding; it is FIFO-pruned so it can't grow unbounded.
    """

    def __init__(self) -> None:
        self._active: "dict[int, list]" = {}  # in_sid -> [out_ref, bytes_so_far]
        self._dropped: "dict[int, None]" = {}  # insertion-ordered FIFO set

    def get(self, in_sid: int) -> Optional[R]:
        entry = self._active.get(in_sid)
        return entry[0] if entry is not None else None

    def can_open(self) -> bool:
        return len(self._active) < MAX_FORWARDS_PER_CONN

    def open(self, in_sid: int, out_ref: R) -> None:
        self._active[in_sid] = [out_ref, 0]

    def note_bytes(self, in_sid: int, n: int) -> bool:
        """Account ``n`` payload bytes; False when the stream is over budget
        (caller kills both halves)."""
        entry = self._active.get(in_sid)
        if entry is None:
            return False
        entry[1] += n
        return entry[1] <= MAX_FORWARD_STREAM_BYTES

    def remove(self, in_sid: int) -> Optional[R]:
        """Forget a mapping (finished or reset); returns its out_ref."""
        entry = self._active.pop(in_sid, None)
        return entry[0] if entry is not None else None

    def is_dropped(self, in_sid: int) -> bool:
        return in_sid in self._dropped

    def drop(self, in_sid: int) -> None:
        self._dropped[in_sid] = None
        while len(self._dropped) > MAX_DROPPED_TRACKED:
            self._dropped.pop(next(iter(self._dropped)))

    def forget_dropped(self, in_sid: int) -> None:
        self._dropped.pop(in_sid, None)

    def drain(self) -> "list[R]":
        """Teardown: forget everything, returning the out_refs so the caller
        can RESET the outgoing halves."""
        refs = [entry[0] for entry in self._active.values()]
        self._active.clear()
        self._dropped.clear()
        return refs

    def active_count(self) -> int:
        return len(self._active)
