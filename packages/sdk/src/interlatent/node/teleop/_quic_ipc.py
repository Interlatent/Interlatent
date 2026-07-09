"""Loopback UDP protocol between QuicTeleopChannel and its child process.

The QUIC connection runs in a dumb-pipe child process (``_quic_proc``) so the
robot process's GIL contention can't starve aioquic's handshake/loss timers
(see the ADR 0017 amendment). Parent and child exchange datagrams over a
loopback UDP socket pair using the 1-byte-type framing defined here — kept in
one aioquic-free module so both sides (and the unit tests) share a single
source of truth.

Framing (both directions): ``type_byte + payload``.
  * ``TYPE_DATA`` — payload is a raw relay datagram, verbatim. Parent→child:
    a state datagram to ship to the relay. Child→parent: a target datagram
    received from the relay. All protocol logic (codec, dedupe, pacing) stays
    in the parent; the child never inspects DATA payloads.
  * ``TYPE_CTRL`` — payload is one UTF-8 JSON object. Child→parent only
    (the parent's control plane toward the child is its stdin pipe):
      {"t": "hello", "cookie": <hex>, "pid": <int>}   bind + 1s heartbeat
      {"t": "connected"}                              WT CONNECT accepted
      {"t": "disconnected", "reason": <str>}          session ended (best-effort)

The parent pins the child's address from the first hello whose ``cookie``
matches the one it passed via env, and drops datagrams from anyone else.
"""
from __future__ import annotations

import json
from typing import Optional, Tuple

TYPE_DATA = 0x00
TYPE_CTRL = 0x01

# Loopback socket buffers, both ends. The parent's reader thread is still
# GIL-exposed; at ~60 Hz dup'd ~1.2 KB datagrams this absorbs >1 s of stall
# kernel-side instead of dropping.
SOCK_BUF_BYTES = 262144

# Spawn contract: env vars the parent sets on the child (full os.environ is
# inherited underneath, so e.g. INTERLATENT_TELEOP_INSECURE keeps working).
ENV_PARENT_PORT = "INTERLATENT_QUIC_PROC_PARENT_PORT"
ENV_COOKIE = "INTERLATENT_QUIC_PROC_COOKIE"
ENV_API_BASE = "INTERLATENT_QUIC_PROC_API_BASE"
ENV_API_KEY = "INTERLATENT_QUIC_PROC_API_KEY"
ENV_SESSION_ID = "INTERLATENT_QUIC_PROC_SESSION_ID"
ENV_TOKEN_PATH = "INTERLATENT_QUIC_PROC_TOKEN_PATH"
ENV_BYPASS_KEY = "INTERLATENT_QUIC_PROC_BYPASS_KEY"


def encode_data(payload: bytes) -> bytes:
    return bytes((TYPE_DATA,)) + payload


def encode_ctrl(obj: dict) -> bytes:
    return bytes((TYPE_CTRL,)) + json.dumps(obj).encode("utf-8")


def parse(datagram: bytes) -> Optional[Tuple[int, bytes]]:
    """Split a loopback datagram into (type, payload). None on empty input;
    unknown type bytes are returned as-is for the caller to ignore."""
    if not datagram:
        return None
    return datagram[0], datagram[1:]


def parse_ctrl(payload: bytes) -> Optional[dict]:
    """Decode a TYPE_CTRL payload. None on garbage (never raises)."""
    try:
        obj = json.loads(payload.decode("utf-8"))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


__all__ = [
    "TYPE_DATA",
    "TYPE_CTRL",
    "SOCK_BUF_BYTES",
    "ENV_PARENT_PORT",
    "ENV_COOKIE",
    "ENV_API_BASE",
    "ENV_API_KEY",
    "ENV_SESSION_ID",
    "ENV_TOKEN_PATH",
    "ENV_BYPASS_KEY",
    "encode_data",
    "encode_ctrl",
    "parse",
    "parse_ctrl",
]
