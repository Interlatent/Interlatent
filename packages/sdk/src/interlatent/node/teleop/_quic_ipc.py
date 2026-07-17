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
  * ``TYPE_VIDEO`` — parent→child only: one framed video frame to ship to the
    browser on its own WebTransport unidirectional stream. Payload is
    ``uint8 cam-name length + cam name UTF-8 + wire bytes``; the cam prefix
    lets the child enforce its per-camera in-flight cap without parsing the
    wire bytes (which stay opaque, like DATA payloads). One loopback datagram
    per camera per preview tick (~8-15 KB JPEG, well under the 64 KB limit).
  * ``TYPE_SPEC`` — parent→child only: the framed kinematic_spec to ship to the
    browser on its own uni stream, in answer to a browser ``request_spec``
    datagram (QUIC path: the browser builds its IK solver from the node's
    installed robot data instead of the platform backend). Payload is the
    browser-facing wire bytes verbatim (same ``uint16 header + JSON header +
    body`` envelope as video, header ``type:"spec"``), opaque to the child.
    Unlike video it is NOT governed/shed — it is one-shot and load-bearing, so
    the child opens its stream unconditionally.

The parent pins the child's address from the first hello whose ``cookie``
matches the one it passed via env, and drops datagrams from anyone else.
"""
from __future__ import annotations

import json
from typing import Optional, Tuple

TYPE_DATA = 0x00
TYPE_CTRL = 0x01
TYPE_VIDEO = 0x02
TYPE_SPEC = 0x03

# Loopback socket buffers, both ends. The parent's reader thread is still
# GIL-exposed; sized so control datagrams (~60 Hz dup'd ~1.2 KB) plus the
# video tee (10 Hz × cams × ~15 KB ≈ 300 KB/s on a 2-cam rig) absorb a
# multi-second stall kernel-side instead of dropping.
SOCK_BUF_BYTES = 1048576

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


def encode_video(cam: str, wire: bytes) -> bytes:
    """Frame one video frame for the child: cam-name prefix + opaque wire
    bytes (the browser-facing framed payload, sent verbatim on a uni stream)."""
    name = cam.encode("utf-8")[:255]
    return bytes((TYPE_VIDEO, len(name))) + name + wire


def parse_video(payload: bytes) -> Optional[Tuple[str, bytes]]:
    """Decode a TYPE_VIDEO payload into (cam, wire). None on garbage —
    truncated, empty cam name, or undecodable UTF-8 (never raises)."""
    if len(payload) < 2:
        return None
    name_len = payload[0]
    if name_len == 0 or len(payload) < 1 + name_len:
        return None
    try:
        cam = payload[1:1 + name_len].decode("utf-8")
    except UnicodeDecodeError:
        return None
    wire = payload[1 + name_len:]
    if not wire:
        return None
    return cam, wire


def encode_spec(wire: bytes) -> bytes:
    """Frame the kinematic_spec wire bytes for the child (TYPE_SPEC). ``wire``
    is the browser-facing envelope, sent verbatim on a uni stream — opaque
    here, exactly like a video frame's wire bytes."""
    return bytes((TYPE_SPEC,)) + wire


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
    "TYPE_VIDEO",
    "TYPE_SPEC",
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
    "encode_video",
    "encode_spec",
    "parse",
    "parse_ctrl",
    "parse_video",
]
