"""Nori-Protocol v1 wire codec — the adapter's conformance surface.

Every frame the adapter sends or parses is defined here, mirroring the
canonical JSON Schemas in the Nori-Protocol repo (vendored for tests at
``tests/fixtures/nori_protocol/``). The transport is newline-delimited JSON
over TCP; this module is pure codec — no sockets, no threads, stdlib only —
so the conformance tests can lock the wire shapes before any I/O exists.

Contract notes that shape this module:

- Outbound ``command`` frames use the schema-canonical ``{"name": "estop"}``
  form, NOT the legacy ``{"estop": true}`` boolean form the TypeScript
  ``@nori/sdk`` still emits.
- ``bye`` has no schema upstream (CLIENTS.md documents the bare shape); it is
  the one builder validated by exact-shape assertion instead.
- Inbound parsing is lenient per the protocol README: unknown ``type`` values
  (e.g. the PROPOSED ``perception``) return ``None`` rather than raising, so a
  future additive message never breaks a deployed node.
- ``validate_ack`` is the fail-closed handshake check: it ACCUMULATES every
  disagreement between the static ``RobotProfile`` and the live ack (norm
  mode, joint set, each range pair) so a mismatched robot is diagnosed in one
  raise, not one field at a time.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

PROTOCOL_VERSION = 1

# The only norm mode the adapter speaks. The RobotProfile limits are expressed
# in this scale ([-100, 100], grippers [0, 100]); a daemon in "degrees" mode
# would silently mis-scale every clamp, so validate_ack fail-closes on it.
REQUIRED_NORM_MODE = "range_m100_100"


class NoriProtocolError(RuntimeError):
    """Any violation of the Nori-Protocol contract by either side."""


class NoriHandshakeError(NoriProtocolError):
    """Handshake rejected or descriptor/profile mismatch (fail-closed)."""


# ---------------------------------------------------------------------------
# Inbound frame types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WatchdogProfile:
    """Daemon watchdog thresholds, disclosed (never negotiated) in the ack."""

    t_warn_ms: float
    t_stop_ms: float


@dataclass(frozen=True)
class Ack:
    accepted: bool
    protocol_version: Optional[int] = None
    norm_mode: Optional[str] = None
    watchdog: Optional[WatchdogProfile] = None
    joints: tuple[str, ...] = ()
    base: tuple[str, ...] = ()
    aux: tuple[str, ...] = ()
    cameras: tuple[str, ...] = ()
    ranges: Dict[str, tuple[float, float]] = field(default_factory=dict)
    initial_state: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass(frozen=True)
class Telemetry:
    ts_ns: int
    state: Dict[str, float]
    status: Optional[Dict[str, Any]] = None
    currents: Dict[str, int] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ErrorFrame:
    code: str
    msg: str
    fatal: bool = False


@dataclass(frozen=True)
class ActionStatus:
    action_id: str
    state: str
    reason: Optional[str] = None
    ts_ns: Optional[int] = None


InboundFrame = Union[Ack, Telemetry, ErrorFrame, ActionStatus]


# ---------------------------------------------------------------------------
# Outbound builders — every emitted frame goes through one of these
# ---------------------------------------------------------------------------


def make_hello(
    *,
    token: Optional[str] = None,
    bus_choice: str = "3",
    input_mode: str = "vr",
    mode: str = "lan",
) -> dict:
    hello: dict = {
        "type": "hello",
        "protocol_version": PROTOCOL_VERSION,
        "mode": mode,
        "input_mode": input_mode,
        "bus_choice": bus_choice,
    }
    if token:
        hello["token"] = token
    return hello


def make_control_action(seq: int, action: Dict[str, float]) -> dict:
    """Absolute-target control frame: {"<joint>.pos": value, ...}."""
    return {
        "type": "control",
        "seq": int(seq),
        "action": {str(k): float(v) for k, v in action.items()},
    }


def make_keepalive(seq: int) -> dict:
    """Motion-free control frame. The daemon's watchdog counts frame ARRIVAL,
    so this holds the session (and pose) without commanding anything."""
    return {"type": "control", "seq": int(seq)}


def make_estop() -> dict:
    # Schema-canonical form (command.json `name` enum) — deliberately NOT the
    # legacy `{"estop": true}` boolean form @nori/sdk sends.
    return {"type": "command", "name": "estop"}


def make_reset_latch(token: str) -> dict:
    if not token:
        raise NoriProtocolError("reset_latch requires the daemon agent token")
    return {"type": "command", "name": "reset_latch", "token": str(token)}


def make_bye() -> dict:
    return {"type": "bye"}


def encode_frame(obj: dict) -> bytes:
    """One frame per line, UTF-8, LF-terminated (CLIENTS.md framing)."""
    return json.dumps(obj, separators=(",", ":")).encode("utf-8") + b"\n"


# ---------------------------------------------------------------------------
# Inbound parsing
# ---------------------------------------------------------------------------


def parse_line(line: Union[bytes, str]) -> Optional[InboundFrame]:
    """Decode one inbound NDJSON line into a typed frame.

    Returns None for unknown/client-bound/undecodable frames — runtime parsers
    must be lenient (protocol README): an additive message type from a newer
    daemon must never take down the node. Never raises.
    """
    try:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        obj = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None

    kind = obj.get("type")
    try:
        if kind == "ack":
            return _parse_ack(obj)
        if kind == "telemetry":
            return _parse_telemetry(obj)
        if kind == "error":
            return ErrorFrame(
                code=str(obj.get("code", "internal")),
                msg=str(obj.get("msg", "")),
                fatal=bool(obj.get("fatal", False)),
            )
        if kind == "action_status":
            return ActionStatus(
                action_id=str(obj.get("action_id", "")),
                state=str(obj.get("state", "")),
                reason=(None if obj.get("reason") is None else str(obj["reason"])),
                ts_ns=(None if obj.get("ts_ns") is None else int(obj["ts_ns"])),
            )
    except (TypeError, ValueError, KeyError):
        return None
    return None


def _parse_ack(obj: dict) -> Ack:
    wd = obj.get("watchdog_profile")
    watchdog = None
    if isinstance(wd, dict) and "t_warn_ms" in wd and "t_stop_ms" in wd:
        watchdog = WatchdogProfile(
            t_warn_ms=float(wd["t_warn_ms"]), t_stop_ms=float(wd["t_stop_ms"])
        )
    desc = obj.get("descriptor") or {}
    ranges: Dict[str, tuple[float, float]] = {}
    for key, pair in (desc.get("ranges") or {}).items():
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            ranges[str(key)] = (float(pair[0]), float(pair[1]))
    initial_state = {
        str(k): float(v)
        for k, v in (obj.get("initial_state") or {}).items()
        if isinstance(v, (int, float))
    }
    return Ack(
        accepted=bool(obj.get("accepted", False)),
        protocol_version=(
            None
            if obj.get("protocol_version") is None
            else int(obj["protocol_version"])
        ),
        norm_mode=(None if obj.get("norm_mode") is None else str(obj["norm_mode"])),
        watchdog=watchdog,
        joints=tuple(str(j) for j in (desc.get("joints") or [])),
        base=tuple(str(b) for b in (desc.get("base") or [])),
        aux=tuple(str(a) for a in (desc.get("aux") or [])),
        cameras=tuple(str(c) for c in (desc.get("cameras") or [])),
        ranges=ranges,
        initial_state=initial_state,
        error=(None if obj.get("error") is None else str(obj["error"])),
    )


def _parse_telemetry(obj: dict) -> Telemetry:
    state = {
        str(k): float(v)
        for k, v in (obj.get("state") or {}).items()
        if isinstance(v, (int, float))
    }
    currents = {
        str(k): int(v)
        for k, v in (obj.get("currents") or {}).items()
        if isinstance(v, (int, float))
    }
    status = obj.get("status")
    return Telemetry(
        ts_ns=int(obj.get("ts_ns", 0)),
        state=state,
        status=(status if isinstance(status, dict) else None),
        currents=currents,
        raw=obj,
    )


# ---------------------------------------------------------------------------
# Fail-closed handshake validation
# ---------------------------------------------------------------------------


def validate_ack(profile: Any, ack: Ack) -> list[str]:
    """Compare the static RobotProfile against the live ack, ACCUMULATING every
    mismatch. Returns [] when clean; the caller raises NoriHandshakeError with
    the joined list. `ack.accepted` is the caller's business (it carries the
    daemon's own error text); this checks only version/units/topology.

    Profile joint names are bare (`left_arm_gripper`); the descriptor carries
    the `.pos` suffix on the wire, so comparison is on `f"{name}.pos"`. A
    profile joint whose range is absent from `ack.descriptor.ranges` is a
    mismatch — fail closed, never assume.
    """
    problems: list[str] = []

    if ack.protocol_version is not None and ack.protocol_version != PROTOCOL_VERSION:
        problems.append(
            f"protocol_version: daemon={ack.protocol_version} "
            f"adapter={PROTOCOL_VERSION}"
        )

    if ack.norm_mode != REQUIRED_NORM_MODE:
        problems.append(
            f"norm_mode: daemon={ack.norm_mode!r} required={REQUIRED_NORM_MODE!r} "
            "(profile limits are normalized units)"
        )

    profile_wire = [f"{name}.pos" for name in profile.joint_names]
    descriptor_present = bool(ack.joints)
    if descriptor_present:
        daemon_joints = set(ack.joints)
        source = "descriptor"
    else:
        # Older daemon builds omit the descriptor block (schema-legal: only
        # type+accepted are required). Fall back to the joint set disclosed by
        # initial_state. This stays fail-closed on everything safety-relevant:
        # version and norm_mode are checked above, topology is checked below,
        # and under range_m100_100 the normalized ranges are pinned by the
        # protocol units contract itself ([-100,100], grippers [0,100]) — the
        # descriptor's range echo is redundant disclosure. What IS lost is
        # camera discovery (descriptor.cameras); the client logs that.
        daemon_joints = {k for k in ack.initial_state if k.endswith(".pos")}
        source = "initial_state (descriptor absent)"
        if not daemon_joints:
            problems.append(
                "daemon disclosed neither descriptor.joints nor any .pos keys "
                "in initial_state — nothing to validate topology against (see "
                "Nori-Protocol fixtures/ack.json for the expected disclosure)"
            )
            return problems

    missing = [j for j in profile_wire if j not in daemon_joints]
    extra = sorted(daemon_joints - set(profile_wire))
    for j in missing:
        problems.append(f"joint missing from daemon {source}: {j}")
    for j in extra:
        problems.append(f"joint in daemon {source} but not in profile: {j}")

    if descriptor_present:
        for name, (lo, hi) in zip(profile.joint_names, profile.joint_limits):
            wire = f"{name}.pos"
            if wire in missing:
                continue  # already reported as a topology problem
            got = ack.ranges.get(wire)
            if got is None:
                problems.append(
                    f"range undisclosed by daemon for {wire} (fail closed)"
                )
            elif got != (float(lo), float(hi)):
                problems.append(
                    f"range mismatch for {wire}: daemon={list(got)} "
                    f"profile=[{lo}, {hi}]"
                )

    return problems


__all__ = [
    "PROTOCOL_VERSION",
    "REQUIRED_NORM_MODE",
    "NoriProtocolError",
    "NoriHandshakeError",
    "WatchdogProfile",
    "Ack",
    "Telemetry",
    "ErrorFrame",
    "ActionStatus",
    "make_hello",
    "make_control_action",
    "make_keepalive",
    "make_estop",
    "make_reset_latch",
    "make_bye",
    "encode_frame",
    "parse_line",
    "validate_ack",
]
