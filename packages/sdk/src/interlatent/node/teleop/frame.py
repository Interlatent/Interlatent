"""The teleop wire frame — the authoritative node-side parser.

A teleop frame is the JSON message a *producer* (the dashboard keyboard overlay
or the VR bridge) sends over the WebSocket relay to the node. The node owns this
parser; each producer re-implements the encoder against this contract (the
contract is duplicated across the WS boundary by design — see ADR 0009 — so the
node never depends on a producer package).

Three modes exist on the wire; the node executes two of them
(`control.py`):

- ``mode="keys"``    — ``held_keys`` is a set of currently-held key strings; the
                       node integrates them into an absolute joint target
                       (``keyboard.next_target``). The dashboard keyboard overlay.
- ``mode="pose"``    — ``ee_pos``/``ee_quat`` is an absolute 6-DoF end-effector
                       TARGET in the arm-base frame (browser clutch mapper
                       output) and ``pinch`` a gripper close amount. Consumed by
                       the POD-side retarget stage in the relay, which solves IK
                       and forwards ``mode="targets"`` — a pose frame should
                       never reach the node (ADR 0009, second amendment); if one
                       does, the node holds pose.
- ``mode="targets"`` — ``joint_targets`` is an absolute joint-target vector
                       (``action_features`` order, robot-native units) the
                       producer or the pod retarget stage already computed.
                       Routed through the SafetyGate and executed.

Back-compat: a frame with no ``mode`` is treated as ``"keys"``, so the existing
overlay (which sends no ``mode``) keeps working untouched.

``estop`` (additive, default False) is the operator's HARD stop — orthogonal to
``deadman``, whose release is a soft hold. On receipt the control loop latches
the SafetyGate (and a robot with a hardware latch, e.g. Nori, forwards it);
clearing is an explicit human act, never the loop's. Because a single estop
datagram must survive the 250 ms staleness window and channel reconnects, both
channels also latch a sticky ``estop_seen`` flag at decode time — see
``consume_estop()`` — so a panic press can never be lost to frame freshness
rules (ADR 0016).
"""
from __future__ import annotations

import json
import struct
import time
from dataclasses import dataclass, field
from typing import Optional

_VALID_MODES = ("keys", "pose", "targets")


def frame_with_header(header: dict, body: bytes) -> bytes:
    """The node→browser length-prefixed wire envelope, shared by every hop that
    ships JSON-header + opaque body on one stream: ``uint16-BE header length +
    UTF-8 JSON header + body``. Used for the WS preview tee, the QUIC video
    tee, and the QUIC kinematic_spec — the header ``type`` distinguishes them,
    so the browser's inbound reader is one parser. Pure + unit-tested."""
    head = json.dumps(header).encode("utf-8")
    return struct.pack(">H", len(head)) + head + body


@dataclass
class TeleopFrame:
    """Decoded WS frame from a producer.

    ``engaged`` decides whether the control loop overrides the policy.
    ``deadman``, when False, forces engaged=False at the gate.
    ``mode`` selects how the node derives the joint target this tick.
    """
    engaged: bool
    deadman: bool
    seq: int
    received_at_ns: int            # monotonic_ns at decode time
    mode: str = "keys"
    estop: bool = False            # operator hard stop (latches; human-cleared)
    held_keys: set[str] = field(default_factory=set)
    joint_targets: Optional[list[float]] = None
    ee_pos: Optional[list[float]] = None       # mode="pose": [x, y, z] (meters, WebXR frame)
    ee_quat: Optional[list[float]] = None       # mode="pose": [x, y, z, w] (optional)
    pinch: float = 0.0                          # mode="pose": 0 open .. 1 closed
    confidence: float = 1.0

    @classmethod
    def from_json(cls, raw: str) -> "Optional[TeleopFrame]":
        try:
            obj = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(obj, dict):
            return None

        mode = str(obj.get("mode") or "keys").lower()
        if mode not in _VALID_MODES:
            mode = "keys"  # unknown mode → safe default

        held_keys = {
            str(k).lower()
            for k in (obj.get("held_keys") or [])
            if isinstance(k, str)
        }

        joint_targets: Optional[list[float]] = None
        raw_targets = obj.get("joint_targets")
        if isinstance(raw_targets, (list, tuple)):
            try:
                joint_targets = [float(x) for x in raw_targets]
            except (TypeError, ValueError):
                joint_targets = None

        ee_pos = _float_list(obj.get("ee_pos"), 3)
        ee_quat = _float_list(obj.get("ee_quat"), 4)

        try:
            pinch = float(obj.get("pinch", 0.0))
        except (TypeError, ValueError):
            pinch = 0.0

        try:
            confidence = float(obj.get("confidence", 1.0))
        except (TypeError, ValueError):
            confidence = 1.0

        return cls(
            engaged=bool(obj.get("engaged", False)),
            deadman=bool(obj.get("deadman", True)),  # default True: assume armed
            seq=int(obj.get("seq", 0) or 0),
            received_at_ns=time.monotonic_ns(),
            mode=mode,
            estop=bool(obj.get("estop", False)),  # absent -> False (back-compat)
            held_keys=held_keys,
            joint_targets=joint_targets,
            ee_pos=ee_pos,
            ee_quat=ee_quat,
            pinch=pinch,
            confidence=confidence,
        )


def _float_list(raw: object, length: int) -> Optional[list[float]]:
    """Parse a fixed-length float vector, or None if absent/malformed."""
    if not isinstance(raw, (list, tuple)) or len(raw) != length:
        return None
    try:
        return [float(x) for x in raw]
    except (TypeError, ValueError):
        return None


__all__ = ["TeleopFrame", "frame_with_header"]
