"""The teleop wire frame — the authoritative node-side parser.

A teleop frame is the JSON message a *producer* (the dashboard keyboard overlay
or the VR bridge) sends over the WebSocket relay to the node. The node owns this
parser; each producer re-implements the encoder against this contract (the
contract is duplicated across the WS boundary by design — see ADR 0009 — so the
node never depends on a producer package).

Three modes converge on one node-side gated path (`control.py`):

- ``mode="keys"``    — ``held_keys`` is a set of currently-held key strings; the
                       node integrates them into an absolute joint target
                       (``keyboard.next_target``). The dashboard keyboard overlay.
- ``mode="pose"``    — ``ee_pos`` (+ optional ``ee_quat``) is a raw 6-DoF
                       end-effector pose and ``pinch`` a gripper close amount;
                       the node retargets to joint targets via ``retarget`` /
                       node-side IK. The WebXR browser producer (see ADR 0009).
- ``mode="targets"`` — ``joint_targets`` is an absolute joint-target vector the
                       producer already computed. Used directly. (Retained for
                       producers that do their own IK; the WebXR path uses
                       ``pose`` so IK stays node-side.)

Back-compat: a frame with no ``mode`` is treated as ``"keys"``, so the existing
overlay (which sends no ``mode``) keeps working untouched.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Optional

_VALID_MODES = ("keys", "pose", "targets")


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


__all__ = ["TeleopFrame"]
