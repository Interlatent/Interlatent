"""Browser-keyboard → joint-target action for DAgger intervention.

The dashboard's :class:`TeleopOverlay` captures keydown/keyup and sends
the held-key *set* (not actions) to the node at ~30 Hz. The node calls
:func:`next_target` once per control tick to integrate that held set
into a smooth joint-target ramp, exactly matching the laptop CLI's
held-key model (see ``interlatent_teleop.laptop.keyboard_cli``):

  A D     shoulder_pan       -/+
  W S     shoulder_lift      +/-
  E C     elbow_flex         +/-
  I K     wrist_flex         +/-
  J L     wrist_roll         -/+
  SPACE [ gripper close
  N     ] gripper open
  SHIFT   3x speed multiplier while held

Why a held-set instead of a pre-computed action vector? Two reasons:

1. The browser's RAF / event loop produces jittery dt, but the node
   ticks at a steady 30 Hz. Integrating on the node yields a smooth
   target ramp; integrating in the browser would alias against network
   jitter.

2. Dropped/delayed frames are a non-event: the node carries the last
   known held set forward across stale frames, and a human holding W
   for hundreds of ms is unaffected by a missed 33-ms frame.

This module is **joint mode only**. The CLI's cartesian + IK path is
not lifted here because the dashboard does not need it for DAgger
intervention; if it ever does, the same pattern extends naturally.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np


# Held-key strings the dashboard is allowed to send. Anything else is
# ignored so a stray key in the browser cannot move the arm.
_VALID_KEYS = frozenset({
    "a", "d", "w", "s", "e", "c", "i", "k", "j", "l",
    "[", "]", "n",
    "shift",
    "space",  # SPACE = close; the overlay maps the JS " " keycode to "space"
})

# Per-joint key bindings: char -> (joint_idx, direction_sign).
# Same map as ``interlatent_teleop.laptop.keyboard_cli.JOINT_KEYS``.
_JOINT_KEYS: dict[str, tuple[int, int]] = {
    "a": (0, -1), "d": (0, +1),
    "w": (1, +1), "s": (1, -1),
    "e": (2, +1), "c": (2, -1),
    "i": (3, +1), "k": (3, -1),
    "j": (4, -1), "l": (4, +1),
}


@dataclass
class KeyboardActionConfig:
    """Tunables for the joint-target integrator.

    Defaults match ``interlatent_teleop.laptop.keyboard_cli`` so the
    feel is identical whether a user is driving via the dashboard or
    the laptop CLI. The lerobot SO-101 calibration runs every joint's
    positive direction opposite to the natural keymap intent (W=up,
    A=left, etc.), so ``joint_key_signs`` defaults to all -1.
    """

    joint_rate_deg_per_s: float = 60.0
    gripper_rate_pct_per_s: float = 120.0
    shift_mul: float = 3.0
    max_joint_lead_deg: float = 45.0
    joint_key_signs: Sequence[float] = field(
        default_factory=lambda: (-1.0, -1.0, -1.0, -1.0, -1.0, -1.0)
    )
    # The gripper convention is fixed at lerobot's 0=closed / 100=open
    # regardless of joint_key_signs.
    gripper_invert: bool = False


def _normalize_held(held: Iterable[str]) -> set[str]:
    out: set[str] = set()
    for k in held or ():
        if not isinstance(k, str):
            continue
        kk = k.lower()
        if kk == " ":
            kk = "space"
        if kk in _VALID_KEYS:
            out.add(kk)
    return out


def next_target(
    *,
    target_joints: np.ndarray,
    actual_joints: np.ndarray,
    held_keys: Iterable[str],
    dt: float,
    cfg: KeyboardActionConfig,
    joint_min: Optional[np.ndarray] = None,
    joint_max: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Advance the joint-target ramp by one tick.

    ``target_joints`` is the previous tick's commanded target (length
    6 for the SO-101: pan, lift, elbow, wflex, wroll, gripper). The
    function mutates a copy and returns the new target — the caller
    keeps state between ticks.

    ``actual_joints`` is the robot's current measured joint vector;
    used for lead-clipping so the commanded target cannot run too far
    ahead of where the arm physically is.

    ``held_keys`` is the set of currently-held key names from the
    browser; case-insensitive, JS-space-bar (" ") is folded to
    ``"space"``, unknown keys are ignored.

    ``dt`` is the elapsed time since the last call, in seconds.
    """
    held = _normalize_held(held_keys)
    out = np.asarray(target_joints, dtype=np.float32).copy()
    if out.shape[0] < 6:
        # Pad short vectors so we can index out[5] for the gripper.
        # Real callers pass shape-(6,) — this guards tests.
        out = np.concatenate([out, np.zeros(6 - out.shape[0], dtype=np.float32)])

    actual = np.asarray(actual_joints, dtype=np.float32).reshape(-1)
    if actual.shape[0] < out.shape[0]:
        pad = np.zeros(out.shape[0] - actual.shape[0], dtype=np.float32)
        actual = np.concatenate([actual, pad])

    mul = cfg.shift_mul if "shift" in held else 1.0
    signs = list(cfg.joint_key_signs)
    # Defensive: pad signs if the caller passed too few.
    while len(signs) < 6:
        signs.append(-1.0)

    # --- joint deltas
    rate = cfg.joint_rate_deg_per_s * mul
    for k, (idx, sign) in _JOINT_KEYS.items():
        if k in held:
            out[idx] += sign * signs[idx] * rate * dt

    # --- gripper (separate; lerobot convention 0=closed / 100=open)
    grip_rate = cfg.gripper_rate_pct_per_s * mul
    if cfg.gripper_invert:
        grip_rate = -grip_rate
    open_held = ("n" in held) or ("]" in held)
    close_held = ("space" in held) or ("[" in held)
    if open_held:
        out[5] = float(np.clip(out[5] + grip_rate * dt, 0.0, 100.0))
    if close_held:
        out[5] = float(np.clip(out[5] - grip_rate * dt, 0.0, 100.0))

    # --- lead-clip per non-gripper joint
    for i in range(5):
        delta = float(out[i] - actual[i])
        if abs(delta) > cfg.max_joint_lead_deg:
            out[i] = float(
                actual[i] + (cfg.max_joint_lead_deg if delta > 0 else -cfg.max_joint_lead_deg)
            )

    # --- final clamp to joint limits if provided
    if joint_min is not None and joint_max is not None:
        out = np.clip(out, joint_min, joint_max).astype(np.float32)

    return out


__all__ = ["KeyboardActionConfig", "next_target"]
