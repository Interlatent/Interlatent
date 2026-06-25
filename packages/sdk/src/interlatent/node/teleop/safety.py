"""Safety envelope: workspace clamp + velocity clamp + deadman + staleness.

Salvaged from the retired `interlatent_teleop` Pi path. The node control loop
calls `SafetyGate.step(...)` once per tick with the latest received target. The
gate returns the *commanded* joint vector — either a velocity-limited step
toward the target, or a hold on the current pose if any safety condition is
violated.

This is the single safety authority for the node's teleop path: both the
keyboard and the VR producer ultimately produce an absolute joint target that
flows through this gate before reaching the robot.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .robot_profile import RobotProfile


@dataclass
class TargetSample:
    joints: np.ndarray          # shape (n_joints,)
    deadman_active: bool
    confidence: float
    received_at: float          # monotonic seconds when received
    producer_timestamp_ns: int  # producer monotonic ns


@dataclass
class SafetyConfig:
    # Hold pose if no target newer than this has been received.
    # 200 ms covers a few dropped packets without latching estop.
    staleness_timeout_s: float = 0.2
    # Minimum producer confidence to act on a target.
    min_confidence: float = 0.5
    # Hard estop latch must be cleared by an explicit reset; for now we
    # auto-clear on deadman release + re-press. Keep latched if a hardware
    # fault was reported.
    estop_latched: bool = False


@dataclass
class SafetyGate:
    profile: RobotProfile
    control_dt: float
    config: SafetyConfig = field(default_factory=SafetyConfig)
    _latest: Optional[TargetSample] = None
    _last_deadman: bool = False
    # Last position we *commanded* the motor to go to. The velocity clamp
    # advances this in time toward the target instead of basing each step on
    # the motor's actual position. This is the same trajectory pattern as the
    # startup home ramp — and the reason that ramp works smoothly even on the
    # gravity-loaded shoulder_lift joint. If we based the clamp on
    # `current_joints` instead, a sluggish motor (e.g. lift fighting gravity)
    # would deadlock: actual doesn't move so commanded doesn't either, position
    # error stays tiny, torque stays tiny.
    _last_commanded: Optional[np.ndarray] = None

    def submit(self, sample: TargetSample) -> None:
        """Called when a new target arrives.

        The most recent submitted sample wins; older samples are discarded. The
        control thread reads the latest at its own rate.
        """
        prev = self._latest
        if prev is not None and sample.producer_timestamp_ns < prev.producer_timestamp_ns:
            # Out-of-order delivery; ignore.
            return
        self._latest = sample

    def step(self, current_joints: np.ndarray, now: Optional[float] = None) -> tuple[np.ndarray, str]:
        """Compute the joint vector to command this tick.

        Returns (commanded_joints, status). `status` is a short machine-readable
        reason for the choice; useful both for client feedback and logging.
        """
        if now is None:
            now = time.monotonic()

        # When we're not actively driving the arm, anchor the commanded
        # trajectory to the motor's actual position so resuming doesn't snap
        # from a stale commanded value.
        def _idle(status: str) -> tuple[np.ndarray, str]:
            self._last_commanded = current_joints.copy()
            return current_joints.copy(), status

        if self.config.estop_latched:
            return _idle("estop_latched")

        sample = self._latest
        if sample is None:
            return _idle("no_target_yet")

        age = now - sample.received_at
        if age > self.config.staleness_timeout_s:
            return _idle(f"stale({age*1000:.0f}ms)")

        if not sample.deadman_active:
            # On deadman release, blend the held target back to current so the
            # next press doesn't snap from a stale faraway pose.
            self._latest = TargetSample(
                joints=current_joints.copy(),
                deadman_active=False,
                confidence=sample.confidence,
                received_at=now,
                producer_timestamp_ns=sample.producer_timestamp_ns,
            )
            self._last_deadman = False
            return _idle("deadman_released")

        if sample.confidence < self.config.min_confidence:
            return _idle(f"low_confidence({sample.confidence:.2f})")

        target = sample.joints
        if target.shape != current_joints.shape:
            return _idle("shape_mismatch")

        # Workspace clamp first.
        lo = np.array([lim[0] for lim in self.profile.joint_limits], dtype=np.float32)
        hi = np.array([lim[1] for lim in self.profile.joint_limits], dtype=np.float32)
        target_clamped = np.clip(target, lo, hi)

        # Velocity clamp based on the *last commanded* position, not the motor's
        # actual position. See the field comment on `_last_commanded` for why
        # this matters for gravity-loaded joints like shoulder_lift.
        if self._last_commanded is None or not self._last_deadman:
            self._last_commanded = current_joints.copy()
        max_step = np.array(self.profile.max_velocity, dtype=np.float32) * self.control_dt
        delta = target_clamped - self._last_commanded
        delta = np.clip(delta, -max_step, max_step)
        commanded = self._last_commanded + delta
        self._last_commanded = commanded.copy()

        self._last_deadman = True
        return commanded.astype(np.float32), "ok"

    def reset(self) -> None:
        """Drop target + commanded-trajectory state.

        Call on teleop **disengage** so the next engage starts the velocity
        clamp from the *live* pose rather than a stale commanded anchor. The
        node loop only steps the gate while engaged (unlike the always-on Pi
        loop this was salvaged from), so without this a re-engage would clamp
        from wherever the arm was when the human last released.
        """
        self._latest = None
        self._last_deadman = False
        self._last_commanded = None

    def latch_estop(self, reason: str) -> None:
        self.config.estop_latched = True

    def clear_estop(self) -> None:
        self.config.estop_latched = False


__all__ = ["SafetyGate", "SafetyConfig", "TargetSample"]
