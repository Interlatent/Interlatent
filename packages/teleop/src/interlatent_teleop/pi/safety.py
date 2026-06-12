"""Safety envelope: workspace clamp + velocity clamp + deadman + staleness.

The Pi's control loop calls `SafetyGate.step(...)` once per tick with
the latest received target. The gate returns the *commanded* joint
vector — either a velocity-limited step toward the target, or a hold
on the current pose if any safety condition is violated.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..common.config import RobotProfile

_LOG = logging.getLogger("interlatent_teleop.pi.safety")


@dataclass
class TargetSample:
    joints: np.ndarray          # shape (n_joints,)
    deadman_active: bool
    confidence: float
    received_at: float          # monotonic seconds at Pi when received
    producer_timestamp_ns: int  # producer monotonic ns


@dataclass
class SafetyConfig:
    # Hold pose if no target newer than this has been received.
    # 200 ms covers a few dropped packets without latching estop.
    staleness_timeout_s: float = 0.2
    # Minimum producer confidence to act on a target.
    min_confidence: float = 0.5
    # Hard estop latch. Auto-clears when the operator releases and
    # re-presses the deadman (an explicit acknowledgment). If the
    # underlying fault persists (e.g. the motor bus is unplugged), the
    # next driver write fails and immediately re-latches, so a broken
    # arm never resumes for more than one tick.
    estop_latched: bool = False
    estop_reason: str = ""


@dataclass
class SafetyGate:
    profile: RobotProfile
    control_dt: float
    config: SafetyConfig = field(default_factory=SafetyConfig)
    _latest: Optional[TargetSample] = None
    _last_deadman: bool = False
    # Last position we *commanded* the motor to go to. The velocity clamp
    # advances this in time toward the target instead of basing each step
    # on the motor's actual position. This is the same trajectory pattern
    # as the startup home ramp — and the reason that ramp works smoothly
    # even on the gravity-loaded shoulder_lift joint. If we based the
    # clamp on `current_joints` instead, a sluggish motor (e.g. lift
    # fighting gravity) would deadlock: actual doesn't move so commanded
    # doesn't either, position error stays tiny, torque stays tiny.
    _last_commanded: Optional[np.ndarray] = None
    # True once we've seen the deadman released while estopped; the
    # next press after that clears the latch.
    _estop_release_seen: bool = False
    # Guards all mutable state above. submit() runs on the gRPC reader
    # thread, step() on the control-loop thread.
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def submit(self, sample: TargetSample) -> None:
        """Called from the network thread when a new target arrives.

        The most recent submitted sample wins; older samples are
        discarded. The control thread reads the latest at its own rate.
        """
        with self._lock:
            prev = self._latest
            if prev is not None and sample.producer_timestamp_ns < prev.producer_timestamp_ns:
                # Out-of-order delivery; ignore.
                return
            self._latest = sample

    def step(self, current_joints: np.ndarray, now: Optional[float] = None) -> tuple[np.ndarray, str]:
        """Compute the joint vector to command this tick.

        Returns (commanded_joints, status). `status` is a short
        machine-readable reason for the choice; useful both for client
        feedback and Pi-side logging.
        """
        if now is None:
            now = time.monotonic()
        with self._lock:
            return self._step_locked(current_joints, now)

    def _step_locked(self, current_joints: np.ndarray, now: float) -> tuple[np.ndarray, str]:

        # When we're not actively driving the arm, anchor the commanded
        # trajectory to the motor's actual position so resuming doesn't
        # snap from a stale commanded value.
        def _idle(status: str) -> tuple[np.ndarray, str]:
            self._last_commanded = current_joints.copy()
            return current_joints.copy(), status

        sample = self._latest
        if self.config.estop_latched:
            # Deadman release followed by a re-press is the operator's
            # explicit acknowledgment — clear the latch and resume. If
            # the fault persists, the next driver write re-latches.
            if sample is not None and not sample.deadman_active:
                self._estop_release_seen = True
            elif sample is not None and self._estop_release_seen:
                _LOG.warning(
                    "estop cleared by deadman re-press (was: %s)",
                    self.config.estop_reason or "unknown",
                )
                self.config.estop_latched = False
                self.config.estop_reason = ""
                self._estop_release_seen = False
            if self.config.estop_latched:
                reason = self.config.estop_reason
                return _idle(f"estop_latched({reason})" if reason else "estop_latched")

        if sample is None:
            return _idle("no_target_yet")

        age = now - sample.received_at
        if age > self.config.staleness_timeout_s:
            return _idle(f"stale({age*1000:.0f}ms)")

        if not sample.deadman_active:
            # On deadman release, blend the held target back to current
            # so the next press doesn't snap from a stale faraway pose.
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

        # Velocity clamp based on the *last commanded* position, not the
        # motor's actual position. See the field comment on
        # `_last_commanded` for why this matters for gravity-loaded
        # joints like shoulder_lift.
        if self._last_commanded is None or not self._last_deadman:
            self._last_commanded = current_joints.copy()
        max_step = np.array(self.profile.max_velocity, dtype=np.float32) * self.control_dt
        delta = target_clamped - self._last_commanded
        delta = np.clip(delta, -max_step, max_step)
        commanded = self._last_commanded + delta
        self._last_commanded = commanded.copy()

        self._last_deadman = True
        return commanded.astype(np.float32), "ok"

    def latch_estop(self, reason: str) -> None:
        with self._lock:
            if not self.config.estop_latched:
                _LOG.error("estop latched: %s (release + re-press deadman to clear)", reason)
            self.config.estop_latched = True
            self.config.estop_reason = reason
            self._estop_release_seen = False

    def clear_estop(self) -> None:
        with self._lock:
            self.config.estop_latched = False
            self.config.estop_reason = ""
            self._estop_release_seen = False
