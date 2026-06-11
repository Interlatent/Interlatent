"""Shared robot profile + safety envelope.

LeRobot reports SO-101 follower joints as `<motor>.pos` scalars in
*degrees* (the feetech driver wraps the raw motor encoder to a
calibrated angular range). Limits and velocity caps here are in
degrees and degrees/second to match what the driver expects.
"""
from __future__ import annotations

from dataclasses import dataclass


SO101_JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

# Per-joint software limits in degrees, sized to match SO-101's
# physical envelope (slightly tighter than the motor-stop range so the
# software clamp triggers before the hardware does). The Pi enforces
# these on every target via SafetyGate.
SO101_JOINT_LIMITS: tuple[tuple[float, float], ...] = (
    (-180.0, 180.0),  # shoulder_pan       (full rotation in lerobot convention)
    (-110.0, 110.0),  # shoulder_lift
    (-110.0, 110.0),  # elbow_flex
    (-100.0, 100.0),  # wrist_flex
    (-180.0, 180.0),  # wrist_roll
    (   0.0, 100.0),  # gripper            (0 closed, 100 open; matches lerobot SO-101)
)

# Per-joint maximum velocity in deg/sec.
#
# Important: the Feetech servos in the SO-101 use position-only PID
# with limited torque. Under gravity, the servos can't slew at their
# unloaded maximum speed — especially for `shoulder_lift` and
# `elbow_flex` when the arm is reaching up against gravity. If our
# software cap is faster than what the motor can actually achieve
# under load, the commanded position runs ahead of the actual
# position and the arm "feels stuck" or "won't go up".
#
# These conservative defaults are below the gravity-loaded speed of
# the motors so the commanded target tracks closely with the actual
# arm pose. Bump these up only if you've added counterweights or
# tuned the motor PID gains higher.
SO101_MAX_VELOCITY: tuple[float, ...] = (
    120.0,  # shoulder_pan         (no gravity load)
    50.0,   # shoulder_lift        (gravity-loaded; smaller per-tick step
            #                       keeps the motor from ringing around
            #                       each commanded position under P~32)
    80.0,   # elbow_flex           (partially gravity-loaded; similar story)
    180.0,  # wrist_flex
    240.0,  # wrist_roll           (no load)
    400.0,  # gripper              (small, fast)
)


@dataclass(frozen=True)
class RobotProfile:
    name: str
    joint_names: tuple[str, ...]
    joint_limits: tuple[tuple[float, float], ...]
    max_velocity: tuple[float, ...]

    def __post_init__(self) -> None:
        n = len(self.joint_names)
        if len(self.joint_limits) != n or len(self.max_velocity) != n:
            raise ValueError(
                f"profile {self.name!r}: joint_names/limits/velocity length mismatch"
            )


SO101_PROFILE = RobotProfile(
    name="so101_follower",
    joint_names=SO101_JOINT_NAMES,
    joint_limits=SO101_JOINT_LIMITS,
    max_velocity=SO101_MAX_VELOCITY,
)
