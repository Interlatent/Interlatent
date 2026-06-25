"""Static per-robot teleop profiles: joint limits, velocity caps, rest pose.

These are the parts of the teleop schema that lerobot *cannot* supply at
inference time. lerobot gives us joint *names* (`robot.action_features`) and the
live joint *positions* (`robot.get_observation()`), but it exposes no per-joint
safe velocity cap and no declared "home"/rest pose. The `SafetyGate` needs both,
so they live here as static, hand-tuned robot config keyed by robot kind.

LeRobot reports SO-101 follower joints as `<motor>.pos` scalars in *degrees*
(the feetech driver wraps the raw encoder to a calibrated angular range), so
limits and velocity caps here are in degrees and degrees/second.

Adding a new robot = add a `RobotProfile` and register it in `_PROFILES`. This
is the single place the multi-robot teleop goal expands.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


SO101_JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

# Per-joint software limits in degrees, sized to match SO-101's physical
# envelope (slightly tighter than the motor-stop range so the software clamp
# triggers before the hardware does). The node SafetyGate enforces these on
# every target.
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
# Important: the Feetech servos in the SO-101 use position-only PID with limited
# torque. Under gravity, the servos can't slew at their unloaded maximum speed —
# especially for `shoulder_lift` and `elbow_flex` when reaching up against
# gravity. If the software cap is faster than the motor can achieve under load,
# the commanded position runs ahead of the actual position and the arm "feels
# stuck". These conservative defaults sit below the gravity-loaded speed so the
# commanded target tracks closely. Bump them only with counterweights or higher
# motor P-gains (see `control.py`'s opt-in P_Coefficient bump).
SO101_MAX_VELOCITY: tuple[float, ...] = (
    120.0,  # shoulder_pan         (no gravity load)
    50.0,   # shoulder_lift        (gravity-loaded; small per-tick step)
    80.0,   # elbow_flex           (partially gravity-loaded)
    180.0,  # wrist_flex
    240.0,  # wrist_roll           (no load)
    400.0,  # gripper              (small, fast)
)

# Static neutral/rest pose (degrees), used as the producer-side calibration
# reference in the pose-target teleop path (Milestone 2): the producer
# calibrates a neutral input against this pose, and the SafetyGate velocity-
# clamps from the *actual* current pose toward each commanded target, so the
# engage transient is absorbed without needing a live home pose shipped back.
# Placeholder mid-range pose (within all limits); tune on real hardware.
SO101_REST_POSE: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class RobotProfile:
    name: str
    joint_names: tuple[str, ...]
    joint_limits: tuple[tuple[float, float], ...]
    max_velocity: tuple[float, ...]
    rest_pose: tuple[float, ...]

    def __post_init__(self) -> None:
        n = len(self.joint_names)
        if (
            len(self.joint_limits) != n
            or len(self.max_velocity) != n
            or len(self.rest_pose) != n
        ):
            raise ValueError(
                f"profile {self.name!r}: joint_names/limits/velocity/rest_pose "
                f"length mismatch"
            )

    def to_schema_dict(self) -> dict:
        """JSON-friendly teleop schema.

        The single shape used for both the control-plane report (node →
        backend → ``Environment.teleop_profile``) and the producer-facing
        ``robot_schema`` in the teleop-token response. Limits are split into
        ``joint_min``/``joint_max`` (mirrors the retired ``OpenTeleopResponse``).
        """
        return {
            "robot_kind": self.name,
            "joint_names": list(self.joint_names),
            "joint_min": [float(lo) for lo, _ in self.joint_limits],
            "joint_max": [float(hi) for _, hi in self.joint_limits],
            "max_velocity": [float(v) for v in self.max_velocity],
            "rest_pose": [float(p) for p in self.rest_pose],
        }


SO101_PROFILE = RobotProfile(
    name="so101_follower",
    joint_names=SO101_JOINT_NAMES,
    joint_limits=SO101_JOINT_LIMITS,
    max_velocity=SO101_MAX_VELOCITY,
    rest_pose=SO101_REST_POSE,
)


# ---------------------------------------------------------------------------
# Koch v1.1 follower
# ---------------------------------------------------------------------------
#
# Same 6-joint topology and naming as SO-101 (LeRobot reports Koch follower joints
# as the same `<motor>.pos` scalars in degrees), so the order below matches LeRobot's
# `koch_follower` action_features. Koch uses Dynamixel XL330/XL430 servos, which hold
# position better under gravity than SO-101's Feetech motors — but these values are a
# **conservative starting envelope**, not hardware-measured. Verify against the
# `DRTC-DEBUG joints` log on a real arm before widening limits or raising velocities;
# the SafetyGate fails safe when limits are too tight, not too loose.
KOCH_JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

# Degrees. Slightly tighter than the mechanical range so the software clamp triggers
# before the hardware stop. Tune per arm.
KOCH_JOINT_LIMITS: tuple[tuple[float, float], ...] = (
    (-180.0, 180.0),  # shoulder_pan
    (-100.0, 100.0),  # shoulder_lift
    (-100.0, 100.0),  # elbow_flex
    (-100.0, 100.0),  # wrist_flex
    (-180.0, 180.0),  # wrist_roll
    (   0.0, 100.0),  # gripper   (0 closed, 100 open; LeRobot normalized convention)
)

# deg/sec. Conservative — Dynamixels can slew faster, but a low cap keeps the
# commanded trajectory tracking the actual pose closely under load. Raise once
# verified on hardware.
KOCH_MAX_VELOCITY: tuple[float, ...] = (
    120.0,  # shoulder_pan
    80.0,   # shoulder_lift
    100.0,  # elbow_flex
    180.0,  # wrist_flex
    240.0,  # wrist_roll
    400.0,  # gripper
)

# Placeholder mid-range neutral pose (within all limits); tune on real hardware.
KOCH_REST_POSE: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


KOCH_PROFILE = RobotProfile(
    name="koch_follower",
    joint_names=KOCH_JOINT_NAMES,
    joint_limits=KOCH_JOINT_LIMITS,
    max_velocity=KOCH_MAX_VELOCITY,
    rest_pose=KOCH_REST_POSE,
)


# Registry keyed by robot kind. Keys match the `--robot` kinds resolved in
# `control.py._make_lerobot_robot` (and their aliases). Each new teleop-capable
# robot adds an entry here.
_PROFILES: dict[str, RobotProfile] = {
    "so101": SO101_PROFILE,
    "so101_follower": SO101_PROFILE,
    "koch": KOCH_PROFILE,
    "koch_follower": KOCH_PROFILE,
}


def get_profile(robot_kind: str) -> Optional[RobotProfile]:
    """Return the teleop profile for a robot kind, or None if unknown.

    A None result means "no static safety envelope for this robot" — the caller
    must refuse to run the gated teleop path rather than command an unclamped
    robot.
    """
    return _PROFILES.get(str(robot_kind).lower().strip())


__all__ = ["RobotProfile", "SO101_PROFILE", "KOCH_PROFILE", "get_profile"]
