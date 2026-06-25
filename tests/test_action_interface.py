"""Manual ``action()`` seam (interlatent.adapters.base.ManualActionInterface).

Exercises the shared block-then-settle logic against a fake adapter that reuses the
real SO-101 RobotProfile (6 joints), with no hardware. Covers the named contract,
hold_missing, range pre-validation, SafetyGate velocity-limited approach, per-control-
mode settle, and timeout — see ADR 0013.
"""
from __future__ import annotations

import numpy as np
import pytest

from interlatent.adapters.base import JointSpec, ManualActionInterface
from interlatent.node.teleop.robot_profile import get_profile

# Match the SO-101 profile's joint order exactly.
_FEATURES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


class FakeAdapter(ManualActionInterface):
    """In-memory adapter. ``track`` controls how send_action moves the joints:

    - "full":    commanded pose is reached exactly (instant tracking).
    - "none":    commands are ignored (joints never move) — to test timeout.
    - "stuck_gripper": arm tracks, gripper never moves — to test effort exclusion.
    """

    robot_kind = "so101"

    def __init__(self, track: str = "full", start: float = 0.0) -> None:
        self._pos = np.full(len(_FEATURES), float(start), dtype=np.float32)
        self._track = track
        self.commands: list[np.ndarray] = []

    @property
    def action_features(self) -> list[str]:
        return list(_FEATURES)

    @property
    def joint_specs(self) -> list[JointSpec]:
        specs = []
        for f in _FEATURES:
            name = f.rsplit(".", 1)[0]
            if name == "gripper":
                specs.append(JointSpec(name=name, control_mode="gripper"))
            else:
                specs.append(JointSpec(name=name, control_mode="position", settle_tolerance=2.0))
        return specs

    def connect(self) -> None:  # pragma: no cover - trivial
        pass

    def disconnect(self) -> None:  # pragma: no cover - trivial
        pass

    def get_observation(self):
        return {f: float(self._pos[i]) for i, f in enumerate(_FEATURES)}

    def send_action(self, action):
        vec = np.array([action[f] for f in _FEATURES], dtype=np.float32)
        self.commands.append(vec.copy())
        if self._track == "full":
            self._pos = vec
        elif self._track == "stuck_gripper":
            self._pos[:-1] = vec[:-1]  # arm tracks, gripper (last) frozen
        # "none": ignore — joints never move
        return action


_FAST = dict(rate_hz=500.0, timeout=3.0)


def test_unknown_joint_raises():
    a = FakeAdapter()
    with pytest.raises(ValueError, match="unknown joint"):
        a.action(elbow=10.0, **_FAST)


def test_omitted_joint_raises_without_hold():
    a = FakeAdapter()
    with pytest.raises(ValueError, match="missing joint"):
        a.action(shoulder_pan=10.0, **_FAST)


def test_hold_missing_fills_and_settles(caplog):
    a = FakeAdapter(track="full", start=5.0)
    with caplog.at_level("INFO"):
        a.action(shoulder_pan=10.0, hold_missing=True, **_FAST)
    # Held joints logged.
    assert any("holding" in r.message for r in caplog.records)
    obs = a.get_observation()
    assert obs["shoulder_pan.pos"] == pytest.approx(10.0, abs=2.0)
    # An unspecified joint stayed at its held (start) value.
    assert obs["elbow_flex.pos"] == pytest.approx(5.0, abs=2.0)


def test_out_of_range_raises_before_motion():
    a = FakeAdapter()
    # shoulder_pan limit is [-180, 180].
    with pytest.raises(ValueError, match="outside its limit"):
        a.action(shoulder_pan=999.0, hold_missing=True, **_FAST)
    assert a.commands == []  # nothing was ever sent


def test_no_profile_refuses():
    a = FakeAdapter()
    a.robot_kind = "no_such_robot"
    with pytest.raises(RuntimeError, match="no RobotProfile"):
        a.action(shoulder_pan=1.0, hold_missing=True, **_FAST)


def test_velocity_limited_approach():
    a = FakeAdapter(track="full", start=0.0)
    a.action(shoulder_pan=30.0, hold_missing=True, **_FAST)
    # shoulder_pan max_velocity = 120 deg/s, control_dt = 1/500 -> 0.24 deg/step.
    max_step = get_profile("so101").max_velocity[0] / 500.0
    cmds = np.array([c[0] for c in a.commands])
    deltas = np.diff(cmds)
    assert np.all(deltas <= max_step + 1e-3), deltas.max()
    assert len(a.commands) > 1  # took several gated steps, not one jump


def test_timeout_raises_when_not_tracking():
    a = FakeAdapter(track="none", start=0.0)
    with pytest.raises(TimeoutError, match="did not settle"):
        a.action(shoulder_pan=30.0, hold_missing=True, rate_hz=500.0, timeout=0.3)


def test_koch_profile_registered_and_consistent():
    for kind in ("koch", "koch_follower"):
        prof = get_profile(kind)
        assert prof is not None, kind
        n = len(prof.joint_names)
        assert len(prof.joint_limits) == n == len(prof.max_velocity) == len(prof.rest_pose)
    # Koch shares the SO-101 6-joint topology, so the fake adapter drives it too.
    a = FakeAdapter(track="full", start=0.0)
    a.robot_kind = "koch"
    a.action(shoulder_pan=20.0, hold_missing=True, **_FAST)
    assert a.get_observation()["shoulder_pan.pos"] == pytest.approx(20.0, abs=2.0)


def test_gripper_commanded_when_arm_already_in_tolerance():
    # Regression: the arm starts within settle_tolerance of every position target,
    # but the gripper must still move. The loop must NOT return before commanding
    # the gripper just because the position joints look settled.
    a = FakeAdapter(track="full", start=0.0)
    # All position joints commanded to ~their current pose (within 2 deg), gripper 0->80.
    a.action(
        shoulder_pan=1.0, shoulder_lift=0.0, elbow_flex=0.0,
        wrist_flex=0.0, wrist_roll=0.0, gripper=80.0,
        **_FAST,
    )
    assert a.commands, "expected at least one command to be sent"
    assert a.get_observation()["gripper.pos"] == pytest.approx(80.0, abs=1.0)


def test_at_target_still_sends_one_command():
    # Commanding the exact current pose should still issue a command (not no-op).
    a = FakeAdapter(track="full", start=0.0)
    a.action(
        shoulder_pan=0.0, shoulder_lift=0.0, elbow_flex=0.0,
        wrist_flex=0.0, wrist_roll=0.0, gripper=0.0,
        **_FAST,
    )
    assert len(a.commands) >= 1


def test_gripper_excluded_from_settle():
    # Arm reaches target; gripper is frozen far from its commanded value. The move
    # must still settle because the gripper (control_mode != position) is excluded.
    a = FakeAdapter(track="stuck_gripper", start=0.0)
    a.action(shoulder_pan=10.0, gripper=80.0, hold_missing=True, rate_hz=500.0, timeout=2.0)
    assert a.get_observation()["gripper.pos"] == pytest.approx(0.0)  # never moved
