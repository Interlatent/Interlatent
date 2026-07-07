"""Deterministic behaviors: interpolation, registry, validation, execution.

All hardware-free — a :class:`behavior_fakes.FakeAdapter` stands in for the arm and
records the emitted command stream. Covers min-jerk boundary conditions, registry
load/override precedence, validation errors, speed scaling, blocking vs non-blocking,
cancellation, and the saturation abort.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from behavior_fakes import FakeAdapter

from interlatent.behaviors import registry as registry_mod
from interlatent.behaviors.executor import TrajectoryExecutor, _Plan
from interlatent.behaviors.interpolation import (
    build_samples,
    peak_velocity_factor,
    shape,
)
from interlatent.behaviors.registry import BehaviorRegistry, behavior
from interlatent.behaviors.schema import (
    BehaviorExecutionError,
    BehaviorValidationError,
    PoseBehavior,
)
from interlatent.node.teleop.robot_profile import get_profile

PROFILE = get_profile("so101")


def _executor(track: str = "full", start: float = 0.0, **kw) -> tuple[FakeAdapter, TrajectoryExecutor]:
    a = FakeAdapter(track=track, start=start)
    a.connect()
    ex = TrajectoryExecutor(a, PROFILE, realtime=False, **kw)
    return a, ex


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------


def test_min_jerk_boundary_conditions():
    """Min-jerk has zero velocity AND zero acceleration at both keyframes."""
    s = shape("min_jerk")
    tau = np.linspace(0, 1, 20001)
    y = s(tau)
    assert y[0] == pytest.approx(0.0) and y[-1] == pytest.approx(1.0)
    dt = tau[1] - tau[0]
    vel = np.gradient(y, dt)
    acc = np.gradient(vel, dt)
    assert vel[0] == pytest.approx(0.0, abs=1e-3)
    assert vel[-1] == pytest.approx(0.0, abs=1e-3)
    assert acc[0] == pytest.approx(0.0, abs=1e-2)
    assert acc[-1] == pytest.approx(0.0, abs=1e-2)
    # Peak velocity factor is the analytic max of s'(tau) = 1.875 at tau=0.5.
    assert vel.max() == pytest.approx(1.875, abs=1e-2)
    assert peak_velocity_factor("min_jerk") == pytest.approx(1.875)


def test_profiles_hit_endpoints_and_are_monotonic():
    for name in ("min_jerk", "linear", "trapezoidal"):
        s = shape(name)
        tau = np.linspace(0, 1, 1000)
        y = s(tau)
        assert y[0] == pytest.approx(0.0, abs=1e-9)
        assert y[-1] == pytest.approx(1.0, abs=1e-9)
        assert np.all(np.diff(y) >= -1e-9), name  # non-decreasing


def test_build_samples_hits_every_keyframe_exactly():
    wps = [np.zeros(2), np.array([10.0, -5.0]), np.array([10.0, 20.0])]
    samples = build_samples(wps, [1.0, 0.5], "min_jerk", dt=1 / 30)
    assert np.allclose(samples[0], wps[0])
    # Segment boundaries land exactly on tau==1 by construction.
    n0 = int(np.ceil(1.0 * 30))
    assert np.allclose(samples[n0], wps[1])
    assert np.allclose(samples[-1], wps[2])


# ---------------------------------------------------------------------------
# Registry: built-ins, load order, validation
# ---------------------------------------------------------------------------


def test_builtin_home_and_hello_present():
    reg = BehaviorRegistry.for_robot("so101")
    assert "home" in reg.names()
    assert "hello" in reg.names()
    home = reg.resolve("home")
    # home is derived from the profile rest pose.
    assert home.targets["shoulder_pan"] == PROFILE.rest_pose[0]
    assert home.duration is None  # auto-sized


def test_user_file_and_explicit_path_override_by_name(tmp_path):
    user = tmp_path / "behaviors.toml"
    user.write_text(
        "[home]\ntype='pose'\nduration=2.0\nshoulder_pan=10.0\n"
        "[wiggle]\ntype='pose'\nduration=0.5\nwrist_roll=15.0\n"
    )
    explicit = tmp_path / "extra.toml"
    explicit.write_text("[wiggle]\ntype='pose'\nduration=0.5\nwrist_roll=25.0\n")
    reg = BehaviorRegistry.for_robot("so101", user_file=user, explicit=explicit)
    # user file overrode the built-in `home` (now an explicit-duration pose).
    home = reg.resolve("home")
    assert home.duration == 2.0 and home.targets["shoulder_pan"] == 10.0
    # explicit path overrode the user-file `wiggle`.
    assert reg.resolve("wiggle").targets["wrist_roll"] == 25.0


def test_unknown_joint_raises_naming_behavior_and_joint(tmp_path):
    p = tmp_path / "b.toml"
    p.write_text("[bad]\ntype='pose'\nduration=1.0\nnope=10.0\n")
    with pytest.raises(BehaviorValidationError) as e:
        BehaviorRegistry.for_robot("so101", explicit=p, user_file=None)
    assert "bad" in str(e.value) and "nope" in str(e.value)


def test_out_of_limit_names_behavior_joint_value_and_limit(tmp_path):
    p = tmp_path / "b.toml"
    p.write_text("[reach]\ntype='pose'\nduration=1.0\nshoulder_pan=999.0\n")
    with pytest.raises(BehaviorValidationError) as e:
        BehaviorRegistry.for_robot("so101", explicit=p, user_file=None)
    msg = str(e.value)
    assert "reach" in msg and "shoulder_pan" in msg and "999" in msg and "180" in msg


def test_trajectory_velocity_cap_violation_raises_at_load(tmp_path):
    # shoulder_lift cap is 50 deg/s; 100 deg over 0.5 s min-jerk peaks at 375 deg/s.
    p = tmp_path / "b.toml"
    p.write_text(
        "[fast]\ntype='trajectory'\n"
        "keyframes=[{t=0.0, shoulder_lift=0.0},{t=0.5, shoulder_lift=100.0}]\n"
    )
    with pytest.raises(BehaviorValidationError) as e:
        BehaviorRegistry.for_robot("so101", explicit=p, user_file=None)
    assert "shoulder_lift" in str(e.value) and "cap" in str(e.value)


def test_trajectory_requires_moved_joints_initialized_at_t0(tmp_path):
    p = tmp_path / "b.toml"
    p.write_text(
        "[t]\ntype='trajectory'\n"
        "keyframes=[{t=0.0, wrist_roll=0.0},{t=1.0, elbow_flex=20.0}]\n"
    )
    with pytest.raises(BehaviorValidationError) as e:
        BehaviorRegistry.for_robot("so101", explicit=p, user_file=None)
    assert "elbow_flex" in str(e.value) and "t=0" in str(e.value)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def test_home_moves_to_rest_pose():
    a, ex = _executor(start=20.0)
    reg = BehaviorRegistry.for_robot("so101")
    res = ex.act(reg.resolve("home"), wait=True)
    assert res.reached and not res.aborted
    final = a.command_array[-1]
    assert np.allclose(final, PROFILE.rest_pose, atol=1e-6)


def test_hello_respects_velocity_caps_and_hits_keyframes():
    a, ex = _executor()
    reg = BehaviorRegistry.for_robot("so101")
    res = ex.act(reg.resolve("hello"), wait=True)
    assert res.reached
    cmds = a.command_array
    caps = np.array(PROFILE.max_velocity)
    dt = 1 / 30
    per_tick = np.abs(np.diff(cmds, axis=0))
    # Acceptance criterion 3: no per-tick delta exceeds cap * tick duration.
    assert np.all(per_tick <= caps * dt + 1e-6)
    # wrist_roll (index 4) swings to exactly ±35 (keyframes hit within tolerance).
    assert cmds[:, 4].max() == pytest.approx(35.0, abs=1e-3)
    assert cmds[:, 4].min() == pytest.approx(-35.0, abs=1e-3)


def test_speed_scaling_slower_is_safe_faster_raises():
    a, ex = _executor()
    reg = BehaviorRegistry.for_robot("so101")
    # 0.5x is gentler — fine.
    assert ex.act(reg.resolve("hello"), speed=0.5, wait=True).reached
    # 2.0x would exceed wrist_roll's cap → raises before any motion.
    a2, ex2 = _executor()
    with pytest.raises(BehaviorValidationError):
        ex2.act(reg.resolve("hello"), speed=2.0, wait=True)
    assert a2.commands == []  # nothing sent


def test_explicit_duration_pose_too_fast_raises_before_motion():
    a, ex = _executor()
    # shoulder_lift cap 50 deg/s; 100 deg over 0.2 s min-jerk peaks at ~937 deg/s.
    beh = PoseBehavior(name="snap", targets={"shoulder_lift": -100.0}, duration=0.2)
    with pytest.raises(BehaviorValidationError):
        ex.act(beh, wait=True)
    assert a.commands == []


def test_non_blocking_returns_handle_then_result():
    a, ex = _executor()
    reg = BehaviorRegistry.for_robot("so101")
    handle = ex.act(reg.resolve("home"), wait=False)
    res = handle.wait(timeout=5)
    assert handle.done and res.reached


def test_saturation_aborts_and_raises():
    # track="none": commanded targets advance but the arm never moves → the position
    # joints lag past the saturation margin for N ticks → abort + raise.
    a, ex = _executor(track="none", saturation_ticks=5)
    beh = PoseBehavior(name="reach", targets={"shoulder_pan": 100.0}, duration=2.0)
    with pytest.raises(BehaviorExecutionError) as e:
        ex.act(beh, wait=True)
    assert "saturation" in str(e.value) or "lagged" in str(e.value)


def test_decelerate_ramps_velocity_to_zero_without_jump():
    a, ex = _executor()
    last = np.zeros(6)
    prev = np.array([-2.0, 0, 0, 0, 0, 0])  # 2 deg/tick on shoulder_pan
    plan = _Plan(behavior="x", samples=np.zeros((1, 6)), target=np.zeros(6), dt=1 / 30)
    ex._decelerate(last, prev, plan, start=0.0, tick=0)
    cmds = a.command_array[:, 0]
    # No instantaneous jump from the last commanded position.
    assert abs(cmds[0] - last[0]) <= 2.0 + 1e-6
    # Per-tick motion shrinks monotonically to ~0 (a smooth stop, not a freeze).
    steps = np.abs(np.diff(np.concatenate([[last[0]], cmds])))
    assert steps[-1] == pytest.approx(0.0, abs=1e-6)
    assert steps[0] > steps[-1]


def test_cancel_midrun_returns_aborted(monkeypatch):
    a = FakeAdapter()
    a.connect()
    ex = TrajectoryExecutor(a, PROFILE, realtime=True)  # real pacing so cancel lands mid-run
    beh = PoseBehavior(name="sweep", targets={"shoulder_pan": 100.0}, duration=2.0)
    handle = ex.act(beh, wait=False)
    time.sleep(0.2)
    handle.cancel()
    res = handle.wait(timeout=5)
    assert res.aborted and "cancel" in res.reason
    # Stopped well before the full 2 s / 60-tick plan.
    assert len(a.commands) < 60


# ---------------------------------------------------------------------------
# Procedural (decorator) behaviors
# ---------------------------------------------------------------------------


def test_procedural_behavior_registration_and_scope():
    calls = []

    @behavior("nod_test", robot="so101")
    def nod(robot):
        calls.append(robot)

    try:
        reg = BehaviorRegistry.for_robot("so101")
        assert "nod_test" in reg.names()
        proc = reg.resolve("nod_test")
        proc.fn("sentinel")
        assert calls == ["sentinel"]
        # Not visible to a different robot kind.
        reg_koch = BehaviorRegistry.for_robot("koch")
        assert "nod_test" not in reg_koch.names()
    finally:
        registry_mod._PROCEDURAL.pop(("so101", "nod_test"), None)
