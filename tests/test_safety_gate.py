"""Teleop safety envelope: clamps, staleness, deadman, estop latch/clear."""
import numpy as np
import pytest

from interlatent_teleop.common.config import SO101_PROFILE
from interlatent_teleop.pi.safety import SafetyGate, TargetSample

DT = 0.02  # 50 Hz
N = len(SO101_PROFILE.joint_names)


@pytest.fixture
def gate() -> SafetyGate:
    return SafetyGate(profile=SO101_PROFILE, control_dt=DT)


def sample(joints, *, deadman=True, confidence=1.0, now=100.0, ts_ns=None):
    return TargetSample(
        joints=np.asarray(joints, dtype=np.float32),
        deadman_active=deadman,
        confidence=confidence,
        received_at=now,
        producer_timestamp_ns=int(now * 1e9) if ts_ns is None else ts_ns,
    )


def test_holds_pose_before_any_target(gate):
    current = np.ones(N, dtype=np.float32)
    cmd, status = gate.step(current, now=100.0)
    np.testing.assert_array_equal(cmd, current)
    assert status == "no_target_yet"


def test_velocity_clamp_limits_step_size(gate):
    current = np.zeros(N, dtype=np.float32)
    gate.submit(sample(np.full(N, 1000.0), now=100.0))
    cmd, status = gate.step(current, now=100.0)
    assert status == "ok"
    max_step = np.array(SO101_PROFILE.max_velocity) * DT
    limits_hi = np.array([lim[1] for lim in SO101_PROFILE.joint_limits])
    expected = np.minimum(max_step, limits_hi)  # workspace clamp applies first
    np.testing.assert_allclose(cmd, expected, rtol=1e-5)


def test_workspace_clamp_converges_to_limit(gate):
    current = np.zeros(N, dtype=np.float32)
    target = np.full(N, 1e6, dtype=np.float32)  # absurd target
    hi = np.array([lim[1] for lim in SO101_PROFILE.joint_limits], dtype=np.float32)
    cmd = current
    for i in range(2000):
        gate.submit(sample(target, now=100.0 + i * DT))
        cmd, status = gate.step(cmd, now=100.0 + i * DT)
        assert status == "ok"
    np.testing.assert_allclose(cmd, hi, atol=1e-2)  # never exceeds software limits


def test_stale_target_holds_pose(gate):
    current = np.zeros(N, dtype=np.float32)
    gate.submit(sample(np.ones(N), now=100.0))
    cmd, status = gate.step(current, now=101.0)  # 1s later >> 200ms staleness
    np.testing.assert_array_equal(cmd, current)
    assert status.startswith("stale")


def test_deadman_released_holds_pose(gate):
    current = np.zeros(N, dtype=np.float32)
    gate.submit(sample(np.ones(N), deadman=False, now=100.0))
    cmd, status = gate.step(current, now=100.0)
    np.testing.assert_array_equal(cmd, current)
    assert status == "deadman_released"


def test_low_confidence_holds_pose(gate):
    current = np.zeros(N, dtype=np.float32)
    gate.submit(sample(np.ones(N), confidence=0.1, now=100.0))
    _, status = gate.step(current, now=100.0)
    assert status.startswith("low_confidence")


def test_out_of_order_samples_ignored(gate):
    current = np.zeros(N, dtype=np.float32)
    gate.submit(sample(np.full(N, 10.0), now=100.0, ts_ns=2_000))
    gate.submit(sample(np.full(N, 99.0), now=100.0, ts_ns=1_000))  # older, ignored
    cmd, status = gate.step(current, now=100.0)
    assert status == "ok"
    max_step = np.array(SO101_PROFILE.max_velocity) * DT
    assert np.all(cmd <= np.minimum(np.full(N, 10.0), max_step) + 1e-5)


def test_estop_latch_freezes_arm(gate):
    current = np.zeros(N, dtype=np.float32)
    gate.latch_estop("driver_write_failed")
    gate.submit(sample(np.ones(N), now=100.0))
    cmd, status = gate.step(current, now=100.0)
    np.testing.assert_array_equal(cmd, current)
    assert status.startswith("estop_latched")
    assert "driver_write_failed" in status


def test_estop_not_cleared_by_holding_deadman(gate):
    """Merely keeping the deadman held must NOT clear the latch."""
    current = np.zeros(N, dtype=np.float32)
    gate.submit(sample(np.ones(N), deadman=True, now=100.0))
    gate.latch_estop("driver_write_failed")
    for i in range(10):
        gate.submit(sample(np.ones(N), deadman=True, now=100.0 + i * DT))
        _, status = gate.step(current, now=100.0 + i * DT)
        assert status.startswith("estop_latched")


def test_estop_cleared_by_deadman_release_then_repress(gate):
    current = np.zeros(N, dtype=np.float32)
    gate.latch_estop("driver_write_failed")

    # Release the deadman: still latched, but the release is registered.
    gate.submit(sample(np.ones(N), deadman=False, now=100.0))
    _, status = gate.step(current, now=100.0)
    assert status.startswith("estop_latched")

    # Re-press: latch clears and the gate resumes driving.
    gate.submit(sample(np.full(N, 0.5), deadman=True, now=100.1))
    cmd, status = gate.step(current, now=100.1)
    assert status == "ok"
    assert np.any(cmd != 0)


def test_estop_relatch_resets_acknowledgment(gate):
    """A new latch after a release requires a fresh release+re-press."""
    current = np.zeros(N, dtype=np.float32)
    gate.latch_estop("fault-1")
    gate.submit(sample(np.ones(N), deadman=False, now=100.0))
    gate.step(current, now=100.0)  # release seen
    gate.latch_estop("fault-2")  # re-latch wipes the acknowledgment
    gate.submit(sample(np.ones(N), deadman=True, now=100.1))
    _, status = gate.step(current, now=100.1)
    assert status.startswith("estop_latched")


def test_clear_estop_explicit(gate):
    gate.latch_estop("x")
    gate.clear_estop()
    current = np.zeros(N, dtype=np.float32)
    _, status = gate.step(current, now=100.0)
    assert status == "no_target_yet"
