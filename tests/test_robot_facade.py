"""The ``interlatent.Robot`` facade + bus arbitration — all hardware-free.

The adapter is faked and the bus lock is stubbed (or pointed at a tmp dir) so these
run in CI with no serial port and no cloud.
"""
from __future__ import annotations

import os

import pytest

from behavior_fakes import FakeAdapter

import interlatent as il
import interlatent.adapters as adapters_mod
import interlatent.robot as robot_mod
from interlatent.behaviors import arbitration as arb
from interlatent.behaviors.registry import behavior
from interlatent.behaviors.schema import BehaviorValidationError


@pytest.fixture
def fake_env(monkeypatch):
    """Patch adapter resolution to the fake and neuter the bus lock."""
    created: list[FakeAdapter] = []

    def _resolve(robot_type, *, port=None, extra=None, cameras=None):
        a = FakeAdapter()
        created.append(a)
        return a

    monkeypatch.setattr(adapters_mod, "resolve_adapter", _resolve)
    monkeypatch.setattr(robot_mod, "acquire_bus_lock", lambda *a, **k: arb.BusLock(None))
    return created


def test_context_manager_pose_and_behaviors(fake_env):
    with il.Robot("so101", port="/dev/null", realtime=False) as robot:
        assert set(["home", "hello"]).issubset(robot.behaviors())
        pose = robot.pose()
        assert set(pose) == {
            "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"
        }
        res = robot.act("home")
        assert res.reached
    # Adapter was disconnected on __exit__.
    assert fake_env[0].connected is False


def test_move_ad_hoc_and_validation(fake_env):
    robot = il.Robot("so101", port="/dev/null", realtime=False)
    try:
        res = robot.move(wrist_roll=30.0, duration=0.5)
        assert res.reached
        assert robot.pose()["wrist_roll"] == pytest.approx(30.0, abs=1e-3)
        with pytest.raises(BehaviorValidationError):
            robot.move(nope=1.0)  # unknown joint
        with pytest.raises(BehaviorValidationError):
            robot.move(shoulder_pan=999.0)  # out of limit
        with pytest.raises(ValueError):
            robot.move()  # nothing to move
    finally:
        robot.close()


def test_act_unknown_behavior_lists_available(fake_env):
    robot = il.Robot("so101", port="/dev/null", realtime=False)
    try:
        with pytest.raises(BehaviorValidationError) as e:
            robot.act("does_not_exist")
        assert "home" in str(e.value)
    finally:
        robot.close()


def test_procedural_behavior_via_facade(fake_env):
    @behavior("nod_face", robot="so101")
    def nod(robot):
        robot.move(wrist_flex=-10.0, duration=0.2)
        robot.move(wrist_flex=0.0, duration=0.2)

    from interlatent.behaviors import registry as registry_mod

    try:
        robot = il.Robot("so101", port="/dev/null", realtime=False)
        res = robot.act("nod_face")
        assert res.reached
        assert robot.pose()["wrist_flex"] == pytest.approx(0.0, abs=1e-3)
        robot.close()
    finally:
        registry_mod._PROCEDURAL.pop(("so101", "nod_face"), None)


def test_close_is_idempotent(fake_env):
    robot = il.Robot("so101", port="/dev/null", realtime=False)
    robot.close()
    robot.close()  # no raise


# ---------------------------------------------------------------------------
# Arbitration
# ---------------------------------------------------------------------------


def test_bus_lock_detects_conflict_and_force_overrides(tmp_path, monkeypatch):
    monkeypatch.setattr(arb, "_LOCK_DIR", tmp_path)
    monkeypatch.setattr(arb, "_pid_alive", lambda pid: True)  # pretend the holder lives
    # A different process already holds the lock.
    (tmp_path / "dev_ttyACM0.lock").write_text("999999")
    with pytest.raises(arb.RobotBusyError):
        arb.acquire_bus_lock("so101", "/dev/ttyACM0")
    # force=True overrides and takes the lock for us.
    lock = arb.acquire_bus_lock("so101", "/dev/ttyACM0", force=True)
    assert (tmp_path / "dev_ttyACM0.lock").read_text() == str(os.getpid())
    lock.release()
    assert not (tmp_path / "dev_ttyACM0.lock").exists()


def test_bus_lock_free_when_no_holder(tmp_path, monkeypatch):
    monkeypatch.setattr(arb, "_LOCK_DIR", tmp_path)
    lock = arb.acquire_bus_lock("so101", "/dev/ttyACM0")
    assert (tmp_path / "dev_ttyACM0.lock").is_file()
    lock.release()
