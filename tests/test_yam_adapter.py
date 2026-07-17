"""YAM adapter (interlatent.adapters.yam) — no hardware, no i2rt/camera SDKs.

Covers the parts that must hold before a real arm is ever attached: the
profile/adapter joint-name contract `base.py` enforces, the configurable topology,
and the pure `_motor_targets` action-writing seam (gripper post-step + delta clamp).
"""
from __future__ import annotations

import queue
import time

import numpy as np
import pytest

from interlatent.adapters.yam.cameras import (
    CameraSpec,
    ThreadedCamera,
    UVCCamera,
    build_camera,
    parse_camera_device,
)
from interlatent.adapters.yam.config import build_adapter_config
from interlatent.adapters.yam.robot import FOLLOWER_HOME_POS, YAMNativeRobot
from interlatent.node.teleop.robot_profile import get_profile


def _adapter(arms: str = "both", **extra) -> YAMNativeRobot:
    return YAMNativeRobot(build_adapter_config({"arms": arms, **extra}, None))


class _FakeArm:
    """Stand-in for an i2rt YAM Robot (7-vector: 6 joints + gripper)."""

    def __init__(self, pos=None):
        self.pos = np.array(pos if pos is not None else [0.0] * 7, dtype=np.float32)
        self.commanded: list[np.ndarray] = []

    def get_joint_pos(self):
        return self.pos.copy()

    def command_joint_pos(self, vec):
        self.commanded.append(np.asarray(vec, dtype=np.float32).copy())

    def update_kp_kd(self, kp, kd):  # pragma: no cover - not exercised here
        pass

    def close(self):  # pragma: no cover
        pass


class _FakeCamera:
    def __init__(self, frame):
        self._frame = frame

    def read(self):
        return self._frame


# --------------------------------------------------------------------------- #
# Topology + the profile/adapter joint-name contract                          #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "arms,kind,n,first,last",
    [
        ("both", "yam", 14, "left_joint_0", "right_gripper"),
        ("left", "yam_left", 7, "left_joint_0", "left_gripper"),
        ("right", "yam_right", 7, "right_joint_0", "right_gripper"),
    ],
)
def test_topology_and_profile_alignment(arms, kind, n, first, last):
    r = _adapter(arms)
    assert r.robot_kind == kind
    feats = r.action_features
    assert len(feats) == n
    bare = [f[:-4] for f in feats]
    assert bare[0] == first and bare[-1] == last
    # joint_specs align 1:1 with action_features; grippers settle on command-issued.
    assert len(r.joint_specs) == n
    assert all(s.name == b for s, b in zip(r.joint_specs, bare))
    assert all(
        (s.control_mode == "gripper") == b.endswith("_gripper")
        for s, b in zip(r.joint_specs, bare)
    )
    # The exact invariant base.py:166 enforces at runtime.
    prof = get_profile(kind)
    assert prof is not None
    assert list(prof.joint_names) == bare


# --------------------------------------------------------------------------- #
# Pure action-writing seam: gripper post-step + delta clamp                   #
# --------------------------------------------------------------------------- #


def test_motor_targets_continuous_passthrough():
    r = _adapter("left", max_step_rad="inf")
    action = {f"left_joint_{i}.pos": 0.1 * i for i in range(6)}
    action["left_gripper.pos"] = 0.3
    targets = r._motor_targets(action)
    vec = targets["left"]
    assert vec.shape == (7,)
    assert vec[6] == pytest.approx(0.3)  # gripper passes through


def test_motor_targets_bangbang_snaps_gripper():
    r = _adapter("left", gripper_mode="bangbang", gripper_threshold="0.5")
    base = {f"left_joint_{i}.pos": 0.0 for i in range(6)}
    assert r._motor_targets({**base, "left_gripper.pos": 0.8})["left"][6] == 1.0
    r2 = _adapter("left", gripper_mode="bangbang", gripper_threshold="0.5")
    assert r2._motor_targets({**base, "left_gripper.pos": 0.2})["left"][6] == 0.0


def test_delta_clamp_limits_arm_step_but_not_gripper():
    r = _adapter("left", max_step_rad="0.1")
    r._last["left"] = np.zeros(6, dtype=np.float32)  # last accepted arm pose
    action = {f"left_joint_{i}.pos": 0.0 for i in range(6)}
    action["left_joint_0.pos"] = 5.0  # huge jump on one joint
    action["left_gripper.pos"] = 1.0  # gripper must NOT be delta-clamped
    vec = r._motor_targets(action)["left"]
    assert vec[0] == pytest.approx(0.1)  # clamped to last + max_step_rad
    assert vec[6] == pytest.approx(1.0)
    # last-accepted updated to where the arm was actually commanded.
    assert r._last["left"][0] == pytest.approx(0.1)


def test_send_action_commands_each_arm():
    r = _adapter("both", max_step_rad="inf")
    r._arms = {"left": _FakeArm(), "right": _FakeArm()}
    action = {f: 0.0 for f in r.action_features}
    action["right_joint_2.pos"] = 0.5
    r.send_action(action)
    assert len(r._arms["left"].commanded) == 1
    assert r._arms["right"].commanded[0][2] == pytest.approx(0.5)


def test_get_observation_joints_and_cameras():
    r = _adapter("left")
    r._arms = {"left": _FakeArm(pos=[1, 2, 3, 4, 5, 6, 0.7])}
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    r._cameras = {"wrist": _FakeCamera(frame)}
    obs = r.get_observation()
    assert obs["left_joint_0.pos"] == pytest.approx(1.0)
    assert obs["left_gripper.pos"] == pytest.approx(0.7)
    assert obs["wrist"] is frame


# --------------------------------------------------------------------------- #
# Manual action() contract (fails before any motion)                          #
# --------------------------------------------------------------------------- #


def test_action_unknown_joint_raises():
    r = _adapter("left")
    r._arms = {"left": _FakeArm()}
    with pytest.raises(ValueError, match="unknown joint"):
        r.action(nope=1.0, hold_missing=True)


def test_action_out_of_range_raises_before_motion():
    r = _adapter("left")
    arm = _FakeArm()
    r._arms = {"left": arm}
    # left_joint_0 limit is ±2.0 rad; 99 is out of range -> raises before commanding.
    with pytest.raises(ValueError, match="outside its limit"):
        r.action(left_joint_0=99.0, hold_missing=True)
    assert arm.commanded == []  # nothing moved


# --------------------------------------------------------------------------- #
# Config / camera parsing                                                      #
# --------------------------------------------------------------------------- #


def test_home_pose_matches_raiden_follower():
    assert FOLLOWER_HOME_POS.tolist() == [0, 0, 0, 0, 0, 0, 1.0]


def test_camera_device_parsing():
    assert parse_camera_device("w", "realsense:123") == CameraSpec("w", "realsense", "123")
    assert parse_camera_device("t", "zed") == CameraSpec("t", "zed", "")
    # Generic UVC/V4L2 webcams: explicit prefix, bare /dev path, and bare index.
    assert parse_camera_device("f", "uvc:/dev/video2") == CameraSpec("f", "uvc", "/dev/video2")
    assert parse_camera_device("f", "/dev/video2") == CameraSpec("f", "uvc", "/dev/video2")
    assert parse_camera_device("f", "webcam:9") == CameraSpec("f", "uvc", "9")
    assert parse_camera_device("f", "2") == CameraSpec("f", "uvc", "2")
    with pytest.raises(ValueError, match="a UVC camera needs a device"):
        parse_camera_device("x", "uvc:")
    with pytest.raises(ValueError, match="camera must be a vendor type"):
        parse_camera_device("x", "potato")


def test_invalid_arms_and_gripper_mode_rejected():
    with pytest.raises(ValueError, match="arms must be"):
        build_adapter_config({"arms": "tri"}, None)
    with pytest.raises(ValueError, match="gripper_mode must be"):
        build_adapter_config({"gripper_mode": "snap"}, None)


# --------------------------------------------------------------------------- #
# ThreadedCamera (per-camera reader thread; read() = latest-frame snapshot)    #
# --------------------------------------------------------------------------- #


class _PumpedCamera:
    """Inner camera for ThreadedCamera tests: read() blocks on a queue, like a
    real driver waiting for the next frame. Queueing an Exception makes the
    next read() raise it (also how tests unblock the reader before disconnect)."""

    def __init__(self):
        self.spec = CameraSpec("fake", "uvc", "9")
        self.frames: queue.Queue = queue.Queue()
        self.disconnected = False

    def connect(self):
        pass

    def read(self):
        item = self.frames.get()
        if isinstance(item, Exception):
            raise item
        return item

    def disconnect(self):
        self.disconnected = True


def test_build_camera_wraps_every_backend_threaded():
    cam = build_camera(CameraSpec("w", "uvc", "/dev/video9"))
    assert isinstance(cam, ThreadedCamera)
    assert isinstance(cam.inner, UVCCamera)


def test_threaded_camera_read_returns_latest_frame():
    inner = _PumpedCamera()
    cam = ThreadedCamera(inner)
    cam.connect()
    try:
        f1 = np.zeros((2, 2, 3), np.uint8)
        inner.frames.put(f1)
        assert cam.read() is f1  # blocks only for the FIRST frame

        f2 = np.ones((2, 2, 3), np.uint8)
        f3 = np.full((2, 2, 3), 2, np.uint8)
        inner.frames.put(f2)
        inner.frames.put(f3)
        deadline = time.monotonic() + 2.0
        while cam.read() is not f3 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert cam.read() is f3  # newest wins; f2 was superseded
    finally:
        inner.frames.put(RuntimeError("shutdown"))
        cam.disconnect()
    assert inner.disconnected


def test_threaded_camera_propagates_reader_error():
    inner = _PumpedCamera()
    cam = ThreadedCamera(inner)
    cam.connect()
    inner.frames.put(RuntimeError("device unplugged"))
    with pytest.raises(RuntimeError, match="reader thread failed"):
        cam.read()
    cam.disconnect()
    assert inner.disconnected


def test_threaded_camera_first_frame_timeout():
    inner = _PumpedCamera()
    cam = ThreadedCamera(inner)
    cam.first_frame_timeout_s = 0.05
    cam.connect()
    try:
        with pytest.raises(RuntimeError, match="no frame within"):
            cam.read()
    finally:
        inner.frames.put(RuntimeError("shutdown"))
        cam.disconnect()


def test_threaded_camera_stale_frame_raises():
    inner = _PumpedCamera()
    cam = ThreadedCamera(inner)
    cam.stale_after_s = 0.05
    cam.connect()
    try:
        inner.frames.put(np.zeros((2, 2, 3), np.uint8))
        assert cam.read() is not None
        time.sleep(0.15)  # device "stalls": reader alive, no new frames
        with pytest.raises(RuntimeError, match="old"):
            cam.read()
    finally:
        inner.frames.put(RuntimeError("shutdown"))
        cam.disconnect()
