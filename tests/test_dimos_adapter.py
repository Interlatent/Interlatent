"""Unit tests for the dimos adapter's bus, robot, and episode-marker layers.

Everything is fake-injected (no dimos installed): a fake transport factory for
DimosBus, a fake bus + fake coordinator RPC for DimosNativeRobot.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest

from interlatent.adapters.dimos.bus import CachedMsg, DimosBus, image_to_rgb
from interlatent.adapters.dimos.config import build_adapter_config
from interlatent.adapters.dimos.episode import EpisodeMarker, publish_marker
from interlatent.adapters.dimos.robot import DimosNativeRobot
from interlatent.node.teleop.robot_profile import get_profile


class FakeJointState:
    """Stands in for dimos.msgs.sensor_msgs.JointState (kwarg constructor)."""

    def __init__(self, name=(), position=(), ts=None):
        self.name = list(name)
        self.position = list(position)
        self.ts = time.time() if ts is None else ts


class FakeTransport:
    def __init__(self, topic, msg_type):
        self.topic = topic
        self.msg_type = msg_type
        self.callbacks = []
        self.broadcasts = []
        self.stopped = False

    def subscribe(self, callback):
        self.callbacks.append(callback)
        return lambda: self.callbacks.remove(callback)

    def broadcast(self, _stream, msg):
        self.broadcasts.append(msg)

    def stop(self):
        self.stopped = True

    def deliver(self, msg):
        for cb in list(self.callbacks):
            cb(msg)


class FakeFactory:
    def __init__(self):
        self.transports: dict[str, FakeTransport] = {}

    def __call__(self, topic, msg_type):
        t = FakeTransport(topic, msg_type)
        self.transports[topic] = t
        return t


def make_bus(cameras=None, **extra):
    cfg = build_adapter_config({"kind": "xarm7", **extra}, cameras or {})
    factory = FakeFactory()
    bus = DimosBus(
        cfg, transport_factory=factory, joint_state_cls=FakeJointState,
        image_cls=object,
    )
    bus.open()
    return cfg, factory, bus


# ---------------------------------------------------------------------------
# bus
# ---------------------------------------------------------------------------


def test_bus_opens_expected_topics():
    _, factory, bus = make_bus(cameras={"wrist": "/camera/wrist/color"})
    assert set(factory.transports) == {
        "/coordinator_joint_state",
        "/camera/wrist/color",
        "/joint_command",
        "/interlatent/episode",
    }
    # Episode topic is the pickled transport (msg_type None).
    assert factory.transports["/interlatent/episode"].msg_type is None
    bus.close()
    assert all(t.stopped for t in factory.transports.values())


def test_bus_latest_wins_and_age():
    _, factory, bus = make_bus()
    js = factory.transports["/coordinator_joint_state"]
    assert bus.latest_joint_state() is None
    assert bus.joint_state_age_ms() is None
    js.deliver(FakeJointState(name=["arm/joint1"], position=[0.1]))
    js.deliver(FakeJointState(name=["arm/joint1"], position=[0.2]))
    cached = bus.latest_joint_state()
    assert cached.msg.position == [0.2]  # newest wins
    assert bus.joint_state_age_ms() < 1000.0


def test_bus_publish_joint_command_builds_typed_msg():
    _, factory, bus = make_bus()
    bus.publish_joint_command(["arm/joint1", "arm/joint2"], [0.1, 0.2])
    (msg,) = factory.transports["/joint_command"].broadcasts
    assert isinstance(msg, FakeJointState)
    assert msg.name == ["arm/joint1", "arm/joint2"]
    assert msg.position == [0.1, 0.2]


def test_image_to_rgb_formats():
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    out = image_to_rgb(SimpleNamespace(data=rgb, format=SimpleNamespace(value="RGB")))
    assert out[0, 0].tolist() == [255, 0, 0]
    bgr = np.zeros((4, 4, 3), dtype=np.uint8)
    bgr[..., 0] = 255  # blue in BGR
    out = image_to_rgb(SimpleNamespace(data=bgr, format=SimpleNamespace(value="BGR")))
    assert out[0, 0].tolist() == [0, 0, 255]
    rgba = np.zeros((2, 2, 4), dtype=np.uint8)
    out = image_to_rgb(SimpleNamespace(data=rgba, format=SimpleNamespace(value="RGBA")))
    assert out.shape == (2, 2, 3)
    gray = np.full((2, 2), 7, dtype=np.uint8)
    out = image_to_rgb(SimpleNamespace(data=gray, format=SimpleNamespace(value="GRAY")))
    assert out.shape == (2, 2, 3) and out.dtype == np.uint8


# ---------------------------------------------------------------------------
# robot (fake bus + fake coordinator RPC)
# ---------------------------------------------------------------------------


class FakeBus:
    def __init__(self):
        self.joint_state = None
        self.images = {}
        self.commands = []
        self.markers = []
        self.opened = False
        self.closed = False

    def open(self):
        self.opened = True

    def close(self):
        self.closed = True

    def latest_joint_state(self):
        return self.joint_state

    def latest_image(self, name):
        return self.images.get(name)

    def joint_state_age_ms(self):
        if self.joint_state is None:
            return None
        return (time.monotonic() - self.joint_state.arrival_monotonic) * 1000.0

    def image_age_ms(self, name):
        cached = self.images.get(name)
        if cached is None:
            return None
        return (time.monotonic() - cached.arrival_monotonic) * 1000.0

    def publish_joint_command(self, names, positions):
        self.commands.append((list(names), list(positions)))

    def publish_episode_marker(self, marker):
        self.markers.append(marker)


class FakeCoordinator:
    def __init__(self):
        self.gripper_calls = []
        self.activated = True

    def set_gripper_position(self, hardware_id, position):
        self.gripper_calls.append((hardware_id, position))

    def get_gripper_position(self, hardware_id):
        return 0.4

    def set_activated(self, active):
        self.activated = active


def fresh_joint_state(positions=None):
    names = [f"arm/joint{i}" for i in range(1, 8)]
    pos = positions if positions is not None else [0.0] * 7
    return CachedMsg(
        FakeJointState(name=names, position=pos), time.monotonic(), time.time()
    )


def make_robot(cameras=None, **extra):
    cfg = build_adapter_config({"kind": "xarm7", **extra}, cameras or {})
    bus = FakeBus()
    bus.joint_state = fresh_joint_state()
    coordinator = FakeCoordinator()
    verify_calls = []
    robot = DimosNativeRobot(
        cfg,
        bus=bus,
        verify_fn=lambda *a, **k: verify_calls.append(a),
        coordinator_client_factory=lambda: coordinator,
    )
    return robot, bus, coordinator, verify_calls


def test_action_features_match_profile_order():
    robot, *_ = make_robot()
    profile = get_profile("dimos_xarm7")
    assert [f[: -len(".pos")] for f in robot.action_features] == list(
        profile.joint_names
    )
    # The exact invariant ManualActionInterface.action() enforces.
    assert robot.robot_kind == "dimos_xarm7"


def test_connect_runs_verify_and_starts(monkeypatch):
    robot, bus, _, verify_calls = make_robot()
    robot.connect()
    assert bus.opened and verify_calls
    assert robot.is_connected
    robot.disconnect()
    assert bus.closed and not robot.is_connected


def test_connect_verify_failure_closes_bus():
    cfg = build_adapter_config({"kind": "xarm7"}, {})
    bus = FakeBus()
    bus.joint_state = fresh_joint_state()

    def boom(*a, **k):
        raise RuntimeError("declared kind does not match")

    robot = DimosNativeRobot(
        cfg, bus=bus, verify_fn=boom, coordinator_client_factory=FakeCoordinator
    )
    with pytest.raises(RuntimeError, match="does not match"):
        robot.connect()
    assert bus.closed and not robot.is_connected


def test_first_send_unclamped_then_delta_clamped():
    robot, bus, _, _ = make_robot()
    robot.connect()
    action = {f: 0.0 for f in robot.action_features}
    action["arm_joint1.pos"] = 1.0  # huge first target: allowed (no baseline)
    robot.send_action(action)
    names, positions = bus.commands[-1]
    assert positions[0] == pytest.approx(1.0)
    assert names[0] == "arm/joint1"
    assert names[-1] == "arm/gripper"  # gripper rides the same message
    # Second send: delta beyond max_step_rad (0.05) clamps against last-accepted.
    action["arm_joint1.pos"] = 2.0
    robot.send_action(action)
    _, positions = bus.commands[-1]
    assert positions[0] == pytest.approx(1.05)


def test_gripper_rides_joint_command_and_is_exempt_from_clamp():
    """The gripper is a claimed joint on the wire (dimos's per-tick hardware
    write stomps any out-of-band gripper RPC while streaming), and it is
    exempt from the max_step_rad delta clamp."""
    robot, bus, coordinator, _ = make_robot()
    robot.connect()
    action = {f: 0.0 for f in robot.action_features}
    action["arm_gripper.pos"] = 0.85  # full-range jump, must NOT be clamped
    robot.send_action(action)
    names, positions = bus.commands[-1]
    assert names[-1] == "arm/gripper"
    assert positions[-1] == pytest.approx(0.85)
    # And never via the coordinator RPC (read-only territory).
    time.sleep(0.1)
    assert coordinator.gripper_calls == []
    robot.disconnect()


def test_observation_maps_names_and_serves_last_commanded_gripper():
    robot, bus, _, _ = make_robot()
    robot.connect()
    bus.joint_state = fresh_joint_state([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    obs = robot.get_observation()
    assert obs["arm_joint1.pos"] == pytest.approx(0.1)
    assert obs["arm_joint7.pos"] == pytest.approx(0.7)
    # Gripper not in the stream: last commanded (none yet -> profile rest).
    assert obs["arm_gripper.pos"] == pytest.approx(0.85)
    action = {f: 0.0 for f in robot.action_features}
    action["arm_gripper.pos"] = 0.2
    robot.send_action(action)
    assert robot.get_observation()["arm_gripper.pos"] == pytest.approx(0.2)
    robot.disconnect()


def test_telemetry_fresh_gates_on_arrival_age():
    robot, bus, _, _ = make_robot(staleness_ms="50")
    robot.connect()
    assert robot.telemetry_fresh
    stale = CachedMsg(bus.joint_state.msg, time.monotonic() - 1.0, time.time())
    bus.joint_state = stale
    assert not robot.telemetry_fresh
    robot.disconnect()


def test_estop_deactivates_coordinator():
    robot, _, coordinator, _ = make_robot()
    robot.connect()
    robot.estop()
    assert coordinator.activated is False
    robot.disconnect()


# ---------------------------------------------------------------------------
# episode markers
# ---------------------------------------------------------------------------


def test_publish_marker_best_effort_never_raises():
    bus = FakeBus()
    publish_marker(bus, "ep-1", "start", "dimos_xarm7")
    (marker,) = bus.markers
    assert isinstance(marker, EpisodeMarker)
    assert (marker.episode_id, marker.event) == ("ep-1", "start")
    assert marker.schema == 1 and marker.source == "interlatent"

    class ExplodingBus:
        def publish_episode_marker(self, marker):
            raise RuntimeError("bus down")

    publish_marker(ExplodingBus(), "ep-1", "stop", "dimos_xarm7")  # must not raise


def test_loop_publishes_start_and_stop_markers(monkeypatch):
    """The native loop brackets every episode with bus markers (ADR 0018)."""
    import interlatent.adapters.dimos.robot as robot_mod
    from interlatent.adapters.dimos.loop import control_loop

    bus = FakeBus()
    bus.joint_state = fresh_joint_state()

    class LoopFakeRobot:
        robot_kind = "dimos_xarm7"
        _bus = bus
        telemetry_fresh = True
        obs_age_ms = 0.0
        action_features = [f"arm_joint{i}.pos" for i in range(1, 8)] + [
            "arm_gripper.pos"
        ]

        def __init__(self, cfg):
            pass

        def connect(self):
            pass

        def disconnect(self):
            pass

        def get_observation(self):
            return {f: 0.0 for f in self.action_features}

        def send_action(self, action):
            return action

    monkeypatch.setattr(robot_mod, "DimosNativeRobot", LoopFakeRobot)

    class FakeClient:
        schedule = SimpleNamespace(flush=lambda: None)

        def step(self, encode, codec="npz"):
            return None  # no action yet

    ticks = {"n": 0}

    def should_stop():
        ticks["n"] += 1
        return ticks["n"] > 2

    control_loop(
        client=FakeClient(),
        session={"id": "ep-42", "fps": 100},
        should_stop=should_stop,
        robot_kind="dimos",
        robot_extra={"kind": "xarm7"},
        robot_cameras={},
    )
    events = [(m.episode_id, m.event) for m in bus.markers]
    assert events[0] == ("ep-42", "start")
    assert events[-1] == ("ep-42", "stop")
