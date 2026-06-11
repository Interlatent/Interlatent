"""SO-101 follower driver wrapper + mock driver.

The real driver uses lerobot. Imports are lazy so the package stays
importable on machines without lerobot installed (e.g. for laptop-only
unit tests, or for running the mock driver to demo the gRPC plumbing
without a real arm attached).
"""
from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

from ..common.config import SO101_JOINT_NAMES

_LOG = logging.getLogger("interlatent_teleop.pi.driver")


class RobotDriver(Protocol):
    joint_names: tuple[str, ...]

    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def read_joints(self) -> np.ndarray: ...
    def write_joints(self, joints: np.ndarray) -> None: ...


class MockDriver:
    """Fake SO-101 that integrates whatever we command, with no dynamics.

    Useful for running the Pi server on a dev laptop with no hardware.
    """

    def __init__(self, joint_names: tuple[str, ...] = SO101_JOINT_NAMES) -> None:
        self.joint_names = joint_names
        self._state = np.zeros(len(joint_names), dtype=np.float32)

    def connect(self) -> None:
        _LOG.info("MockDriver connected (no hardware)")

    def disconnect(self) -> None:
        _LOG.info("MockDriver disconnected")

    def read_joints(self) -> np.ndarray:
        return self._state.copy()

    def write_joints(self, joints: np.ndarray) -> None:
        self._state = joints.astype(np.float32).copy()


class SO101Driver:
    """LeRobot-backed SO-101 follower driver.

    Mirrors the import pattern used in
    interlatent-sdk/src/interlatent/node/control.py so it works against
    both old and new lerobot layouts.
    """

    joint_names: tuple[str, ...] = SO101_JOINT_NAMES

    def __init__(self, port: str, robot_id: str = "so101_teleop") -> None:
        self.port = port
        self.robot_id = robot_id
        self._robot = None  # set in connect()

    def connect(self) -> None:
        try:
            from lerobot.robots import make_robot_from_config
        except ImportError as e:
            raise RuntimeError(
                "lerobot is not installed. `pip install interlatent-teleop[so101]` "
                "or install lerobot manually."
            ) from e

        try:
            from lerobot.robots.so_follower import SO101FollowerConfig
        except ImportError:
            from lerobot.robots.so101_follower import SO101FollowerConfig

        cfg = SO101FollowerConfig(port=self.port, id=self.robot_id)
        self._robot = make_robot_from_config(cfg)
        self._robot.connect()
        _LOG.info("SO101Driver connected on %s", self.port)

    def disconnect(self) -> None:
        if self._robot is not None:
            try:
                self._robot.disconnect()
            finally:
                self._robot = None

    def set_motor_p_gains(self, gains: dict[str, int]) -> None:
        """Override Feetech P_Coefficient on selected motors.

        LeRobot defaults SO-101 motors to P_Coefficient=16 to avoid
        shakiness (see `lerobot/robots/so_follower/so_follower.py`),
        which works fine for a static arm but is too soft for the
        gravity-loaded joints to lift the arm against gravity.

        Bumping P_Coefficient to 32 (Feetech default) or higher gives
        shoulder_lift and elbow_flex enough holding torque to actually
        track upward commands.

        Args:
            gains: motor_name -> P_Coefficient (0-255). Names must
                match the motors registered with the bus
                (e.g. "shoulder_lift", "elbow_flex").
        """
        if self._robot is None:
            raise RuntimeError("driver not connected")
        bus = self._robot.bus
        with bus.torque_disabled():
            for motor_name, gain in gains.items():
                if motor_name not in bus.motors:
                    _LOG.warning("motor %r not on the bus (have: %s)",
                                 motor_name, list(bus.motors.keys()))
                    continue
                bus.write("P_Coefficient", motor_name, int(gain))
                _LOG.info("set P_Coefficient[%s] = %d", motor_name, int(gain))

    def read_joints(self) -> np.ndarray:
        if self._robot is None:
            raise RuntimeError("SO101Driver not connected")
        obs = self._robot.get_observation()
        # lerobot returns `<motor>.pos` scalars in degrees.
        return np.array(
            [float(obs[f"{name}.pos"]) for name in self.joint_names],
            dtype=np.float32,
        )

    def write_joints(self, joints: np.ndarray) -> None:
        if self._robot is None:
            raise RuntimeError("SO101Driver not connected")
        action = {f"{name}.pos": float(j) for name, j in zip(self.joint_names, joints)}
        self._robot.send_action(action)


def build_driver(kind: str, port: str = "", robot_id: str = "so101_teleop") -> RobotDriver:
    if kind == "mock":
        return MockDriver()
    if kind == "so101":
        if not port:
            raise ValueError("--port is required for so101 driver")
        return SO101Driver(port=port, robot_id=robot_id)
    raise ValueError(f"unknown driver kind: {kind!r}")
