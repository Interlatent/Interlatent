"""``YAMNativeRobot`` — drives I2RT's YAM bimanual arms via the ``i2rt`` driver.

A thin, synchronous adapter behind the same interface the inference loop uses
(``connect`` / ``get_observation`` / ``send_action`` / ``disconnect`` /
``action_features``). Each YAM follower is a 7-DOF arm (6 revolute + 1 gripper)
spoken to over its own CAN bus through I2RT's ``get_yam_robot`` — the exact path
raiden's ``scripts/read_arm_poses.py`` uses — so this adapter pulls in **only**
``i2rt`` (+ a camera SDK), never raiden's heavier teleop/IK/serving stack.

The observation/action dict uses bare ``<side>_<joint>.pos`` keys, left arm then
right arm, with the gripper last in each 7-block (matching raiden's
``FOLLOWER_HOME_POS = [0]*6 + [1.0]``). All actions are joint-space (radians for the
revolute joints, gripper in [0, 1]); there is no IK here.

``send_action`` adds the swappable obs/action seam: a configurable gripper post-step
and a ``max_step_rad`` per-step delta clamp on the arm joints (execution safety). A
non-joint action space (e.g. EE poses) would decode here — but that is deliberately
out of scope (ADR 0013: the robot side is joint-space only).

Import-weight note: ``i2rt`` and the camera SDKs are imported lazily inside methods,
so importing this module never requires the ``[yam]`` extra.
"""
from __future__ import annotations

import logging
import subprocess
import time
from typing import Any

import numpy as np

from ..._clamp_log import warn_clamp
from ..base import JointSpec, ManualActionInterface
from .cameras import build_camera
from .config import YAMAdapterConfig

_logger = logging.getLogger(__name__)

# Per-joint PD gains (6 revolute + gripper), pinned to raiden's follower values.
FOLLOWER_KP = np.array([80.0, 80.0, 80.0, 40.0, 10.0, 10.0, 20.0])
FOLLOWER_KD = np.array([5.0, 5.0, 5.0, 1.5, 1.5, 1.5, 0.5])
# Rest/home pose (6 zeros + gripper open), from raiden's FOLLOWER_HOME_POS.
FOLLOWER_HOME_POS = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

_ARM_DOF = 6  # revolute joints; gripper is index 6 of the 7-vector
_GRIPPER_IDX = 6


class YAMNativeRobot(ManualActionInterface):
    """Native-``i2rt`` YAM robot for the DRTC inference loop.

    Implements the formal :class:`~interlatent.adapters.base.RobotAdapter` duck type
    and inherits the manual block-then-settle ``action()`` from
    :class:`~interlatent.adapters.base.ManualActionInterface`. Manual ``action()``
    requires a YAM :class:`RobotProfile` (``yam`` / ``yam_left`` / ``yam_right`` in
    :mod:`interlatent.node.teleop.robot_profile`); the per-instance ``robot_kind``
    selects which one. The engine path (``send_action`` per tick) needs no profile.
    """

    def __init__(self, config: YAMAdapterConfig) -> None:
        self.config = config
        self._sides = config.active_sides  # ("left",) / ("right",) / ("left","right")

        # robot_kind selects the matching profile topology.
        self.robot_kind = "yam" if self._sides == ("left", "right") else f"yam_{self._sides[0]}"

        # Per-side ordered "<side>_<joint>.pos" keys (arm joints then gripper).
        self._side_keys: dict[str, list[str]] = {
            side: [f"{side}_joint_{i}.pos" for i in range(_ARM_DOF)] + [f"{side}_gripper.pos"]
            for side in self._sides
        }
        self._channels: dict[str, str] = {
            "left": config.left_channel,
            "right": config.right_channel,
        }

        self._arms: dict[str, Any] = {}  # side -> i2rt Robot
        self._cameras: dict[str, Any] = {}  # name -> Camera
        self._gripper_mode = config.gripper_mode
        self._gripper_threshold = config.gripper_threshold
        self._max_step_rad = float(config.max_step_rad)
        self._last: dict[str, np.ndarray | None] = {s: None for s in self._sides}
        self._drop_count = 0

    @property
    def is_connected(self) -> bool:
        return bool(self._arms)

    @property
    def action_features(self) -> list[str]:
        """Ordered action-feature names (left block then right, gripper last)."""
        keys: list[str] = []
        for side in self._sides:
            keys.extend(self._side_keys[side])
        return keys

    @property
    def joint_specs(self) -> list[JointSpec]:
        """Per-joint settle metadata, aligned with :attr:`action_features`.

        Arm joints settle by position tolerance (radians); the gripper settles on
        "command issued" (a gripper closing on an object never reaches a position
        target). Joint *ranges* are not declared here — they come from the YAM
        :class:`RobotProfile`.
        """
        specs: list[JointSpec] = []
        for side in self._sides:
            for key in self._side_keys[side]:
                name = key[: -len(".pos")]
                if key.endswith("_gripper.pos"):
                    specs.append(JointSpec(name=name, control_mode="gripper"))
                else:
                    specs.append(
                        JointSpec(name=name, control_mode="position", settle_tolerance=0.05)
                    )
        return specs

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Preflight CAN, open each arm, set gains, open cameras, optionally home."""
        from i2rt.robots.get_robot import get_yam_robot
        from i2rt.robots.utils import GripperType

        self._check_can_interfaces()

        for side in self._sides:
            channel = self._channels[side]
            arm = get_yam_robot(channel=channel, gripper_type=GripperType.LINEAR_4310)
            arm.update_kp_kd(kp=FOLLOWER_KP, kd=FOLLOWER_KD)
            self._arms[side] = arm
            _logger.info("YAM %s follower connected on %s", side, channel)

        self._open_cameras()

        if self.config.auto_home:
            self._home()

        _logger.info(
            "YAMNativeRobot connected (arms=%s, cameras=%s, auto_home=%s).",
            list(self._arms), list(self._cameras), self.config.auto_home,
        )

    def _check_can_interfaces(self) -> None:
        """Ensure every required CAN bus is up, bringing it up ourselves if we can.

        A down (but present) interface gets one non-interactive
        ``sudo ip link set <iface> up type can bitrate 1000000`` attempt —
        ``sudo -n`` so a password prompt can never hang connect(). Only if the
        interface is still not up afterwards do we raise.
        """
        missing: list[str] = []
        for side in self._sides:
            channel = self._channels[side]
            up, detail = self._can_interface_up(channel)
            if detail is not None:  # no `ip` at all (non-Linux host)
                missing.append(detail)
                continue
            if up:
                continue

            _logger.warning(
                "YAM CAN interface %s is not up; attempting "
                "`sudo ip link set %s up type can bitrate 1000000`",
                channel, channel,
            )
            result = subprocess.run(
                ["sudo", "-n", "ip", "link", "set", channel,
                 "up", "type", "can", "bitrate", "1000000"],
                capture_output=True, text=True, check=False,
            )
            up, _ = self._can_interface_up(channel)
            if up:
                _logger.info("YAM CAN interface %s brought up successfully.", channel)
                continue
            err = (result.stderr or result.stdout).strip()
            missing.append(f"{channel} ({err})" if err else channel)

        if missing:
            raise RuntimeError(
                "YAM CAN adapter(s) are not currently set up, and bringing them "
                "up automatically failed: "
                + ", ".join(missing)
                + ". Bring them up manually (persistent udev names + "
                "`sudo ip link set <iface> up type can bitrate 1000000`, e.g. via "
                "raiden's `rd reset_can` or the i2rt udev setup)."
            )

    @staticmethod
    def _can_interface_up(channel: str) -> tuple[bool, str | None]:
        """(is_up, error) for one interface; error is set only when `ip` is missing."""
        try:
            result = subprocess.run(
                ["ip", "link", "show", channel],
                capture_output=True, text=True, check=False,
            )
        except FileNotFoundError:  # no `ip` (non-Linux host)
            return False, f"{channel} (could not run `ip`; YAM needs Linux + SocketCAN)"
        up = result.returncode == 0 and (
            "state UP" in result.stdout or "state UNKNOWN" in result.stdout
        )
        return up, None

    def _open_cameras(self) -> None:
        for name, spec in self.config.cameras.items():
            cam = build_camera(spec)
            cam.connect()
            self._cameras[name] = cam

    def _home(self) -> None:
        """Smoothly move every active arm to the rest pose (raiden FOLLOWER_HOME_POS)."""
        for side, arm in self._arms.items():
            self._smooth_move(arm, FOLLOWER_HOME_POS)
            self._last[side] = FOLLOWER_HOME_POS[:_ARM_DOF].astype(np.float32).copy()

    @staticmethod
    def _smooth_move(arm: Any, target: np.ndarray, *, steps: int = 200, duration_s: float = 5.0) -> None:
        """Linearly interpolate from the measured pose to ``target`` (raiden-style)."""
        current = np.asarray(arm.get_joint_pos(), dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        for i in range(steps + 1):
            alpha = i / steps
            arm.command_joint_pos((1 - alpha) * current + alpha * target)
            if i < steps:
                time.sleep(duration_s / steps)

    def disconnect(self) -> None:
        """Close cameras and arms (motors power down on close)."""
        for cam in self._cameras.values():
            try:
                cam.disconnect()
            except Exception:  # noqa: BLE001
                _logger.warning("YAM camera disconnect failed", exc_info=True)
        self._cameras = {}

        for side, arm in self._arms.items():
            try:
                arm.close()
            except ValueError:
                # i2rt raises ValueError when the motors no longer answer on
                # the CAN bus — i.e. the yams are already powered off.
                _logger.info("YAM %s arm already powered off; nothing to close.", side)
            except Exception:  # noqa: BLE001
                _logger.warning("YAM %s arm close failed", side, exc_info=True)
        self._arms = {}
        _logger.info("YAMNativeRobot disconnected.")

    # ------------------------------------------------------------------
    # Observation / action
    # ------------------------------------------------------------------

    def get_observation(self) -> dict[str, Any]:
        """Return the active arms' joint positions + camera RGB frames."""
        obs: dict[str, Any] = {}
        for side in self._sides:
            pos = np.asarray(self._arms[side].get_joint_pos(), dtype=np.float32)
            for i, key in enumerate(self._side_keys[side]):
                obs[key] = float(pos[i])
        for name, cam in self._cameras.items():
            obs[name] = cam.read()
        return obs

    def _motor_targets(self, action: dict[str, Any]) -> dict[str, np.ndarray]:
        """Map an action dict to per-side ``(7,)`` joint-command arrays.

        Pure (no hardware): applies the configurable gripper post-step and the
        ``max_step_rad`` delta clamp on the arm joints, and returns exactly the arrays
        ``send_action`` hands to ``command_joint_pos``. The gripper (index 6) is not
        delta-clamped. Factored out so the action-writing behaviour is unit-testable.
        """
        targets: dict[str, np.ndarray] = {}
        for side in self._sides:
            keys = self._side_keys[side]
            vec = np.array([float(action[k]) for k in keys], dtype=np.float32)
            if self._gripper_mode == "bangbang":
                vec[_GRIPPER_IDX] = 1.0 if vec[_GRIPPER_IDX] >= self._gripper_threshold else 0.0
            arm = self._clamp_oversized_step(side, vec[:_ARM_DOF], keys, self._last[side])
            vec[:_ARM_DOF] = arm
            self._last[side] = arm
            targets[side] = vec
        return targets

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Apply a joint-target action dict via i2rt ``command_joint_pos`` per arm."""
        targets = self._motor_targets(action)
        for side, vec in targets.items():
            self._arms[side].command_joint_pos(vec)
        return action

    def _clamp_oversized_step(
        self, side: str, arm_cur: np.ndarray, keys: list[str], last: np.ndarray | None
    ) -> np.ndarray:
        """Clamp any arm-joint delta exceeding ``max_step_rad`` (execution safety).

        Limits each arm joint so ``|target - last| <= max_step_rad`` — the arm
        advances toward the commanded target by at most one step instead of slamming
        on a model glitch, bad chunk, or teleop frame. The returned array is both sent
        and remembered as "last accepted" so the next delta is measured from where the
        arm was actually told to go.
        """
        if last is None or not np.isfinite(self._max_step_rad):
            return arm_cur.astype(np.float32)
        delta = arm_cur - last
        if np.any(np.abs(delta) > self._max_step_rad):
            self._drop_count += 1
            j = int(np.argmax(np.abs(delta)))
            warn_clamp(
                f"yam:{side}",
                "YAM %s action exceeds max_step_rad=%.3f at %s (Δ=%.3f rad); clamped "
                "to the per-step limit (adapter clamp #%d).",
                side, self._max_step_rad, keys[j], float(delta[j]), self._drop_count,
            )
            return (last + np.clip(delta, -self._max_step_rad, self._max_step_rad)).astype(np.float32)
        return arm_cur.astype(np.float32)


__all__ = ["YAMNativeRobot", "FOLLOWER_KP", "FOLLOWER_KD", "FOLLOWER_HOME_POS"]
