"""``AxolNativeRobot`` — drives the Almond Axol via its native async SDK.

Wraps ``almond_axol.robot.Axol`` (CAN/impedance motor control, gravity comp,
telemetry) behind a small synchronous interface the inference loop uses:
``connect`` / ``get_observation`` / ``send_action`` / ``disconnect`` /
``action_features``. Cameras are the native ``almond_axol.lerobot.camera``
``ZedCamera`` / ``ZedStereoCamera``, opened **onboard the Jetson by serial
number** (this is why the adapter now pulls in ``lerobot`` through those camera
classes — the two-box ZED-stream receiver it used before is gone).

Like the vendor's ``AxolRobot``, the async ``Axol`` runs on a dedicated asyncio
event loop in a background thread so CAN telemetry keeps streaming while the
synchronous loop blocks on ``get_observation`` / ``send_action``.

The observation/action dict uses the same 16 ``*.pos`` keys in the same order
as the vendor wrapper (left Joint-order, then right), so ``observation.state``,
``action``, and the reported feature-element-names are identical to data
recorded through ``AxolRobot`` — the policy sees what it was trained on.

``send_action`` adds two adapter concerns (the swappable obs/action seam): a
configurable gripper post-step and ``max_step_rad`` drop-logging. A non-joint
action space (e.g. EE poses → ``KinematicsSolver.ik``) would decode here.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any

import numpy as np

from ..base import JointSpec, ManualActionInterface
from .config import AxolAdapterConfig

_logger = logging.getLogger(__name__)

# Joint-position keys, derived from the Joint enum exactly as the vendor wrapper
# does (left arm 7 + gripper, then right). Built lazily to keep almond_axol off
# the import path until a robot is actually constructed.
_VALID_GRIPPER_MODES = ("continuous", "bangbang")


def _pos_keys() -> tuple[list[str], list[str], list[str], list[str], str, str]:
    from almond_axol.utils.shared import ARM_JOINTS, Joint

    joints = list(Joint)
    left = [f"left_{j.value}.pos" for j in joints]
    right = [f"right_{j.value}.pos" for j in joints]
    left_arm = [f"left_{j.value}.pos" for j in ARM_JOINTS]
    right_arm = [f"right_{j.value}.pos" for j in ARM_JOINTS]
    left_grip = f"left_{Joint.GRIPPER.value}.pos"
    right_grip = f"right_{Joint.GRIPPER.value}.pos"
    return left, right, left_arm, right_arm, left_grip, right_grip


class AxolNativeRobot(ManualActionInterface):
    """Native-SDK Axol robot for the DRTC inference loop.

    Implements the formal :class:`~interlatent.adapters.base.RobotAdapter` duck type
    and inherits the manual block-then-settle ``action()`` from
    :class:`~interlatent.adapters.base.ManualActionInterface`. Manual ``action()``
    additionally requires an axol :class:`RobotProfile` (joint limits / velocity caps)
    in ``interlatent.node.teleop.robot_profile``; until one is added with real
    hardware values it fails closed (``action()`` raises rather than driving the arm
    unguarded). The engine path (``send_action`` per tick) does not need a profile.
    """

    robot_kind = "axol"

    def __init__(self, config: AxolAdapterConfig) -> None:
        if config.gripper_mode not in _VALID_GRIPPER_MODES:
            raise ValueError(
                f"gripper_mode must be one of {_VALID_GRIPPER_MODES}, got "
                f"{config.gripper_mode!r}"
            )
        self.config = config
        (
            self._left_keys,
            self._right_keys,
            self._left_arm_keys,
            self._right_arm_keys,
            self._left_grip_key,
            self._right_grip_key,
        ) = _pos_keys()

        self._axol: Any | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        # ZED cameras (one per device) + obs-key → view map (stereo expands).
        self._cameras: list[Any] = []
        self._cam_views: dict[str, Any] = {}

        self._gripper_mode = config.gripper_mode
        self._gripper_threshold = config.gripper_threshold
        self._max_step_rad = float(getattr(config.axol_config, "max_step_rad", 0.5))
        self._last_left: np.ndarray | None = None
        self._last_right: np.ndarray | None = None
        self._drop_count = 0

    @property
    def is_connected(self) -> bool:
        return self._axol is not None

    @property
    def action_features(self) -> list[str]:
        """Ordered action-feature names (the 16 ``*.pos`` keys)."""
        return list(self._left_keys + self._right_keys)

    @property
    def joint_specs(self) -> list[JointSpec]:
        """Per-joint settle metadata, aligned with :attr:`action_features`.

        Arm joints settle by position tolerance; the two grippers settle on
        "command issued" (a gripper closing on an object never reaches a position
        target). Joint *ranges* are not declared here — they come from the axol
        :class:`RobotProfile`. Tolerance is in the joint's native unit (radians for
        the axol arm joints).
        """
        grippers = {self._left_grip_key, self._right_grip_key}
        specs: list[JointSpec] = []
        for key in self._left_keys + self._right_keys:
            name = key.rsplit(".", 1)[0]
            if key in grippers:
                specs.append(JointSpec(name=name, control_mode="gripper"))
            else:
                specs.append(JointSpec(name=name, control_mode="position", settle_tolerance=0.05))
        return specs

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Start the event loop, enable Axol + telemetry, and open the cameras."""
        loop = asyncio.new_event_loop()
        self._loop = loop
        self._loop_thread = threading.Thread(
            target=loop.run_forever, name="axol-native-event-loop", daemon=True
        )
        self._loop_thread.start()
        asyncio.run_coroutine_threadsafe(self._connect_async(), loop).result(timeout=30)

        self._open_cameras()
        _logger.info(
            "AxolNativeRobot connected (cameras=%s).", list(self._cam_views)
        )

    def _open_cameras(self) -> None:
        """Open the onboard ZED cameras by serial and wire the obs-key views.

        Mono → obs key = name; stereo → ``name_left`` / ``name_right``. Cameras
        are opened sequentially because the ZED SDK's open/grab path touches
        shared NVENC state on the Jetson and isn't safe to drive concurrently.
        """
        from almond_axol.lerobot.camera import ZedCamera, ZedStereoCamera

        # A GMSL camera plugged in after boot stays invisible to the SDK until
        # the zed_x_daemon re-enumerates; restart it first (needs sudo).
        if self.config.restart_zed_daemon:
            try:
                from almond_axol.zed.daemon import restart_zed_daemon

                restart_zed_daemon()
            except Exception:  # noqa: BLE001
                _logger.warning(
                    "restart_zed_daemon failed; continuing (set --robot-arg "
                    "restart_zed_daemon=false to skip).",
                    exc_info=True,
                )

        for name, cam_cfg in self.config.cameras.items():
            cam = ZedStereoCamera(cam_cfg) if cam_cfg.stereo else ZedCamera(cam_cfg)
            cam.connect()
            self._cameras.append(cam)
            if cam_cfg.stereo:
                self._cam_views[f"{name}_left"] = cam.left_view
                self._cam_views[f"{name}_right"] = cam.right_view
            else:
                self._cam_views[name] = cam

    async def _connect_async(self) -> None:
        from almond_axol.robot.axol import Axol

        self._axol = Axol(
            self.config.axol_config,
            left_channel=self.config.left_channel,
            right_channel=self.config.right_channel,
        )
        await self._axol.enable()
        await self._axol.start_telemetry(
            self.config.telemetry_hz, torque=self.config.observe_torques
        )

    def disconnect(self) -> None:
        """Stop cameras, disable motors, and tear down the event loop."""
        for cam in self._cameras:
            try:
                cam.disconnect()
            except Exception:  # noqa: BLE001
                _logger.warning("ZED camera disconnect failed", exc_info=True)
        self._cameras = []
        self._cam_views = {}

        if self._loop is not None and self._axol is not None:
            try:
                asyncio.run_coroutine_threadsafe(
                    self._disconnect_async(), self._loop
                ).result(timeout=10)
            except Exception:  # noqa: BLE001
                _logger.warning("Axol disable failed", exc_info=True)

        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=5)
        self._loop = None
        self._loop_thread = None
        _logger.info("AxolNativeRobot disconnected.")

    async def _disconnect_async(self) -> None:
        if self._axol is None:
            return
        await self._axol.disable()
        self._axol = None

    # ------------------------------------------------------------------
    # Observation / action
    # ------------------------------------------------------------------

    def get_observation(self) -> dict[str, Any]:
        """Return the 16 joint positions + timestamp-aligned camera frames.

        Cameras are sampled with ``read_at_or_after`` against one shared
        ``perf_counter`` instant so every frame + the joint sample share the
        sender-clock moment, matching how training data was recorded. Falls back
        to ``read_latest`` if a camera misses the target within its timeout.
        """
        assert self._axol is not None and self._axol.left is not None
        assert self._axol.right is not None

        target_ts = time.perf_counter()
        left = self._axol.left.positions  # (8,) cached telemetry, no await
        right = self._axol.right.positions

        obs: dict[str, Any] = {}
        for i, key in enumerate(self._left_keys):
            obs[key] = float(left[i])
        for i, key in enumerate(self._right_keys):
            obs[key] = float(right[i])

        for cam_key, view in self._cam_views.items():
            cam_fps = getattr(view, "fps", None) or 30
            timeout_ms = int(2 * 1000.0 / cam_fps + 200)
            try:
                frame, _cap, _recv = view.read_at_or_after(target_ts, timeout_ms=timeout_ms)
            except (TimeoutError, RuntimeError) as exc:
                _logger.debug("%s read_at_or_after failed (%s); using latest.", cam_key, exc)
                frame = view.read_latest()
            obs[cam_key] = frame
        return obs

    def _motor_targets(self, action: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        """Map an action dict to ``(left, right)`` ``(8,)`` motion_control arrays.

        Pure (no hardware/event-loop): applies the configurable gripper
        post-step and the ``max_step_rad`` delta clamp, and returns exactly the
        two arrays ``send_action`` hands to ``motion_control``. Factored out so
        the action-writing behaviour is unit-testable without a live loop. The
        gripper occupies index 7 of each arm array (``Joint.GRIPPER`` is last)
        and is not delta-clamped.
        """
        action = dict(action)
        if self._gripper_mode == "bangbang":
            for key in (self._left_grip_key, self._right_grip_key):
                if key in action:
                    action[key] = (
                        1.0 if float(action[key]) >= self._gripper_threshold else 0.0
                    )

        left = np.array([action[k] for k in self._left_keys], dtype=np.float32)
        right = np.array([action[k] for k in self._right_keys], dtype=np.float32)
        n_l = len(self._left_arm_keys)
        n_r = len(self._right_arm_keys)
        left_arm = self._clamp_oversized_step(
            "left", left[:n_l], self._left_arm_keys, self._last_left
        )
        right_arm = self._clamp_oversized_step(
            "right", right[:n_r], self._right_arm_keys, self._last_right
        )
        # Write the clamped arm joints back and remember them as "last accepted"
        # so the next delta is measured from where the arm was actually commanded.
        left[:n_l] = left_arm
        right[:n_r] = right_arm
        self._last_left = left_arm
        self._last_right = right_arm
        return left, right

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Apply a joint-target action dict via native ``motion_control``.

        Applies the configurable gripper post-step and the ``max_step_rad``
        delta clamp (execution safety) before commanding ``motion_control``.
        """
        assert self._axol is not None and self._loop is not None
        left, right = self._motor_targets(action)
        asyncio.run_coroutine_threadsafe(
            self._axol.motion_control(left=left, right=right), self._loop
        ).result(timeout=1.0)
        return action

    def _clamp_oversized_step(
        self, side: str, arm_cur: np.ndarray, arm_keys: list[str], last: np.ndarray | None
    ) -> np.ndarray:
        """Clamp any arm-joint delta exceeding ``max_step_rad`` (execution safety).

        Limits each arm joint so ``|target - last| <= max_step_rad`` — the arm
        advances toward the commanded target by at most one step instead of
        slamming on a model glitch, bad chunk, or teleop frame. Returns the
        (possibly clamped) array, which is both sent to ``motion_control`` and
        remembered as "last accepted" so the next delta is measured from where
        the arm was actually told to go.
        """
        if last is None or not np.isfinite(self._max_step_rad):
            return arm_cur
        delta = arm_cur - last
        if np.any(np.abs(delta) > self._max_step_rad):
            self._drop_count += 1
            j = int(np.argmax(np.abs(delta)))
            _logger.warning(
                "Axol %s action exceeds max_step_rad=%.3f at %s (Δ=%.3f rad); "
                "clamped to the per-step limit (adapter clamp #%d).",
                side, self._max_step_rad, arm_keys[j], float(delta[j]), self._drop_count,
            )
            return last + np.clip(delta, -self._max_step_rad, self._max_step_rad)
        return arm_cur
