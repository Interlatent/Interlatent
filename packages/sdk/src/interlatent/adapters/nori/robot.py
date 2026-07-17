"""``NoriNativeRobot`` — drives the Nori robot through its on-Pi daemon.

A thin, synchronous adapter behind the same interface the inference loop uses
(``connect`` / ``get_observation`` / ``send_action`` / ``disconnect`` /
``action_features``). Unlike YAM/Axol there is no motor driver here at all: the
adapter speaks the Nori-Protocol v1 NDJSON contract to ``NoriCoreAgent`` on TCP
(default ``localhost:7777``) via :class:`~.client.NoriSessionClient`, and
subscribes camera frames from the daemon's companion ZeroMQ MJPEG channel.

The observation/action dict uses the daemon's own ``<side>_arm_<joint>.pos``
keys, left arm then right arm (12 joints total), in the daemon-normalized
``range_m100_100`` units ([-100, 100]; grippers [0, 100]). All actions are
joint-space absolute targets; there is no IK here (ADR 0013).

Safety posture (the point of this adapter): every enforcement mechanism —
range clamping, e-stop hard latch, watchdog safe-stop — lives in the daemon.
This class only *discloses* daemon state (``last_status`` / ``obs_age_ms`` /
``telemetry_fresh`` / ``session_dead``) and adds the same source-agnostic
per-send delta clamp the other adapters carry. ``get_observation()`` doubles
as the control loop's liveness proof for the keep-alive pump (ADR 0015).

Import-weight note: ``zmq``/``cv2`` are imported lazily inside the camera
backend, so importing this module never requires the ``[nori]`` extra.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

import numpy as np

from ..._clamp_log import warn_clamp
from ..base import JointSpec, ManualActionInterface
from .cameras import NoriCamera, resolve_camera_specs
from .client import NoriSessionClient
from .config import NoriAdapterConfig, resolve_token

_logger = logging.getLogger(__name__)

_SIDES = ("left", "right")
_ARM_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


class NoriNativeRobot(ManualActionInterface):
    """Nori robot for the DRTC inference loop, over the daemon wire contract.

    Implements the formal :class:`~interlatent.adapters.base.RobotAdapter` duck
    type and inherits the manual block-then-settle ``action()`` from
    :class:`~interlatent.adapters.base.ManualActionInterface` (requires the
    ``nori`` :class:`RobotProfile`). ``connect()`` fail-closes if the live ack
    descriptor disagrees with that profile — every mismatch is reported at once.
    """

    robot_kind = "nori"

    def __init__(self, config: NoriAdapterConfig) -> None:
        self.config = config
        # Ordered wire keys, left block then right, gripper last per block —
        # must equal the "nori" RobotProfile joint order (base.py enforces).
        self._keys: list[str] = [
            f"{side}_arm_{joint}.pos" for side in _SIDES for joint in _ARM_JOINTS
        ]
        self._gripper_mask = np.array(
            [k.endswith("_gripper.pos") for k in self._keys]
        )
        from ...node.teleop.robot_profile import get_profile

        profile = get_profile(self.robot_kind)
        assert profile is not None, "nori RobotProfile missing from _PROFILES"
        self._profile = profile
        self._client = NoriSessionClient(config, profile)
        self._cameras: dict[str, NoriCamera] = {}
        self._max_step = float(config.max_step)
        self._last: Optional[np.ndarray] = None  # last-accepted non-gripper vec
        self._drop_count = 0
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected and self._client.connected

    @property
    def action_features(self) -> list[str]:
        """Ordered action-feature names (left arm block then right)."""
        return list(self._keys)

    @property
    def joint_specs(self) -> list[JointSpec]:
        """Per-joint settle metadata, aligned with :attr:`action_features`.

        Arm joints settle by position tolerance in normalized units (2.0 on the
        [-100, 100] scale — the same relative tightness as SO-101's 2.0 deg /
        YAM's 0.05 rad defaults); grippers settle on "command issued". Ranges
        are not declared here — they come from the ``nori`` RobotProfile and
        are cross-checked against the live ack at connect.
        """
        specs: list[JointSpec] = []
        for key in self._keys:
            name = key[: -len(".pos")]
            if key.endswith("_gripper.pos"):
                specs.append(JointSpec(name=name, control_mode="gripper"))
            else:
                specs.append(
                    JointSpec(name=name, control_mode="position", settle_tolerance=2.0)
                )
        return specs

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Handshake with the daemon (fail-closed validation inside), then open
        the descriptor's cameras. No auto-home: the daemon owns startup poses."""
        ack = self._client.connect()
        try:
            for spec in resolve_camera_specs(
                self.config, list(self._client.descriptor_cameras)
            ):
                cam = NoriCamera(spec)
                cam.connect()
                self._cameras[spec.obs_key] = cam
            self._warm_cameras()
        except Exception:
            for cam in self._cameras.values():
                try:
                    cam.disconnect()
                except Exception:  # noqa: BLE001
                    pass
            self._cameras = {}
            self._client.close()
            raise
        self._connected = True
        wd = ack.watchdog
        _logger.info(
            "NoriNativeRobot connected (joints=%d, cameras=%s, watchdog "
            "warn/stop=%s/%s ms — the daemon safe-stops on frame silence; "
            "keep-alive is liveness-tied, ADR 0015).",
            len(self._keys), list(self._cameras),
            wd.t_warn_ms if wd else "?", wd.t_stop_ms if wd else "?",
        )

    def _warm_cameras(self) -> None:
        """Block until every configured camera has delivered one frame.

        The Pi's capture bridge publishes lazily (it may spin up seconds after
        a session starts), so a camera with no frame at connect is not yet an
        error — but a camera with no frame after ``camera_warmup_s`` IS one,
        and failing here (session setup) with an actionable message beats
        crashing on the control loop's first tick. After warm-up, ``read()``
        always has a last frame to serve, so a mid-session publisher stall
        shows up as a frozen image, never a dead session.
        """
        if not self._cameras or self.config.camera_warmup_s <= 0:
            return
        deadline = time.monotonic() + self.config.camera_warmup_s
        pending = dict(self._cameras)
        while pending and time.monotonic() < deadline:
            for key, cam in list(pending.items()):
                try:
                    cam.read()
                except RuntimeError:
                    continue
                del pending[key]
            if pending:
                time.sleep(0.2)
        if pending:
            details = ", ".join(
                f"{key} (tcp://{cam.spec.host}:{cam.spec.port})"
                for key, cam in pending.items()
            )
            raise RuntimeError(
                f"Nori camera(s) produced no frames within "
                f"{self.config.camera_warmup_s:.0f}s: {details}. Is the "
                "capture bridge publishing? It normally runs only during a "
                "Nori teleop session — for interlatent sessions start it "
                "standalone (external-bridge mode) and check listeners with "
                "`ss -tlnp | grep 555`. Or drop the --camera flag / raise "
                "--robot-arg camera_warmup_s."
            )

    def disconnect(self) -> None:
        """Close cameras first, then the session (bye + pump stop)."""
        for name, cam in self._cameras.items():
            try:
                cam.disconnect()
            except Exception:  # noqa: BLE001
                _logger.warning("Nori camera %s disconnect failed", name, exc_info=True)
        self._cameras = {}
        self._client.close()
        self._connected = False
        _logger.info("NoriNativeRobot disconnected.")

    # ------------------------------------------------------------------
    # Observation / daemon-state disclosure
    # ------------------------------------------------------------------

    def get_observation(self) -> dict[str, Any]:
        """The 12 joint positions + camera RGB frames.

        Calling this is the control loop's LIVENESS PROOF: it feeds the
        keep-alive pump's gate, so the daemon's watchdog keeps guarding "is the
        brain alive" (ADR 0015). Safety status deliberately stays OUT of this
        dict — every float here lands in the recorded ``observation_state``
        vector; use :attr:`last_status` / :attr:`telemetry_fresh` instead.
        """
        self._client.note_liveness()
        state, _age = self._client.latest_state()
        obs: dict[str, Any] = {k: float(state.get(k, 0.0)) for k in self._keys}
        for name, cam in self._cameras.items():
            obs[name] = cam.read()
        return obs

    @property
    def last_status(self) -> Optional[dict[str, Any]]:
        """The daemon's most recent periodic ``telemetry.status`` block
        (``safety``/``watchdog``/``latch_reason``/``link``), or None."""
        return self._client.latest_status()

    @property
    def obs_age_ms(self) -> float:
        return self._client.latest_state()[1]

    @property
    def telemetry_fresh(self) -> bool:
        """False while reconnecting or when telemetry has gone stale — the loop
        must hold (no motion, no capture) rather than act on old joints."""
        return self._client.connected and self.obs_age_ms <= self.config.staleness_ms

    @property
    def session_dead(self) -> bool:
        """Fatal daemon error or reconnect window exhausted: episode is over."""
        return self._client.session_dead

    @property
    def dead_reason(self) -> str:
        return self._client.dead_reason

    # ------------------------------------------------------------------
    # Action
    # ------------------------------------------------------------------

    def _motor_targets(self, action: dict[str, Any]) -> dict[str, float]:
        """Map an action dict to the wire ``action`` payload.

        Pure (no I/O): applies the ``max_step`` per-send delta clamp on the
        non-gripper joints (normalized units; grippers exempt, mirroring YAM)
        and returns exactly the dict ``send_action`` puts on the wire. The
        daemon re-clamps to its ranges robot-side regardless — this guard
        exists to stop single-tick slams at the source, same as every adapter.
        """
        vec = np.array([float(action[k]) for k in self._keys], dtype=np.float32)
        arm = self._clamp_oversized_step(vec[~self._gripper_mask])
        vec[~self._gripper_mask] = arm
        self._last = arm
        return {k: float(v) for k, v in zip(self._keys, vec)}

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """One absolute-target control frame; non-blocking, latest-wins."""
        self._client.send_action(self._motor_targets(action))
        return action

    def _clamp_oversized_step(self, target: np.ndarray) -> np.ndarray:
        """Clamp any non-gripper delta exceeding ``max_step`` (execution safety),
        measured against the last-ACCEPTED command, not the measured pose."""
        last = self._last
        if last is None or not np.isfinite(self._max_step):
            return target.astype(np.float32)
        delta = target - last
        if np.any(np.abs(delta) > self._max_step):
            self._drop_count += 1
            nk = [k for k, g in zip(self._keys, self._gripper_mask) if not g]
            j = int(np.argmax(np.abs(delta)))
            warn_clamp(
                "nori",
                "Nori action exceeds max_step=%.2f at %s (Δ=%.2f normalized); "
                "clamped to the per-step limit (adapter clamp #%d).",
                self._max_step, nk[j], float(delta[j]), self._drop_count,
            )
            return (last + np.clip(delta, -self._max_step, self._max_step)).astype(
                np.float32
            )
        return target.astype(np.float32)

    # ------------------------------------------------------------------
    # Safety commands (see CONTEXT.md "E-stop ingress (teleop)")
    # ------------------------------------------------------------------

    def estop(self) -> None:
        """Trip the daemon's hard e-stop latch (schema-canonical command)."""
        self._client.send_estop()

    def reset_latch(self, token: Optional[str] = None) -> None:
        """Clear the daemon's e-stop latch. Human-initiated only — never called
        by the control loop. Token: explicit arg > config > the daemon's token
        file; the daemon enforces it either way."""
        tok = token or resolve_token(self.config)
        if not tok:
            raise RuntimeError(
                "reset_latch needs the daemon agent token (pass --token, set "
                f"--robot-arg token=..., or make {self.config.token_path} readable)"
            )
        self._client.send_reset_latch(tok)


__all__ = ["NoriNativeRobot"]
