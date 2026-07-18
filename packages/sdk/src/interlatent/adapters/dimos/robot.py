"""``DimosNativeRobot`` — drives a robot managed by a running dimos stack.

Like nori, there is no motor driver here: the "vendor SDK" is the dimos process.
The adapter is an external bus peer — ``coordinator_joint_state`` + camera Image
topics in, ``joint_command`` out, consumed by a dimos servo task at its 100 Hz
tick. The GRIPPER RIDES ``joint_command`` like any other joint: dimos's per-tick
hardware write re-sends its last-commanded gripper value whenever any task
streams to the hardware, so an out-of-band ``set_gripper_position`` RPC is
stomped at tick rate — the RPC is read-only territory for this adapter
(observation fallback and connect-time verification).

Identity is declare-then-verify (ADR 0018): ``connect()`` runs
:func:`~.verify.verify_connect` against the live stack and fail-closes,
accumulating every mismatch into one raise, BEFORE any command can flow. The
biggest trap it guards: a stock dimos coordinator blueprint has no servo task
and silently ignores ``joint_command``.

Safety posture: dimos applies NO limits to streamed joint commands — the
``max_step_rad`` delta clamp here is the ONLY clamp in the whole path, so this
adapter is the last hand that touches a command before the bus. The gripper is
exempt from the delta clamp (yam/nori precedent: a gripper is commanded across
its whole range in one step by design).

Import-weight note: dimos is imported lazily inside ``connect()``-time
factories; importing this module never requires the ``[dimos]`` extra.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional

import numpy as np

from ..._clamp_log import warn_clamp
from ..base import JointSpec, ManualActionInterface
from .bus import DimosBus, image_to_rgb
from .config import DimosAdapterConfig

_logger = logging.getLogger(__name__)

# Coordinator RPC proxy duck type (subset the robot itself needs):
#   get_gripper_position(hardware_id) / set_activated(active)
CoordinatorClientFactory = Callable[[], Any]


def _default_coordinator_client_factory() -> Any:
    from dimos.control.coordinator import ControlCoordinator  # [dimos] extra
    from dimos.core.rpc_client import RPCClient

    return RPCClient(None, ControlCoordinator)


class DimosNativeRobot(ManualActionInterface):
    """Dimos-mediated robot for the DRTC inference loop.

    Implements the :class:`~interlatent.adapters.base.RobotAdapter` duck type;
    the manual block-then-settle ``action()`` comes from
    :class:`~interlatent.adapters.base.ManualActionInterface` (requires the
    kind's :class:`RobotProfile`, e.g. ``dimos_xarm7``).
    """

    def __init__(
        self,
        config: DimosAdapterConfig,
        *,
        bus: DimosBus | None = None,
        verify_fn: Callable[..., None] | None = None,
        coordinator_client_factory: CoordinatorClientFactory | None = None,
    ) -> None:
        self.config = config
        self.kind = config.kind
        self.robot_kind = self.kind.profile_name  # keys the profile registry
        self._keys: list[str] = list(self.kind.feature_keys)
        self._gripper_key = _gripper_feature(self.kind)
        self._arm_features = [k for k in self._keys if k != self._gripper_key]
        from ...node.teleop.robot_profile import get_profile

        profile = get_profile(self.robot_kind)
        assert profile is not None, f"{self.robot_kind} RobotProfile missing"
        self._profile = profile
        self._bus = bus if bus is not None else DimosBus(config)
        self._verify_fn = verify_fn
        self._client_factory = (
            coordinator_client_factory or _default_coordinator_client_factory
        )
        self._coordinator: Any = None  # estop / gripper reads; built lazily
        self._last_gripper_cmd: float | None = None
        self._max_step = float(config.max_step_rad)
        self._last: Optional[np.ndarray] = None  # last-accepted arm command
        self._drop_count = 0
        self._stale_warned: set[str] = set()
        self._connected = False

    # ------------------------------------------------------------------
    # metadata
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def action_features(self) -> list[str]:
        """Ordered action features: arm joints in dimos hardware order, gripper last."""
        return list(self._keys)

    @property
    def joint_specs(self) -> list[JointSpec]:
        """Aligned with :attr:`action_features`. Arm joints settle by position
        (0.05 rad ≈ 2.9 deg — same relative tightness as the other adapters);
        the gripper settles on "command issued" (it closes on objects)."""
        specs: list[JointSpec] = []
        for key in self._keys:
            name = key[: -len(".pos")]
            if key == self._gripper_key:
                specs.append(JointSpec(name=name, control_mode="gripper"))
            else:
                specs.append(
                    JointSpec(name=name, control_mode="position", settle_tolerance=0.05)
                )
        return specs

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Bind to the running stack: transport override -> subscriptions ->
        declare-then-verify (fail-closed) -> first state + camera warmup.
        On any failure everything opened so far is closed."""
        self._apply_transport_override()
        self._bus.open()
        try:
            self._run_verify()
            self._await_first_joint_state()
            self._warm_cameras()
        except Exception:
            self._bus.close()
            raise
        self._connected = True
        _logger.info(
            "DimosNativeRobot connected (kind=%s, joints=%d, cameras=%s). "
            "Reminder: the adapter's max_step_rad=%.3f clamp is the ONLY limit "
            "in this path — dimos does not clamp streamed joint commands.",
            self.kind.name, len(self._keys), list(self.config.cameras),
            self._max_step,
        )

    def _apply_transport_override(self) -> None:
        if self.config.transport is None:
            return
        from dimos.core.global_config import global_config  # [dimos] extra

        global_config.transport = self.config.transport

    def _run_verify(self) -> None:
        verify = self._verify_fn
        if verify is None:
            from .verify import verify_connect

            verify = verify_connect
        verify(self.config, self._bus)

    def _await_first_joint_state(self) -> None:
        deadline = time.monotonic() + self.config.connect_timeout_s
        while self._bus.latest_joint_state() is None:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"no JointState arrived on {self.config.joint_state_topic!r} "
                    f"within {self.config.connect_timeout_s:.0f}s. Is the dimos "
                    "coordinator publishing (check `dimos topic echo "
                    f"{self.config.joint_state_topic.lstrip('/')}`), and do both "
                    "sides use the same transport backend (DIMOS_TRANSPORT)?"
                )
            time.sleep(0.05)

    def _warm_cameras(self) -> None:
        """Block until every camera topic delivered one frame (nori pattern:
        fail at connect with an actionable message, not on the loop's first
        tick; after warmup a stalled publisher is a frozen image, never a dead
        session)."""
        if not self.config.cameras or self.config.camera_warmup_s <= 0:
            return
        deadline = time.monotonic() + self.config.camera_warmup_s
        pending = set(self.config.cameras)
        while pending and time.monotonic() < deadline:
            pending = {n for n in pending if self._bus.latest_image(n) is None}
            if pending:
                time.sleep(0.2)
        if pending:
            details = ", ".join(
                f"{n} ({self.config.cameras[n]})" for n in sorted(pending)
            )
            raise RuntimeError(
                f"camera topic(s) produced no frames within "
                f"{self.config.camera_warmup_s:.0f}s: {details}. Is a camera "
                "module publishing on that topic in the running blueprint "
                "(`dimos topic echo <topic>`)? Or drop the --camera flag / "
                "raise --robot-arg camera_warmup_s."
            )

    def disconnect(self) -> None:
        if self._coordinator is not None:
            try:
                self._coordinator.stop_rpc_client()
            except Exception:  # noqa: BLE001
                pass
            self._coordinator = None
        self._bus.close()
        # dimos pools zenoh sessions process-wide and never closes them; their
        # non-daemon threads would keep a one-shot process (interlatent-act)
        # alive after exit. Close the pool best-effort — a later connect()
        # re-acquires sessions on demand.
        try:
            from dimos.protocol.service.zenohservice import default_session_pool

            default_session_pool.close_all()
        except Exception:  # noqa: BLE001
            pass
        self._connected = False
        _logger.info("DimosNativeRobot disconnected.")

    # ------------------------------------------------------------------
    # observation
    # ------------------------------------------------------------------

    def get_observation(self) -> dict[str, Any]:
        """Joint positions (mapped to feature keys) + camera RGB frames.

        The gripper position comes from ``coordinator_joint_state`` when the
        stack reports it there; otherwise the last commanded value is served
        (disclosed in CONFIG.md — some stacks do not fold the gripper into the
        joint state stream).
        """
        cached = self._bus.latest_joint_state()
        if cached is None:
            raise RuntimeError("no joint state received yet (connect() not run?)")
        msg = cached.msg
        by_name = dict(zip(list(msg.name), [float(p) for p in msg.position]))
        obs: dict[str, Any] = {}
        for key in self._keys:
            dimos_name = self.kind.dimos_name_for(key)
            if dimos_name in by_name:
                obs[key] = by_name[dimos_name]
            elif key == self._gripper_key:
                obs[key] = float(
                    self._last_gripper_cmd
                    if self._last_gripper_cmd is not None
                    else self._profile.rest_pose[self._keys.index(key)]
                )
            else:
                obs[key] = 0.0  # verified joints should always be present
        for name in self.config.cameras:
            img = self._bus.latest_image(name)
            if img is None:
                continue  # warmup guarantees one frame unless cameras were empty
            age = self._bus.image_age_ms(name) or 0.0
            if age > self.config.camera_staleness_ms and name not in self._stale_warned:
                self._stale_warned.add(name)
                _logger.warning(
                    "camera %r stale (%.0f ms > %.0f ms); serving last frame",
                    name, age, self.config.camera_staleness_ms,
                )
            elif age <= self.config.camera_staleness_ms:
                self._stale_warned.discard(name)
            obs[name] = image_to_rgb(img.msg)
        return obs

    @property
    def obs_age_ms(self) -> float:
        age = self._bus.joint_state_age_ms()
        return float("inf") if age is None else age

    @property
    def telemetry_fresh(self) -> bool:
        """False when joint state has gone stale — the loop must hold (no
        motion, no capture) rather than act on old joints."""
        return self._connected and self.obs_age_ms <= self.config.staleness_ms

    # ------------------------------------------------------------------
    # action
    # ------------------------------------------------------------------

    def _motor_targets(self, action: dict[str, Any]) -> tuple[list[str], list[float]]:
        """Map an action dict to the wire (dimos names, positions) pair.

        Pure (no I/O): applies the ``max_step_rad`` per-send delta clamp on the
        arm joints, measured against the last-ACCEPTED command; the gripper is
        exempt and appended last. This is the only clamp in the path.
        """
        arm_vec = np.array(
            [float(action[k]) for k in self._arm_features], dtype=np.float32
        )
        arm_vec = self._clamp_oversized_step(arm_vec)
        self._last = arm_vec
        names = [self.kind.dimos_name_for(k) for k in self._arm_features]
        positions = [float(v) for v in arm_vec]
        if self._gripper_key:
            # The wire message must ALWAYS carry the full claimed joint set —
            # dimos's servo task rejects a command missing even one claimed
            # joint (set_target_by_name returns False without updating). An
            # action without the gripper key holds the last commanded value
            # (rest pose before any command).
            if self._gripper_key in action:
                self._last_gripper_cmd = float(action[self._gripper_key])
            value = (
                self._last_gripper_cmd
                if self._last_gripper_cmd is not None
                else float(self._profile.rest_pose[self._keys.index(self._gripper_key)])
            )
            names.append(self.kind.dimos_name_for(self._gripper_key))
            positions.append(value)
        return names, positions

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """One absolute-target command; non-blocking, latest-wins. The gripper
        rides the same joint_command message (the servo task claims it)."""
        names, positions = self._motor_targets(action)
        self._bus.publish_joint_command(names, positions)
        return action

    def _clamp_oversized_step(self, target: np.ndarray) -> np.ndarray:
        last = self._last
        if last is None or not np.isfinite(self._max_step):
            return target.astype(np.float32)
        delta = target - last
        if np.any(np.abs(delta) > self._max_step):
            self._drop_count += 1
            j = int(np.argmax(np.abs(delta)))
            warn_clamp(
                "dimos",
                "dimos action exceeds max_step_rad=%.3f at %s (Δ=%.3f rad); "
                "clamped to the per-step limit (adapter clamp #%d).",
                self._max_step, self._arm_features[j], float(delta[j]),
                self._drop_count,
            )
            return (last + np.clip(delta, -self._max_step, self._max_step)).astype(
                np.float32
            )
        return target.astype(np.float32)

    # ------------------------------------------------------------------
    # safety (best-effort; see CONFIG.md)
    # ------------------------------------------------------------------

    def estop(self) -> None:
        """Deactivate the coordinator's tick output (best-effort). Human reset:
        `reset_runtime_state` + `set_activated(True)` via dimos tooling."""
        if self._coordinator is None:
            self._coordinator = self._client_factory()
        self._coordinator.set_activated(False)


def _gripper_feature(kind) -> str | None:
    from .kinds import feature_key_for

    if kind.dimos_gripper_joint is None:
        return None
    return feature_key_for(kind.dimos_gripper_joint)


__all__ = ["DimosNativeRobot"]
