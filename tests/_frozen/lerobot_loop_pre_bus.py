"""``lerobot_control_loop`` frozen immediately before the runner+bus migration.

A copy of ``interlatent/node/control.py::lerobot_control_loop`` as of commit
``af714bc`` (PR 2 — the last commit where the inline while-loop was the live
implementation). ``tests/test_loop_equivalence.py`` drives this against the
migrated loop and asserts the traces match tick-for-tick.

One mechanical transform was applied, and only one: every reference to a
``control.py`` module-level name (``_capture_tick``, ``_clamp_action_delta``,
``_make_lerobot_robot``, ``NodeTeleopProfiler``, …) is rewritten to
``_ctrl.<name>`` so the equivalence harness's monkeypatches route through both
the frozen and the live loop identically. Late imports inside the body
(``get_profile``, ``SafetyGate``, ``ButterworthLowPass``) are kept verbatim.
Control flow, ordering, and every expression are otherwise unchanged — do not
"fix" anything here; a frozen reference that drifts stops being a reference.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional

import numpy as np

from interlatent.node import control as _ctrl
from interlatent.node.movement import CommandBus, MovementSource

_LOG = logging.getLogger(__name__)


def lerobot_control_loop(
    *,
    client,
    session: dict,
    should_stop: Callable[[], bool],
    robot_kind: str,
    robot_port: Optional[str] = None,
    robot_extra: Optional[dict[str, str]] = None,
    robot_cameras: Optional[dict[str, str]] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    node_id: Optional[str] = None,
    bypass_key: Optional[str] = None,
    teleop_channel: Optional[Any] = None,
    image_resize: Optional[int] = None,
    policy_enabled: bool = True,
    **_: Any,
) -> None:
    """Verbatim pre-migration loop body; see the module docstring."""
    _ctrl._AUTO_CALIB_PRESET = (
        "so101_pre777"
        if "molmoact" in str(session.get("policy_uri", "")).lower()
        else ""
    )

    robot = _ctrl._make_lerobot_robot(
        robot_kind, port=robot_port, extra=robot_extra or {}, cameras=robot_cameras or {}
    )

    session_id = session.get("id", "")
    fps = int(session.get("fps", 30) or 30)

    robot.connect()
    action_keys = list(getattr(robot, "action_features", None) or [])
    _LOG.info(
        "LeRobot %r connected; action_keys=%s; entering control loop "
        "(streaming RecordTick → server) episode=%s",
        robot_kind, action_keys, session_id,
    )

    try:
        _modes = {}
        _bus = getattr(robot, "bus", None)
        _motors = getattr(_bus, "motors", None) or {}
        for _name, _motor in _motors.items():
            _nm = getattr(_motor, "norm_mode", None)
            _modes[_name] = getattr(_nm, "name", str(_nm))
        _LOG.info(
            "DRTC-DEBUG robot units | use_degrees=%s | image_resize=%s | "
            "calib_preset=%r calib_map=%s | motor_norm_modes=%s",
            getattr(getattr(robot, "config", None), "use_degrees", "?"),
            image_resize,
            _ctrl._resolve_calib_preset_name() or None,
            _ctrl._active_calib_map() or None,
            _modes,
        )
    except Exception:
        _LOG.warning("DRTC-DEBUG robot-units introspection failed", exc_info=True)

    from interlatent.node.teleop.robot_profile import get_profile
    from interlatent.node.teleop.safety import SafetyGate, TargetSample

    _teleop_dt = 1.0 / fps if fps > 0 else 1.0 / 30.0
    teleop_profile = get_profile(robot_kind)
    teleop_gate = (
        SafetyGate(profile=teleop_profile, control_dt=_teleop_dt)
        if teleop_profile is not None
        else None
    )
    _teleop_schema = teleop_profile.to_schema_dict() if teleop_profile is not None else None
    teleop_warned = False

    command_bus = CommandBus(
        teleop_channel=teleop_channel,
        teleop_gate=teleop_gate,
        teleop_profile=teleop_profile,
        policy_enabled=policy_enabled,
    )

    _max_step = _ctrl._parse_max_step(robot_extra or {})
    if _max_step is None:
        _LOG.warning(
            "Delta clamp DISABLED: no --robot.max_step set. A single-tick joint "
            "slam will execute unclamped. Set --robot.max_step=<units> (motor-norm "
            "units, e.g. degrees for MolmoAct2) to enable the execution-safety guard."
        )

    from interlatent.node.smoothing import ButterworthLowPass

    _filter_hz = _ctrl._parse_action_filter_hz(robot_extra or {})
    action_filter = (
        ButterworthLowPass(cutoff_hz=_filter_hz, sample_hz=float(fps if fps > 0 else 30))
        if _filter_hz is not None
        else None
    )

    node_profiler = _ctrl.NodeTeleopProfiler(
        session_id=session_id, robot_kind=robot_kind, fps=fps,
        teleop_configured=teleop_gate is not None,
    )

    try:
        period = 1.0 / fps if fps > 0 else 1.0 / 30.0
        step_counter = 0
        features_reported = False
        features_report_attempts = 0
        _tl_n = 0
        _tl_age_sum_ms = 0.0
        _tl_age_max_ms = 0.0
        _tl_window_started = time.monotonic()
        _pv_empty_warned = False
        while not should_stop():
            loop_start = time.perf_counter()
            _cmd_at: Optional[float] = None
            _capture_at: Optional[float] = None
            _frame_age_ms: Optional[float] = None

            obs = robot.get_observation()

            if teleop_channel is not None and action_keys:
                _send_state = getattr(teleop_channel, "send_state", None)
                if _send_state is not None:
                    try:
                        _send_state(
                            _ctrl._extract_joint_state(obs, action_keys).tolist()
                        )
                    except Exception:
                        pass

            frame = command_bus.sample_teleop()

            _consume_estop = getattr(teleop_channel, "consume_estop", None)
            estop_hit = bool(frame and frame.estop) or bool(
                _consume_estop is not None and _consume_estop()
            )
            if (
                estop_hit
                and teleop_gate is not None
                and not teleop_gate.config.estop_latched
            ):
                teleop_gate.latch_estop("teleop_frame")
                _LOG.warning(
                    "Operator e-stop received — SafetyGate latched; motion and "
                    "capture suspended until an explicit reset."
                )

            state_keys = None

            _ready = command_bus.readiness(frame, action_keys)
            engaged = _ready.engaged
            teleop_ok = _ready.teleop_available
            estop_latched = (
                teleop_gate is not None and teleop_gate.config.estop_latched
            )

            source = command_bus.arbitrate(frame, action_keys)
            if estop_latched:
                try:
                    client.schedule.flush()
                except Exception:
                    pass
                if action_filter is not None:
                    action_filter.reset()
            elif source is MovementSource.TELEOP:
                actual_joints = _ctrl._extract_joint_state(obs, action_keys)
                if (
                    frame.mode == "targets"
                    and frame.joint_targets is not None
                    and len(frame.joint_targets) == len(action_keys)
                ):
                    target = np.asarray(frame.joint_targets, dtype=np.float32)
                else:
                    if frame.mode == "pose" and not teleop_warned:
                        _LOG.warning(
                            "Teleop frame mode='pose' reached the node — the "
                            "pod-side retarget stage should have converted it "
                            "to 'targets' (is the relay running without a "
                            "teleop_view hook?); holding pose. See ADR 0009, "
                            "second amendment.",
                        )
                        teleop_warned = True
                    target = actual_joints.copy()

                teleop_gate.submit(TargetSample(
                    joints=target.reshape(-1),
                    deadman_active=frame.deadman,
                    confidence=frame.confidence,
                    received_at=loop_start,
                    producer_timestamp_ns=time.monotonic_ns(),
                ))
                commanded, _gate_status = teleop_gate.step(actual_joints, now=loop_start)
                action_arr = np.asarray(commanded, dtype=np.float32).reshape(-1)
                action_arr = _ctrl._clamp_action_delta(
                    action_arr, actual_joints, _max_step, action_keys,
                    step_counter, source="teleop",
                )
                robot.send_action(_ctrl._coerce_action_for_robot(action_arr, action_keys))
                _cmd_at = time.perf_counter()

                _note_applied = getattr(teleop_channel, "note_applied", None)
                if _note_applied is not None:
                    try:
                        _note_applied(int(frame.seq))
                    except Exception:
                        pass

                _tl_age_ms = (time.monotonic_ns() - frame.received_at_ns) / 1e6
                _tl_n += 1
                _tl_age_sum_ms += _tl_age_ms
                if _tl_age_ms > _tl_age_max_ms:
                    _tl_age_max_ms = _tl_age_ms
                _frame_age_ms = _tl_age_ms

                try:
                    client.schedule.flush()
                except Exception:
                    pass

                if action_filter is not None:
                    action_filter.reset()

                state_keys = _ctrl._capture_tick(
                    client, obs, action_arr, step_counter,
                    control_source="teleop",
                )
                _capture_at = time.perf_counter()
                step_counter += 1
            elif source is MovementSource.HOLD:
                if teleop_gate is not None:
                    teleop_gate.reset()
                actual_joints = _ctrl._extract_joint_state(obs, action_keys)
                state_keys = _ctrl._capture_tick(
                    client, obs, actual_joints, step_counter,
                    control_source="hold",
                )
                _capture_at = time.perf_counter()
                step_counter += 1
            else:  # MovementSource.POLICY
                if teleop_gate is not None:
                    teleop_gate.reset()

                action = client.step(
                    lambda o=obs: _ctrl._encode_npz(
                        _ctrl._to_policy_schema(o), image_resize=image_resize
                    ),
                    codec="npz",
                )
                if action is not None:
                    action_arr = np.asarray(action, dtype=np.float32).reshape(-1)

                    if action_filter is not None:
                        action_arr = action_filter.filter(action_arr)

                    if action_keys:
                        actual_joints = _ctrl._extract_joint_state(obs, action_keys)
                        action_arr = _ctrl._clamp_action_delta(
                            action_arr, actual_joints, _max_step, action_keys,
                            step_counter, source="policy",
                        )

                    robot.send_action(_ctrl._coerce_action_for_robot(action_arr, action_keys))
                    _cmd_at = time.perf_counter()

                    state_keys = _ctrl._capture_tick(
                        client, obs, action_arr, step_counter,
                        control_source="policy",
                    )
                    _capture_at = time.perf_counter()
                    step_counter += 1

            if teleop_channel is not None:
                _pv_due = getattr(teleop_channel, "preview_due", None)
                _pv_send = getattr(teleop_channel, "send_preview", None)
                if _pv_due is not None and _pv_send is not None:
                    try:
                        if _pv_due():
                            _pv = _ctrl._encode_preview_jpegs(obs)
                            if _pv:
                                _pv_send(_pv, time.monotonic_ns())
                            elif not _pv_empty_warned:
                                _pv_empty_warned = True
                                _LOG.warning(
                                    "preview encode produced no frames "
                                    "(cv2/PIL missing, or no uint8 camera "
                                    "arrays in obs keys=%s) — headset video "
                                    "will ride the recording uplink",
                                    sorted(obs.keys()),
                                )
                    except Exception:
                        pass

            if (
                state_keys is not None
                and not features_reported
                and features_report_attempts < 5
            ):
                features_report_attempts += 1
                if _ctrl._report_robot_features(
                    api_base, node_id, api_key, state_keys, action_keys,
                    teleop_profile=_teleop_schema,
                    bypass_key=bypass_key,
                ):
                    features_reported = True

            _tl_now = time.monotonic()
            if _tl_now - _tl_window_started >= 5.0:
                if _tl_n > 0:
                    _LOG.info(
                        "teleop exec latency (%.0fs): n=%d age mean/max=%.0f/%.0fms "
                        "(WS receive -> send_action)",
                        _tl_now - _tl_window_started, _tl_n,
                        _tl_age_sum_ms / _tl_n, _tl_age_max_ms,
                    )
                _tl_n = 0
                _tl_age_sum_ms = 0.0
                _tl_age_max_ms = 0.0
                _tl_window_started = _tl_now

            elapsed = time.perf_counter() - loop_start
            node_profiler.record_tick(
                loop_dt_s=elapsed,
                cmd_dt_s=(_cmd_at - loop_start) if _cmd_at is not None else None,
                capture_dt_s=(
                    (_capture_at - _cmd_at)
                    if (_capture_at is not None and _cmd_at is not None) else None
                ),
                frame_age_ms=_frame_age_ms,
                engaged=engaged,
                teleop_ok=teleop_ok,
                estop=estop_latched,
                over_period=elapsed >= period,
            )
            if elapsed < period:
                time.sleep(period - elapsed)
    finally:
        try:
            robot.disconnect()
        except Exception:
            _LOG.warning("Robot disconnect failed", exc_info=True)
        node_profiler.close()
