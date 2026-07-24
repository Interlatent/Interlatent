"""``adapters/yam/loop.py::control_loop`` frozen before the runner+bus migration.

A copy of the native YAM loop as of commit ``9070b75`` (PR 3 — the last commit
where the inline while-loop was the live implementation), including the PR 1
safety hotfix (e-stop rung, policy-path delta clamp).
``tests/test_loop_equivalence.py`` drives this against the migrated loop and
asserts the traces match tick-for-tick.

One mechanical transform, and only one: the package-relative imports
(``from .config import …``, ``from .robot import …``) are rewritten absolute.
They stay *inside* the function, exactly as in the original, so a monkeypatch
installed on those modules before the call is picked up identically by the
frozen and the live loop. Everything else is unchanged — do not "fix" anything
here; a frozen reference that drifts stops being a reference.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional

import numpy as np

_logger = logging.getLogger(__name__)


def control_loop(
    *,
    client: Any,
    session: dict,
    should_stop: Callable[[], bool],
    robot_kind: Optional[str] = None,
    robot_port: Optional[str] = None,
    robot_extra: Optional[dict[str, str]] = None,
    robot_cameras: Optional[dict[str, str]] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    teleop_channel: Any = None,
    node_id: Optional[str] = None,
    image_resize: Optional[int] = None,
    bypass_key: Optional[str] = None,
    policy_enabled: bool = True,
    **_: Any,
) -> None:
    """Verbatim pre-migration loop body; see the module docstring."""
    from interlatent.node import control as _ctrl
    from interlatent.node.teleop_profiler import NodeTeleopProfiler

    from interlatent.adapters.yam.config import build_adapter_config
    from interlatent.adapters.yam.robot import YAMNativeRobot

    _ctrl._AUTO_CALIB_PRESET = ""

    cfg = build_adapter_config(robot_extra or {}, robot_cameras or {})
    robot = YAMNativeRobot(cfg)

    session_id = session.get("id", "")
    fps = int(session.get("fps", 30) or 30)
    period = 1.0 / fps if fps > 0 else 1.0 / 30.0

    robot.connect()
    action_keys = robot.action_features
    _logger.info(
        "YAMNativeRobot connected; action_keys=%s; entering native control loop "
        "(streaming RecordTick → server) episode=%s",
        action_keys, session_id,
    )

    from interlatent.node.teleop.robot_profile import get_profile
    from interlatent.node.teleop.safety import SafetyGate, TargetSample

    teleop_profile = get_profile(robot_kind or "yam")
    teleop_gate = (
        SafetyGate(profile=teleop_profile, control_dt=period)
        if teleop_profile is not None
        else None
    )
    _teleop_schema = (
        teleop_profile.to_schema_dict() if teleop_profile is not None else None
    )
    teleop_warned = False
    _max_step = _ctrl._parse_max_step(robot_extra or {})

    from interlatent.node.smoothing import ButterworthLowPass

    _filter_hz = _ctrl._parse_action_filter_hz(robot_extra or {})
    action_filter = (
        ButterworthLowPass(cutoff_hz=_filter_hz, sample_hz=float(fps if fps > 0 else 30))
        if _filter_hz is not None
        else None
    )

    features_reported = False
    features_report_attempts = 0
    step_counter = 0
    _pv_empty_warned = False

    node_profiler = NodeTeleopProfiler(
        session_id=session_id, robot_kind=robot_kind or "yam", fps=fps,
        teleop_configured=teleop_gate is not None,
    )

    try:
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
                    except Exception:  # noqa: BLE001
                        pass

            frame = teleop_channel.latest_frame() if teleop_channel is not None else None
            engaged = bool(frame and frame.engaged and frame.deadman)

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
                _logger.warning(
                    "Operator e-stop received — SafetyGate latched; motion and "
                    "capture suspended until an explicit reset."
                )

            state_keys = None
            teleop_ok = (
                engaged
                and teleop_gate is not None
                and action_keys
                and len(action_keys) == len(teleop_profile.joint_names)
            )
            estop_latched = (
                teleop_gate is not None and teleop_gate.config.estop_latched
            )
            if estop_latched:
                try:
                    client.schedule.flush()
                except Exception:  # noqa: BLE001
                    pass
                if action_filter is not None:
                    action_filter.reset()
            elif teleop_ok:
                actual_joints = _ctrl._extract_joint_state(obs, action_keys)
                if (
                    frame.mode == "targets"
                    and frame.joint_targets is not None
                    and len(frame.joint_targets) == len(action_keys)
                ):
                    target = np.asarray(frame.joint_targets, dtype=np.float32)
                else:
                    if frame.mode == "pose" and not teleop_warned:
                        _logger.warning(
                            "Teleop frame mode='pose' reached the node — the "
                            "pod-side retarget stage should have converted it "
                            "to 'targets'; holding pose. See ADR 0009, second "
                            "amendment.",
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
                action_dict = {k: float(action_arr[i]) for i, k in enumerate(action_keys)}
                robot.send_action(action_dict)
                _cmd_at = time.perf_counter()

                _received_at_ns = getattr(frame, "received_at_ns", None)
                if _received_at_ns is not None:
                    _frame_age_ms = (time.monotonic_ns() - _received_at_ns) / 1e6

                try:
                    client.schedule.flush()
                except Exception:  # noqa: BLE001
                    pass
                if action_filter is not None:
                    action_filter.reset()

                state_keys = _ctrl._capture_tick(
                    client, obs, action_arr, step_counter, control_source="teleop"
                )
                _capture_at = time.perf_counter()
                step_counter += 1
            elif not policy_enabled:
                if teleop_gate is not None:
                    teleop_gate.reset()
                actual_joints = _ctrl._extract_joint_state(obs, action_keys)
                state_keys = _ctrl._capture_tick(
                    client, obs, actual_joints, step_counter, control_source="hold"
                )
                _capture_at = time.perf_counter()
                step_counter += 1
            else:
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
                    action_dict = {k: float(action_arr[i]) for i, k in enumerate(action_keys)}
                    robot.send_action(action_dict)
                    _cmd_at = time.perf_counter()
                    state_keys = _ctrl._capture_tick(
                        client, obs, action_arr, step_counter, control_source="policy"
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
                                _logger.warning(
                                    "preview encode produced no frames "
                                    "(cv2/PIL missing, or no uint8 camera "
                                    "arrays in obs keys=%s) — headset video "
                                    "will ride the recording uplink",
                                    sorted(obs.keys()),
                                )
                    except Exception:  # noqa: BLE001
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
        except Exception:  # noqa: BLE001
            _logger.warning("YAMNativeRobot disconnect failed", exc_info=True)
        node_profiler.close()
