"""Native YAM DRTC control loop (the ``--robot yam`` registry entry point).

A standalone control-loop function in the shape the node daemon invokes
(``import_callable`` → ``loop_fn(**kwargs)``). It drives the robot through the
native :class:`~interlatent.adapters.yam.robot.YAMNativeRobot` and reuses the
LeRobot-free DRTC wire helpers from :mod:`interlatent.node.control` so the
observation payload and recording are byte-identical to the built-in loop.

Scope: inference + per-tick recording (``control_source="policy"``), plus the
hosted DAgger teleop receiver: ``mode="targets"`` frames from the platform
route through the node-side SafetyGate exactly as in the built-in LeRobot
loop (``node/control.py``), and intervened ticks record
``control_source="teleop"``.
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
    # False for teleop-recording assignments (no policy loaded): never
    # client.step(); disengaged ticks hold pose but still record.
    policy_enabled: bool = True,
    **_: Any,
) -> None:
    """Observe → DRTC step → i2rt command_joint_pos, with per-tick recording.

    The ``client`` is an already-opened ``DRTCClient`` (the daemon opens it and
    closes it in its own finally-block — we must not close it here).
    """
    from interlatent.node import control as _ctrl
    from interlatent.node.teleop_profiler import NodeTeleopProfiler

    from .config import build_adapter_config
    from .robot import YAMNativeRobot

    # YAM uses no SO101 joint-zero calibration; clear the module's auto-preset so the
    # shared encoder applies an identity map.
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

    # --- Teleop receiver setup (hosted DAgger path) ----------------------
    # Mirrors node/control.py: the SafetyGate is the single safety authority
    # for human-driven motion. The platform streams absolute joint targets
    # (``mode="targets"``); they route through the gate's workspace +
    # velocity clamp here. Without a registered profile for this robot kind
    # we refuse the gated teleop path and stay on policy.
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

    # --- Action smoothing (policy path) ---------------------------------
    # Low-pass the per-tick policy action stream to attenuate chunk-boundary /
    # model jitter before it reaches the motors, mirroring the built-in loop.
    # 2nd-order Butterworth designed at the control rate; default 3 Hz cutoff,
    # tunable via ``--robot.action_filter_hz`` (0/none disables). Smoothing runs
    # BEFORE send_action, so the robot's per-step delta clamp remains the final
    # execution-safety guard. Reset on every teleop engage (discontinuity in
    # the policy stream), matching the built-in loop.
    from interlatent.node.smoothing import ButterworthLowPass

    _filter_hz = _ctrl._parse_action_filter_hz(robot_extra or {})
    action_filter = (
        ButterworthLowPass(cutoff_hz=_filter_hz, sample_hz=float(fps if fps > 0 else 30))
        if _filter_hz is not None
        else None
    )
    if action_filter is not None:
        _logger.info(
            "Action smoothing ENABLED: 2nd-order Butterworth low-pass, cutoff=%.2f Hz "
            "@ %d Hz control rate (policy path). Set --robot.action_filter_hz=none to "
            "disable.", action_filter.cutoff_hz, int(fps if fps > 0 else 30),
        )
    else:
        _logger.info("Action smoothing DISABLED (--robot.action_filter_hz=none).")

    features_reported = False
    features_report_attempts = 0
    step_counter = 0
    # One-shot: a preview encode that yields {} means cv2/PIL are missing
    # (or obs has no camera arrays) — the headset video then silently rides
    # the seconds-stale recording uplink. Say so once. (Mirrors control.py.)
    _pv_empty_warned = False

    # Local (node-side) per-second profiler — see node/teleop_profiler.py.
    # Logs a TELEOP-PROFILE line every second (grep-able wherever this
    # node's other logs already go) plus a best-effort CSV copy; never
    # raises into the loop. Purely additive: clock reads placed strictly
    # AFTER work this loop is already doing, nothing in the command path
    # below is touched.
    node_profiler = NodeTeleopProfiler(
        session_id=session_id, robot_kind=robot_kind or "yam", fps=fps,
        teleop_configured=teleop_gate is not None,
    )

    try:
        while not should_stop():
            loop_start = time.perf_counter()
            # Set only if this tick actually reaches that stage (the
            # policy path skips both when client.step() returns no
            # action yet) — record_tick() treats a None as "no sample".
            _cmd_at: Optional[float] = None
            _capture_at: Optional[float] = None
            _frame_age_ms: Optional[float] = None
            obs = robot.get_observation()

            # Feed the pod-side retarget stage's staleness gate directly over
            # the teleop WS (~15 Hz, rate-limited inside send_state). RecordTick
            # state rides the batched JPEG uplink and can lag seconds behind on
            # a slow link. Mirrors node/control.py — this native loop replaces
            # that one wholesale, so every teleop-channel tee must exist here
            # too or YAM silently loses it.
            if teleop_channel is not None and action_keys:
                _send_state = getattr(teleop_channel, "send_state", None)
                if _send_state is not None:
                    try:
                        _send_state(
                            _ctrl._extract_joint_state(obs, action_keys).tolist()
                        )
                    except Exception:  # noqa: BLE001
                        pass

            # Sample the latest teleop frame. None when no producer is
            # connected or the last frame is stale (channel drops > 250 ms).
            frame = teleop_channel.latest_frame() if teleop_channel is not None else None
            engaged = bool(frame and frame.engaged and frame.deadman)

            state_keys = None
            teleop_ok = (
                engaged
                and teleop_gate is not None
                and action_keys
                and len(action_keys) == len(teleop_profile.joint_names)
            )
            if teleop_ok:
                # --- TELEOP PATH (mode="targets" only; see control.py) ---
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

                # How old was the frame we just executed? Same metric
                # node/control.py's "teleop exec latency" window computes;
                # this loop has no such window, so just feed it straight
                # into the profiler. getattr-guarded in case this frame
                # type predates the field.
                _received_at_ns = getattr(frame, "received_at_ns", None)
                if _received_at_ns is not None:
                    _frame_age_ms = (time.monotonic_ns() - _received_at_ns) / 1e6

                # Drop policy chunks queued or landing during teleop so they
                # don't apply when the human releases. (schedule.flush() — a
                # long-standing `client.flush_buffer()` call here named a
                # method DRTCClient never had, so the drop silently never
                # happened. Mirrors node/control.py.)
                try:
                    client.schedule.flush()
                except Exception:  # noqa: BLE001
                    pass
                # Discontinuity: drop the smoother's state so the first
                # post-release policy action warm-starts from the live pose.
                if action_filter is not None:
                    action_filter.reset()

                state_keys = _ctrl._capture_tick(
                    client, obs, action_arr, step_counter, control_source="teleop"
                )
                _capture_at = time.perf_counter()
                step_counter += 1
            elif not policy_enabled:
                # --- HOLD PATH (teleop recording, disengaged) ---
                # No policy to fall back to: send nothing (motors hold),
                # but record every tick so the episode stays continuous.
                if teleop_gate is not None:
                    teleop_gate.reset()
                actual_joints = _ctrl._extract_joint_state(obs, action_keys)
                state_keys = _ctrl._capture_tick(
                    client, obs, actual_joints, step_counter, control_source="hold"
                )
                _capture_at = time.perf_counter()
                step_counter += 1
            else:
                # Reset the gate so the next engage starts from the live pose.
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
                    # Low-pass the policy stream before send_action; the robot's
                    # per-step delta clamp (inside send_action) stays the final
                    # execution-safety guard. Warm-started, so no startup ramp. The
                    # recorded action is the smoothed command actually sent.
                    if action_filter is not None:
                        action_arr = action_filter.filter(action_arr)
                    action_dict = {k: float(action_arr[i]) for i, k in enumerate(action_keys)}
                    robot.send_action(action_dict)
                    _cmd_at = time.perf_counter()
                    state_keys = _ctrl._capture_tick(
                        client, obs, action_arr, step_counter, control_source="policy"
                    )
                    _capture_at = time.perf_counter()
                    step_counter += 1

            # Live-preview tee (headset video): small downscaled JPEGs pushed
            # over the teleop WS, decoupled from the batched full-resolution
            # recording uplink (which over a real link runs seconds behind —
            # the operator must never steer off that). preview_due() is
            # checked FIRST so idle sessions (no viewer) pay zero encode
            # cost, and this block runs AFTER send_action so it never delays
            # pose→motion. (Mirrors node/control.py.)
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
                        # Previews are best-effort; never break the loop.
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
        _logger.info(
            "Native YAM loop exiting for session %s; daemon's client.close() flushes "
            "the recorder queue and triggers server-side upload.",
            session_id,
        )
