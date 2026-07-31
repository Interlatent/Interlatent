"""Native dimos DRTC control loop (the ``--robot dimos`` registry entry point).

A standalone control-loop function in the shape the node daemon invokes
(``import_callable`` → ``loop_fn(**kwargs)``). It drives the robot through
:class:`~interlatent.adapters.dimos.robot.DimosNativeRobot` and reuses the
LeRobot-free DRTC wire helpers from :mod:`interlatent.node.control`, so the
observation payload and recording are byte-identical to the built-in loop.

Ported from the YAM loop (the canonical native-loop template) with two
dimos-specific additions:

- **Episode markers**: an :class:`~.episode.EpisodeMarker` is published on the
  dimos bus at episode start/stop so a dimos-side memory2 recorder can segment
  its local low-level recording to match the episode of record (ADR 0018).
  Best-effort — marker failures never touch the control path.
- **Staleness hold** (nori pattern): when ``coordinator_joint_state`` goes
  stale the loop holds — no motion, no capture — because stale joints must not
  drive the gate, feed the policy, or be recorded as live state. The dimos
  servo task's own timeout owns the robot meanwhile (hold-last semantics).

Every teleop-channel tee from ``node/control.py`` must exist here too — this
native loop replaces that one wholesale, and a missing tee is silently lost.
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
    """Observe → DRTC step → dimos joint_command, with per-tick recording.

    The ``client`` is an already-opened ``DRTCClient`` (the daemon opens it and
    closes it in its own finally-block — we must not close it here).
    """
    from interlatent.node import control as _ctrl
    from interlatent.node.teleop_profiler import NodeTeleopProfiler

    from .config import build_adapter_config
    from .episode import publish_marker
    from .robot import DimosNativeRobot

    # dimos speaks radians end-to-end; no SO101 joint-zero calibration. Clear
    # the module's auto-preset so the shared encoder applies an identity map.
    _ctrl._AUTO_CALIB_PRESET = ""

    cfg = build_adapter_config(robot_extra or {}, robot_cameras or {})
    robot = DimosNativeRobot(cfg)

    session_id = session.get("id", "")
    fps = int(session.get("fps", 30) or 30)
    period = 1.0 / fps if fps > 0 else 1.0 / 30.0

    robot.connect()  # declare-then-verify happens inside (fail-closed)
    publish_marker(robot._bus, session_id, "start", robot.robot_kind)
    action_keys = robot.action_features
    _logger.info(
        "DimosNativeRobot connected (kind=%s); action_keys=%s; entering native "
        "control loop (streaming RecordTick → server) episode=%s",
        cfg.kind.name, action_keys, session_id,
    )

    # --- Teleop receiver setup (hosted relay path) -----------------------
    # Mirrors node/control.py: the SafetyGate is the single safety authority
    # for human-driven motion. Profile lookup uses the robot's PER-INSTANCE
    # kind ("dimos_xarm7"), not the daemon's --robot value ("dimos") — the
    # declared kind selects the safety envelope.
    from interlatent.node.teleop.robot_profile import get_profile
    from interlatent.node.teleop.safety import SafetyGate, TargetSample

    teleop_profile = get_profile(robot.robot_kind)
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

    # --- Action smoothing (policy path), mirrors the built-in loop --------
    from interlatent.node.smoothing import ButterworthLowPass

    _filter_hz = _ctrl._parse_action_filter_hz(robot_extra or {})
    action_filter = (
        ButterworthLowPass(cutoff_hz=_filter_hz, sample_hz=float(fps if fps > 0 else 30))
        if _filter_hz is not None
        else None
    )
    _logger.info(
        "Action smoothing %s.",
        f"ENABLED: Butterworth cutoff={_filter_hz} Hz" if action_filter else "DISABLED",
    )

    features_reported = False
    features_report_attempts = 0
    step_counter = 0
    _pv_empty_warned = False
    _stale_warned = False

    node_profiler = NodeTeleopProfiler(
        session_id=session_id, robot_kind=robot.robot_kind, fps=fps,
        teleop_configured=teleop_gate is not None,
    )

    try:
        while not should_stop():
            loop_start = time.perf_counter()
            _cmd_at: Optional[float] = None
            _capture_at: Optional[float] = None
            _frame_age_ms: Optional[float] = None

            # --- Staleness hold: stale joints must not drive the gate, feed
            # the policy, or be recorded as live state. Send nothing; the
            # dimos servo task's timeout owns the robot meanwhile.
            if not robot.telemetry_fresh:
                if not _stale_warned:
                    _stale_warned = True
                    _logger.warning(
                        "dimos joint state stale (%.0f ms) — holding (no "
                        "motion, no capture) until the stream recovers.",
                        robot.obs_age_ms,
                    )
                if teleop_gate is not None:
                    teleop_gate.reset()
                if action_filter is not None:
                    action_filter.reset()
                elapsed = time.perf_counter() - loop_start
                if elapsed < period:
                    time.sleep(period - elapsed)
                continue
            if _stale_warned:
                _stale_warned = False
                _logger.info("dimos joint state recovered — resuming.")

            obs = robot.get_observation()

            # Feed the pod-side retarget stage's staleness gate directly over
            # the teleop WS (rate-limited inside send_state). Mirrors
            # node/control.py — every teleop-channel tee must exist here too.
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
                            "to 'targets'; holding pose. See ADR 0009."
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

                # Drop policy chunks queued during teleop so they don't apply
                # when the human releases. (Mirrors node/control.py.)
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
                # --- HOLD PATH (teleop recording, disengaged) ---
                if teleop_gate is not None:
                    teleop_gate.reset()
                actual_joints = _ctrl._extract_joint_state(obs, action_keys)
                state_keys = _ctrl._capture_tick(
                    client, obs, actual_joints, step_counter, control_source="hold"
                )
                _capture_at = time.perf_counter()
                step_counter += 1
            else:
                # --- POLICY PATH ---
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
                    action_dict = {k: float(action_arr[i]) for i, k in enumerate(action_keys)}
                    robot.send_action(action_dict)
                    _cmd_at = time.perf_counter()
                    state_keys = _ctrl._capture_tick(
                        client, obs, action_arr, step_counter, control_source="policy"
                    )
                    _capture_at = time.perf_counter()
                    step_counter += 1

            # Live-preview tee (headset video). Mirrors node/control.py; runs
            # AFTER send_action so it never delays pose→motion.
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
                                    "preview encode produced no frames — "
                                    "headset video will ride the recording "
                                    "uplink (obs keys=%s)", sorted(obs.keys()),
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
                over_period=elapsed >= period,
            )
            if elapsed < period:
                time.sleep(period - elapsed)
    finally:
        publish_marker(robot._bus, session_id, "stop", robot.robot_kind)
        try:
            robot.disconnect()
        except Exception:  # noqa: BLE001
            _logger.warning("DimosNativeRobot disconnect failed", exc_info=True)
        node_profiler.close()
        _logger.info(
            "Native dimos loop exiting for session %s; daemon's client.close() "
            "flushes the recorder queue and triggers server-side upload.",
            session_id,
        )
