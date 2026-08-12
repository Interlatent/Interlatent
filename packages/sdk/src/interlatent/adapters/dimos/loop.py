"""Native dimos DRTC control loop (the ``--robot dimos`` registry entry point).

A standalone control-loop function in the shape the node daemon invokes
(``import_callable`` → ``loop_fn(**kwargs)``). A thin shim now: it constructs
the native :class:`~interlatent.adapters.dimos.robot.DimosNativeRobot`, wires
the per-session collaborators (SafetyGate, delta clamp, Butterworth smoother,
profiler) into a full-motion
:class:`~interlatent.node.movement.CommandBus`, and hands the tick to
:func:`~interlatent.node.looprunner.run_control_loop`. Per-tick behavior lives
there and in ``CommandBus.drive()``, not here — which is the point: dimos can
no longer silently miss a safety rung the shared path grows.

This loop was a fork of the pre-ADR-0022 YAM loop and had drifted exactly the
way ADR 0022 predicts a hand-maintained copy drifts: **no e-stop handling at
all** (the gate was never latched and ``DimosNativeRobot.estop()`` was never
forwarded), no delta clamp on the policy path, no teleop seq echo. Those come
free from the bus now.

Two dimos-specific pieces moved onto the robot rather than staying here:

- **Episode markers** (ADR 0018) are published by ``connect()`` /
  ``disconnect()`` off ``robot.episode_id``, set below. The shared runner
  disconnects the robot in its own ``finally``, and ``disconnect()`` closes the
  bus — so a "stop" published from this file would land on a closed bus.
- **The staleness hold** is ``DimosNativeRobot.pre_tick`` returning
  ``TickVerdict.HOLD_NO_CAPTURE``; the bus does the gate/smoother reset that
  the inline version used to hand-roll.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

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
    from interlatent.node.looprunner import run_control_loop
    from interlatent.node.movement import CommandBus, WireHelpers, dict_coerce
    from interlatent.node.teleop_profiler import NodeTeleopProfiler

    from .config import build_adapter_config
    from .robot import DimosNativeRobot

    # dimos speaks radians end-to-end; no SO101 joint-zero calibration. Clear
    # the module's auto-preset so the shared encoder applies an identity map.
    _ctrl._AUTO_CALIB_PRESET = ""

    cfg = build_adapter_config(robot_extra or {}, robot_cameras or {})
    robot = DimosNativeRobot(cfg)

    session_id = session.get("id", "")
    fps = int(session.get("fps", 30) or 30)
    period = 1.0 / fps if fps > 0 else 1.0 / 30.0

    # Must precede connect(): the robot brackets the episode with ADR 0018 bus
    # markers across its own bus lifetime (see the module docstring).
    robot.episode_id = session_id
    robot.connect()  # declare-then-verify happens inside (fail-closed)
    action_keys = robot.action_features
    _logger.info(
        "DimosNativeRobot connected (kind=%s); action_keys=%s; entering native "
        "control loop (streaming RecordTick → server) episode=%s",
        robot.robot_kind, action_keys, session_id,
    )

    # --- Teleop receiver setup (hosted relay path) -----------------------
    # The SafetyGate is the single safety authority for human-driven motion.
    # Profile lookup uses the robot's PER-INSTANCE kind ("dimos_xarm7"), not the
    # daemon's --robot value ("dimos") — one adapter family, several
    # kinematically distinct arms, and the declared kind selects the envelope.
    from interlatent.node.teleop.robot_profile import get_profile
    from interlatent.node.teleop.safety import SafetyGate

    teleop_profile = get_profile(robot.robot_kind)
    teleop_gate = (
        SafetyGate(profile=teleop_profile, control_dt=period)
        if teleop_profile is not None
        else None
    )
    _teleop_schema = (
        teleop_profile.to_schema_dict() if teleop_profile is not None else None
    )
    _max_step = _ctrl._parse_max_step(robot_extra or {})

    # --- Action smoothing (policy path) ---------------------------------
    # Low-pass the per-tick policy action stream before the shared delta clamp;
    # the robot's own per-step clamp inside send_action (last-accepted-command
    # anchored, gripper-exempt) stays the final guard below the Protocol — and
    # for dimos it is the ONLY clamp downstream, since dimos applies no limits
    # to streamed joint commands.
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

    node_profiler = NodeTeleopProfiler(
        session_id=session_id, robot_kind=robot.robot_kind, fps=fps,
        teleop_configured=teleop_gate is not None,
    )

    command_bus = CommandBus(
        teleop_channel=teleop_channel,
        teleop_gate=teleop_gate,
        teleop_profile=teleop_profile,
        policy_enabled=policy_enabled,
        robot=robot,
        client=client,
        action_keys=list(action_keys),
        helpers=WireHelpers(
            extract=_ctrl._extract_joint_state,
            clamp=_ctrl._clamp_action_delta,
            coerce=dict_coerce,
            encode=lambda o: _ctrl._encode_npz(
                _ctrl._to_policy_schema(o), image_resize=image_resize
            ),
        ),
        max_step=_max_step,
        action_filter=action_filter,
    )

    def _capture(obs, action, step, *, control_source=None):
        return _ctrl._capture_tick(
            client, obs, action, step, control_source=control_source
        )

    def _report(state_keys, act_keys):
        return _ctrl._report_robot_features(
            api_base, node_id, api_key, state_keys, act_keys,
            teleop_profile=_teleop_schema,
        )

    try:
        run_control_loop(
            robot=robot,
            bus=command_bus,
            should_stop=should_stop,
            fps=fps,
            action_keys=list(action_keys),
            capture_fn=_capture,
            teleop_channel=teleop_channel,
            preview_fn=_ctrl._encode_preview_jpegs,
            report_features_fn=_report,
            extract_fn=_ctrl._extract_joint_state,
            profiler=node_profiler,
        )
    finally:
        _logger.info(
            "Native dimos loop exiting for session %s; daemon's client.close() "
            "flushes the recorder queue and triggers server-side upload.",
            session_id,
        )
