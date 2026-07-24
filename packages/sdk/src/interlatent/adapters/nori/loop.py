"""Native Nori DRTC control loop (the ``--robot nori`` registry entry point).

A standalone control-loop function in the shape the node daemon invokes
(``import_callable`` → ``loop_fn(**kwargs)``). A thin shim now: it constructs
the native :class:`~interlatent.adapters.nori.robot.NoriNativeRobot`, wires the
per-session collaborators into a full-motion
:class:`~interlatent.node.movement.CommandBus`, and hands the tick to
:func:`~interlatent.node.looprunner.run_control_loop`.

The Nori safety composition survives in two robot-owned seams instead of three
inline rungs:

- ``NoriNativeRobot.pre_tick`` — the per-tick episode guard (session death,
  the daemon safety FSM, telemetry staleness → ``TickVerdict``), discovered by
  the runner. ``get_observation()`` at the top of every tick stays the
  keep-alive pump's liveness proof (ADR 0015).
- ``NoriNativeRobot.estop`` — the daemon hard-latch forward
  (``command{name:"estop"}``, ADR 0016), called once per latch (with
  retry-on-failure) by ``CommandBus.observe_estop``.

Tick-for-tick equivalence with the pre-migration inline loop was proven by a
frozen-copy harness (retired after the 2026-07 hardware soak; see ADR 0022);
``tests/test_loop_contract.py`` remains the ongoing guard.
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
    bypass_key: Optional[str] = None,
    # False for teleop-recording assignments (no policy loaded): never
    # client.step(); disengaged ticks hold pose but still record.
    policy_enabled: bool = True,
    **_: Any,
) -> None:
    """Observe → DRTC step → daemon control frame, with per-tick recording.

    The ``client`` is an already-opened ``DRTCClient`` (the daemon opens it and
    closes it in its own finally-block — we must not close it here).
    """
    from interlatent.node import control as _ctrl
    from interlatent.node.looprunner import run_control_loop
    from interlatent.node.movement import CommandBus, WireHelpers, dict_coerce
    from interlatent.node.teleop_profiler import NodeTeleopProfiler

    from .config import build_adapter_config
    from .robot import NoriNativeRobot

    # Nori uses no SO101 joint-zero calibration; clear the module's auto-preset
    # so the shared encoder applies an identity map.
    _ctrl._AUTO_CALIB_PRESET = ""

    cfg = build_adapter_config(robot_extra or {}, robot_cameras or {})
    robot = NoriNativeRobot(cfg)

    session_id = session.get("id", "")
    fps = int(session.get("fps", 30) or 30)
    period = 1.0 / fps if fps > 0 else 1.0 / 30.0

    robot.connect()
    action_keys = robot.action_features
    _logger.info(
        "NoriNativeRobot connected; action_keys=%s; entering native control loop "
        "(streaming RecordTick → server) episode=%s",
        action_keys, session_id,
    )

    # --- Teleop receiver setup (hosted relay path) -----------------------
    # The SafetyGate is the single safety authority for human-driven motion;
    # the daemon re-clamps everything robot-side.
    from interlatent.node.teleop.robot_profile import get_profile
    from interlatent.node.teleop.safety import SafetyGate

    teleop_profile = get_profile(robot_kind or "nori")
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
    # Low-pass the policy stream before the shared delta clamp; the adapter's
    # per-send clamp (last-accepted-command anchored, gripper-exempt) and the
    # daemon's robot-side re-clamp stay in place below.
    from interlatent.node.smoothing import ButterworthLowPass

    _filter_hz = _ctrl._parse_action_filter_hz(robot_extra or {})
    action_filter = (
        ButterworthLowPass(cutoff_hz=_filter_hz, sample_hz=float(fps if fps > 0 else 30))
        if _filter_hz is not None
        else None
    )

    # The old inline loop had no profiler; the shared runner brings the same
    # purely-additive per-second CSV/log telemetry lerobot and YAM already had.
    node_profiler = NodeTeleopProfiler(
        session_id=session_id, robot_kind=robot_kind or "nori", fps=fps,
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
            teleop_profile=_teleop_schema, bypass_key=bypass_key,
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
            "Native Nori loop exiting for session %s; daemon's client.close() "
            "flushes the recorder queue and triggers server-side upload.",
            session_id,
        )


__all__ = ["control_loop"]
