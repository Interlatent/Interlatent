"""Native Axol DRTC control loop (the ``--robot axol`` / registry entry point).

A standalone control-loop function in the shape the node daemon invokes
(``import_callable`` → ``loop_fn(**kwargs)``). A thin shim now: it constructs
the native :class:`~interlatent.adapters.axol.robot.AxolNativeRobot`, wires a
full-motion :class:`~interlatent.node.movement.CommandBus`, and hands the tick
to :func:`~interlatent.node.looprunner.run_control_loop`.

Scope: inference + per-tick recording (``control_source="policy"``), plus the
policy-less hold path (``control_source="hold"``) a teleop-recording assignment
needs. Teleop itself is intentionally not wired (no SafetyGate/RobotProfile for
Axol yet, ADR 0011:111): the ``teleop_channel`` kwarg is accepted and dropped
before the bus ever sees it, so a recording here captures a held pose rather
than human-driven motion. Wiring Axol teleop later is exactly one step —
register a RobotProfile and stop dropping the channel; the bus already owns the
rest of the path.

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
    teleop_channel: Any = None,  # accepted, dropped (no teleop for Axol yet)
    node_id: Optional[str] = None,
    image_resize: Optional[int] = None,
    bypass_key: Optional[str] = None,
    # False for teleop-recording assignments (no policy loaded): never
    # client.step(); every tick holds pose and still records. Declaring it
    # explicitly matters — while it fell into ``**_`` this loop ran inference
    # against a policy-less session (CONTEXT.md's TeleopRecording contract).
    policy_enabled: bool = True,
    **_: Any,
) -> None:
    """Observe → DRTC step → native motion_control, with per-tick recording.

    The ``client`` is an already-opened ``DRTCClient`` (the daemon opens it and
    closes it in its own finally-block — we must not close it here).
    """
    # Canonical wire helpers. (The ZED cameras pull in lerobot via the native
    # almond_axol camera classes; the node.control helpers themselves do not.)
    from interlatent.node import control as _ctrl
    from interlatent.node.looprunner import run_control_loop
    from interlatent.node.movement import CommandBus, WireHelpers, dict_coerce

    from .config import build_adapter_config
    from .robot import AxolNativeRobot

    # Axol uses no SO101 joint-zero calibration; clear the module's auto-preset
    # so the shared encoder applies an identity map (the INTERLATENT_CALIB_PRESET
    # env var, if an operator sets it, still overrides — not expected on Axol).
    _ctrl._AUTO_CALIB_PRESET = ""

    cfg = build_adapter_config(robot_extra or {}, robot_cameras or {})
    robot = AxolNativeRobot(cfg)

    session_id = session.get("id", "")
    fps = int(session.get("fps", 30) or 30)

    robot.connect()
    action_keys = robot.action_features
    _logger.info(
        "AxolNativeRobot connected; action_keys=%s; entering native control loop "
        "(streaming RecordTick → server) episode=%s",
        action_keys, session_id,
    )

    # Last-line guard against a single-tick joint slam, shared with every other
    # loop via --robot.max_step. Unset ⇒ disabled (the helper logs it).
    _max_step = _ctrl._parse_max_step(robot_extra or {})

    # No gate (no RobotProfile), no smoother (Axol has never run one — adding
    # it would change motion on hardware, which a migration must not), and the
    # teleop channel is dropped here so the bus cannot sample frames or consume
    # the channel's e-stop latch a path Axol doesn't have.
    command_bus = CommandBus(
        teleop_channel=None,
        teleop_gate=None,
        teleop_profile=None,
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
        action_filter=None,
    )

    def _capture(obs, action, step, *, control_source=None):
        return _ctrl._capture_tick(
            client, obs, action, step, control_source=control_source
        )

    def _report(state_keys, act_keys):
        # No teleop_profile: Axol reports feature names only (ADR 0003).
        return _ctrl._report_robot_features(
            api_base, node_id, api_key, state_keys, act_keys,
            bypass_key=bypass_key,
        )

    try:
        run_control_loop(
            robot=robot,
            bus=command_bus,
            should_stop=should_stop,
            fps=fps,
            action_keys=list(action_keys),
            capture_fn=_capture,
            teleop_channel=None,
            report_features_fn=_report,
        )
    finally:
        _logger.info(
            "Native Axol loop exiting for session %s; daemon's client.close() "
            "flushes the recorder queue and triggers server-side upload.",
            session_id,
        )
