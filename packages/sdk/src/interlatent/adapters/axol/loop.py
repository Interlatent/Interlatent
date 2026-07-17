"""Native Axol DRTC control loop (the ``--robot axol`` / registry entry point).

A standalone control-loop function in the shape the node daemon invokes
(``import_callable`` → ``loop_fn(**kwargs)``). It drives the robot through the
native :class:`~interlatent.adapters.axol.robot.AxolNativeRobot` and reuses the
LeRobot-free DRTC wire helpers from :mod:`interlatent.node.control` so the
observation payload and recording are byte-identical to the built-in loop.

Scope: inference + per-tick recording (``control_source="policy"``). Teleop
is intentionally not wired (no SafetyGate/RobotProfile for Axol yet); the
``teleop_channel`` kwarg is accepted and ignored.
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
    teleop_channel: Any = None,  # accepted, ignored (no teleop for Axol yet)
    node_id: Optional[str] = None,
    image_resize: Optional[int] = None,
    bypass_key: Optional[str] = None,
    **_: Any,
) -> None:
    """Observe → DRTC step → native motion_control, with per-tick recording.

    The ``client`` is an already-opened ``DRTCClient`` (the daemon opens it and
    closes it in its own finally-block — we must not close it here).
    """
    # Canonical wire helpers. (The ZED cameras pull in lerobot via the native
    # almond_axol camera classes; the node.control helpers themselves do not.)
    from interlatent.node import control as _ctrl

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
    period = 1.0 / fps if fps > 0 else 1.0 / 30.0

    robot.connect()
    action_keys = robot.action_features
    _logger.info(
        "AxolNativeRobot connected; action_keys=%s; entering native control loop "
        "(streaming RecordTick → server) episode=%s",
        action_keys, session_id,
    )

    features_reported = False
    features_report_attempts = 0
    step_counter = 0
    try:
        while not should_stop():
            loop_start = time.perf_counter()
            obs = robot.get_observation()

            # Encode lazily — client.step() only builds the payload on ticks
            # where DRTC actually sends an observation.
            action = client.step(
                lambda o=obs: _ctrl._encode_npz(
                    _ctrl._to_policy_schema(o), image_resize=image_resize
                ),
                codec="npz",
            )

            state_keys = None
            if action is not None:
                action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
                # Build the joint-target dict directly (the 16 *.pos keys in
                # order) — no SO101 calibration coercion for Axol.
                action_dict = {
                    k: float(action_arr[i]) for i, k in enumerate(action_keys)
                }
                robot.send_action(action_dict)
                state_keys = _ctrl._capture_tick(
                    client, obs, action_arr, step_counter, control_source="policy"
                )
                step_counter += 1

            # One-time feature-element-names report (ADR 0003). state_keys come
            # from the first capture so they align with observation.state.
            if (
                state_keys is not None
                and not features_reported
                and features_report_attempts < 5
            ):
                features_report_attempts += 1
                if _ctrl._report_robot_features(
                    api_base, node_id, api_key, state_keys, action_keys,
                    bypass_key=bypass_key,
                ):
                    features_reported = True

            elapsed = time.perf_counter() - loop_start
            if elapsed < period:
                time.sleep(period - elapsed)
    finally:
        try:
            robot.disconnect()
        except Exception:  # noqa: BLE001
            _logger.warning("AxolNativeRobot disconnect failed", exc_info=True)
        _logger.info(
            "Native Axol loop exiting for session %s; daemon's client.close() "
            "flushes the recorder queue and triggers server-side upload.",
            session_id,
        )
