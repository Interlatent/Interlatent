"""Native YAM DRTC control loop (the ``--robot yam`` registry entry point).

A standalone control-loop function in the shape the node daemon invokes
(``import_callable`` → ``loop_fn(**kwargs)``). It drives the robot through the
native :class:`~interlatent.adapters.yam.robot.YAMNativeRobot` and reuses the
LeRobot-free DRTC wire helpers from :mod:`interlatent.node.control` so the
observation payload and recording are byte-identical to the built-in loop.

Scope: inference + per-tick recording (``control_source="policy"``). Teleop/DAgger
is intentionally not wired; the ``teleop_channel`` kwarg is accepted and ignored.
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
    teleop_channel: Any = None,  # accepted, ignored (no teleop for YAM yet)
    node_id: Optional[str] = None,
    image_resize: Optional[int] = None,
    **_: Any,
) -> None:
    """Observe → DRTC step → i2rt command_joint_pos, with per-tick recording.

    The ``client`` is an already-opened ``DRTCClient`` (the daemon opens it and
    closes it in its own finally-block — we must not close it here).
    """
    from interlatent.node import control as _ctrl

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

    # --- Action smoothing (policy path) ---------------------------------
    # Low-pass the per-tick policy action stream to attenuate chunk-boundary /
    # model jitter before it reaches the motors, mirroring the built-in loop.
    # 2nd-order Butterworth designed at the control rate; default 3 Hz cutoff,
    # tunable via ``--robot.action_filter_hz`` (0/none disables). Smoothing runs
    # BEFORE send_action, so the robot's per-step delta clamp remains the final
    # execution-safety guard. No teleop path here, so the filter never needs a
    # mid-stream reset() — the warm start on the first action is enough.
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
    try:
        while not should_stop():
            loop_start = time.perf_counter()
            obs = robot.get_observation()

            action = client.step(
                lambda o=obs: _ctrl._encode_npz(
                    _ctrl._to_policy_schema(o), image_resize=image_resize
                ),
                codec="npz",
            )

            state_keys = None
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
                state_keys = _ctrl._capture_tick(
                    client, obs, action_arr, step_counter, control_source="policy"
                )
                step_counter += 1

            if (
                state_keys is not None
                and not features_reported
                and features_report_attempts < 5
            ):
                features_report_attempts += 1
                if _ctrl._report_robot_features(
                    api_base, node_id, api_key, state_keys, action_keys
                ):
                    features_reported = True

            elapsed = time.perf_counter() - loop_start
            if elapsed < period:
                time.sleep(period - elapsed)
    finally:
        try:
            robot.disconnect()
        except Exception:  # noqa: BLE001
            _logger.warning("YAMNativeRobot disconnect failed", exc_info=True)
        _logger.info(
            "Native YAM loop exiting for session %s; daemon's client.close() flushes "
            "the recorder queue and triggers server-side upload.",
            session_id,
        )
