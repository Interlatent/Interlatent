"""``adapters/axol/loop.py::control_loop`` frozen before the runner+bus migration.

A copy of the native Axol loop as of commit ``de521b7`` (PR 5 — the last
commit where the inline while-loop was the live implementation), including
the PR 1 hotfix (explicit ``policy_enabled`` + HOLD path, policy-path delta
clamp). ``tests/test_loop_equivalence.py`` drives this against the migrated
loop and asserts the traces match tick-for-tick.

One mechanical transform, and only one: the package-relative imports are
rewritten absolute; they stay inside the function so monkeypatches installed
before the call are picked up identically by both generations. Do not "fix"
anything here. Original module docstring follows.



A standalone control-loop function in the shape the node daemon invokes
(``import_callable`` → ``loop_fn(**kwargs)``). It drives the robot through the
native :class:`~interlatent.adapters.axol.robot.AxolNativeRobot` and reuses the
LeRobot-free DRTC wire helpers from :mod:`interlatent.node.control` so the
observation payload and recording are byte-identical to the built-in loop.

Scope: inference + per-tick recording (``control_source="policy"``), plus the
policy-less hold path (``control_source="hold"``) a teleop-recording assignment
needs. Teleop itself is intentionally not wired (no SafetyGate/RobotProfile for
Axol yet); the ``teleop_channel`` kwarg is accepted and ignored, so a recording
here captures a held pose rather than human-driven motion.
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

    from interlatent.adapters.axol.config import build_adapter_config
    from interlatent.adapters.axol.robot import AxolNativeRobot

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

    # Last-line guard against a single-tick joint slam, shared with every other
    # loop via --robot.max_step. Unset ⇒ disabled (the helper logs it).
    _max_step = _ctrl._parse_max_step(robot_extra or {})

    features_reported = False
    features_report_attempts = 0
    step_counter = 0
    try:
        while not should_stop():
            loop_start = time.perf_counter()
            obs = robot.get_observation()

            state_keys = None
            if not policy_enabled:
                # --- HOLD PATH (teleop recording, no policy loaded) ---
                # Send nothing (the motors hold) but record every tick so the
                # episode stays continuous. Axol has no teleop path yet (no
                # RobotProfile ⇒ no SafetyGate, ADR 0011:111), so a policy-less
                # assignment is hold-only here — never "policy".
                actual_joints = _ctrl._extract_joint_state(obs, action_keys)
                state_keys = _ctrl._capture_tick(
                    client, obs, actual_joints, step_counter, control_source="hold"
                )
                step_counter += 1
            else:
                # Encode lazily — client.step() only builds the payload on ticks
                # where DRTC actually sends an observation.
                action = client.step(
                    lambda o=obs: _ctrl._encode_npz(
                        _ctrl._to_policy_schema(o), image_resize=image_resize
                    ),
                    codec="npz",
                )

                if action is not None:
                    action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
                    # Execution-safety delta clamp (+ DRTC-DEBUG glass-box log),
                    # mirroring node/control.py. A huge delta on the first policy
                    # command is the slam: the arm leaps from its current pose to
                    # the model's absolute target.
                    if action_keys:
                        actual_joints = _ctrl._extract_joint_state(obs, action_keys)
                        action_arr = _ctrl._clamp_action_delta(
                            action_arr, actual_joints, _max_step, action_keys,
                            step_counter, source="policy",
                        )
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
