"""Native Nori DRTC control loop (the ``--robot nori`` registry entry point).

A standalone control-loop function in the shape the node daemon invokes
(``import_callable`` → ``loop_fn(**kwargs)``). It drives the robot through the
native :class:`~interlatent.adapters.nori.robot.NoriNativeRobot` and reuses the
LeRobot-free DRTC wire helpers from :mod:`interlatent.node.control` so the
observation payload and recording are byte-identical to the built-in loop.

Beyond the YAM template this loop adds the Nori safety composition:

- ``get_observation()`` each tick doubles as the liveness proof for the
  adapter's keep-alive pump — if this loop wedges, the pump stops and the
  daemon safe-stops (ADR 0015).
- Operator e-stop (frame flag or sticky channel latch) latches the SafetyGate
  AND forwards the daemon's hard latch (``command{name:"estop"}``, ADR 0016).
- A daemon-reported latch/safe-stop is a HARD EPISODE BOUNDARY: flush the DRTC
  schedule and return — one DRTC session is one episode (the daemon's runner
  finally-block fires CloseSession/upload), and returning frees the Nori
  daemon's single control-client slot so a human can run
  ``interlatent-act --robot nori --reset-latch``.
- Stale telemetry (mid-reconnect) holds: no motion, no capture — stale joints
  must never be recorded as live state.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional

import numpy as np

_logger = logging.getLogger(__name__)

# An idle daemon is ALWAYS watchdog-stopped when a session begins (nobody was
# streaming control frames), and it recovers as soon as our keep-alive flows.
# Give the pump this long to feed it back to ok before declaring the session
# unstartable. Verified on hardware 2026-07-10: the first status block after
# connect reads safety=safe_hold watchdog=stop with latch_reason=None.
_STARTUP_RECOVERY_S = 10.0


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

    from .config import build_adapter_config
    from .robot import NoriNativeRobot

    # Nori uses no SO101 joint-zero calibration; clear the module's auto-preset
    # so the shared encoder applies an identity map. (Mirrors yam/loop.py.)
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
    # Mirrors node/control.py: the SafetyGate is the single safety authority
    # for human-driven motion; the daemon re-clamps everything robot-side.
    from interlatent.node.teleop.robot_profile import get_profile
    from interlatent.node.teleop.safety import SafetyGate, TargetSample

    teleop_profile = get_profile(robot_kind or "nori")
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

    # --- Action smoothing (policy path; mirrors yam/loop.py) -------------
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
    # One-shot warnings (mirror node/control.py).
    _pv_empty_warned = False
    _stale_warned = False
    _estop_forwarded = False
    _was_healthy = False  # daemon seen safety=ok this session (see safety gate)
    _loop_t0 = time.monotonic()
    try:
        while not should_stop():
            loop_start = time.perf_counter()
            # LIVENESS PROOF: this call feeds the keep-alive pump's gate
            # (ADR 0015). It must stay the first thing every tick does.
            obs = robot.get_observation()

            # --- Session death check (fatal daemon error / reconnect window
            # exhausted): the episode is over. Returning lets the daemon's
            # finally-block close the DRTC client (upload) and frees the Nori
            # daemon's single control-client slot.
            if robot.session_dead:
                _logger.error(
                    "Nori session dead (%s) — ending episode %s.",
                    robot.dead_reason, session_id,
                )
                return

            # --- Daemon safety state. Three distinct situations:
            #   latched            -> hard episode boundary, human-only reset
            #                         (`interlatent-act --robot nori --reset-latch`).
            #   safe_hold/wd-stop AFTER the session was healthy -> the frame
            #                         stream broke mid-session; hard boundary
            #                         (recovers by itself once a client streams
            #                         again — start a new session).
            #   safe_hold/wd-stop BEFORE first health -> the idle daemon's
            #                         normal resting state; hold and let the
            #                         keep-alive pump feed it back to ok,
            #                         bounded by _STARTUP_RECOVERY_S.
            st = robot.last_status
            _safety = (st or {}).get("safety")
            _wd = (st or {}).get("watchdog")
            _latched = _safety == "latched"
            _stopped = _safety == "safe_hold" or _wd == "stop"
            if _latched or (_stopped and _was_healthy):
                try:
                    client.schedule.flush()
                except Exception:  # noqa: BLE001
                    pass
                if _latched:
                    _logger.warning(
                        "Nori daemon reports latched (latch_reason=%s) — hard "
                        "episode boundary; ending episode %s. Clear with "
                        "`interlatent-act --robot nori --reset-latch`.",
                        st.get("latch_reason"), session_id,
                    )
                else:
                    _logger.warning(
                        "Nori daemon safe-stopped mid-session (safety=%s, "
                        "watchdog=%s) — the control-frame stream broke; hard "
                        "episode boundary; ending episode %s. It self-recovers "
                        "when a client streams again — start a new session.",
                        _safety, _wd, session_id,
                    )
                return
            if _stopped:
                if time.monotonic() - _loop_t0 > _STARTUP_RECOVERY_S:
                    _logger.error(
                        "Nori daemon still safe-stopped %.0fs after session "
                        "start (safety=%s, watchdog=%s) — keep-alive frames are "
                        "not reviving it; ending episode %s.",
                        _STARTUP_RECOVERY_S, _safety, _wd, session_id,
                    )
                    return
                # Startup hold: no motion, no capture. get_observation at the
                # top of the tick keeps proving liveness, so the pump streams
                # keep-alives and the daemon's watchdog walks back to ok.
                elapsed = time.perf_counter() - loop_start
                if elapsed < period:
                    time.sleep(period - elapsed)
                continue
            if st is not None and _safety == "ok" and _wd in ("ok", "warn"):
                if not _was_healthy:
                    _logger.info(
                        "Nori daemon healthy (safety=ok, watchdog=%s) — session "
                        "live.", _wd,
                    )
                _was_healthy = True

            # --- Staleness hold (mid-reconnect or telemetry gap): stale
            # joints must not drive the gate, feed the policy, or be recorded
            # as live state. Send nothing; the daemon watchdog owns the robot
            # meanwhile (pump is silent while disconnected).
            if not robot.telemetry_fresh:
                if not _stale_warned:
                    _stale_warned = True
                    _logger.warning(
                        "Nori telemetry stale (%.0f ms) — holding (no motion, "
                        "no capture) until the link recovers.", robot.obs_age_ms,
                    )
                if teleop_gate is not None:
                    teleop_gate.reset()
                if action_filter is not None:
                    action_filter.reset()
                elapsed = time.perf_counter() - loop_start
                if elapsed < period:
                    time.sleep(period - elapsed)
                continue
            _stale_warned = False

            # Feed the pod-side retarget stage's staleness gate directly over
            # the teleop channel (~15 Hz, rate-limited inside send_state).
            # (Mirrors node/control.py — this native loop replaces that one
            # wholesale, so every teleop-channel tee must exist here too or
            # Nori silently loses it.)
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

            # --- OPERATOR E-STOP (ADR 0016; mirrors node/control.py, plus the
            # Nori hardware forward). Sticky channel latch first — it survives
            # frame staleness and reconnects — then the live frame flag.
            _consume_estop = getattr(teleop_channel, "consume_estop", None)
            estop_hit = bool(frame and frame.estop) or bool(
                _consume_estop is not None and _consume_estop()
            )
            if estop_hit:
                if teleop_gate is not None and not teleop_gate.config.estop_latched:
                    teleop_gate.latch_estop("teleop_frame")
                if not _estop_forwarded:
                    _estop_forwarded = True
                    _logger.warning(
                        "Operator e-stop — latching SafetyGate and forwarding "
                        "the daemon hard latch (command name=estop)."
                    )
                    try:
                        robot.estop()
                    except Exception:  # noqa: BLE001
                        _logger.error(
                            "Failed to forward e-stop to the Nori daemon",
                            exc_info=True,
                        )
                        _estop_forwarded = False  # retry next tick

            # Read the LATCH, not the event. ``consume_estop()`` is one-shot and
            # ``frame.estop`` only holds while the operator's frames say so, so
            # gating solely on ``estop_hit`` above resumed driving on the very
            # next tick — and the policy branch below never consults the gate,
            # so a queued chunk could execute after an e-stop whenever the
            # daemon's own latch hadn't yet surfaced in telemetry. This rung
            # sits ABOVE every movement source, exactly as node/control.py's.
            estop_latched = (
                teleop_gate is not None and teleop_gate.config.estop_latched
            )
            if estop_latched:
                # The daemon telemetry flips to safety=latched, and the hard-
                # boundary check above ends the episode on a following tick.
                # Until then: hold — no motion, no capture, nothing queued.
                try:
                    client.schedule.flush()
                except Exception:  # noqa: BLE001
                    pass
                if action_filter is not None:
                    action_filter.reset()
                elapsed = time.perf_counter() - loop_start
                if elapsed < period:
                    time.sleep(period - elapsed)
                continue

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

                # Echo the executed target's seq back to the producer so it
                # can compute command round-trip latency against its own
                # clock. (Mirrors node/control.py; only the QUIC channel
                # defines note_applied.)
                _note_applied = getattr(teleop_channel, "note_applied", None)
                if _note_applied is not None:
                    try:
                        _note_applied(int(frame.seq))
                    except Exception:  # noqa: BLE001
                        pass

                # Drop policy chunks queued or landing during teleop so they
                # don't apply when the human releases. (Mirrors node/control.py.)
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
                step_counter += 1
            elif not policy_enabled:
                # --- HOLD PATH (teleop recording, disengaged) ---
                # No policy to fall back to: send nothing — the adapter's
                # keep-alive pump keeps the daemon session (and pose) alive,
                # so "send nothing" still means "hold", not "safe-stop".
                # Record every tick so the episode stays continuous.
                if teleop_gate is not None:
                    teleop_gate.reset()
                actual_joints = _ctrl._extract_joint_state(obs, action_keys)
                state_keys = _ctrl._capture_tick(
                    client, obs, actual_joints, step_counter, control_source="hold"
                )
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
                    # Low-pass the policy stream before send_action; the
                    # adapter's per-step delta clamp (inside send_action) stays
                    # the final client-side guard, and the daemon re-clamps
                    # robot-side. (Mirrors yam/loop.py.)
                    if action_filter is not None:
                        action_arr = action_filter.filter(action_arr)
                    # Execution-safety delta clamp (+ DRTC-DEBUG glass-box log),
                    # mirroring node/control.py. The adapter's own per-send clamp
                    # (gripper-exempt) and the daemon's robot-side clamp stay in
                    # place; this one is source-agnostic, so policy and teleop
                    # are bounded by the same --robot.max_step knob.
                    if action_keys:
                        actual_joints = _ctrl._extract_joint_state(obs, action_keys)
                        action_arr = _ctrl._clamp_action_delta(
                            action_arr, actual_joints, _max_step, action_keys,
                            step_counter, source="policy",
                        )
                    action_dict = {k: float(action_arr[i]) for i, k in enumerate(action_keys)}
                    robot.send_action(action_dict)
                    state_keys = _ctrl._capture_tick(
                        client, obs, action_arr, step_counter, control_source="policy"
                    )
                    step_counter += 1

            # Live-preview tee (headset video): small downscaled JPEGs pushed
            # over the teleop channel, decoupled from the batched recording
            # uplink. preview_due() first so idle sessions pay zero encode
            # cost; runs AFTER send_action so it never delays pose→motion.
            # (Mirrors node/control.py.)
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
            if elapsed < period:
                time.sleep(period - elapsed)
    finally:
        try:
            robot.disconnect()  # cameras, then bye + pump stop (provably dead)
        except Exception:  # noqa: BLE001
            _logger.warning("NoriNativeRobot disconnect failed", exc_info=True)
        _logger.info(
            "Native Nori loop exiting for session %s; daemon's client.close() "
            "flushes the recorder queue and triggers server-side upload.",
            session_id,
        )


__all__ = ["control_loop"]
