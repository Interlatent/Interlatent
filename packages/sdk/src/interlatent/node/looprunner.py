"""The one robot-agnostic tick skeleton.

Every control loop in this tree runs the same per-tick sequence: observe, drive
the robot, record, tee video to the operator, report features once, instrument,
pace. Only *driving* varies by robot, and that lives behind
:meth:`~interlatent.node.movement.CommandBus.drive`. What remains — this module —
is identical for every arm, which is why it exists exactly once.

The split:

* :mod:`interlatent.node.movement` owns **motion** — arbitration, action
  production, the SafetyGate, the delta clamp, ``send_action``.
* This module owns **everything else** — capture, preview, feature reporting,
  latency accounting, profiling, pacing.
* An adapter owns only what is genuinely robot-specific: its ``robot.py``, and
  optionally a ``pre_tick`` guard for conditions the generic path cannot know
  about (see :class:`~interlatent.node.movement.TickVerdict`).

Kept free of the optional extras so it imports on a barebones Pi; the caller
constructs the robot and passes it in, so this module never resolves a robot
kind and never reaches for lerobot.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional

from .movement import CommandBus, TickVerdict

_LOG = logging.getLogger(__name__)

#: Give up reporting robot features after this many tries; a miss only means the
#: analysis pipeline falls back to bare indices for this environment.
_FEATURE_REPORT_ATTEMPTS = 5
#: Teleop execution-latency log cadence.
_LATENCY_WINDOW_S = 5.0


def run_control_loop(
    *,
    robot: Any,
    bus: CommandBus,
    should_stop: Callable[[], bool],
    fps: int,
    action_keys: list,
    capture_fn: Callable[..., Optional[list]],
    teleop_channel: Any = None,
    preview_fn: Optional[Callable[[dict], dict]] = None,
    report_features_fn: Optional[Callable[[list, list], bool]] = None,
    extract_fn: Optional[Callable[[dict, list], Any]] = None,
    profiler: Any = None,
) -> None:
    """Drive one episode. Returns when ``should_stop()`` or a guard ends it.

    ``capture_fn(obs, action, step, control_source=...)`` records a tick and
    returns the ordered observation-state keys (or ``None`` on failure).
    ``report_features_fn(state_keys, action_keys)`` returns True once accepted.
    Both are supplied by the caller so this module stays free of the wire
    helpers' import weight.
    """
    period = 1.0 / fps if fps > 0 else 1.0 / 30.0
    pre_tick = getattr(robot, "pre_tick", None)

    step = 0
    features_reported = False
    feature_attempts = 0
    preview_empty_warned = False
    lat_n = 0
    lat_sum_ms = 0.0
    lat_max_ms = 0.0
    lat_window_started = time.monotonic()

    try:
        while not should_stop():
            loop_start = time.perf_counter()

            # The observation read comes first and unconditionally: for a robot
            # driven through a supervising daemon it doubles as the keep-alive
            # pump's liveness proof, so skipping it would trip a watchdog.
            obs = robot.get_observation()

            # Robot-specific pre-flight, before any movement is arbitrated.
            if pre_tick is not None:
                verdict = pre_tick(obs)
                if verdict is TickVerdict.END_EPISODE:
                    _LOG.info("pre_tick ended the episode")
                    return
                if verdict is TickVerdict.HOLD_NO_CAPTURE:
                    _pace(loop_start, period)
                    continue

            _tee_state(teleop_channel, obs, action_keys, extract_fn)

            outcome = bus.drive(obs, step=step, now=loop_start)

            state_keys = None
            if outcome.should_record:
                state_keys = capture_fn(
                    obs, outcome.action, step, control_source=outcome.control_source
                )
                step += 1
            capture_at = time.perf_counter() if outcome.should_record else None

            preview_empty_warned = _tee_preview(
                teleop_channel, obs, preview_fn, preview_empty_warned
            )

            # One-time feature-element-names report (ADR 0003), gated on a
            # capture so observation.state names are present and aligned.
            if (
                state_keys is not None
                and not features_reported
                and report_features_fn is not None
                and feature_attempts < _FEATURE_REPORT_ATTEMPTS
            ):
                feature_attempts += 1
                features_reported = bool(report_features_fn(state_keys, action_keys))

            # Teleop execution latency: age of each executed frame, from receive
            # to send_action. The node-local half of teleop latency — pair it
            # with the channel's inter-arrival summary to tell "the network is
            # slow" apart from "the control loop is slow".
            if outcome.frame_age_ms is not None:
                lat_n += 1
                lat_sum_ms += outcome.frame_age_ms
                lat_max_ms = max(lat_max_ms, outcome.frame_age_ms)
            now = time.monotonic()
            if now - lat_window_started >= _LATENCY_WINDOW_S:
                if lat_n:
                    _LOG.info(
                        "teleop exec latency (%.0fs): n=%d age mean/max=%.0f/%.0fms "
                        "(receive -> send_action)",
                        now - lat_window_started, lat_n, lat_sum_ms / lat_n, lat_max_ms,
                    )
                lat_n, lat_sum_ms, lat_max_ms = 0, 0.0, 0.0
                lat_window_started = now

            elapsed = time.perf_counter() - loop_start
            if profiler is not None:
                profiler.record_tick(
                    loop_dt_s=elapsed,
                    cmd_dt_s=(
                        outcome.cmd_at - loop_start if outcome.cmd_at is not None else None
                    ),
                    capture_dt_s=(
                        capture_at - outcome.cmd_at
                        if (capture_at is not None and outcome.cmd_at is not None)
                        else None
                    ),
                    frame_age_ms=outcome.frame_age_ms,
                    engaged=outcome.engaged,
                    teleop_ok=outcome.teleop_ok,
                    estop=outcome.estop_latched,
                    over_period=elapsed >= period,
                )
            if elapsed < period:
                time.sleep(period - elapsed)
    finally:
        try:
            robot.disconnect()
        except Exception:
            _LOG.warning("Robot disconnect failed", exc_info=True)
        if profiler is not None:
            profiler.close()


def _pace(loop_start: float, period: float) -> None:
    elapsed = time.perf_counter() - loop_start
    if elapsed < period:
        time.sleep(period - elapsed)


def _tee_state(teleop_channel, obs, action_keys, extract_fn) -> None:
    """Feed the pod-side retarget stage's staleness gate directly (~15 Hz,
    rate-limited inside ``send_state``).

    RecordTick state rides the batched JPEG uplink and can lag seconds behind on
    a slow link, which made that stage flap between ready and stale mid-engage.
    Guarded so a version-skewed channel can never take down the loop.
    """
    if teleop_channel is None or not action_keys or extract_fn is None:
        return
    send_state = getattr(teleop_channel, "send_state", None)
    if send_state is None:
        return
    try:
        send_state(extract_fn(obs, action_keys).tolist())
    except Exception:
        pass


def _tee_preview(teleop_channel, obs, preview_fn, empty_warned: bool) -> bool:
    """Push small downscaled JPEGs to the operator, decoupled from the batched
    recording uplink (which over a real link runs seconds behind — the operator
    must never steer off that).

    ``preview_due()`` is checked FIRST so idle sessions pay zero encode cost, and
    this runs after the send so it never delays pose→motion.
    """
    if teleop_channel is None or preview_fn is None:
        return empty_warned
    due = getattr(teleop_channel, "preview_due", None)
    send = getattr(teleop_channel, "send_preview", None)
    if due is None or send is None:
        return empty_warned
    try:
        if due():
            frames = preview_fn(obs)
            if frames:
                send(frames, time.monotonic_ns())
            elif not empty_warned:
                _LOG.warning(
                    "preview encode produced no frames (cv2/PIL missing, or no "
                    "uint8 camera arrays in obs keys=%s) — operator video will "
                    "ride the recording uplink", sorted(obs.keys()),
                )
                return True
    except Exception:
        # Previews are best-effort; never break the loop.
        pass
    return empty_warned
