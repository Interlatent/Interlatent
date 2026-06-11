"""Built-in control-loop wrappers used by the Node daemon.

The daemon calls one of these on every assignment. The functions are
deliberately written to be drop-in replacements for one another and
to never import their heavy deps at module load — the daemon should
be importable on a barebones Pi.

Custom integrations: write your own `control_loop(client, session,
should_stop, **_)` and pass `--loop my_module:control_loop` to the
daemon. `import_callable` resolves it.

Episode recording: the LeRobot wrapper does NOT stage anything on the
Pi anymore. Each Infer call already ships the full observation to the
GPU container; when the OpenSession metadata carries ``record=true``
the server persists per-step rows + raw JPEG bytes, builds the LeRobot
dataset on shutdown, and uploads it through the same inbox protocol
the SDK upload path uses. The Pi loop is therefore pure inference:
``robot.get_observation() -> client.step() -> robot.send_action()``.
"""
from __future__ import annotations

import importlib
import io
import logging
import os
import time
from typing import Any, Callable, Optional

import numpy as np

_LOG = logging.getLogger("interlatent.node.control")

# One-shot guard for the "camera arrays present but no frames encoded" warning
# in _capture_tick — avoids spamming the log at 30 Hz.
_FRAMELESS_WARNED = False


def import_callable(spec: str) -> Callable[..., Any]:
    """Resolve a `module.path:attr` spec to the attribute itself.

    Used for `--loop` overrides on the CLI.
    """
    if ":" not in spec:
        raise ValueError(
            f"--loop expects 'module:function', got {spec!r}"
        )
    module_path, _, attr = spec.partition(":")
    mod = importlib.import_module(module_path)
    fn = getattr(mod, attr, None)
    if fn is None:
        raise AttributeError(
            f"module {module_path!r} has no attribute {attr!r}"
        )
    if not callable(fn):
        raise TypeError(f"{spec!r} is not callable")
    return fn


# ---------------------------------------------------------------------------
# Built-in LeRobot wrapper
# ---------------------------------------------------------------------------


def lerobot_control_loop(
    *,
    client,
    session: dict,
    should_stop: Callable[[], bool],
    robot_kind: str,
    robot_port: Optional[str] = None,
    robot_extra: Optional[dict[str, str]] = None,
    robot_cameras: Optional[dict[str, str]] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    teleop_channel: Any = None,
    node_id: Optional[str] = None,
    # Pre-encode square-resize target for camera frames (pixels per side).
    # None keeps native camera resolution. Set by the daemon when the
    # GPU-side policy is known to downsample anyway (e.g. MolmoAct2 →
    # 256), saving uplink bandwidth without losing information.
    image_resize: Optional[int] = None,
    **_: Any,
) -> None:
    """Generic LeRobot observe/act loop.

    Reads frames + joint state from a LeRobot Robot, npz-encodes them
    for DRTC, and dispatches the returned action chunk.

    **Recording flow:** after a successful step() we capture the
    observation + executed action + JPEG-encoded camera frames and hand
    them to ``client.record_tick(...)``. That call is non-blocking —
    a background thread inside :class:`DRTCClient` drains the queue and
    ships each tick to the server via the ``RecordTick`` RPC, where the
    server-side recorder builds + uploads the LeRobot dataset on
    ``CloseSession``. Inference latency is unaffected.

    **DAgger / teleop override:** when ``teleop_channel`` carries an
    engaged frame from the dashboard, this loop short-circuits the
    policy path. The current held-key set is integrated locally into a
    joint target, sent to the robot, and recorded with
    ``control_source="teleop"`` so the resulting LeRobot dataset
    distinguishes policy vs human-driven steps. The DRTC action buffer
    is flushed every engaged tick so policy chunks that land late
    don't fight the human.

    Pacing is governed by the DRTC client's internal RTC cooldown plus
    the explicit period below: without it the loop busy-spins while
    waiting for an action chunk (e.g. during a DRTC cold start),
    starving the camera capture thread until lerobot's frame-freshness
    check fails with a TimeoutError.
    """
    # Auto-enable the pre-#777 calibration migration for MolmoAct2 (its
    # released SO100/SO101 data predates lerobot's joint-zero convention
    # change). The env var still overrides — set INTERLATENT_CALIB_PRESET=none
    # to force it off. See the "Calibration migration" section below.
    global _AUTO_CALIB_PRESET
    _AUTO_CALIB_PRESET = (
        "so101_pre777"
        if "molmoact" in str(session.get("policy_uri", "")).lower()
        else ""
    )

    robot = _make_lerobot_robot(
        robot_kind, port=robot_port, extra=robot_extra or {}, cameras=robot_cameras or {}
    )

    session_id = session.get("id", "")
    fps = int(session.get("fps", 30) or 30)

    robot.connect()
    # Ordered action-feature names — used to turn the flat policy action
    # vector into the {name: value} dict send_action expects.
    action_keys = list(getattr(robot, "action_features", None) or [])
    _LOG.info(
        "LeRobot %r connected; action_keys=%s; entering control loop "
        "(streaming RecordTick → server) episode=%s",
        robot_kind, action_keys, session_id,
    )

    # DRTC-DEBUG glass-box: the motor norm mode decides how send_action
    # interprets the policy's numbers. MolmoAct2 emits ABSOLUTE joint pose in
    # DEGREES (ranges up to ±270). If a body joint is in RANGE_M100_100 (i.e.
    # use_degrees=False) instead of DEGREES, a value like 124/186/270 is sent
    # as an out-of-range normalized command and the joint slams to its stop.
    # Logged once at connect so the units question is answered on every run.
    try:
        _modes = {}
        _bus = getattr(robot, "bus", None)
        _motors = getattr(_bus, "motors", None) or {}
        for _name, _motor in _motors.items():
            _nm = getattr(_motor, "norm_mode", None)
            _modes[_name] = getattr(_nm, "name", str(_nm))
        _LOG.info(
            "DRTC-DEBUG robot units | use_degrees=%s | image_resize=%s | "
            "calib_preset=%r calib_map=%s | motor_norm_modes=%s",
            getattr(getattr(robot, "config", None), "use_degrees", "?"),
            image_resize,
            _resolve_calib_preset_name() or None,
            _active_calib_map() or None,
            _modes,
        )
    except Exception:
        _LOG.warning("DRTC-DEBUG robot-units introspection failed", exc_info=True)

    # Teleop integrator state. ``teleop_target`` is the running joint
    # target we accumulate while engaged. It resets to ``None`` on
    # disengage so the next engage starts from the current actual
    # joint position (the human doesn't want their last commanded
    # target to come back after they release).
    from .keyboard_action import KeyboardActionConfig, next_target as kb_next_target

    teleop_cfg = KeyboardActionConfig()
    teleop_target: Optional[np.ndarray] = None
    teleop_last_t: Optional[float] = None
    teleop_warned_action_dim = False

    try:
        period = 1.0 / fps if fps > 0 else 1.0 / 30.0
        step_counter = 0
        # Report the robot's per-element feature names once (env-constant
        # robot config). state_keys come from the first capture so they
        # align exactly with the recorded observation.state vector; action
        # names are the robot's ordered action_features. Best-effort with a
        # few retries — a miss just means the analysis pipeline falls back
        # to bare indices for this env.
        features_reported = False
        features_report_attempts = 0
        while not should_stop():
            loop_start = time.perf_counter()

            obs = robot.get_observation()

            # Sample the latest browser teleop frame. None when no
            # dashboard is connected or the last frame is stale.
            frame = teleop_channel.latest_frame() if teleop_channel is not None else None
            engaged = bool(frame and frame.engaged and frame.deadman)

            if engaged and action_keys and len(action_keys) == 6:
                # --- TELEOP PATH ---
                actual_joints = _extract_joint_state(obs, action_keys)
                if teleop_target is None:
                    # First engaged tick — start the integrator at the
                    # robot's current pose so there is no jump.
                    teleop_target = actual_joints.copy()
                    teleop_last_t = loop_start
                dt = max(0.0, loop_start - (teleop_last_t or loop_start))
                teleop_last_t = loop_start
                teleop_target = kb_next_target(
                    target_joints=teleop_target,
                    actual_joints=actual_joints,
                    held_keys=frame.held_keys,
                    dt=dt,
                    cfg=teleop_cfg,
                )
                action_arr = np.asarray(teleop_target, dtype=np.float32).reshape(-1)
                robot.send_action(_coerce_action_for_robot(action_arr, action_keys))

                # Drop policy chunks that are already queued or land
                # during teleop so they don't apply when the human
                # releases.
                try:
                    client.flush_buffer()
                except Exception:
                    pass

                _capture_tick(
                    client, obs, action_arr, step_counter,
                    control_source="teleop",
                )
                # Per-tick capture — non-blocking; queues to a background
                # thread in the DRTC client that ships via RecordTick.
                state_keys = _capture_tick(client, obs, action_arr, step_counter)

                # One-time feature-name report (retry a few ticks on failure).
                if not features_reported and features_report_attempts < 5:
                    features_report_attempts += 1
                    if _report_robot_features(
                        api_base, node_id, api_key, state_keys, action_keys,
                    ):
                        features_reported = True

                step_counter += 1
            else:
                if engaged and not teleop_warned_action_dim:
                    _LOG.warning(
                        "Teleop engage ignored: keyboard_action expects 6 joints "
                        "but robot has %d action keys (%s). DAgger is only "
                        "wired for the SO-101 in this release.",
                        len(action_keys), action_keys,
                    )
                    teleop_warned_action_dim = True

                # --- POLICY PATH ---
                # Reset the integrator so the next engage starts fresh.
                teleop_target = None
                teleop_last_t = None

                # Encode lazily: client.step() only builds the payload on
                # ticks where DRTC actually sends an observation, so we
                # skip the encode on the majority of ticks. With
                # ``image_resize`` set, frames are downsampled to a
                # square before JPEG — for MolmoAct2 (224 input) sending
                # 256x256 cuts payload ~5-10x vs raw 640x480 with no
                # measurable accuracy loss.
                action = client.step(
                    lambda o=obs: _encode_npz(
                        _to_policy_schema(o), image_resize=image_resize
                    ),
                    codec="npz",
                )
                if action is not None:
                    action_arr = np.asarray(action, dtype=np.float32).reshape(-1)

                    # DRTC-DEBUG glass-box: log the actual joint angles, the
                    # commanded ones, and the per-joint jump — for the first
                    # chunk-boundaries of a session and periodically after.
                    # A huge delta on the first policy command is the slam:
                    # the arm leaps from its current pose to the model's
                    # absolute target. Units here are whatever the motor norm
                    # mode reports above (degrees for MolmoAct2).
                    if action_keys and (step_counter < 10 or step_counter % 100 == 0):
                        try:
                            _actual = _extract_joint_state(obs, action_keys)
                            _cmd = action_arr[: len(action_keys)]
                            _delta = _cmd - _actual[: len(_cmd)]
                            _pairs = ", ".join(
                                "%s: %.2f->%.2f (Δ%+.2f)"
                                % (action_keys[i], _actual[i], _cmd[i], _delta[i])
                                for i in range(len(_cmd))
                            )
                            _LOG.info(
                                "DRTC-DEBUG joints #%d | max|Δ|=%.2f | %s",
                                step_counter, float(np.abs(_delta).max()), _pairs,
                            )
                        except Exception:
                            _LOG.warning(
                                "DRTC-DEBUG joint dump failed", exc_info=True
                            )

                    robot.send_action(_coerce_action_for_robot(action_arr, action_keys))

                    _capture_tick(
                        client, obs, action_arr, step_counter,
                        control_source="policy",
                    )
                    step_counter += 1

            elapsed = time.perf_counter() - loop_start
            if elapsed < period:
                time.sleep(period - elapsed)
    finally:
        try:
            robot.disconnect()
        except Exception:
            _LOG.warning("Robot disconnect failed", exc_info=True)
        _LOG.info(
            "Control loop exiting for session %s; client.close() will "
            "flush recorder queue and trigger server-side upload.",
            session_id,
        )


def _extract_joint_state(obs: dict, action_keys: list) -> np.ndarray:
    """Pull joint positions out of a lerobot observation in action order.

    For follower robots the action features (e.g. ``shoulder_pan.pos``)
    match the observation joint keys exactly. Missing keys are filled
    with zero — the lead-clip in the keyboard integrator then prevents
    the target from running away while the robot is still publishing.
    """
    out = np.zeros(len(action_keys), dtype=np.float32)
    for i, key in enumerate(action_keys):
        v = obs.get(key)
        if v is None:
            continue
        try:
            out[i] = float(np.asarray(v).reshape(-1)[0])
        except (TypeError, ValueError, IndexError):
            continue
    return out


def _capture_tick(
    client: Any,
    obs: Any,
    action: "np.ndarray",
    step: int,
    *,
    control_source: Optional[str] = None,
) -> None:
    """JPEG-encode camera frames and queue a RecordTick on the DRTC client.

    Returns the ordered list of observation keys whose scalars went into the
    recorded ``observation_state`` vector (so a caller can report them as the
    per-element feature names — they align with the vector by construction),
    or ``None`` if the capture raised.

    lerobot returns ``{"<motor>.pos": float, "<cam>": np.ndarray (HxWx3 RGB)}``.
    We split numeric scalars (joint state) from numpy arrays (camera frames),
    JPEG-encode each frame, and hand the lot to :meth:`DRTCClient.record_tick`.

    JPEG encoding runs on the control thread but it's cheap (~1–2 ms for a
    640×480 frame on a Pi 5, OpenCV releases the GIL during ``imencode``).
    If this turns out to be a real budget hit on slower hardware we can
    move it into the DRTC client's recorder thread — but for now keeping
    it here lets the in-flight buffer hold compressed bytes, not raw RGB,
    cutting memory ~10× per queued frame.
    """
    try:
        import time as _time

        try:
            import cv2  # type: ignore
        except ImportError:
            cv2 = None  # type: ignore

        jpegs: dict[str, bytes] = {}
        state: list[float] = []
        state_keys: list[str] = []
        cam_arrays = 0
        for k, v in obs.items():
            # Coerce first, then detect — match _to_policy_schema's image
            # rule (uint8 + ndim>=2) EXACTLY, so recording sees the same
            # camera frames inference does. lerobot's get_observation can
            # hand back torch tensors / PIL images rather than bare ndarrays
            # (it varies by version); a plain ``isinstance(v, np.ndarray)``
            # check then silently drops those frames into the state branch,
            # which is how an episode ends up with observations but a blank
            # video while inference still works.
            try:
                arr = v if isinstance(v, np.ndarray) else np.asarray(v)
            except Exception:
                arr = None
            if arr is not None and arr.dtype == np.uint8 and arr.ndim >= 2:
                cam_arrays += 1
                data: Optional[bytes] = None
                # Fast path: OpenCV (releases the GIL during imencode).
                if cv2 is not None:
                    img = arr
                    if img.ndim == 3 and img.shape[2] == 3:
                        img = np.ascontiguousarray(img[..., ::-1])  # RGB->BGR
                    ok, buf = cv2.imencode(
                        ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85],
                    )
                    if ok:
                        data = bytes(buf)
                # Fallback: PIL — the SAME encoder the inference path uses
                # (_jpeg_encode). Without this, a node that has Pillow but
                # not OpenCV records state-only ticks (empty jpegs) while
                # inference still works, yielding a black/gappy video. Pass
                # the ORIGINAL RGB array (PIL expects RGB, not the BGR flip).
                if data is None:
                    try:
                        data = _jpeg_encode(arr).tobytes()
                    except Exception:
                        _LOG.debug("PIL JPEG encode failed for %r", k, exc_info=True)
                        data = None
                if data:
                    jpegs[k] = data
            else:
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                state.append(fv)
                state_keys.append(k)

        # Surface the silent-frameless case once: if the observation carried
        # camera arrays but none encoded, recording will be observation-only
        # (black video). The most common cause is neither cv2 nor Pillow
        # being importable on the node.
        global _FRAMELESS_WARNED
        if cam_arrays and not jpegs and not _FRAMELESS_WARNED:
            _LOG.warning(
                "record_tick: observation had %d camera array(s) but 0 frames "
                "encoded (cv2=%s) — episodes will record observations only and "
                "the video will be blank. Install Pillow or opencv-python on "
                "the node.",
                cam_arrays, cv2 is not None,
            )
            _FRAMELESS_WARNED = True

        client.record_tick(
            step=step,
            observation_state=state if state else None,
            action=action.tolist(),
            jpegs=jpegs,
            control_timestamp_ns=_time.monotonic_ns(),
            control_source=control_source,
        )
        return state_keys
    except Exception:
        # Capture must never break inference.
        _LOG.exception("record_tick failed at step %d", step)
        return None


def _report_robot_features(
    api_base: Optional[str],
    node_id: Optional[str],
    token: Optional[str],
    state_names: Optional[list],
    action_names: Optional[list],
) -> bool:
    """POST the robot's per-element feature names to the node endpoint.

    Returns True when the report was sent (or there is nothing actionable to
    send / no config to send it with — so the caller stops retrying), and
    False only on a network error worth one more attempt. The backend stores
    them first-writer-wins, so re-reports are harmless.

    Names are reported verbatim from the lerobot robot — modern LeRobot's
    own flat-list convention (e.g. ``["shoulder_pan.pos", ...]``) — keyed by
    the canonical feature keys ``observation.state`` / ``action``.
    """
    if not api_base or not node_id or not token:
        return True  # unconfigured — nothing to do, don't spin
    names: dict[str, list] = {}
    if state_names:
        names["observation.state"] = list(state_names)
    if action_names:
        names["action"] = list(action_names)
    if not names:
        return True  # robot exposed no labelable features
    try:
        import httpx

        url = f"{api_base.rstrip('/')}/api/v1/nodes/{node_id}/robot-features"
        resp = httpx.post(
            url,
            headers={"x-api-key": token},
            json={"feature_element_names": names},
            timeout=10.0,
        )
        _LOG.info(
            "Reported robot feature names (%s) -> %s",
            {k: len(v) for k, v in names.items()}, resp.status_code,
        )
        return True
    except Exception:
        _LOG.warning("Failed to report robot feature names (will retry)", exc_info=True)
        return False


def _build_camera_configs(cameras: dict[str, str]) -> dict[str, Any]:
    """Build lerobot OpenCV camera configs from a ``{name: device}`` map.

    Each ``name`` becomes the lerobot camera key, so the robot emits
    ``observation.images.<name>``. Pick names that match the policy's
    expected image keys and no ``rename_map`` is needed.

    ``device`` is a /dev path ("/dev/video0") or a bare index ("0").
    Resolution/fps default to 640x480@30 — the SO101 camera default.

    When the installed lerobot supports it, cameras are opened with the
    MJPG fourcc (compressed) instead of raw YUYV. On a shared USB 2.0 bus
    (e.g. a Raspberry Pi) raw YUYV from multiple cameras overruns the bus
    and connect() fails with ``VIDIOC_QBUF: Bad file descriptor``; MJPG
    cuts per-stream bandwidth ~10x and avoids that.
    """
    if not cameras:
        return {}
    from lerobot.cameras.opencv import OpenCVCameraConfig
    out: dict[str, Any] = {}
    for name, device in cameras.items():
        idx_or_path: Any = int(device) if str(device).isdigit() else device
        # width/height/fps are required by current lerobot — leaving
        # them unset raises "Specifying 'width' is required". MJPG keeps
        # us within USB 2.0 bandwidth on a Pi when multiple cameras
        # share the bus; if a specific camera rejects it the user can
        # swap in a custom --loop adapter.
        kwargs: dict[str, Any] = dict(
            index_or_path=idx_or_path,
            width=640,
            height=480,
            fps=30,
            fourcc="MJPG",
        )
        out[name] = OpenCVCameraConfig(**kwargs)
    return out


def _make_lerobot_robot(
    kind: str,
    *,
    port: Optional[str],
    extra: dict[str, str],
    cameras: Optional[dict[str, str]] = None,
):
    """Instantiate a LeRobot Robot by name.

    For v1 we support the two follower configs we've verified — extending
    to more is a 3-line addition per type.
    """
    # Importing here keeps the daemon importable without lerobot installed.
    try:
        from lerobot.robots import make_robot_from_config
    except ImportError as e:
        raise RuntimeError(
            "lerobot is not installed. Install with `pip install "
            "interlatent[lerobot]` (or pass --loop module:fn for a "
            "custom adapter)."
        ) from e

    cam_configs = _build_camera_configs(cameras or {})

    kind_norm = kind.lower().strip()
    if kind_norm in ("so101", "so101_follower"):
        # lerobot consolidated its per-robot modules: SO101FollowerConfig
        # now lives in the shared `so_follower` module (covers SO100 +
        # SO101). Older lerobot shipped a dedicated `so101_follower`
        # module — support both layouts.
        try:
            from lerobot.robots.so_follower import SO101FollowerConfig
        except ImportError:
            from lerobot.robots.so101_follower import SO101FollowerConfig
        kwargs = _filter_kwargs(SO101FollowerConfig, extra)
        if cam_configs:
            kwargs["cameras"] = cam_configs
        cfg = SO101FollowerConfig(port=port or "", **kwargs)
        return make_robot_from_config(cfg)

    if kind_norm in ("koch", "koch_follower"):
        from lerobot.robots.koch_follower import KochFollowerConfig
        kwargs = _filter_kwargs(KochFollowerConfig, extra)
        if cam_configs:
            kwargs["cameras"] = cam_configs
        cfg = KochFollowerConfig(port=port or "", **kwargs)
        return make_robot_from_config(cfg)

    raise ValueError(
        f"Unsupported --robot {kind!r}. Built-in support: so101_follower, "
        f"koch_follower. For other LeRobot robots, write a thin adapter and "
        f"pass --loop module:fn."
    )


def _filter_kwargs(cfg_cls, extra: dict[str, str]) -> dict[str, Any]:
    """Pick only the keys `cfg_cls` actually declares.

    Lets the user pass --robot-arg key=value without us hardcoding which
    keys are valid for which config class.
    """
    try:
        import dataclasses
        if dataclasses.is_dataclass(cfg_cls):
            valid = {f.name for f in dataclasses.fields(cfg_cls)}
            return {k: v for k, v in extra.items() if k in valid}
    except Exception:
        pass
    return dict(extra)


# ---------------------------------------------------------------------------
# Calibration migration (lerobot PR #777)
# ---------------------------------------------------------------------------
#
# lerobot PR #777 changed the SO100/SO101 joint zero-position convention from
# "arm fully extended horizontal = 0" (old) to "middle of each joint's range
# = 0" (new). A policy trained on pre-#777 data (e.g. allenai/MolmoAct2-*)
# both *expects proprio* and *emits actions* in the OLD frame, while a current
# lerobot robot reads/commands in the NEW frame. Feeding new-frame state to
# such a policy pushes some joints outside the trained quantile range -> the
# normalizer clamps -> the policy emits a constant boundary action -> the arm
# slams toward it (observed: shoulder_lift commanded +147 deg in one chunk).
# See https://huggingface.co/docs/lerobot/backwardcomp
#
# Fix: a per-joint affine mapping OLD<->NEW, applied on BOTH boundaries:
#   - state-in  (robot NEW -> model OLD): inverse, so the policy sees
#     in-distribution proprio and stops clamping;
#   - action-out (model OLD -> robot NEW): forward, so commands land in the
#     robot's frame.
# Stored as the forward (OLD->NEW) affine new = scale*old + offset; the inverse
# is old = (new - offset)/scale. Keyed by joint NAME, so it is independent of
# action-feature ordering; joints absent from the map pass through unchanged.
#
# Off by default. Enable per-run with INTERLATENT_CALIB_PRESET=so101_pre777
# (set on the node, alongside --robot.use_degrees=true). Verify with the
# DRTC-DEBUG joints log: a correct map collapses max|Δ| from ~147 deg to a few
# degrees. If your robot has residual calibration offset vs the dataset, tune
# the per-joint (scale, offset) below.
#
# From the lerobot backward-compat doc (SO101 pre-#777 -> current):
#   shoulder_lift_new = -(old - 90)   ->  scale=-1, offset=90
#   elbow_flex_new    =  old - 90     ->  scale= 1, offset=-90
_CALIB_PRESETS: dict[str, dict[str, tuple[float, float]]] = {
    "so101_pre777": {
        "shoulder_lift": (-1.0, 90.0),
        "elbow_flex": (1.0, -90.0),
    },
}

# Preset auto-selected per session by the control loop (e.g. so101_pre777 when
# the policy is a MolmoAct2 checkpoint, whose released SO100/SO101 data predates
# lerobot PR #777). The env var overrides this both ways: a preset name forces
# that map; "none"/"off"/"0"/"false" force the migration OFF for debugging.
_AUTO_CALIB_PRESET: str = ""


def _resolve_calib_preset_name() -> str:
    env = os.environ.get("INTERLATENT_CALIB_PRESET", "").strip()
    if env:
        return "" if env.lower() in ("none", "off", "0", "false") else env
    return _AUTO_CALIB_PRESET


def _active_calib_map() -> dict[str, tuple[float, float]]:
    """Per-joint OLD->NEW affine for the active session.

    Resolved from the env-var override or the session auto-preset.
    Empty/unknown preset -> {} (identity: no migration applied).
    """
    return _CALIB_PRESETS.get(_resolve_calib_preset_name(), {})


def _joint_name(key: str) -> str:
    """``"shoulder_lift.pos"`` -> ``"shoulder_lift"`` (bare keys pass through)."""
    return key.rsplit(".", 1)[0] if "." in key else key


def _calib_old_to_new(joint: str, value: float, calib: dict) -> float:
    """Model frame -> robot frame (for the action sent to the robot)."""
    scale, offset = calib.get(joint, (1.0, 0.0))
    return scale * value + offset


def _calib_new_to_old(joint: str, value: float, calib: dict) -> float:
    """Robot frame -> model frame (for the proprio state fed to the policy)."""
    scale, offset = calib.get(joint, (1.0, 0.0))
    return (value - offset) / scale


# ---------------------------------------------------------------------------
# Codecs
# ---------------------------------------------------------------------------


def _to_policy_schema(obs: dict) -> dict:
    """Map a raw lerobot robot observation to the policy input schema.

    lerobot robots return joints as bare ``<motor>.pos`` scalars and
    cameras as bare ``<name>`` image arrays. lerobot policies instead
    expect a single ``observation.state`` vector plus
    ``observation.images.<name>`` image keys.

    Joint scalars are concatenated in robot-observation order — which
    is the motor order the policy's ``observation.state`` was trained
    on, so no explicit joint-name map is needed. Images are detected by
    uint8 dtype + 2-D-or-more shape and re-keyed under
    ``observation.images.``.
    """
    state_vals: list[float] = []
    out: dict[str, Any] = {}
    calib = _active_calib_map()  # robot(NEW) -> model(OLD); {} when disabled
    for key, value in obs.items():
        if key == "task":
            out["task"] = value
            continue
        arr = np.asarray(value)
        if arr.dtype == np.uint8 and arr.ndim >= 2:
            name = key.rsplit(".", 1)[-1]
            out[f"observation.images.{name}"] = arr
        else:
            joint = _joint_name(key)
            state_vals.extend(
                _calib_new_to_old(joint, float(x), calib) for x in arr.flatten()
            )
    if state_vals:
        out["observation.state"] = np.asarray(state_vals, dtype=np.float32)
    return out


def _jpeg_encode(
    arr: np.ndarray,
    quality: int = 85,
    target_size: Optional[int] = None,
) -> np.ndarray:
    """Encode an HWC/HW uint8 image as JPEG bytes (a 1-D uint8 array).

    Camera frames dominate the DRTC observation payload. JPEG already
    shrinks raw frames ~15-30x; ``target_size`` (square edge in pixels)
    additionally pre-resizes the image with BILINEAR before encoding,
    which is the right move when the GPU-side policy is going to
    downsample anyway (e.g. MolmoAct2's image processor squashes
    everything to ~224×224 internally — sending 640×480 wastes uplink
    and decode cycles for no signal gain).

    The server auto-detects the JPEG magic bytes (FF D8 FF) and decodes —
    no codec negotiation needed.
    """
    from PIL import Image

    img = arr
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img[:, :, 0]
    pil = Image.fromarray(img)
    if target_size is not None and target_size > 0:
        # Square resize. We don't preserve aspect ratio because the
        # downstream processor doesn't either — its own resize is also
        # square. Matching that here keeps the wire bytes minimal.
        pil = pil.resize((int(target_size), int(target_size)), Image.BILINEAR)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    return np.frombuffer(buf.getvalue(), dtype=np.uint8)


def _encode_npz(obs: dict, image_resize: Optional[int] = None) -> bytes:
    """Encode a LeRobot observation dict into npz bytes.

    LeRobot observations are dicts of numpy arrays (joints, cameras).
    Camera frames (uint8, 2-D+) are JPEG-compressed before packing;
    everything else (the state vector) goes in raw. The DRTC server's
    npz codec produces a dict for the lerobot backend to consume via
    `_to_batch`, decoding any JPEG blobs on the way in.

    ``image_resize`` is forwarded to :func:`_jpeg_encode` for each
    camera frame; None means keep native resolution.
    """
    flat: dict[str, np.ndarray] = {}
    for k, v in obs.items():
        if isinstance(v, np.ndarray) and v.dtype == np.uint8 and v.ndim >= 2:
            # Camera frame — JPEG-compress it.
            try:
                flat[k] = _jpeg_encode(v, target_size=image_resize)
                continue
            except Exception:
                _LOG.debug("JPEG encode failed for %r; sending raw", k, exc_info=True)
                flat[k] = v
                continue
        if isinstance(v, np.ndarray):
            flat[k] = v
        else:
            try:
                flat[k] = np.asarray(v)
            except Exception:
                # Skip un-encodable keys rather than crashing the whole loop.
                _LOG.debug("Skipping unencodable obs key %r (type=%s)", k, type(v).__name__)
    buf = io.BytesIO()
    np.savez(buf, **flat)
    return buf.getvalue()


def _coerce_action_for_robot(action: np.ndarray, action_keys: list) -> Any:
    """Convert the DRTC action vector into the shape the robot wants.

    lerobot follower robots (so_follower, koch_follower, ...) expect a
    dict mapping action-feature names to floats, e.g.
    ``{"shoulder_pan.pos": 0.1, ...}``. We zip the flat policy action
    vector with the robot's ordered ``action_features`` keys.

    If the lengths don't match (or no keys were supplied) we fall back
    to the bare array — some robots accept that.
    """
    arr = np.asarray(action, dtype=np.float32).reshape(-1)
    if action_keys and len(action_keys) == len(arr):
        calib = _active_calib_map()  # model(OLD) -> robot(NEW); {} when disabled
        return {
            k: _calib_old_to_new(_joint_name(k), float(v), calib)
            for k, v in zip(action_keys, arr)
        }
    return arr
