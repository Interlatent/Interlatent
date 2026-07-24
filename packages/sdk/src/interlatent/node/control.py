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
from typing import Any, Callable, Optional

import numpy as np

from . import _env
from .._clamp_log import warn_clamp
from .jpeg import backend_name as _jpeg_backend_name
from .jpeg import encode_jpeg as _encode_jpeg
from .looprunner import run_control_loop
from .movement import CommandBus, WireHelpers
from .teleop_profiler import NodeTeleopProfiler

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
    node_id: Optional[str] = None,
    # Protection-bypass secret for a protected preview/test domain (mirrors
    # NodeDaemonConfig.bypass_key) — needed here because robot-features
    # reporting makes its own request rather than reusing the daemon's
    # shared httpx client.
    bypass_key: Optional[str] = None,
    # Browser/VR teleop receiver (set by the daemon when a session is
    # teleop-capable). The node consumes ``mode="targets"`` frames — absolute
    # joint vectors the hosted teleop engine already computed — and routes them
    # through the SafetyGate. None disables teleop (pure policy). See
    # docs/adr/0009 (hosted relay + its amendments).
    teleop_channel: Optional[Any] = None,
    # Pre-encode square-resize target for camera frames (pixels per side).
    # None keeps native camera resolution. Set by the daemon when the
    # GPU-side policy is known to downsample anyway (e.g. MolmoAct2 →
    # 256), saving uplink bandwidth without losing information.
    image_resize: Optional[int] = None,
    # False for teleop-recording assignments (no policy loaded): the loop
    # never calls client.step() — all motion comes from the teleop channel,
    # and disengaged ticks hold pose while still recording (so the episode
    # is continuous). See the TeleopRecording resource.
    policy_enabled: bool = True,
    **_: Any,
) -> None:
    """Generic LeRobot observe/act loop.

    A thin shim now: this function builds the robot and the per-session
    collaborators (SafetyGate, delta clamp, action smoother, profiler), wires
    them into a full-motion :class:`~interlatent.node.movement.CommandBus`, and
    hands the tick to :func:`~interlatent.node.looprunner.run_control_loop` —
    the one tick skeleton every loop shares. Per-tick behavior lives there and
    in ``CommandBus.drive()``, not here.

    **Recording flow:** each recorded tick hands the observation + executed
    action to ``_capture_tick`` → ``client.record_tick(...)``. That call is
    non-blocking — a background thread inside :class:`DRTCClient` drains the
    queue and ships each tick to the server via the ``RecordTick`` RPC, where
    the server-side recorder builds + uploads the LeRobot dataset on
    ``CloseSession``. Inference latency is unaffected.

    Pacing is governed by the DRTC client's internal RTC cooldown plus the
    runner's explicit period: without it the loop busy-spins while waiting for
    an action chunk (e.g. during a DRTC cold start), starving the camera
    capture thread until lerobot's frame-freshness check fails with a
    TimeoutError.
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

    # --- Teleop receiver setup (hosted relay path) ----------------------
    # The SafetyGate is the single safety authority for human-driven motion:
    # the platform streams absolute joint targets (``mode="targets"``) and they
    # route through the gate's workspace + velocity clamp here on the node. It
    # needs a static per-robot profile (limits + velocity cap + rest pose) that
    # lerobot cannot supply; without one for this robot kind we refuse the gated
    # teleop path and stay on policy. The pose-modality compute (clutch mapping
    # + IK) lives on the compute pod (ADR 0009, second amendment), so the node
    # handles only ``mode="keys"`` / ``mode="targets"``.
    from .teleop.robot_profile import get_profile
    from .teleop.safety import SafetyGate

    _teleop_dt = 1.0 / fps if fps > 0 else 1.0 / 30.0
    teleop_profile = get_profile(robot_kind)
    teleop_gate = (
        SafetyGate(profile=teleop_profile, control_dt=_teleop_dt)
        if teleop_profile is not None
        else None
    )
    # Reported to the backend (→ Environment.teleop_profile → the teleop-token
    # response) so the producer can retarget against this robot's schema.
    _teleop_schema = teleop_profile.to_schema_dict() if teleop_profile is not None else None

    # --- Delta clamp (execution safety, all action sources) -------------
    # Last-line guard against a single-tick joint slam (model glitch, bad
    # chunk, or teleop frame). The per-step limit is configured by the adapter
    # via the ``--robot.max_step`` extra (units match the motor-norm mode the
    # loop logs above). Unset ⇒ disabled with a one-time warning. See A6 /
    # ADR 0012. ``None`` disables.
    _max_step = _parse_max_step(robot_extra or {})
    if _max_step is None:
        _LOG.warning(
            "Delta clamp DISABLED: no --robot.max_step set. A single-tick joint "
            "slam will execute unclamped. Set --robot.max_step=<units> (motor-norm "
            "units, e.g. degrees for MolmoAct2) to enable the execution-safety guard."
        )

    # --- Action smoothing (policy path) ---------------------------------
    # Low-pass the per-tick policy action stream to attenuate chunk-boundary /
    # model jitter before it reaches the motors. 2nd-order Butterworth, designed
    # at the control rate; default 3 Hz cutoff, tunable via
    # ``--robot.action_filter_hz`` (0/none disables). Smoothing runs BEFORE the
    # delta clamp so the clamp remains the final execution-safety guard. Not
    # applied on the teleop path (human input is already velocity-clamped by the
    # SafetyGate and should stay responsive). See node/smoothing.py.
    from .smoothing import ButterworthLowPass

    _filter_hz = _parse_action_filter_hz(robot_extra or {})
    action_filter = (
        ButterworthLowPass(cutoff_hz=_filter_hz, sample_hz=float(fps if fps > 0 else 30))
        if _filter_hz is not None
        else None
    )
    if action_filter is not None:
        _LOG.info(
            "Action smoothing ENABLED: 2nd-order Butterworth low-pass, cutoff=%.2f Hz "
            "@ %d Hz control rate (policy path). Set --robot.action_filter_hz=none to "
            "disable.", action_filter.cutoff_hz, int(fps if fps > 0 else 30),
        )
    else:
        _LOG.info("Action smoothing DISABLED (--robot.action_filter_hz=none).")

    # Local (node-side) per-second CSV profiler — see teleop_profiler.py.
    # Purely additive: read-only clock samples around work the loop is
    # already doing, written to a local file on this machine. Never
    # raises; disables itself silently on any internal failure. Distinct
    # from the "teleop exec latency" 5s log block below (which it also
    # captures into the CSV as frame_age_*): that block only logs, it
    # doesn't persist a file you can open after the session ends.
    node_profiler = NodeTeleopProfiler(
        session_id=session_id, robot_kind=robot_kind, fps=fps,
        teleop_configured=teleop_gate is not None,
    )

    # --- Wire the bus and hand the tick to the shared runner -------------
    # The bus owns the whole motion path (arbitrate → produce → SafetyGate →
    # delta clamp → send_action → flush/smoother-reset); the runner owns
    # everything else (capture, preview tee, feature report, latency window,
    # profiler, pacing). See node/movement.py and node/looprunner.py.
    #
    # ``coerce`` is where the calibration frame is decided: this path applies
    # the OLD→NEW affine via _coerce_action_for_robot because the policy
    # commands in *model* frame. The manual LeRobotAdapter path is a raw
    # robot-frame passthrough by design — never route this loop through it.
    command_bus = CommandBus(
        teleop_channel=teleop_channel,
        teleop_gate=teleop_gate,
        teleop_profile=teleop_profile,
        policy_enabled=policy_enabled,
        robot=robot,
        client=client,
        action_keys=action_keys,
        helpers=WireHelpers(
            extract=_extract_joint_state,
            clamp=_clamp_action_delta,
            coerce=_coerce_action_for_robot,
            encode=lambda o: _encode_npz(
                _to_policy_schema(o), image_resize=image_resize
            ),
        ),
        max_step=_max_step,
        action_filter=action_filter,
    )

    def _capture(obs, action, step, *, control_source=None):
        # Non-blocking: queues to a background thread in the DRTC client that
        # ships each tick to the server via the RecordTick RPC.
        return _capture_tick(client, obs, action, step, control_source=control_source)

    def _report(state_keys, act_keys):
        return _report_robot_features(
            api_base, node_id, api_key, state_keys, act_keys,
            teleop_profile=_teleop_schema, bypass_key=bypass_key,
        )

    try:
        run_control_loop(
            robot=robot,
            bus=command_bus,
            should_stop=should_stop,
            fps=fps,
            action_keys=action_keys,
            capture_fn=_capture,
            teleop_channel=teleop_channel,
            preview_fn=_encode_preview_jpegs,
            report_features_fn=_report,
            extract_fn=_extract_joint_state,
            profiler=node_profiler,
        )
    finally:
        _LOG.info(
            "Control loop exiting for session %s; client.close() will "
            "flush recorder queue and trigger server-side upload.",
            session_id,
        )


def _extract_joint_state(obs: dict, action_keys: list) -> np.ndarray:
    """Pull joint positions out of a lerobot observation in action order.

    For follower robots the action features (e.g. ``shoulder_pan.pos``)
    match the observation joint keys exactly. Missing keys are filled
    with zero.
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
                # Capability-adaptive encoder (ADR 0023, SDK ADR 0019):
                # nvjpeg on CUDA boxes, else turbojpeg, else cv2 (releases
                # the GIL), else PIL. Same encoder as the inference uplink,
                # so recorded and served frames match byte-for-byte behavior.
                data = _encode_jpeg(arr)
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
                "encoded (jpeg backend=%s) — episodes will record observations "
                "only and the video will be blank. Install PyTurboJPEG, "
                "opencv-python, or Pillow on the node.",
                cam_arrays, _jpeg_backend_name(),
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


# Preview tee tunables: 320px longest side at q70 is ~8-15 KB per frame —
# visually identical on the headset's 0.8 m quad, and 3-5× cheaper on the
# uplink than the 640×480 q85 recording frames. Both are env-overridable
# because the preview byte-budget is the single lever for coexistence on a
# thin uplink: on a bufferbloated WiFi link the preview stream is what
# jitters control-datagram delivery (targets queue behind video bytes,
# spiking applied frame_age) and what the congestion backoff sheds, so
# dialing bytes/frame down attacks the fps shed AND the latency jitter at
# once. Lower resolution or quality on a struggling Jetson; the defaults
# are the visually-lossless ceiling, not a floor.
_PREVIEW_MAX_DIM_DEFAULT = 320
_PREVIEW_JPEG_QUALITY_DEFAULT = 70


def _preview_max_dim() -> int:
    """Preview longest-side cap from INTERLATENT_PREVIEW_MAX_DIM (px)."""
    return _env.env_int(
        "INTERLATENT_PREVIEW_MAX_DIM", _PREVIEW_MAX_DIM_DEFAULT, 64, 1280
    )


def _preview_jpeg_quality() -> int:
    """Preview JPEG quality from INTERLATENT_PREVIEW_JPEG_QUALITY (1-95)."""
    return _env.env_int(
        "INTERLATENT_PREVIEW_JPEG_QUALITY", _PREVIEW_JPEG_QUALITY_DEFAULT, 10, 95
    )


def _encode_preview_jpegs(obs: dict) -> dict[str, bytes]:
    """Downscale + JPEG-encode the observation's camera frames for the
    live headset preview.

    Camera detection matches ``_capture_tick`` exactly (uint8, ndim>=2)
    and keys match RecordTick's raw camera names, so the pod merges both
    feeds per camera. Encoding goes through the capability-adaptive
    encoder (node/jpeg.py); a frame that fails to encode is simply
    skipped. ~1-2 ms per camera at 10 Hz — negligible against a 33 ms
    tick budget. Resolution + quality are read per-set so an operator can
    dial the uplink budget live (INTERLATENT_PREVIEW_MAX_DIM /
    INTERLATENT_PREVIEW_JPEG_QUALITY) without restarting the node.
    """
    max_dim = _preview_max_dim()
    quality = _preview_jpeg_quality()
    out: dict[str, bytes] = {}
    for k, v in obs.items():
        try:
            arr = v if isinstance(v, np.ndarray) else np.asarray(v)
        except Exception:
            continue
        if arr is None or arr.dtype != np.uint8 or arr.ndim < 2:
            continue
        data = _encode_jpeg(arr, quality=quality, max_dim=max_dim)
        if data:
            out[k] = data
    return out


def _report_robot_features(
    api_base: Optional[str],
    node_id: Optional[str],
    token: Optional[str],
    state_names: Optional[list],
    action_names: Optional[list],
    teleop_profile: Optional[dict] = None,
    bypass_key: Optional[str] = None,
) -> bool:
    """POST the robot's per-element feature names + teleop profile to the node endpoint.

    Returns True when the report was sent (or there is nothing actionable to
    send / no config to send it with — so the caller stops retrying), and
    False only on a network error worth one more attempt. The backend stores
    both first-writer-wins, so re-reports are harmless.

    Names are reported verbatim from the lerobot robot — modern LeRobot's
    own flat-list convention (e.g. ``["shoulder_pan.pos", ...]``) — keyed by
    the canonical feature keys ``observation.state`` / ``action``.
    ``teleop_profile`` is the static teleop schema (``RobotProfile.to_schema_dict()``)
    reported so the hosted teleop producer can retarget against this robot's
    schema, or ``None`` for robots with no registered profile.
    """
    if not api_base or not node_id or not token:
        return True  # unconfigured — nothing to do, don't spin
    names: dict[str, list] = {}
    if state_names:
        names["observation.state"] = list(state_names)
    if action_names:
        names["action"] = list(action_names)
    if not names and not teleop_profile:
        return True  # nothing labelable to report
    try:
        import httpx

        url = f"{api_base.rstrip('/')}/api/v1/nodes/{node_id}/robot-features"
        payload: dict = {"feature_element_names": names}
        if teleop_profile:
            payload["teleop_profile"] = teleop_profile
        headers = {"x-api-key": token}
        if (bypass_key or "").strip():
            # Protected test domains (Vercel preview deployments) challenge
            # un-bypassed requests; carry the automation bypass secret same
            # as the daemon's shared heartbeat/poll client.
            headers["x-vercel-protection-bypass"] = bypass_key.strip()
        resp = httpx.post(
            url,
            headers=headers,
            json=payload,
            timeout=10.0,
        )
        _LOG.info(
            "Reported robot features (names=%s, teleop_profile=%s) -> %s",
            {k: len(v) for k, v in names.items()},
            bool(teleop_profile), resp.status_code,
        )
        return True
    except Exception:
        _LOG.warning("Failed to report robot features (will retry)", exc_info=True)
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
    no codec negotiation needed. ``target_size`` is a square resize: we
    don't preserve aspect ratio because the downstream processor doesn't
    either — its own resize is also square. Matching that here keeps the
    wire bytes minimal.
    """
    data = _encode_jpeg(arr, quality=quality, target_size=target_size)
    if data is None:
        raise RuntimeError(
            "JPEG encode failed (no encoder backend available — install "
            "PyTurboJPEG, opencv-python, or Pillow)"
        )
    return np.frombuffer(data, dtype=np.uint8)


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


def _parse_max_step(extra: dict) -> Optional[float]:
    """Read the per-tick delta-clamp limit from the ``--robot.*`` extras.

    Configured as part of the adapter (the ``--robot.max_step=<v>`` extra).
    Units match the motor-norm mode the loop logs at connect (e.g. degrees for
    MolmoAct2). Returns ``None`` (clamp disabled) when unset, non-numeric, or
    non-positive.
    """
    raw = extra.get("max_step")
    if raw is None:
        return None
    try:
        v = float(raw)
    except (TypeError, ValueError):
        _LOG.warning("Ignoring non-numeric --robot.max_step=%r", raw)
        return None
    return v if v > 0 else None


# Default low-pass cutoff for the policy action stream. 3 Hz at the typical 30 Hz
# control rate: deliberate arm motion is well below this, per-tick jitter sits above.
_DEFAULT_ACTION_FILTER_HZ = 3.0


def _parse_action_filter_hz(extra: dict) -> Optional[float]:
    """Read the policy action-smoothing cutoff from the ``--robot.*`` extras.

    Configured as part of the adapter (``--robot.action_filter_hz=<hz>``). Returns
    the Butterworth low-pass cutoff in Hz, defaulting to
    :data:`_DEFAULT_ACTION_FILTER_HZ` when unset. ``0`` / ``none`` / ``off`` (or a
    non-positive / non-numeric value) disables smoothing and returns ``None``.
    """
    raw = extra.get("action_filter_hz")
    if raw is None:
        return _DEFAULT_ACTION_FILTER_HZ
    if isinstance(raw, str) and raw.strip().lower() in ("none", "off", "disabled", ""):
        return None
    try:
        v = float(raw)
    except (TypeError, ValueError):
        _LOG.warning(
            "Ignoring non-numeric --robot.action_filter_hz=%r; using default %.1f Hz",
            raw, _DEFAULT_ACTION_FILTER_HZ,
        )
        return _DEFAULT_ACTION_FILTER_HZ
    return v if v > 0 else None


def _clamp_action_delta(
    action: "np.ndarray",
    actual: "np.ndarray",
    max_step: Optional[float],
    action_keys: list,
    step: int,
    *,
    source: str,
) -> "np.ndarray":
    """Execution-safety guard: cap the per-tick joint jump to ``max_step``.

    Applied to every action about to execute, regardless of source (policy or
    teleop), as the last line of defense against a single-tick slam. Each joint
    is limited so ``|cmd_i - actual_i| <= max_step`` (a *clamp*: the arm advances
    toward the target by at most ``max_step`` and never stalls). Also emits the
    sampled DRTC-DEBUG glass-box line. ``max_step is None`` ⇒ log only, no clamp.
    """
    arr = np.asarray(action, dtype=np.float32).reshape(-1)
    act = np.asarray(actual, dtype=np.float32).reshape(-1)
    n = min(len(action_keys), len(arr), len(act))
    if n == 0:
        return arr
    cmd = arr[:n]
    a = act[:n]
    delta = cmd - a
    # Sampled glass-box: first few ticks of a session + periodically after.
    if step < 10 or step % 100 == 0:
        try:
            pairs = ", ".join(
                "%s: %.2f->%.2f (Δ%+.2f)" % (action_keys[i], a[i], cmd[i], delta[i])
                for i in range(n)
            )
            _LOG.info(
                "DRTC-DEBUG joints #%d (%s) | max|Δ|=%.2f | %s",
                step, source, float(np.abs(delta).max()), pairs,
            )
        except Exception:
            _LOG.warning("DRTC-DEBUG joint dump failed", exc_info=True)
    if max_step is None:
        return arr
    exceeded = np.abs(delta) > max_step
    if np.any(exceeded):
        out = arr.copy()
        out[:n] = a + np.clip(delta, -max_step, max_step)
        j = int(np.argmax(np.abs(delta)))
        warn_clamp(
            f"control:{source}",
            "Delta clamp (%s) #%d: %d joint(s) exceeded max_step=%.3f; worst %s "
            "Δ=%.3f capped — sent clamped command.",
            source, step, int(exceeded.sum()), max_step, action_keys[j], float(delta[j]),
        )
        return out
    return arr


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
