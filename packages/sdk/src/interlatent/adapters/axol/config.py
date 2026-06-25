"""Build the native Axol adapter config from the node's passthrough dicts.

The node hands robot construction a flat ``extra`` dict (``--robot-arg
key=value``) and a ``cameras`` dict (``--camera name=device``). The Axol robot
runs **onboard the Jetson** and opens each GMSL-attached ZED **directly by
serial number** via the native ``almond_axol`` ZED camera, so we interpret each
camera "device" as a ZED **serial number** and build native
``almond_axol.lerobot.camera.ZedCameraConfig``s; the remaining knobs populate a
native ``almond_axol.robot.config.AxolConfig`` and the adapter's own settings.

``almond_axol`` (and, through its camera classes, ``lerobot``) is imported
lazily (inside functions) so importing this module never requires the
``[axol]`` extra.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

_logger = logging.getLogger(__name__)


@dataclass
class AxolAdapterConfig:
    """Everything the native Axol inference loop needs to build/drive the robot."""

    axol_config: Any  # almond_axol.robot.config.AxolConfig
    cameras: dict[str, Any]  # name -> almond_axol.lerobot.camera.ZedCameraConfig
    left_channel: str
    right_channel: str
    telemetry_hz: float = 120.0
    observe_torques: bool = False
    gripper_mode: str = "continuous"  # "continuous" | "bangbang"
    gripper_threshold: float = 0.5
    # Restart the Stereolabs ``zed_x_daemon`` before opening cameras so a GMSL
    # camera plugged in after boot is enumerable. Needs (passwordless) sudo on
    # the Jetson; set false on nodes where that isn't available.
    restart_zed_daemon: bool = True


def _as_bool(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _as_stiffness(v: str) -> float | list[float]:
    """Parse a stiffness arg: a scalar, or a comma-separated 7-vector (ARM_JOINTS order)."""
    parts = [p for p in str(v).split(",") if p.strip() != ""]
    vals = [float(p) for p in parts]
    if not vals:
        raise ValueError(f"empty stiffness value {v!r}")
    return vals[0] if len(vals) == 1 else vals


def _load_axol_config_file(path: str) -> Any:
    """Load a full native ``AxolConfig`` from a file via draccus (deep gains)."""
    from almond_axol.robot.config import AxolConfig

    try:
        import draccus
    except ImportError as e:  # pragma: no cover - draccus ships with almond-axol
        raise RuntimeError(
            "Loading --robot-arg config_path requires draccus (installed with almond-axol)."
        ) from e
    try:
        with open(path) as f:
            return draccus.load(AxolConfig, f)
    except Exception as e:
        raise RuntimeError(f"Failed to load Axol config from {path!r}: {e}") from e


def _camera_dims(extra: dict[str, str]) -> dict[str, int]:
    """Resolve optional ZED resolution/fps overrides into ZedCameraConfig kwargs.

    Recognizes ``resolution`` (one of the native ``ZED_RESOLUTION_DIMS`` names,
    e.g. ``SVGA`` / ``HD1080`` / ``HD1200``) and ``camera_fps``. Omitted knobs
    fall through to the native ZedCameraConfig defaults (SVGA 960x600 @ 60).
    """
    from almond_axol.lerobot.camera.configuration_zed import ZED_RESOLUTION_DIMS

    dims: dict[str, int] = {}
    resolution = extra.pop("resolution", None)
    if resolution is not None:
        try:
            width, height = ZED_RESOLUTION_DIMS[resolution]
        except KeyError:
            raise ValueError(
                f"--robot-arg resolution={resolution!r} is not a supported ZED "
                f"resolution ({', '.join(ZED_RESOLUTION_DIMS)})"
            )
        dims["width"], dims["height"] = int(width), int(height)
    if "camera_fps" in extra:
        dims["fps"] = int(extra.pop("camera_fps"))
    return dims


def build_adapter_config(
    extra: dict[str, str], cameras: dict[str, str] | None
) -> AxolAdapterConfig:
    """Build an :class:`AxolAdapterConfig` from ``--robot-arg`` / ``--camera``.

    Recognized ``--robot-arg`` keys: ``config_path`` (a full native AxolConfig
    for deep per-joint gains), ``left_stiffness`` / ``right_stiffness`` (**must
    match data-collection**), ``max_step_rad``, ``telemetry_hz``,
    ``observe_torques``, ``left_channel`` / ``right_channel``, ``stereo`` (open
    all cameras as stereo ZED X), ``resolution`` / ``camera_fps`` (ZED capture
    settings), ``restart_zed_daemon`` (default true), ``gripper_mode`` /
    ``gripper_threshold`` (adapter post-step). Unrecognized keys warn + ignore.

    ``--camera <name>=<serial>`` opens the GMSL-attached ZED with that serial
    number onboard the Jetson; **names must match the policy's training camera
    keys** (e.g. ``overhead`` / ``left_arm`` / ``right_arm``).
    """
    from dataclasses import replace

    from almond_axol.lerobot.camera import ZedCameraConfig
    from almond_axol.robot.config import AxolConfig
    from almond_axol.utils.shared import CAN_LEFT, CAN_RIGHT
    from lerobot.cameras.configs import ColorMode

    extra = dict(extra or {})
    cameras = dict(cameras or {})

    # Adapter post-step knobs.
    gripper_mode = extra.pop("gripper_mode", "continuous")
    gripper_threshold = float(extra.pop("gripper_threshold", "0.5"))
    stereo = _as_bool(extra.pop("stereo", "false"))
    restart_zed_daemon = _as_bool(extra.pop("restart_zed_daemon", "true"))

    # Native gains config: optional file base, then flat overlays.
    config_path = extra.pop("config_path", None)
    axol_cfg = _load_axol_config_file(config_path) if config_path else AxolConfig()
    overlay: dict[str, Any] = {}
    if "max_step_rad" in extra:
        overlay["max_step_rad"] = float(extra.pop("max_step_rad"))
    if "left_stiffness" in extra:
        overlay["left_stiffness"] = _as_stiffness(extra.pop("left_stiffness"))
    if "right_stiffness" in extra:
        overlay["right_stiffness"] = _as_stiffness(extra.pop("right_stiffness"))
    if overlay:
        axol_cfg = replace(axol_cfg, **overlay)

    # Onboard ZED cameras, opened by serial number.
    if not cameras:
        raise ValueError(
            "Axol requires at least one camera, e.g. --camera overhead=41234567 "
            "(the device value is the ZED serial number; names must match the "
            "policy's training camera keys)"
        )
    dims = _camera_dims(extra)
    cam_cfgs: dict[str, ZedCameraConfig] = {}
    for name, device in cameras.items():
        try:
            serial = int(str(device).strip())
        except (TypeError, ValueError):
            raise ValueError(
                f"--camera {name}={device!r}: Axol interprets the device as a "
                f"ZED serial number (integer), e.g. --camera {name}=41234567"
            )
        cam_cfgs[name] = ZedCameraConfig(
            serial=serial,
            color_mode=ColorMode.RGB,
            stereo=stereo,
            **dims,
        )

    telemetry_hz = float(extra.pop("telemetry_hz", "120.0"))
    observe_torques = _as_bool(extra.pop("observe_torques", "false"))
    left_channel = str(extra.pop("left_channel", CAN_LEFT))
    right_channel = str(extra.pop("right_channel", CAN_RIGHT))

    if extra:
        _logger.warning(
            "Ignoring unrecognized --robot-arg key(s) for Axol: %s",
            ", ".join(sorted(extra)),
        )

    return AxolAdapterConfig(
        axol_config=axol_cfg,
        cameras=cam_cfgs,
        left_channel=left_channel,
        right_channel=right_channel,
        telemetry_hz=telemetry_hz,
        observe_torques=observe_torques,
        gripper_mode=gripper_mode,
        gripper_threshold=gripper_threshold,
        restart_zed_daemon=restart_zed_daemon,
    )
