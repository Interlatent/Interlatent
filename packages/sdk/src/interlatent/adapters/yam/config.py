"""Build the YAM adapter config from the node's passthrough dicts.

The node hands robot construction a flat ``extra`` dict (``--robot-arg key=value``)
and a ``cameras`` dict (``--camera name=device``). YAM runs the I2RT arms over CAN
via the ``i2rt`` driver directly (one bus per follower), and captures RGB from
RealSense / ZED or generic UVC/V4L2 webcams (see :mod:`.cameras`).

Nothing here imports ``i2rt`` or a camera SDK — those are resolved lazily inside the
adapter — so importing this module never requires the ``[yam]`` extra.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .cameras import CameraSpec, parse_camera_device

_logger = logging.getLogger(__name__)

# Raiden's persistent CAN interface names (udev-assigned). One bus per follower.
_DEFAULT_LEFT_CHANNEL = "can_follower_l"
_DEFAULT_RIGHT_CHANNEL = "can_follower_r"

_VALID_ARMS = ("both", "left", "right")
_VALID_GRIPPER_MODES = ("continuous", "bangbang")


@dataclass
class YAMAdapterConfig:
    """Everything the native YAM inference loop needs to build/drive the robot."""

    arms: str = "both"  # "both" | "left" | "right"
    left_channel: str = _DEFAULT_LEFT_CHANNEL
    right_channel: str = _DEFAULT_RIGHT_CHANNEL
    cameras: dict[str, CameraSpec] = field(default_factory=dict)
    # Execution-safety per-step clamp on arm joints, radians (see send_action).
    max_step_rad: float = 0.05
    # Move to the rest pose on connect(). Convenient, but moves hardware the instant
    # you connect — set false to skip.
    auto_home: bool = True
    # Adapter gripper post-step (mirrors axol): "continuous" passes the value through,
    # "bangbang" snaps to open/closed at the threshold.
    gripper_mode: str = "continuous"
    gripper_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.arms not in _VALID_ARMS:
            raise ValueError(f"arms must be one of {_VALID_ARMS}, got {self.arms!r}")
        if self.gripper_mode not in _VALID_GRIPPER_MODES:
            raise ValueError(
                f"gripper_mode must be one of {_VALID_GRIPPER_MODES}, got "
                f"{self.gripper_mode!r}"
            )

    @property
    def active_sides(self) -> tuple[str, ...]:
        """Active arm side labels in canonical left-then-right order."""
        if self.arms == "left":
            return ("left",)
        if self.arms == "right":
            return ("right",)
        return ("left", "right")


def _as_bool(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def build_adapter_config(
    extra: dict[str, str] | None, cameras: dict[str, str] | None
) -> YAMAdapterConfig:
    """Build a :class:`YAMAdapterConfig` from ``--robot-arg`` / ``--camera``.

    Recognized ``--robot-arg`` keys: ``arms`` (``both|left|right``),
    ``left_channel`` / ``right_channel``, ``max_step_rad``, ``auto_home``,
    ``gripper_mode`` / ``gripper_threshold``. Unrecognized keys warn + are ignored.

    ``--camera <name>=<device>`` declares an RGB camera: ``realsense[:serial]``,
    ``zed[:serial]``, or a generic UVC/V4L2 webcam given by ``/dev/video*`` path,
    bare index, or ``uvc:<path-or-index>``. Names must match the policy's training
    camera keys. Cameras are optional (a manual ``interlatent-act`` joint move needs
    none).
    """
    extra = dict(extra or {})
    cameras = dict(cameras or {})

    arms = extra.pop("arms", "both").strip().lower()
    left_channel = str(extra.pop("left_channel", _DEFAULT_LEFT_CHANNEL))
    right_channel = str(extra.pop("right_channel", _DEFAULT_RIGHT_CHANNEL))
    max_step_rad = float(extra.pop("max_step_rad", "0.05"))
    auto_home = _as_bool(extra.pop("auto_home", "true"))
    gripper_mode = extra.pop("gripper_mode", "continuous").strip().lower()
    gripper_threshold = float(extra.pop("gripper_threshold", "0.5"))

    cam_specs: dict[str, CameraSpec] = {
        name: parse_camera_device(name, device) for name, device in cameras.items()
    }

    if extra:
        _logger.warning(
            "Ignoring unrecognized --robot-arg key(s) for YAM: %s",
            ", ".join(sorted(extra)),
        )

    return YAMAdapterConfig(
        arms=arms,
        left_channel=left_channel,
        right_channel=right_channel,
        cameras=cam_specs,
        max_step_rad=max_step_rad,
        auto_home=auto_home,
        gripper_mode=gripper_mode,
        gripper_threshold=gripper_threshold,
    )


__all__ = ["YAMAdapterConfig", "build_adapter_config"]
