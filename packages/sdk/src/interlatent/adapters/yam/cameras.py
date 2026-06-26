"""Minimal RGB camera capture for the YAM adapter.

Thin wrappers over Intel RealSense (``pyrealsense2``) and Stereolabs ZED
(``pyzed``) that expose exactly what :meth:`YAMNativeRobot.get_observation` needs:
``connect()`` / ``read() -> uint8 HxWx3 RGB`` / ``disconnect()``. **RGB only** — the
learned-depth backends (FFS / tri-stereo) stay in raiden; the action interface
consumes bare ``uint8`` RGB frames keyed by camera name.

Both vendor SDKs are imported lazily (inside methods) so importing this module — and
therefore ``interlatent.adapters.yam`` — never requires the ``[yam]`` extra.

A camera is declared on the CLI as ``--camera <name>=<type>:<serial>`` (e.g.
``--camera wrist=realsense:1234`` / ``--camera overhead=zed:5678``); ``<name>`` must
match the policy's training camera keys.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

_logger = logging.getLogger(__name__)

_REALSENSE = "realsense"
_ZED = "zed"
_KINDS = (_REALSENSE, _ZED)


@dataclass(frozen=True)
class CameraSpec:
    """A declared camera: backend kind + device serial + capture settings."""

    name: str
    kind: str  # "realsense" | "zed"
    serial: str  # vendor serial number (string; empty = first available)
    width: int = 640
    height: int = 480
    fps: int = 30


def parse_camera_device(name: str, device: str) -> CameraSpec:
    """Parse a ``--camera name=<type>:<serial>`` device string into a CameraSpec.

    ``<type>`` must be ``realsense`` or ``zed``. ``<serial>`` is optional (empty →
    first available device of that kind).
    """
    raw = str(device).strip()
    if ":" in raw:
        kind, _, serial = raw.partition(":")
    else:
        kind, serial = raw, ""
    kind = kind.strip().lower()
    if kind not in _KINDS:
        raise ValueError(
            f"--camera {name}={device!r}: camera type must be one of "
            f"{', '.join(_KINDS)} (e.g. {name}=realsense:1234 or {name}=zed:5678)"
        )
    return CameraSpec(name=name, kind=kind, serial=serial.strip())


class Camera(Protocol):
    """Duck type for a YAM RGB camera."""

    def connect(self) -> None: ...

    def read(self) -> np.ndarray: ...

    def disconnect(self) -> None: ...


def build_camera(spec: CameraSpec) -> "Camera":
    """Construct the concrete camera for a spec (no hardware touched yet)."""
    if spec.kind == _REALSENSE:
        return RealSenseCamera(spec)
    if spec.kind == _ZED:
        return ZedCamera(spec)
    raise ValueError(f"unknown camera kind {spec.kind!r}")  # pragma: no cover


class RealSenseCamera:
    """Intel RealSense color stream → ``uint8 HxWx3`` RGB (``pyrealsense2``)."""

    def __init__(self, spec: CameraSpec) -> None:
        self.spec = spec
        self._pipeline: Any | None = None

    def connect(self) -> None:
        import pyrealsense2 as rs  # lazy: only needed with the [yam] extra

        config = rs.config()
        if self.spec.serial:
            config.enable_device(self.spec.serial)
        # rgb8 hands back channel-ordered RGB directly — no BGR->RGB conversion.
        config.enable_stream(
            rs.stream.color,
            self.spec.width,
            self.spec.height,
            rs.format.rgb8,
            self.spec.fps,
        )
        pipeline = rs.pipeline()
        pipeline.start(config)
        self._pipeline = pipeline
        _logger.info(
            "RealSense %s connected (serial=%s, %dx%d@%d)",
            self.spec.name, self.spec.serial or "<first>", self.spec.width,
            self.spec.height, self.spec.fps,
        )

    def read(self) -> np.ndarray:
        assert self._pipeline is not None, "RealSenseCamera.read before connect()"
        frames = self._pipeline.wait_for_frames()
        color = frames.get_color_frame()
        return np.asanyarray(color.get_data())

    def disconnect(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:  # noqa: BLE001
                _logger.warning("RealSense %s stop failed", self.spec.name, exc_info=True)
            self._pipeline = None


class ZedCamera:
    """Stereolabs ZED left view → ``uint8 HxWx3`` RGB (``pyzed``).

    The ZED SDK retrieves frames as BGRA; we drop alpha and reorder to RGB with a
    numpy gather (no OpenCV dependency).
    """

    def __init__(self, spec: CameraSpec) -> None:
        self.spec = spec
        self._zed: Any | None = None
        self._runtime: Any | None = None
        self._mat: Any | None = None

    def connect(self) -> None:
        import pyzed.sl as sl  # lazy: ZED SDK is host-installed, not on PyPI

        init = sl.InitParameters()
        if self.spec.serial:
            init.set_from_serial_number(int(self.spec.serial))
        init.camera_fps = self.spec.fps
        zed = sl.Camera()
        status = zed.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(
                f"ZED {self.spec.name} (serial={self.spec.serial or '<first>'}) "
                f"open failed: {status}"
            )
        self._zed = zed
        self._runtime = sl.RuntimeParameters()
        self._mat = sl.Mat()
        self._view_left = sl.VIEW.LEFT
        self._success = sl.ERROR_CODE.SUCCESS
        _logger.info(
            "ZED %s connected (serial=%s)", self.spec.name, self.spec.serial or "<first>"
        )

    def read(self) -> np.ndarray:
        assert self._zed is not None, "ZedCamera.read before connect()"
        if self._zed.grab(self._runtime) != self._success:
            raise RuntimeError(f"ZED {self.spec.name} grab failed")
        self._zed.retrieve_image(self._mat, self._view_left)
        bgra = self._mat.get_data()  # HxWx4, BGRA
        rgb = bgra[:, :, [2, 1, 0]]  # drop alpha, BGR->RGB
        return np.ascontiguousarray(rgb, dtype=np.uint8)

    def disconnect(self) -> None:
        if self._zed is not None:
            try:
                self._zed.close()
            except Exception:  # noqa: BLE001
                _logger.warning("ZED %s close failed", self.spec.name, exc_info=True)
            self._zed = None


__all__ = ["CameraSpec", "Camera", "RealSenseCamera", "ZedCamera", "build_camera", "parse_camera_device"]
