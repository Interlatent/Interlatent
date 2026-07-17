"""Minimal RGB camera capture for the YAM adapter.

Thin wrappers over Intel RealSense (``pyrealsense2``), Stereolabs ZED (``pyzed``),
and any generic UVC/V4L2 webcam (OpenCV, e.g. a Logitech on ``/dev/video2``) that
expose exactly what :meth:`YAMNativeRobot.get_observation` needs: ``connect()`` /
``read() -> uint8 HxWx3 RGB`` / ``disconnect()``. **RGB only** — the learned-depth
backends (FFS / tri-stereo) stay in raiden; the action interface consumes bare
``uint8`` RGB frames keyed by camera name.

Every backend SDK is imported lazily (inside methods) so importing this module — and
therefore ``interlatent.adapters.yam`` — never requires the ``[yam]`` extra.

A camera is declared on the CLI as ``--camera <name>=<device>``; ``<name>`` must match
the policy's training camera keys. ``<device>`` is either a vendor camera
``<type>:<serial>`` (``--camera wrist=realsense:1234`` / ``--camera overhead=zed:5678``)
or a generic webcam given by V4L2 path or index (``--camera front=/dev/video2`` /
``--camera front=2``, optionally prefixed ``uvc:``).
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

_logger = logging.getLogger(__name__)

_REALSENSE = "realsense"
_ZED = "zed"
_UVC = "uvc"
_KINDS = (_REALSENSE, _ZED, _UVC)
# Accepted aliases for the generic OpenCV/V4L2 backend.
_UVC_ALIASES = (_UVC, "opencv", "v4l2", "webcam")


@dataclass(frozen=True)
class CameraSpec:
    """A declared camera: backend kind + device id + capture settings.

    ``device`` is the vendor serial for ``realsense``/``zed`` (empty = first
    available), or the V4L2 path/index (e.g. ``/dev/video2`` or ``2``) for ``uvc``.
    """

    name: str
    kind: str  # "realsense" | "zed" | "uvc"
    device: str  # vendor serial (realsense/zed) or V4L2 path/index (uvc)
    width: int = 640
    height: int = 480
    fps: int = 30


def parse_camera_device(name: str, device: str) -> CameraSpec:
    """Parse a ``--camera name=<device>`` string into a CameraSpec.

    Accepts three forms:

    - ``realsense[:<serial>]`` / ``zed[:<serial>]`` — vendor camera by serial
      (serial optional; empty → first available device of that kind).
    - ``uvc:<path-or-index>`` (aliases ``opencv``/``v4l2``/``webcam``) — generic
      UVC/V4L2 webcam through OpenCV.
    - a bare V4L2 path or index (``/dev/video2`` or ``2``) — shorthand for ``uvc``.
    """
    raw = str(device).strip()
    kind_part, sep, rest = raw.partition(":")
    kind = kind_part.strip().lower()

    if kind in (_REALSENSE, _ZED):
        return CameraSpec(name=name, kind=kind, device=rest.strip())
    if kind in _UVC_ALIASES:
        dev = rest.strip()
        if not dev:
            raise ValueError(
                f"--camera {name}={device!r}: a UVC camera needs a device "
                f"(e.g. {name}=uvc:/dev/video2 or {name}=uvc:2)"
            )
        return CameraSpec(name=name, kind=_UVC, device=dev)
    # No recognized prefix: a /dev path or a bare index is a generic webcam.
    if not sep and (raw.startswith("/dev/") or raw.isdigit()):
        return CameraSpec(name=name, kind=_UVC, device=raw)
    raise ValueError(
        f"--camera {name}={device!r}: camera must be a vendor type "
        f"({_REALSENSE}/{_ZED}, e.g. {name}=realsense:1234 or {name}=zed:5678) "
        f"or a UVC/V4L2 webcam (e.g. {name}=/dev/video2 or {name}=uvc:2)"
    )


class Camera(Protocol):
    """Duck type for a YAM RGB camera."""

    def connect(self) -> None: ...

    def read(self) -> np.ndarray: ...

    def disconnect(self) -> None: ...


def build_camera(spec: CameraSpec) -> "Camera":
    """Construct the camera for a spec (no hardware touched yet).

    Every backend is wrapped in :class:`ThreadedCamera`: a per-camera reader
    thread drains the device continuously and ``read()`` is a non-blocking
    latest-frame snapshot. A synchronous ``read()`` on the control thread
    blocks up to a full frame period per camera (the driver hands out frames
    at its own cadence) — three sequential 30 fps cameras cost ~30-45 ms per
    tick, which alone drops a 30 Hz control loop to ~21 Hz.
    """
    inner: Camera
    if spec.kind == _REALSENSE:
        inner = RealSenseCamera(spec)
    elif spec.kind == _ZED:
        inner = ZedCamera(spec)
    elif spec.kind == _UVC:
        inner = UVCCamera(spec)
    else:
        raise ValueError(f"unknown camera kind {spec.kind!r}")  # pragma: no cover
    return ThreadedCamera(inner)


class ThreadedCamera:
    """Non-blocking latest-frame view of any :class:`Camera`.

    ``connect()`` starts a daemon reader thread that calls the inner camera's
    blocking ``read()`` in a loop and keeps only the newest frame. The heavy
    work (driver wait + BGR→RGB conversion) happens in C code that releases
    the GIL, so the readers run genuinely in parallel with the control loop.
    ``read()`` returns the latest frame immediately — it only blocks while
    waiting for the *first* frame after connect.

    Failure semantics match the synchronous backends: if the reader thread
    dies (device read raised) or the device stops producing frames
    (``stale_after_s`` with no new frame), ``read()`` raises instead of
    silently repeating the last image into the recording.

    The inner camera's ``read()`` must return an *owned* array (no view into
    driver memory that the next grab recycles) — all three backends here do.
    """

    # Overridable per-instance (tests patch these; RealSense's own
    # wait_for_frames timeout is 5 s, so first-frame matches it).
    first_frame_timeout_s = 5.0
    stale_after_s = 1.0

    def __init__(self, inner: "Camera") -> None:
        self.inner = inner
        self.spec = getattr(inner, "spec", None)
        self._name = getattr(self.spec, "name", "?")
        self._cond = threading.Condition()
        self._frame: np.ndarray | None = None
        self._frame_at = 0.0
        self._error: Exception | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def connect(self) -> None:
        self.inner.connect()
        self._stop.clear()
        self._frame = None
        self._error = None
        self._thread = threading.Thread(
            target=self._pump, name=f"cam-{self._name}", daemon=True
        )
        self._thread.start()

    def _pump(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self.inner.read()
            except Exception as exc:  # noqa: BLE001
                if self._stop.is_set():
                    return  # disconnect() racing a blocked read — not a fault
                with self._cond:
                    self._error = exc
                    self._cond.notify_all()
                return
            with self._cond:
                self._frame = frame
                self._frame_at = time.monotonic()
                self._cond.notify_all()

    def read(self) -> np.ndarray:
        with self._cond:
            if self._frame is None and self._error is None:
                self._cond.wait_for(
                    lambda: self._frame is not None or self._error is not None,
                    timeout=self.first_frame_timeout_s,
                )
            if self._error is not None:
                raise RuntimeError(
                    f"camera {self._name}: reader thread failed: {self._error}"
                ) from self._error
            frame = self._frame
            frame_at = self._frame_at
        if frame is None:
            raise RuntimeError(
                f"camera {self._name}: no frame within "
                f"{self.first_frame_timeout_s:.1f}s of connect()"
            )
        age = time.monotonic() - frame_at
        if age > self.stale_after_s:
            raise RuntimeError(
                f"camera {self._name}: newest frame is {age:.1f}s old "
                f"(device stalled?)"
            )
        return frame

    def disconnect(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            # A healthy reader unblocks within one frame period; RealSense's
            # wait_for_frames can hold up to its own 5 s timeout.
            thread.join(timeout=6.0)
            if thread.is_alive():
                _logger.warning(
                    "camera %s reader thread did not exit; releasing the "
                    "device anyway", self._name,
                )
            self._thread = None
        self.inner.disconnect()


class RealSenseCamera:
    """Intel RealSense color stream → ``uint8 HxWx3`` RGB (``pyrealsense2``)."""

    def __init__(self, spec: CameraSpec) -> None:
        self.spec = spec
        self._pipeline: Any | None = None

    def connect(self) -> None:
        import pyrealsense2 as rs  # lazy: only needed with the [yam] extra

        config = rs.config()
        if self.spec.device:
            config.enable_device(self.spec.device)
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
            self.spec.name, self.spec.device or "<first>", self.spec.width,
            self.spec.height, self.spec.fps,
        )

    def read(self) -> np.ndarray:
        assert self._pipeline is not None, "RealSenseCamera.read before connect()"
        frames = self._pipeline.wait_for_frames()
        color = frames.get_color_frame()
        # Copy: get_data() views SDK pool memory that the next grab recycles;
        # frames outlive this call now that a reader thread hands them out.
        return np.asanyarray(color.get_data()).copy()

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
        if self.spec.device:
            init.set_from_serial_number(int(self.spec.device))
        init.camera_fps = self.spec.fps
        zed = sl.Camera()
        status = zed.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(
                f"ZED {self.spec.name} (serial={self.spec.device or '<first>'}) "
                f"open failed: {status}"
            )
        self._zed = zed
        self._runtime = sl.RuntimeParameters()
        self._mat = sl.Mat()
        self._view_left = sl.VIEW.LEFT
        self._success = sl.ERROR_CODE.SUCCESS
        _logger.info(
            "ZED %s connected (serial=%s)", self.spec.name, self.spec.device or "<first>"
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


class UVCCamera:
    """Generic UVC/V4L2 webcam (e.g. Logitech) → ``uint8 HxWx3`` RGB via OpenCV.

    ``spec.device`` is a V4L2 path (``/dev/video2``) or a numeric index (``2``).
    OpenCV captures BGR; we reorder to RGB to match the other backends.
    """

    def __init__(self, spec: CameraSpec) -> None:
        self.spec = spec
        self._cap: Any | None = None

    def connect(self) -> None:
        import cv2  # lazy: only needed when a UVC camera is declared

        dev = self.spec.device
        target: Any = int(dev) if dev.isdigit() else dev
        cap = cv2.VideoCapture(target)
        # V4L2 keeps a driver-side queue of captured frames (OpenCV default
        # 4). A control loop that consumes slower than the camera produces
        # (e.g. 27 Hz loop vs 30 fps camera) leaves that queue permanently
        # full, so every read() returns the OLDEST buffered frame — a
        # standing ~130 ms of staleness added BEFORE the capture timestamp,
        # invisible to every downstream latency metric. One buffer = read()
        # always dequeues the freshest frame. Best-effort: backends that
        # don't support the property ignore the set.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.spec.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.spec.height)
        cap.set(cv2.CAP_PROP_FPS, self.spec.fps)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(
                f"UVC camera {self.spec.name} (device={dev}) failed to open — check "
                f"the path/index (e.g. `v4l2-ctl --list-devices`) and permissions."
            )
        self._cap = cap
        _logger.info(
            "UVC %s connected (device=%s, requested %dx%d@%d)",
            self.spec.name, dev, self.spec.width, self.spec.height, self.spec.fps,
        )

    def read(self) -> np.ndarray:
        assert self._cap is not None, "UVCCamera.read before connect()"
        import cv2  # cached module lookup after connect()'s import

        ok, bgr = self._cap.read()
        if not ok or bgr is None:
            raise RuntimeError(f"UVC {self.spec.name} (device={self.spec.device}) read failed")
        # cvtColor over a numpy fancy-index flip: releases the GIL and hands
        # back an owned contiguous array (no view into the capture buffer).
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def disconnect(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:  # noqa: BLE001
                _logger.warning("UVC %s release failed", self.spec.name, exc_info=True)
            self._cap = None


__all__ = [
    "CameraSpec", "Camera", "RealSenseCamera", "ZedCamera", "UVCCamera",
    "ThreadedCamera", "build_camera", "parse_camera_device",
]
