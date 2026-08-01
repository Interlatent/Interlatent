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
``--camera front=2``, optionally prefixed ``uvc:``). Capture settings ride as
comma-separated ``key=val`` extras after the device
(``--camera front=/dev/video2,width=1280,height=720,fps=15,pixel_format=yuyv``);
accepted keys are ``width``/``height``/``fps`` (all backends) and ``pixel_format``
(UVC only: ``mjpg`` (default) / ``yuyv`` / ``default`` = driver's choice).
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

    ``pixel_format`` is UVC-only: the wire format requested from the camera.
    ``mjpg`` (default) keeps 3 cameras + CAN adapters inside a shared USB2
    domain's 480 Mbit/s isochronous budget (uncompressed 640x480@30 YUYV
    reserves ~147 Mbit/s PER camera; xHCI then refuses to enumerate the last
    device). The cost is a per-frame JPEG decode inside OpenCV — a CPU-tight
    rig on an uncongested USB3 bus can set ``yuyv`` or ``default`` (driver's
    choice) per camera.
    """

    name: str
    kind: str  # "realsense" | "zed" | "uvc"
    device: str  # vendor serial (realsense/zed) or V4L2 path/index (uvc)
    width: int = 640
    height: int = 480
    fps: int = 30
    pixel_format: str = "mjpg"  # "mjpg" | "yuyv" | "default" (uvc only)


_PIXEL_FORMATS = ("mjpg", "yuyv", "default")
_SPEC_EXTRA_KEYS = ("width", "height", "fps", "pixel_format")


def _parse_spec_extras(name: str, device: str, parts: list[str]) -> dict:
    """Parse trailing ``key=val`` device extras into CameraSpec kwargs."""
    extras: dict[str, Any] = {}
    for part in parts:
        key, sep, val = part.partition("=")
        key, val = key.strip().lower(), val.strip()
        if not sep or key not in _SPEC_EXTRA_KEYS:
            raise ValueError(
                f"--camera {name}={device!r}: unknown camera option {part!r} "
                f"(accepted: {', '.join(k + '=' for k in _SPEC_EXTRA_KEYS)})"
            )
        if key == "pixel_format":
            val = val.lower()
            if val not in _PIXEL_FORMATS:
                raise ValueError(
                    f"--camera {name}={device!r}: pixel_format must be one of "
                    f"{'/'.join(_PIXEL_FORMATS)}, got {val!r}"
                )
            extras[key] = val
        else:
            try:
                extras[key] = int(val)
            except ValueError:
                raise ValueError(
                    f"--camera {name}={device!r}: {key} must be an integer, "
                    f"got {val!r}"
                ) from None
    return extras


def parse_camera_device(name: str, device: str) -> CameraSpec:
    """Parse a ``--camera name=<device>`` string into a CameraSpec.

    The device accepts three forms, each optionally followed by
    comma-separated ``key=val`` capture extras (``width``/``height``/``fps``/
    ``pixel_format``, e.g. ``/dev/video2,width=1280,pixel_format=yuyv``):

    - ``realsense[:<serial>]`` / ``zed[:<serial>]`` — vendor camera by serial
      (serial optional; empty → first available device of that kind).
    - ``uvc:<path-or-index>`` (aliases ``opencv``/``v4l2``/``webcam``) — generic
      UVC/V4L2 webcam through OpenCV.
    - a bare V4L2 path or index (``/dev/video2`` or ``2``) — shorthand for ``uvc``.
    """
    raw = str(device).strip()
    dev_token, *extra_parts = [p.strip() for p in raw.split(",")]
    extras = _parse_spec_extras(name, device, extra_parts)
    kind_part, sep, rest = dev_token.partition(":")
    kind = kind_part.strip().lower()

    if kind in (_REALSENSE, _ZED):
        return CameraSpec(name=name, kind=kind, device=rest.strip(), **extras)
    if kind in _UVC_ALIASES:
        dev = rest.strip()
        if not dev:
            raise ValueError(
                f"--camera {name}={device!r}: a UVC camera needs a device "
                f"(e.g. {name}=uvc:/dev/video2 or {name}=uvc:2)"
            )
        return CameraSpec(name=name, kind=_UVC, device=dev, **extras)
    # No recognized prefix: a /dev path or a bare index is a generic webcam.
    if not sep and (dev_token.startswith("/dev/") or dev_token.isdigit()):
        return CameraSpec(name=name, kind=_UVC, device=dev_token, **extras)
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


def _fourcc_to_str(code: int) -> str:
    """Decode an OpenCV FOURCC int to its 4-char tag (``"?"`` when unset)."""
    if code <= 0:
        return "?"
    return "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))


class UVCCamera:
    """Generic UVC/V4L2 webcam (e.g. Logitech) → ``uint8 HxWx3`` RGB via OpenCV.

    ``spec.device`` is a V4L2 path (``/dev/video2``) or a numeric index (``2``).
    OpenCV captures BGR; we reorder to RGB to match the other backends. The
    wire format defaults to MJPG (see :class:`CameraSpec.pixel_format`);
    the negotiated format is read back and logged, and a refused request
    falls back to the driver default with a warning rather than failing.
    """

    def __init__(self, spec: CameraSpec) -> None:
        self.spec = spec
        self._cap: Any | None = None

    def connect(self) -> None:
        import cv2  # lazy: only needed when a UVC camera is declared

        dev = self.spec.device
        target: Any = int(dev) if dev.isdigit() else dev
        # Pin the V4L2 backend: GStreamer-built OpenCV silently ignores
        # CAP_PROP_FOURCC, which would make pixel_format a no-op with no
        # signal. With the backend pinned, an unsupported platform fails
        # the isOpened() check below instead.
        cap = cv2.VideoCapture(target, cv2.CAP_V4L2)
        # FOURCC must be set BEFORE frame size/fps: V4L2 renegotiates the
        # format on each property set, and a size chosen against the
        # driver-default format (usually uncompressed YUYV) can lock a
        # size list the requested format doesn't offer.
        fourcc_req: int | None = None
        if self.spec.pixel_format != "default":
            fourcc_req = cv2.VideoWriter_fourcc(*self.spec.pixel_format.upper())
            cap.set(cv2.CAP_PROP_FOURCC, fourcc_req)
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
        got_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        got = _fourcc_to_str(got_fourcc)
        if fourcc_req is not None and got_fourcc != fourcc_req:
            _logger.warning(
                "UVC %s: %s not accepted by the driver, running %s — an "
                "uncompressed format reserves ~147 Mbit/s of USB isochronous "
                "bandwidth at 640x480@30 (per camera)",
                self.spec.name, self.spec.pixel_format.upper(), got,
            )
        self._cap = cap
        _logger.info(
            "UVC %s connected (device=%s, requested %dx%d@%d, negotiated "
            "%s %dx%d@%g)",
            self.spec.name, dev, self.spec.width, self.spec.height,
            self.spec.fps, got,
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            float(cap.get(cv2.CAP_PROP_FPS)),
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
