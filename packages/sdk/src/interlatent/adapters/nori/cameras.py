"""Nori camera capture: the daemon's companion ZeroMQ MJPEG channel.

The Nori Pi publishes each camera as raw JPEG frames on a per-camera ZeroMQ PUB
socket at ``cam_base_port + <descriptor index>`` (default base 5555), wire format
``b"<name> <capture_monotonic_seconds>\\n" + <JPEG bytes>`` — latest-wins via
CONFLATE. This channel is documented in Nori-Protocol/CLIENTS.md but explicitly
OUTSIDE the ``protocol_version`` guarantees (accepted caveat, see the plan/ADR).

Frames are decoded to ``uint8 HxWx3`` RGB — dtype/shape-identical to the YAM
camera backends — so the shared capture path (``_capture_tick``/``_encode_npz``)
and the teleop preview tee consume them unchanged. The `:7777` control socket
never carries images.

``zmq`` and ``cv2`` are imported lazily (inside methods) so importing this
module never requires the ``[nori]`` extra; JPEG decode falls back to PIL when
OpenCV is absent (same dual-decoder convention as ``node/control.py``).
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .config import NoriAdapterConfig

_logger = logging.getLogger(__name__)

# Socket receive timeout: one tick at the daemon's ~30fps camera cadence plus
# slack. read() returns the last frame on timeout rather than blocking the loop.
_RCVTIMEO_MS = 100


@dataclass(frozen=True)
class NoriCameraSpec:
    obs_key: str  # observation key the policy was trained with
    daemon_name: str  # camera name as listed in ack.descriptor.cameras
    index: int  # position in the descriptor list -> port offset
    host: str
    port: int


def resolve_camera_specs(
    cfg: NoriAdapterConfig, descriptor_cameras: list[str] | tuple[str, ...]
) -> list[NoriCameraSpec]:
    """Map ``--camera obs_key=<value>`` onto the daemon's camera publishers.

    ``<value>`` forms:

    - ``<daemon_name>`` — resolved against ``ack.descriptor.cameras``; index =
      position in that list. Fails fast if the daemon doesn't advertise it.
    - ``<daemon_name>:<index>`` or bare ``<index>`` — EXPLICIT port index
      (port = ``cam_base_port + index``), no descriptor needed. The escape
      hatch for daemon builds that send a descriptorless ack: check which
      ports are live with ``ss -tlnp | grep 555`` on the robot.

    With no mapping configured, every descriptor camera is subscribed under
    its native name (descriptorless ack => no cameras).
    """
    names = list(descriptor_cameras)
    mapping = dict(cfg.cameras) or {n: n for n in names}
    specs: list[NoriCameraSpec] = []
    for obs_key, value in mapping.items():
        value = str(value).strip()
        name_part, sep, idx_part = value.partition(":")
        if sep and idx_part.strip().isdigit():
            idx = int(idx_part)
            daemon_name = name_part.strip() or obs_key
        elif value.isdigit():
            idx = int(value)
            daemon_name = obs_key
        elif value in names:
            idx = names.index(value)
            daemon_name = value
        else:
            raise ValueError(
                f"--camera {obs_key}={value!r}: daemon advertises no such "
                f"camera (descriptor.cameras={names})"
                + (
                    " — this ack carried no camera descriptor; use the "
                    "explicit-index form obs_key=<name>:<index> (port = "
                    f"{cfg.cam_base_port}+index)"
                    if not names
                    else ""
                )
            )
        specs.append(
            NoriCameraSpec(
                obs_key=obs_key,
                daemon_name=daemon_name,
                index=idx,
                host=cfg.camera_host,
                port=cfg.cam_base_port + idx,
            )
        )
    return specs


def _decode_jpeg(data: bytes) -> np.ndarray:
    """JPEG bytes -> uint8 HxWx3 RGB. cv2 when available, PIL fallback —
    mirrors the dual-encoder convention in node/control.py."""
    try:
        import cv2

        bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("cv2.imdecode returned None")
        return np.ascontiguousarray(bgr[:, :, ::-1], dtype=np.uint8)  # BGR->RGB
    except ImportError:
        from PIL import Image

        img = Image.open(io.BytesIO(data)).convert("RGB")
        return np.asarray(img, dtype=np.uint8)


def split_camera_frame(payload: bytes) -> tuple[str, float, bytes]:
    """Split the wire format ``b"<name> <ts>\\n" + JPEG`` into its parts."""
    header, sep, jpeg = payload.partition(b"\n")
    if not sep:
        raise ValueError("malformed Nori camera frame: no header newline")
    name, _, ts_raw = header.decode("utf-8", "replace").partition(" ")
    try:
        ts = float(ts_raw)
    except ValueError:
        ts = 0.0
    return name, ts, jpeg


class NoriCamera:
    """One daemon camera as a YAM-shaped RGB camera (connect/read/disconnect).

    Latest-wins by construction: CONFLATE keeps only the newest frame in the
    socket, and ``read()`` keeps returning the last decoded frame when no new
    one arrived within the receive timeout (a stalled publisher shows up as a
    frozen image, never as a blocked control loop).
    """

    def __init__(self, spec: NoriCameraSpec, *, context_factory: Any = None) -> None:
        self.spec = spec
        self._context_factory = context_factory  # injectable for tests
        self._ctx: Any | None = None
        self._sock: Any | None = None
        self._last: Optional[np.ndarray] = None

    def connect(self) -> None:
        if self._context_factory is not None:
            self._ctx = self._context_factory()
        else:
            import zmq  # lazy: only needed with the [nori] extra

            self._ctx = zmq.Context.instance()
        import zmq  # types/constants; cheap once the extra is installed

        sock = self._ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.SUBSCRIBE, b"")
        sock.setsockopt(zmq.CONFLATE, 1)
        sock.setsockopt(zmq.RCVTIMEO, _RCVTIMEO_MS)
        sock.connect(f"tcp://{self.spec.host}:{self.spec.port}")
        self._sock = sock
        _logger.info(
            "Nori camera %s connected (daemon=%s tcp://%s:%d)",
            self.spec.obs_key, self.spec.daemon_name, self.spec.host, self.spec.port,
        )

    def read(self) -> np.ndarray:
        assert self._sock is not None, "NoriCamera.read before connect()"
        import zmq

        try:
            payload = self._sock.recv()
        except zmq.Again:
            if self._last is None:
                raise RuntimeError(
                    f"Nori camera {self.spec.obs_key} "
                    f"(tcp://{self.spec.host}:{self.spec.port}): no frame yet — "
                    "is the daemon's camera publisher running?"
                ) from None
            return self._last
        _, _, jpeg = split_camera_frame(payload)
        self._last = _decode_jpeg(jpeg)
        return self._last

    def disconnect(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close(linger=0)
            except Exception:  # noqa: BLE001
                _logger.warning(
                    "Nori camera %s close failed", self.spec.obs_key, exc_info=True
                )
            self._sock = None


__all__ = [
    "NoriCameraSpec",
    "NoriCamera",
    "resolve_camera_specs",
    "split_camera_frame",
]
