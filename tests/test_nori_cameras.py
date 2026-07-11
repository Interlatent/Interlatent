"""Nori ZMQ camera channel: wire-format split, JPEG decode, spec resolution."""
from __future__ import annotations

import numpy as np
import pytest

from interlatent.adapters.nori.cameras import (
    NoriCameraSpec,
    resolve_camera_specs,
    split_camera_frame,
)
from interlatent.adapters.nori.config import NoriAdapterConfig


def _jpeg_bytes(w: int = 8, h: int = 6) -> bytes:
    PIL = pytest.importorskip("PIL.Image")
    import io

    img = PIL.new("RGB", (w, h), (200, 10, 30))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Wire format + decode                                                         #
# --------------------------------------------------------------------------- #


def test_split_camera_frame():
    payload = b"front 123.456\n" + b"\xff\xd8jpegbytes"
    name, ts, jpeg = split_camera_frame(payload)
    assert name == "front" and ts == pytest.approx(123.456)
    assert jpeg == b"\xff\xd8jpegbytes"


def test_split_camera_frame_rejects_headerless():
    with pytest.raises(ValueError):
        split_camera_frame(b"no-newline-jpeg-bytes")


def test_decode_jpeg_roundtrip():
    from interlatent.adapters.nori.cameras import _decode_jpeg

    rgb = _decode_jpeg(_jpeg_bytes())
    assert rgb.dtype == np.uint8 and rgb.shape == (6, 8, 3)
    # Red-dominant fill must come back red-dominant (channel order check).
    assert rgb[..., 0].mean() > rgb[..., 2].mean() > 0 - 1


# --------------------------------------------------------------------------- #
# Spec resolution against the descriptor                                       #
# --------------------------------------------------------------------------- #


def test_resolve_specs_default_all_cameras():
    cfg = NoriAdapterConfig()
    specs = resolve_camera_specs(cfg, ["front", "right_wrist"])
    assert [(s.obs_key, s.daemon_name, s.port) for s in specs] == [
        ("front", "front", 5555),
        ("right_wrist", "right_wrist", 5556),
    ]
    assert all(s.host == "127.0.0.1" for s in specs)


def test_resolve_specs_mapping_and_port_math():
    cfg = NoriAdapterConfig(
        cameras={"observation.images.top": "right_wrist"},
        cam_base_port=6000,
        cam_host="10.0.0.9",
    )
    (spec,) = resolve_camera_specs(cfg, ["front", "right_wrist"])
    assert spec.obs_key == "observation.images.top"
    assert spec.index == 1 and spec.port == 6001 and spec.host == "10.0.0.9"


def test_resolve_specs_unknown_name_raises():
    cfg = NoriAdapterConfig(cameras={"top": "nonexistent"})
    with pytest.raises(ValueError, match="no such\n?.*camera|no such"):
        resolve_camera_specs(cfg, ["front"])


def test_resolve_specs_explicit_index_without_descriptor():
    # Descriptorless ack (real daemon build, 2026-07-10): name:index form
    # bypasses descriptor discovery entirely.
    cfg = NoriAdapterConfig(cameras={"front": "front:0", "wrist": "right_wrist:2"})
    specs = resolve_camera_specs(cfg, [])
    assert [(s.obs_key, s.daemon_name, s.port) for s in specs] == [
        ("front", "front", 5555),
        ("wrist", "right_wrist", 5557),
    ]


def test_resolve_specs_bare_index_uses_obs_key_as_name():
    cfg = NoriAdapterConfig(cameras={"front": "0"})
    (spec,) = resolve_camera_specs(cfg, [])
    assert spec.daemon_name == "front" and spec.port == 5555


def test_resolve_specs_descriptorless_name_error_teaches_index_form():
    cfg = NoriAdapterConfig(cameras={"front": "front"})
    with pytest.raises(ValueError, match="name>:<index"):
        resolve_camera_specs(cfg, [])


# --------------------------------------------------------------------------- #
# Live inproc SUB/PUB round-trip (only when pyzmq is installed)                #
# --------------------------------------------------------------------------- #


def test_camera_read_over_zmq_inproc():
    zmq = pytest.importorskip("zmq")
    from interlatent.adapters.nori.cameras import NoriCamera

    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    port = pub.bind_to_random_port("tcp://127.0.0.1")
    spec = NoriCameraSpec(
        obs_key="front", daemon_name="front", index=0, host="127.0.0.1", port=port
    )
    cam = NoriCamera(spec, context_factory=lambda: ctx)
    cam.connect()
    try:
        jpeg = _jpeg_bytes()
        frame = None
        # PUB/SUB joins race the first sends; publish until one lands.
        for _ in range(100):
            pub.send(b"front 1.0\n" + jpeg)
            try:
                frame = cam.read()
                break
            except RuntimeError:
                continue
        assert frame is not None, "no frame received over inproc zmq"
        assert frame.dtype == np.uint8 and frame.shape == (6, 8, 3)
        # Latest-wins: with no new frame, read() serves the last one.
        assert cam.read() is not None
    finally:
        cam.disconnect()
        pub.close(linger=0)
        ctx.term()
