"""Capability-adaptive JPEG encoder (node/jpeg.py, ADR 0023).

The historical bug class here is color order: cv2 wants BGR, turbojpeg
wants an explicit pixel-format flag, PIL wants RGB. The parity tests
force each backend and check the DECODED image against the source, so a
channel swap in any branch fails loudly instead of shipping blue robots.
"""
from __future__ import annotations

import numpy as np
import pytest

import interlatent.node.jpeg as jpeg_mod
from interlatent.node.jpeg import encode_jpeg


def _decode(data: bytes) -> np.ndarray:
    """Reference decode via PIL (RGB)."""
    import io

    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


def _test_frame(h: int = 48, w: int = 64) -> np.ndarray:
    """A frame with strongly asymmetric channels: R ramps, G mid, B low.
    A BGR/RGB swap moves per-channel means by >100 — unmissable."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :, 0] = np.linspace(180, 255, w, dtype=np.uint8)[None, :]
    arr[:, :, 1] = 90
    arr[:, :, 2] = 10
    return arr


def _available_backends() -> list[str]:
    out = ["pil"]
    try:
        import cv2  # noqa: F401

        out.append("cv2")
    except ImportError:
        pass
    try:
        from turbojpeg import TurboJPEG

        TurboJPEG()
        out.append("turbojpeg")
    except Exception:
        pass
    # Real GPU only — probe() is exception-safe and returns None on
    # GPU-less CI, which auto-skips the GPU parity cases.
    for gpu_name in ("nvjpeg", "gpujpeg"):
        try:
            import importlib

            mod = importlib.import_module(f"interlatent.node.{gpu_name}")
            enc = mod.probe()
            if enc is not None:
                enc.close()
                out.append(gpu_name)
        except Exception:
            pass
    return out


def _clear_backend_state():
    jpeg_mod._BACKEND = None
    jpeg_mod._CPU_BACKEND = None
    jpeg_mod._NVJPEG_MIN_PIXELS = None
    jpeg_mod._NVJPEG_WARNED = False


@pytest.fixture(autouse=True)
def _reset_backend(monkeypatch):
    monkeypatch.delenv("INTERLATENT_JPEG_BACKEND", raising=False)
    monkeypatch.delenv("INTERLATENT_NVJPEG_MIN_PIXELS", raising=False)
    _clear_backend_state()
    yield
    _clear_backend_state()


@pytest.mark.parametrize("backend", _available_backends())
def test_color_order_parity(backend, monkeypatch):
    """Decoded output of every backend matches the RGB source channelwise."""
    if backend == "pil":
        # Force PIL by making the resolver skip the faster backends.
        jpeg_mod._BACKEND = ("pil", None)
    elif backend == "cv2":
        import cv2

        jpeg_mod._BACKEND = ("cv2", cv2)
    elif backend in ("nvjpeg", "gpujpeg"):
        import importlib

        enc = importlib.import_module(f"interlatent.node.{backend}").probe()
        assert enc is not None  # listed only when the probe succeeded
        jpeg_mod._BACKEND = (backend, enc)
        # The parity frame is 48x64 — route it to the GPU regardless of
        # the size threshold so a channel-order swap fails this test.
        monkeypatch.setattr(jpeg_mod, "_NVJPEG_MIN_PIXELS", 0)
    src = _test_frame()
    data = encode_jpeg(src, quality=95)
    assert data is not None and data[:3] == b"\xff\xd8\xff"
    dec = _decode(data)
    assert dec.shape == src.shape
    # Channel means survive JPEG q95 within a few counts; a swap moves
    # them by ~150+.
    for c in range(3):
        assert abs(float(dec[:, :, c].mean()) - float(src[:, :, c].mean())) < 10, (
            f"channel {c} mean off — color order bug in backend {backend!r}"
        )


def test_mono_frame():
    src = (np.arange(48 * 64, dtype=np.uint8).reshape(48, 64) % 255)
    data = encode_jpeg(src)
    assert data is not None and data[:3] == b"\xff\xd8\xff"


def test_trailing_one_channel_squeezed():
    src = _test_frame()[:, :, :1]  # HW1
    data = encode_jpeg(src)
    assert data is not None


def test_target_size_square():
    data = encode_jpeg(_test_frame(), target_size=32)
    assert data is not None
    dec = _decode(data)
    assert dec.shape[:2] == (32, 32)


def test_max_dim_preserves_aspect():
    data = encode_jpeg(_test_frame(h=48, w=96), max_dim=32)
    assert data is not None
    dec = _decode(data)
    assert dec.shape[:2] == (16, 32)


def test_max_dim_noop_when_small():
    data = encode_jpeg(_test_frame(h=20, w=30), max_dim=320)
    assert data is not None
    dec = _decode(data)
    assert dec.shape[:2] == (20, 30)


def test_quality_orders_size():
    src = _test_frame(h=120, w=160)
    lo = encode_jpeg(src, quality=30)
    hi = encode_jpeg(src, quality=95)
    assert lo is not None and hi is not None and len(lo) < len(hi)


def test_no_backend_returns_none(monkeypatch):
    jpeg_mod._BACKEND = ("none", None)
    assert encode_jpeg(_test_frame()) is None
