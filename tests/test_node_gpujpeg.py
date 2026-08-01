"""GPUJPEG backend resolution + routing (node/jpeg.py + node/gpujpeg.py) — no GPU.

Same contract as test_node_nvjpeg.py: the encoder is faked, so what's
under test is where gpujpeg sits in the resolver chain (after nvjpeg,
before the CPU encoders), the env kill-switch, shared GPU routing, and
fallback. The real-GPU parity case lives in test_node_jpeg.py
(auto-skips without a built libgpujpeg + CUDA device).
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

import interlatent.node.gpujpeg as gpujpeg_mod
import interlatent.node.jpeg as jpeg_mod
import interlatent.node.nvjpeg as nvjpeg_mod
from interlatent.node.jpeg import encode_jpeg

_FAKE_JPEG = b"\xff\xd8\xff-fake-gpujpeg"


class _FakeEncoder:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[tuple, int]] = []

    def encode(self, arr, quality):
        self.calls.append((arr.shape, int(quality)))
        if self.fail:
            raise gpujpeg_mod.GpuJpegError("fake encode failure")
        return _FAKE_JPEG


def _clear_backend_state():
    jpeg_mod._BACKEND = None
    jpeg_mod._CPU_BACKEND = None
    jpeg_mod._NVJPEG_MIN_PIXELS = None
    jpeg_mod._NVJPEG_WARNED = False


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.delenv("INTERLATENT_JPEG_BACKEND", raising=False)
    monkeypatch.delenv("INTERLATENT_NVJPEG_MIN_PIXELS", raising=False)
    monkeypatch.delenv("INTERLATENT_GPU_JPEG_MIN_PIXELS", raising=False)
    _clear_backend_state()
    yield
    _clear_backend_state()


def test_gpujpeg_resolves_when_nvjpeg_absent(monkeypatch):
    fake = _FakeEncoder()
    monkeypatch.setattr(nvjpeg_mod, "probe", lambda: None)
    monkeypatch.setattr(gpujpeg_mod, "probe", lambda: fake)
    assert jpeg_mod._resolve_backend() == ("gpujpeg", fake)
    assert jpeg_mod._CPU_BACKEND is not None


def test_nvjpeg_wins_over_gpujpeg(monkeypatch):
    nv = _FakeEncoder()
    monkeypatch.setattr(nvjpeg_mod, "probe", lambda: nv)
    monkeypatch.setattr(
        gpujpeg_mod, "probe", lambda: pytest.fail("gpujpeg probed after nvjpeg won")
    )
    assert jpeg_mod._resolve_backend() == ("nvjpeg", nv)


def test_env_gpujpeg_skips_nvjpeg_probe(monkeypatch):
    fake = _FakeEncoder()
    monkeypatch.setenv("INTERLATENT_JPEG_BACKEND", "gpujpeg")
    monkeypatch.setattr(
        nvjpeg_mod, "probe", lambda: pytest.fail("nvjpeg probe must not run")
    )
    monkeypatch.setattr(gpujpeg_mod, "probe", lambda: fake)
    assert jpeg_mod._resolve_backend() == ("gpujpeg", fake)


def test_env_forced_gpujpeg_with_failed_probe_warns_and_falls_back(
    monkeypatch, caplog
):
    monkeypatch.setenv("INTERLATENT_JPEG_BACKEND", "gpujpeg")
    monkeypatch.setattr(gpujpeg_mod, "probe", lambda: None)
    with caplog.at_level(logging.WARNING, logger="interlatent.node.jpeg"):
        name, _ = jpeg_mod._resolve_backend()
    assert name in ("turbojpeg", "cv2", "pil")
    assert any(
        "no usable GPU JPEG encoder" in r.getMessage() for r in caplog.records
    )


def test_routing_and_fallback_shared_with_gpu_backends(monkeypatch):
    fake = _FakeEncoder()
    monkeypatch.setattr(nvjpeg_mod, "probe", lambda: None)
    monkeypatch.setattr(gpujpeg_mod, "probe", lambda: fake)
    # Recording-size frame routes to the GPU backend.
    big = np.full((480, 640, 3), 128, dtype=np.uint8)
    assert encode_jpeg(big, quality=85) == _FAKE_JPEG
    # Small frame stays CPU.
    small = encode_jpeg(np.full((100, 100, 3), 128, dtype=np.uint8))
    assert small is not None and small[:3] == b"\xff\xd8\xff"
    assert small != _FAKE_JPEG
    assert fake.calls == [((480, 640, 3), 85)]


def test_generic_min_pixels_env_applies(monkeypatch):
    monkeypatch.setenv("INTERLATENT_GPU_JPEG_MIN_PIXELS", "0")
    fake = _FakeEncoder()
    monkeypatch.setattr(nvjpeg_mod, "probe", lambda: None)
    monkeypatch.setattr(gpujpeg_mod, "probe", lambda: fake)
    assert encode_jpeg(np.full((100, 100, 3), 128, dtype=np.uint8)) == _FAKE_JPEG


def test_encode_error_falls_back_to_cpu(monkeypatch, caplog):
    fake = _FakeEncoder(fail=True)
    monkeypatch.setattr(nvjpeg_mod, "probe", lambda: None)
    monkeypatch.setattr(gpujpeg_mod, "probe", lambda: fake)
    with caplog.at_level(logging.WARNING, logger="interlatent.node.jpeg"):
        out = encode_jpeg(np.full((480, 640, 3), 128, dtype=np.uint8))
    assert out is not None and out[:3] == b"\xff\xd8\xff"
    assert fake.calls  # attempted before falling back
    assert any(
        "gpujpeg encode failed" in r.getMessage() for r in caplog.records
    )


def test_probe_returns_none_without_library(monkeypatch):
    monkeypatch.setattr(gpujpeg_mod, "_LIB", None)

    def _no_lib(*_a, **_k):
        raise OSError("no such library")

    monkeypatch.setattr(gpujpeg_mod, "_load_lib", _no_lib)
    assert gpujpeg_mod.probe() is None
