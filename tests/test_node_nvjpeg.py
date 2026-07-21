"""nvJPEG resolver + routing (node/jpeg.py + node/nvjpeg.py) — no GPU.

Everything here runs on GPU-less CI: the encoder is faked and ``probe``
is monkeypatched, so what's under test is the resolver ordering, the env
kill-switch, the size/mono routing rules, and the fall-back-never-crash
contract. The real-GPU parity case lives in test_node_jpeg.py (it
auto-skips without a CUDA device); the Jetson-side verification recipe
is in SDK ADR 0019.
"""
from __future__ import annotations

import ctypes
import logging

import numpy as np
import pytest

import interlatent.node.jpeg as jpeg_mod
import interlatent.node.nvjpeg as nvjpeg_mod
from interlatent.node.jpeg import encode_jpeg

_FAKE_JPEG = b"\xff\xd8\xff-fake-nvjpeg"


class _FakeEncoder:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[tuple, int]] = []

    def encode(self, arr, quality):
        self.calls.append((arr.shape, int(quality)))
        if self.fail:
            raise nvjpeg_mod.NvJpegError("fake encode failure")
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
    _clear_backend_state()
    yield
    _clear_backend_state()


def _frame(h: int = 480, w: int = 640) -> np.ndarray:
    return np.full((h, w, 3), 128, dtype=np.uint8)


# --------------------------------------------------------------------------- #
# Resolver ordering + env kill-switch                                          #
# --------------------------------------------------------------------------- #


def test_nvjpeg_wins_when_probe_succeeds(monkeypatch):
    fake = _FakeEncoder()
    monkeypatch.setattr(nvjpeg_mod, "probe", lambda: fake)
    assert jpeg_mod._resolve_backend() == ("nvjpeg", fake)
    # The CPU sidecar resolved alongside it, for small/mono/failure routing.
    assert jpeg_mod._CPU_BACKEND is not None


def test_probe_none_falls_to_cpu_chain(monkeypatch):
    monkeypatch.setattr(nvjpeg_mod, "probe", lambda: None)
    name, _ = jpeg_mod._resolve_backend()
    assert name in ("turbojpeg", "cv2", "pil")


def test_env_cv2_excludes_nvjpeg_entirely(monkeypatch):
    monkeypatch.setenv("INTERLATENT_JPEG_BACKEND", "cv2")
    monkeypatch.setattr(
        nvjpeg_mod, "probe", lambda: pytest.fail("nvjpeg probe must not run")
    )
    name, _ = jpeg_mod._resolve_backend()
    assert name == "cv2"


def test_env_forced_nvjpeg_with_failed_probe_warns_and_falls_back(
    monkeypatch, caplog
):
    monkeypatch.setenv("INTERLATENT_JPEG_BACKEND", "nvjpeg")
    monkeypatch.setattr(nvjpeg_mod, "probe", lambda: None)
    with caplog.at_level(logging.WARNING, logger="interlatent.node.jpeg"):
        name, _ = jpeg_mod._resolve_backend()
    assert name in ("turbojpeg", "cv2", "pil")
    assert any(
        "no usable CUDA nvJPEG" in r.getMessage() for r in caplog.records
    )


def test_env_garbage_value_is_ignored(monkeypatch, caplog):
    monkeypatch.setenv("INTERLATENT_JPEG_BACKEND", "quantum")
    monkeypatch.setattr(nvjpeg_mod, "probe", lambda: None)  # auto still probes
    with caplog.at_level(logging.WARNING, logger="interlatent.node.jpeg"):
        name, _ = jpeg_mod._resolve_backend()
    assert name in ("turbojpeg", "cv2", "pil")
    assert any(
        "Ignoring INTERLATENT_JPEG_BACKEND" in r.getMessage()
        for r in caplog.records
    )


# --------------------------------------------------------------------------- #
# Routing: size threshold, mono exclusion, failure fallback                    #
# --------------------------------------------------------------------------- #


def test_recording_size_frame_routes_gpu(monkeypatch):
    fake = _FakeEncoder()
    monkeypatch.setattr(nvjpeg_mod, "probe", lambda: fake)
    assert encode_jpeg(_frame(), quality=85) == _FAKE_JPEG
    assert fake.calls == [((480, 640, 3), 85)]


def test_small_and_mono_frames_stay_cpu(monkeypatch):
    fake = _FakeEncoder()
    monkeypatch.setattr(nvjpeg_mod, "probe", lambda: fake)
    small = encode_jpeg(_frame(100, 100))  # 10k px < 150k threshold
    mono = encode_jpeg(np.full((480, 640), 128, dtype=np.uint8))
    assert small is not None and small[:3] == b"\xff\xd8\xff"
    assert small != _FAKE_JPEG
    assert mono is not None and mono[:3] == b"\xff\xd8\xff"
    assert fake.calls == []


def test_min_pixels_env_override_routes_small_frame_gpu(monkeypatch):
    monkeypatch.setenv("INTERLATENT_NVJPEG_MIN_PIXELS", "0")
    fake = _FakeEncoder()
    monkeypatch.setattr(nvjpeg_mod, "probe", lambda: fake)
    assert encode_jpeg(_frame(100, 100)) == _FAKE_JPEG


def test_encode_error_falls_back_to_cpu_and_warns_once(monkeypatch, caplog):
    fake = _FakeEncoder(fail=True)
    monkeypatch.setattr(nvjpeg_mod, "probe", lambda: fake)
    with caplog.at_level(logging.WARNING, logger="interlatent.node.jpeg"):
        first = encode_jpeg(_frame())
        second = encode_jpeg(_frame())
    # Both frames still encode via the CPU chain — the loop never starves.
    assert first is not None and first[:3] == b"\xff\xd8\xff"
    assert second is not None and second[:3] == b"\xff\xd8\xff"
    assert len(fake.calls) == 2  # nvjpeg attempted each time
    warnings = [
        r for r in caplog.records
        if "nvjpeg encode failed" in r.getMessage()
        and r.levelno == logging.WARNING
    ]
    assert len(warnings) == 1  # one-shot, then debug


# --------------------------------------------------------------------------- #
# Binding-level guards (no CUDA libraries present)                             #
# --------------------------------------------------------------------------- #


def test_cuda_device_count_zero_when_cudart_unloadable(monkeypatch):
    monkeypatch.setattr(nvjpeg_mod, "_CUDART", None)

    def _no_lib(*_a, **_k):
        raise OSError("no such library")

    monkeypatch.setattr(ctypes, "CDLL", _no_lib)
    assert nvjpeg_mod.cuda_device_count() == 0


def test_probe_returns_none_without_cuda(monkeypatch):
    monkeypatch.setattr(nvjpeg_mod, "cuda_device_count", lambda: 0)
    assert nvjpeg_mod.probe() is None
