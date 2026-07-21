"""ctypes binding for CUDA nvJPEG baseline encode (node capture path).

Pure binding, no policy: which frames go to the GPU (size threshold, mono
exclusion, env override) is decided by the resolver in ``node/jpeg.py`` —
this module only knows how to load ``libnvjpeg``/``libcudart`` and drive
one encode. nvJPEG ships with the CUDA toolkit (every JetPack), so there
is deliberately no pip dependency here; a box without CUDA simply fails
:func:`probe` and the CPU encoders take over. See SDK ADR 0019.

Everything is loaded lazily and cached at module level. All entry points
are exception-safe for callers: :func:`cuda_device_count` returns 0 and
:func:`probe` returns None on any failure (missing library, no device,
broken driver), logging the reason at debug.

ctypes gotcha this file is paranoid about: every foreign function gets
explicit ``argtypes``/``restype``. Without them ctypes truncates pointers
to 32-bit ints on LP64 platforms — on the Jetson (aarch64) that is a
crash or silent corruption, not an error. The 16x16 probe encode exists
to surface exactly that class of bug once at resolve time instead of at
30 Hz on the control thread.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import logging
import threading
from typing import Optional, Sequence

import numpy as np

_LOG = logging.getLogger(__name__)

# nvjpeg.h enum values (stable ABI constants since CUDA 10).
NVJPEG_INPUT_RGBI = 5  # interleaved RGB in channel[0], pitch = width*3
NVJPEG_CSS_420 = 2  # 4:2:0 chroma subsampling (turbojpeg's encode default)
# driver_types.h
_CUDA_MEMCPY_HOST_TO_DEVICE = 1

# Versioned sonames first-party CUDA installs actually expose; the bare
# .so only exists with the toolkit's dev symlinks. /usr/local/cuda is the
# JetPack/standard toolkit prefix (find_library misses it when the lib
# isn't in ldconfig's cache).
_NVJPEG_CANDIDATES = (
    ctypes.util.find_library("nvjpeg"),
    "libnvjpeg.so",
    "libnvjpeg.so.13",
    "libnvjpeg.so.12",
    "libnvjpeg.so.11",
    "/usr/local/cuda/lib64/libnvjpeg.so",
)
_CUDART_CANDIDATES = (
    ctypes.util.find_library("cudart"),
    "libcudart.so",
    "libcudart.so.13",
    "libcudart.so.12",
    "libcudart.so.11.0",
    "/usr/local/cuda/lib64/libcudart.so",
)

_c_void_p_p = ctypes.POINTER(ctypes.c_void_p)
_c_size_t_p = ctypes.POINTER(ctypes.c_size_t)


class NvJpegError(RuntimeError):
    """A CUDA or nvJPEG call returned a nonzero status."""


class _NvjpegImage(ctypes.Structure):
    """nvjpegImage_t: NVJPEG_MAX_COMPONENT (4) planes; RGBI uses plane 0."""

    _fields_ = [
        ("channel", ctypes.c_void_p * 4),
        ("pitch", ctypes.c_size_t * 4),
    ]


def _check(status: int, what: str) -> None:
    if status != 0:
        raise NvJpegError(f"{what} failed with status {status}")


def _load_lib(candidates: Sequence[Optional[str]]) -> ctypes.CDLL:
    last: Optional[Exception] = None
    for name in candidates:
        if not name:
            continue
        try:
            return ctypes.CDLL(name)
        except OSError as exc:
            last = exc
    raise OSError(f"no loadable library among {[c for c in candidates if c]}: {last}")


_CUDART: Optional[ctypes.CDLL] = None
_NVJPEG: Optional[ctypes.CDLL] = None


def _cudart() -> ctypes.CDLL:
    global _CUDART
    if _CUDART is None:
        lib = _load_lib(_CUDART_CANDIDATES)
        lib.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        lib.cudaGetDeviceCount.restype = ctypes.c_int
        lib.cudaMalloc.argtypes = [_c_void_p_p, ctypes.c_size_t]
        lib.cudaMalloc.restype = ctypes.c_int
        lib.cudaFree.argtypes = [ctypes.c_void_p]
        lib.cudaFree.restype = ctypes.c_int
        lib.cudaMemcpy.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
        ]
        lib.cudaMemcpy.restype = ctypes.c_int
        lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        lib.cudaStreamSynchronize.restype = ctypes.c_int
        _CUDART = lib
    return _CUDART


def _nvjpeg() -> ctypes.CDLL:
    global _NVJPEG
    if _NVJPEG is None:
        lib = _load_lib(_NVJPEG_CANDIDATES)
        lib.nvjpegCreateSimple.argtypes = [_c_void_p_p]
        lib.nvjpegCreateSimple.restype = ctypes.c_int
        lib.nvjpegEncoderStateCreate.argtypes = [
            ctypes.c_void_p, _c_void_p_p, ctypes.c_void_p,
        ]
        lib.nvjpegEncoderStateCreate.restype = ctypes.c_int
        lib.nvjpegEncoderParamsCreate.argtypes = [
            ctypes.c_void_p, _c_void_p_p, ctypes.c_void_p,
        ]
        lib.nvjpegEncoderParamsCreate.restype = ctypes.c_int
        lib.nvjpegEncoderParamsSetQuality.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
        ]
        lib.nvjpegEncoderParamsSetQuality.restype = ctypes.c_int
        lib.nvjpegEncoderParamsSetSamplingFactors.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
        ]
        lib.nvjpegEncoderParamsSetSamplingFactors.restype = ctypes.c_int
        lib.nvjpegEncodeImage.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.POINTER(_NvjpegImage), ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
        ]
        lib.nvjpegEncodeImage.restype = ctypes.c_int
        lib.nvjpegEncodeRetrieveBitstream.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            _c_size_t_p, ctypes.c_void_p,
        ]
        lib.nvjpegEncodeRetrieveBitstream.restype = ctypes.c_int
        lib.nvjpegEncoderParamsDestroy.argtypes = [ctypes.c_void_p]
        lib.nvjpegEncoderParamsDestroy.restype = ctypes.c_int
        lib.nvjpegEncoderStateDestroy.argtypes = [ctypes.c_void_p]
        lib.nvjpegEncoderStateDestroy.restype = ctypes.c_int
        lib.nvjpegDestroy.argtypes = [ctypes.c_void_p]
        lib.nvjpegDestroy.restype = ctypes.c_int
        _NVJPEG = lib
    return _NVJPEG


def cuda_device_count() -> int:
    """Visible CUDA devices; 0 on any failure (no driver, no libcudart)."""
    try:
        count = ctypes.c_int(0)
        if _cudart().cudaGetDeviceCount(ctypes.byref(count)) != 0:
            return 0
        return max(0, int(count.value))
    except Exception:
        return 0


class NvJpegEncoder:
    """One reusable nvJPEG baseline encoder (handle + state + params).

    Thread-safe via an internal lock, though today every encode happens on
    the control-loop thread (recording, preview tee, and inference uplink
    all encode inline; the teleop sender thread only ships bytes).

    The device input buffer is grow-only and keyed by byte size, so the
    steady state of a session (fixed camera resolutions) does zero
    allocations per tick. Host memory is pageable — on Jetson unified
    memory the H2D copy is cheap; pinned staging is a documented later
    micro-optimization.
    """

    def __init__(self) -> None:
        self._rt = _cudart()
        self._nv = _nvjpeg()
        self._lock = threading.Lock()
        self._handle = ctypes.c_void_p()
        self._state = ctypes.c_void_p()
        self._params = ctypes.c_void_p()
        self._dev_buf: Optional[ctypes.c_void_p] = None
        self._dev_cap = 0
        self._last_quality: Optional[int] = None
        _check(self._nv.nvjpegCreateSimple(ctypes.byref(self._handle)),
               "nvjpegCreateSimple")
        _check(self._nv.nvjpegEncoderStateCreate(
            self._handle, ctypes.byref(self._state), None),
            "nvjpegEncoderStateCreate")
        _check(self._nv.nvjpegEncoderParamsCreate(
            self._handle, ctypes.byref(self._params), None),
            "nvjpegEncoderParamsCreate")
        _check(self._nv.nvjpegEncoderParamsSetSamplingFactors(
            self._params, NVJPEG_CSS_420, None),
            "nvjpegEncoderParamsSetSamplingFactors")

    def encode(self, arr: np.ndarray, quality: int) -> bytes:
        """Encode a uint8 HxWx3 RGB frame to baseline JPEG bytes.

        Raises :class:`NvJpegError` on any CUDA/nvJPEG failure — the
        caller (``node/jpeg.py``) owns falling back to a CPU encoder.
        """
        if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
            raise NvJpegError(f"expected uint8 HxWx3 RGB, got {arr.dtype} {arr.shape}")
        arr = np.ascontiguousarray(arr)  # no-op on the capture path
        h, w = int(arr.shape[0]), int(arr.shape[1])
        need = h * w * 3
        with self._lock:
            if int(quality) != self._last_quality:
                _check(self._nv.nvjpegEncoderParamsSetQuality(
                    self._params, int(quality), None),
                    "nvjpegEncoderParamsSetQuality")
                self._last_quality = int(quality)
            if need > self._dev_cap:
                if self._dev_buf is not None:
                    self._rt.cudaFree(self._dev_buf)
                    self._dev_buf, self._dev_cap = None, 0
                ptr = ctypes.c_void_p()
                _check(self._rt.cudaMalloc(ctypes.byref(ptr), need), "cudaMalloc")
                self._dev_buf, self._dev_cap = ptr, need
            _check(self._rt.cudaMemcpy(
                self._dev_buf, arr.ctypes.data, need,
                _CUDA_MEMCPY_HOST_TO_DEVICE), "cudaMemcpy H2D")
            img = _NvjpegImage()
            img.channel[0] = self._dev_buf.value
            img.pitch[0] = w * 3
            _check(self._nv.nvjpegEncodeImage(
                self._handle, self._state, self._params, ctypes.byref(img),
                NVJPEG_INPUT_RGBI, w, h, None), "nvjpegEncodeImage")
            _check(self._rt.cudaStreamSynchronize(None), "cudaStreamSynchronize")
            length = ctypes.c_size_t(0)
            _check(self._nv.nvjpegEncodeRetrieveBitstream(
                self._handle, self._state, None, ctypes.byref(length), None),
                "nvjpegEncodeRetrieveBitstream (size)")
            buf = (ctypes.c_ubyte * length.value)()
            _check(self._nv.nvjpegEncodeRetrieveBitstream(
                self._handle, self._state, buf, ctypes.byref(length), None),
                "nvjpegEncodeRetrieveBitstream (copy)")
            return bytes(buf[: length.value])

    def close(self) -> None:
        with self._lock:
            if self._dev_buf is not None:
                self._rt.cudaFree(self._dev_buf)
                self._dev_buf, self._dev_cap = None, 0
            if self._params:
                self._nv.nvjpegEncoderParamsDestroy(self._params)
                self._params = ctypes.c_void_p()
            if self._state:
                self._nv.nvjpegEncoderStateDestroy(self._state)
                self._state = ctypes.c_void_p()
            if self._handle:
                self._nv.nvjpegDestroy(self._handle)
                self._handle = ctypes.c_void_p()

    def __del__(self) -> None:  # pragma: no cover - interpreter teardown
        try:
            self.close()
        except Exception:
            pass


def probe() -> Optional[NvJpegEncoder]:
    """A working encoder, or None when this box can't do CUDA JPEG.

    Runs one 16x16 test encode so driver/ABI breakage (the ctypes
    truncation class especially) fails HERE, once, at backend-resolve
    time — never on the 30 Hz control thread.
    """
    enc: Optional[NvJpegEncoder] = None
    try:
        if cuda_device_count() <= 0:
            _LOG.debug("nvjpeg probe: no CUDA device")
            return None
        enc = NvJpegEncoder()
        data = enc.encode(np.zeros((16, 16, 3), dtype=np.uint8), 85)
        if not data.startswith(b"\xff\xd8"):
            raise NvJpegError(f"probe encode returned non-JPEG ({len(data)} bytes)")
        return enc
    except Exception:
        _LOG.debug("nvjpeg probe failed", exc_info=True)
        if enc is not None:
            try:
                enc.close()
            except Exception:
                pass
        return None
