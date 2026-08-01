"""ctypes binding for CESNET GPUJPEG encode (node capture path).

GPUJPEG runs baseline JPEG encode on CUDA SMs — no fixed-function
hardware needed, which makes it the only GPU JPEG path on a Jetson Orin
Nano (no NVJPG block, and JetPack ships no CUDA nvJPEG; the Tegra
``libnvjpeg.so`` is an unrelated libjpeg-API library with the same
soname). On x86 CUDA boxes the nvJPEG backend resolves first and this
module is never consulted. See SDK ADR 0019.

The library is built from source by the node operator (there is no pip
wheel or apt package)::

    git clone --branch v0.27.13 --depth 1 https://github.com/CESNET/GPUJPEG.git
    cd GPUJPEG && cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j$(nproc) && sudo cmake --install build && sudo ldconfig

**ABI pin: v0.25+ struct layout, validated against v0.27.13.** Unlike
nvJPEG's opaque handles, GPUJPEG passes parameter structs by pointer,
so their field layout is load-bearing. Two guards keep a mismatched
build from corrupting frames silently: :func:`probe` refuses versions
older than 0.25 (``comp_count`` moved between structs there), and the
probe encode round-trips a color-asymmetric test frame through PIL to
catch channel-order/layout garbling before the backend is ever chosen.
All defaults are filled by the library's own ``*_set_default_parameters``
initializers, never hand-written, so unknown trailing fields keep their
correct defaults.

Same contract as node/nvjpeg.py: pure binding, no routing policy;
:func:`probe` returns None on any failure and the CPU chain takes over.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import logging
import threading
from typing import Optional, Sequence, Tuple

import numpy as np

_LOG = logging.getLogger(__name__)

# gpujpeg_type.h / gpujpeg_common.h enum + macro values (v0.27.13).
_GPUJPEG_RGB = 1  # enum gpujpeg_color_space
_GPUJPEG_444_U8_P012 = 1  # enum gpujpeg_pixel_format: interleaved RGB
_GPUJPEG_ENCODER_INPUT_IMAGE = 0  # enum gpujpeg_encoder_input_type
# MK_SUBSAMPLING(2,2, 1,1, 1,1, 0,0) — 4:2:0, matching the other backends.
_GPUJPEG_SUBSAMPLING_420 = (
    2 << 28 | 2 << 24 | 1 << 20 | 1 << 16 | 1 << 12 | 1 << 8
)

_LIB_CANDIDATES = (
    ctypes.util.find_library("gpujpeg"),
    "libgpujpeg.so",
    "libgpujpeg.so.0",
    "/usr/local/lib/libgpujpeg.so",
)

_MIN_VERSION = (0, 25)


class GpuJpegError(RuntimeError):
    """A GPUJPEG call failed or the built library is unusable."""


class _SamplingFactor(ctypes.Structure):
    _fields_ = [("horizontal", ctypes.c_uint8), ("vertical", ctypes.c_uint8)]


class _Parameters(ctypes.Structure):
    """struct gpujpeg_parameters, v0.25+ layout (verified v0.27.13)."""

    _fields_ = [
        ("verbose", ctypes.c_int),
        ("perf_stats", ctypes.c_int),
        ("quality", ctypes.c_int),
        ("restart_interval", ctypes.c_int),
        ("interleaved", ctypes.c_int),
        ("segment_info", ctypes.c_int),
        ("comp_count", ctypes.c_int),
        ("sampling_factor", _SamplingFactor * 4),
        ("color_space_internal", ctypes.c_int),
    ]


class _ImageParameters(ctypes.Structure):
    """struct gpujpeg_image_parameters, v0.25+ layout."""

    _fields_ = [
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("color_space", ctypes.c_int),
        ("pixel_format", ctypes.c_int),
        ("width_padding", ctypes.c_int),
    ]


class _EncoderInput(ctypes.Structure):
    """struct gpujpeg_encoder_input."""

    _fields_ = [
        ("type", ctypes.c_int),
        ("image", ctypes.c_void_p),
        ("texture", ctypes.c_void_p),
    ]


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


_LIB: Optional[ctypes.CDLL] = None


def _gpujpeg() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        lib = _load_lib(_LIB_CANDIDATES)
        lib.gpujpeg_init_device.argtypes = [ctypes.c_int, ctypes.c_int]
        lib.gpujpeg_init_device.restype = ctypes.c_int
        lib.gpujpeg_set_default_parameters.argtypes = [ctypes.POINTER(_Parameters)]
        lib.gpujpeg_set_default_parameters.restype = None
        lib.gpujpeg_parameters_chroma_subsampling.argtypes = [
            ctypes.POINTER(_Parameters), ctypes.c_uint32,
        ]
        lib.gpujpeg_parameters_chroma_subsampling.restype = None
        lib.gpujpeg_image_set_default_parameters.argtypes = [
            ctypes.POINTER(_ImageParameters),
        ]
        lib.gpujpeg_image_set_default_parameters.restype = None
        lib.gpujpeg_encoder_create.argtypes = [ctypes.c_void_p]  # cudaStream_t
        lib.gpujpeg_encoder_create.restype = ctypes.c_void_p
        lib.gpujpeg_encoder_input_set_image.argtypes = [
            ctypes.POINTER(_EncoderInput), ctypes.c_void_p,
        ]
        lib.gpujpeg_encoder_input_set_image.restype = None
        lib.gpujpeg_encoder_encode.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(_Parameters),
            ctypes.POINTER(_ImageParameters), ctypes.POINTER(_EncoderInput),
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t),
        ]
        lib.gpujpeg_encoder_encode.restype = ctypes.c_int
        lib.gpujpeg_encoder_destroy.argtypes = [ctypes.c_void_p]
        lib.gpujpeg_encoder_destroy.restype = ctypes.c_int
        _LIB = lib
    return _LIB


def _library_version(lib: ctypes.CDLL) -> Optional[Tuple[int, ...]]:
    """(major, minor, ...) of the loaded library, None when undeterminable."""
    try:
        lib.gpujpeg_version.argtypes = []
        lib.gpujpeg_version.restype = ctypes.c_int
        lib.gpujpeg_version_to_string.argtypes = [ctypes.c_int]
        lib.gpujpeg_version_to_string.restype = ctypes.c_char_p
        raw = lib.gpujpeg_version_to_string(lib.gpujpeg_version())
        text = (raw or b"").decode("ascii", "replace")
        parts = tuple(int(p) for p in text.strip().split(".") if p.isdigit())
        return parts or None
    except Exception:
        return None


class GpuJpegEncoder:
    """One reusable GPUJPEG encoder; same interface as NvJpegEncoder.

    Parameter structs are initialized once by the library and only the
    per-call fields (quality, width, height) are touched afterwards.
    The output buffer returned by ``gpujpeg_encoder_encode`` is owned by
    the encoder and valid only until the next encode, so it is copied
    out under the lock.
    """

    def __init__(self) -> None:
        self._lib = _gpujpeg()
        self._lock = threading.Lock()
        if self._lib.gpujpeg_init_device(0, 0) != 0:
            raise GpuJpegError("gpujpeg_init_device(0) failed — no usable CUDA device")
        self._enc = self._lib.gpujpeg_encoder_create(None)
        if not self._enc:
            raise GpuJpegError("gpujpeg_encoder_create failed")
        self._param = _Parameters()
        self._lib.gpujpeg_set_default_parameters(ctypes.byref(self._param))
        self._param.interleaved = 1
        self._lib.gpujpeg_parameters_chroma_subsampling(
            ctypes.byref(self._param), _GPUJPEG_SUBSAMPLING_420
        )
        self._img = _ImageParameters()
        self._lib.gpujpeg_image_set_default_parameters(ctypes.byref(self._img))
        self._img.color_space = _GPUJPEG_RGB
        self._img.pixel_format = _GPUJPEG_444_U8_P012
        self._img.width_padding = 0

    def encode(self, arr: np.ndarray, quality: int) -> bytes:
        """Encode a uint8 HxWx3 RGB frame to baseline JPEG bytes.

        Raises :class:`GpuJpegError` on failure — the caller
        (``node/jpeg.py``) owns falling back to a CPU encoder.
        """
        if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
            raise GpuJpegError(f"expected uint8 HxWx3 RGB, got {arr.dtype} {arr.shape}")
        arr = np.ascontiguousarray(arr)  # no-op on the capture path
        with self._lock:
            self._param.quality = int(quality)
            self._img.width = int(arr.shape[1])
            self._img.height = int(arr.shape[0])
            inp = _EncoderInput()
            self._lib.gpujpeg_encoder_input_set_image(
                ctypes.byref(inp), arr.ctypes.data
            )
            out = ctypes.c_void_p()
            out_size = ctypes.c_size_t(0)
            rc = self._lib.gpujpeg_encoder_encode(
                self._enc, ctypes.byref(self._param), ctypes.byref(self._img),
                ctypes.byref(inp), ctypes.byref(out), ctypes.byref(out_size),
            )
            if rc != 0 or not out or out_size.value == 0:
                raise GpuJpegError(f"gpujpeg_encoder_encode failed (rc={rc})")
            return ctypes.string_at(out, out_size.value)

    def close(self) -> None:
        with self._lock:
            if self._enc:
                self._lib.gpujpeg_encoder_destroy(self._enc)
                self._enc = None

    def __del__(self) -> None:  # pragma: no cover - interpreter teardown
        try:
            self.close()
        except Exception:
            pass


def _probe_frame() -> np.ndarray:
    """32x32 RGB with strongly asymmetric channels (R~217, G=90, B=10)."""
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    arr[:, :, 0] = np.linspace(180, 255, 32, dtype=np.uint8)[None, :]
    arr[:, :, 1] = 90
    arr[:, :, 2] = 10
    return arr


def probe() -> Optional[GpuJpegEncoder]:
    """A working encoder, or None when this box can't run GPUJPEG.

    Beyond the load/device/create checks, the test encode is decoded
    via PIL (when available) and its channel means verified — a struct
    layout mismatch from an unpinned build shows up as garbled
    parameters (wrong color space / dimensions), and this catches it at
    resolve time instead of shipping green robots.
    """
    enc: Optional[GpuJpegEncoder] = None
    try:
        lib = _gpujpeg()
        version = _library_version(lib)
        if version is not None and version[:2] < _MIN_VERSION:
            raise GpuJpegError(
                f"GPUJPEG {'.'.join(map(str, version))} predates the pinned "
                f"struct ABI (>= {'.'.join(map(str, _MIN_VERSION))}); "
                f"rebuild v0.27.x"
            )
        enc = GpuJpegEncoder()
        src = _probe_frame()
        data = enc.encode(src, 90)
        if not data.startswith(b"\xff\xd8"):
            raise GpuJpegError(f"probe encode returned non-JPEG ({len(data)} bytes)")
        try:
            import io

            from PIL import Image

            dec = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
            if dec.shape != src.shape:
                raise GpuJpegError(
                    f"probe decode shape {dec.shape} != {src.shape}"
                )
            for c in range(3):
                delta = abs(float(dec[:, :, c].mean()) - float(src[:, :, c].mean()))
                if delta > 25.0:
                    raise GpuJpegError(
                        f"probe channel {c} mean off by {delta:.0f} — "
                        f"color order / struct layout mismatch"
                    )
        except ImportError:
            pass  # no PIL on this node — magic check has to do
        if version is not None:
            _LOG.debug("gpujpeg %s probe ok", ".".join(map(str, version)))
        return enc
    except Exception:
        _LOG.debug("gpujpeg probe failed", exc_info=True)
        if enc is not None:
            try:
                enc.close()
            except Exception:
                pass
        return None
