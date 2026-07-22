"""Capability-adaptive JPEG encoding for the node capture path (ADR 0023).

One resolver, five backends, fastest available wins:

1. **nvJPEG** (CUDA toolkit's libnvjpeg via ``node/nvjpeg.py`` — x86
   CUDA boxes only; JetPack ships no CUDA nvJPEG. SDK ADR 0019.)
2. **GPUJPEG** (CESNET, CUDA-SM encode via ``node/gpujpeg.py`` — the
   GPU path on Jetson, where the operator builds libgpujpeg from
   source; probed only when nvJPEG is absent.)
3. **PyTurboJPEG** (libjpeg-turbo — SIMD/NEON, several times faster than
   PIL; install via ``interlatent[turbo]`` plus the system libturbojpeg).
4. **OpenCV** ``imencode`` (releases the GIL during encode).
5. **PIL** (always-works fallback).

A GPU backend takes RGB frames at or above the routing threshold; mono
and small frames stay on the CPU chain, where the fixed per-call GPU
cost would dominate.

The backend is resolved once per process and logged, so a node operator
can see from the log which encoder their hardware ended up with. The
same interface runs on an RPi, a Jetson, or an x86 box — only the
throughput changes.

``INTERLATENT_JPEG_BACKEND`` (``auto`` | ``nvjpeg`` | ``gpujpeg`` |
``turbojpeg`` | ``cv2`` | ``pil``) starts the chain at the named
backend — an ops kill-switch for a misbehaving encoder in the field. A
forced backend that fails to probe logs a WARNING and falls through to
the rest of the chain: the node must never end up encoder-less over a
typo. ``INTERLATENT_GPU_JPEG_MIN_PIXELS`` tunes the GPU routing
threshold (``INTERLATENT_NVJPEG_MIN_PIXELS`` accepted as an alias).

All inputs are uint8 arrays, HW (mono) or HWC with C in {1, 3}, **RGB**
channel order (the capture path's native order). Color-order is the
historical bug class here: cv2 wants BGR, turbojpeg wants an explicit
pixel-format flag, PIL wants RGB, nvJPEG wants an input-format enum —
each branch below handles its own conversion and the cross-backend
parity test pins them to each other.
"""
from __future__ import annotations

import io
import logging
from typing import Any, Optional, Tuple

import numpy as np

_LOG = logging.getLogger(__name__)

# Resolved lazily on first encode: (name, handle). ``handle`` is the
# NvJpegEncoder / TurboJPEG instance / the cv2 module / None for PIL.
_BACKEND: Optional[Tuple[str, Any]] = None
# Best CPU encoder, resolved alongside nvjpeg: small/mono frames and
# per-call nvjpeg failures still want turbojpeg, not PIL.
_CPU_BACKEND: Optional[Tuple[str, Any]] = None

_GPU_BACKENDS = ("nvjpeg", "gpujpeg")
_CPU_CHAIN = ("turbojpeg", "cv2", "pil")
_BACKEND_CHOICES = ("auto",) + _GPU_BACKENDS + _CPU_CHAIN

# Frames below this pixel area (post-resize) stay on the CPU chain even
# when a GPU backend is resolved: the per-call GPU cost (H2D copy +
# launch + sync + bitstream retrieve) is ~fixed while CPU encode cost
# scales with area. 150k pixels splits the real frame classes with ~2x
# margin each side — preview tee 320x240 ≈ 77k and inference uplink
# 256² ≈ 65k stay CPU; recording frames ≥ 640x480 = 307k go GPU.
_NVJPEG_MIN_PIXELS_DEFAULT = 150_000
_NVJPEG_MIN_PIXELS: Optional[int] = None

# One-shot: the first per-call GPU-encode failure is field-visible
# (WARNING), the rest are debug — mirrors _FRAMELESS_WARNED in
# node/control.py.
_NVJPEG_WARNED = False


def _nvjpeg_min_pixels() -> int:
    global _NVJPEG_MIN_PIXELS
    if _NVJPEG_MIN_PIXELS is None:
        import os

        try:
            _NVJPEG_MIN_PIXELS = int(
                os.environ.get("INTERLATENT_GPU_JPEG_MIN_PIXELS", "")
                or os.environ.get("INTERLATENT_NVJPEG_MIN_PIXELS", "")
                or _NVJPEG_MIN_PIXELS_DEFAULT
            )
        except (TypeError, ValueError):
            _NVJPEG_MIN_PIXELS = _NVJPEG_MIN_PIXELS_DEFAULT
    return _NVJPEG_MIN_PIXELS


def _env_backend() -> str:
    import os

    val = (os.environ.get("INTERLATENT_JPEG_BACKEND", "") or "auto").strip().lower()
    if val not in _BACKEND_CHOICES:
        _LOG.warning(
            "Ignoring INTERLATENT_JPEG_BACKEND=%r (accepted: %s)",
            val, "|".join(_BACKEND_CHOICES),
        )
        return "auto"
    return val


def _try_cpu(name: str) -> Optional[Tuple[str, Any]]:
    if name == "turbojpeg":
        try:
            from turbojpeg import TurboJPEG  # type: ignore

            return ("turbojpeg", TurboJPEG())  # OSError when lib is absent
        except Exception:
            return None
    if name == "cv2":
        try:
            import cv2  # type: ignore

            return ("cv2", cv2)
        except Exception:
            return None
    if name == "pil":
        try:
            from PIL import Image  # noqa: F401

            return ("pil", None)
        except Exception:
            return None
    return None


def _resolve_cpu_backend(start: str = "turbojpeg") -> Tuple[str, Any]:
    """Best available CPU encoder from ``start`` down; cached after the
    first resolution (later calls ignore ``start`` — the cached choice
    already honors any env-forced exclusion)."""
    global _CPU_BACKEND
    if _CPU_BACKEND is None:
        chain = _CPU_CHAIN[_CPU_CHAIN.index(start):] if start in _CPU_CHAIN else _CPU_CHAIN
        _CPU_BACKEND = next(
            (got for got in map(_try_cpu, chain) if got), ("none", None)
        )
    return _CPU_BACKEND


def _probe_gpu(name: str) -> Optional[Any]:
    """Probe one GPU backend module by name; None when unusable."""
    if name == "nvjpeg":
        from . import nvjpeg as mod  # lazy: keep import cost off module load
    else:
        from . import gpujpeg as mod
    return mod.probe()


def _resolve_backend() -> Tuple[str, Any]:
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    choice = _env_backend()
    gpu_order = _GPU_BACKENDS if choice == "auto" else (
        (choice,) if choice in _GPU_BACKENDS else ()
    )
    for gpu_name in gpu_order:
        enc = _probe_gpu(gpu_name)
        if enc is not None:
            _BACKEND = (gpu_name, enc)
            # Resolve the CPU sidecar eagerly so the one-shot log names
            # the real fallback the session will use for small frames.
            cpu_name, _ = _resolve_cpu_backend()
            _LOG.info(
                "node JPEG encoder backend: %s (cpu fallback: %s)",
                gpu_name, cpu_name,
            )
            return _BACKEND
    if choice in _GPU_BACKENDS:
        _LOG.warning(
            "INTERLATENT_JPEG_BACKEND=%s but no usable GPU JPEG encoder "
            "(no GPU, missing library, or probe failure); falling back to "
            "the CPU encoder chain", choice,
        )
    start = choice if choice in _CPU_CHAIN else "turbojpeg"
    _BACKEND = _resolve_cpu_backend(start=start)
    _LOG.info("node JPEG encoder backend: %s", _BACKEND[0])
    return _BACKEND


def backend_name() -> str:
    """The resolved backend name (one of ``_BACKEND_CHOICES`` sans "auto",
    or "none")."""
    return _resolve_backend()[0]


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Squeeze a trailing 1-channel axis; ensure C-contiguous uint8."""
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[:, :, 0]
    return np.ascontiguousarray(arr)


def _target_dims(
    h: int, w: int, target_size: Optional[int], max_dim: Optional[int],
) -> Tuple[int, int]:
    """(new_w, new_h) after the requested resize; (w, h) when unchanged.

    ``target_size`` is a square resize (aspect deliberately not
    preserved — the downstream image processor's resize is square too,
    see ``_jpeg_encode`` in node/control.py). ``max_dim`` caps the
    longest side, preserving aspect (the preview tee).
    """
    if target_size is not None and target_size > 0:
        return int(target_size), int(target_size)
    if max_dim is not None and max_dim > 0 and max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        return max(1, int(w * scale)), max(1, int(h * scale))
    return w, h


def _resize(arr: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    """Resize RGB/mono uint8; cv2 INTER_AREA when available, else PIL."""
    try:
        import cv2  # type: ignore

        return cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    except Exception:
        from PIL import Image

        return np.asarray(
            Image.fromarray(arr).resize((new_w, new_h), Image.BILINEAR)
        )


def _encode_pil(arr: np.ndarray, quality: int) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def encode_jpeg(
    arr: np.ndarray,
    quality: int = 85,
    target_size: Optional[int] = None,
    max_dim: Optional[int] = None,
) -> Optional[bytes]:
    """Encode an RGB (HWC) or mono (HW) uint8 frame to JPEG bytes.

    Returns None when no encoder is available or every backend failed —
    callers on the capture path skip the frame rather than crash the
    control loop. ``target_size``/``max_dim`` pre-resize before encoding
    (see :func:`_target_dims`).
    """
    try:
        arr = _normalize(arr)
        h, w = int(arr.shape[0]), int(arr.shape[1])
        new_w, new_h = _target_dims(h, w, target_size, max_dim)
        if (new_w, new_h) != (w, h):
            arr = _normalize(_resize(arr, new_w, new_h))

        name, handle = _resolve_backend()
        if name in _GPU_BACKENDS:
            if arr.ndim == 3 and arr.shape[0] * arr.shape[1] >= _nvjpeg_min_pixels():
                try:
                    return handle.encode(arr, int(quality))
                except Exception:
                    global _NVJPEG_WARNED
                    if not _NVJPEG_WARNED:
                        _NVJPEG_WARNED = True
                        _LOG.warning(
                            "%s encode failed; using the CPU fallback "
                            "(further %s errors logged at debug)",
                            name, name, exc_info=True,
                        )
                    else:
                        _LOG.debug("%s encode failed; falling back", name, exc_info=True)
            # Mono frames, sub-threshold frames, and GPU-encode failures
            # all take the best CPU encoder resolved alongside the GPU
            # backend.
            name, handle = _resolve_cpu_backend()
        if name == "turbojpeg":
            try:
                from turbojpeg import TJPF_GRAY, TJPF_RGB  # type: ignore

                pf = TJPF_GRAY if arr.ndim == 2 else TJPF_RGB
                return handle.encode(arr, quality=quality, pixel_format=pf)
            except Exception:
                _LOG.debug("turbojpeg encode failed; falling back", exc_info=True)
        if name in ("turbojpeg", "cv2"):
            try:
                import cv2  # type: ignore

                img = arr
                if img.ndim == 3:
                    # cvtColor over a numpy flip: releases the GIL and is
                    # faster than the fancy-index copy on the control thread.
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                ok, buf = cv2.imencode(
                    ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)],
                )
                if ok:
                    return bytes(buf)
            except Exception:
                _LOG.debug("cv2 encode failed; falling back", exc_info=True)
        if name != "none":
            try:
                return _encode_pil(arr, int(quality))
            except Exception:
                _LOG.debug("PIL encode failed", exc_info=True)
        return None
    except Exception:
        _LOG.debug("encode_jpeg failed", exc_info=True)
        return None
