"""Capability-adaptive JPEG encoding for the node capture path (ADR 0023).

One resolver, three backends, fastest available wins:

1. **PyTurboJPEG** (libjpeg-turbo — SIMD/NEON, several times faster than
   PIL; install via ``interlatent[turbo]`` plus the system libturbojpeg).
2. **OpenCV** ``imencode`` (releases the GIL during encode).
3. **PIL** (always-works fallback).

The backend is resolved once per process and logged, so a node operator
can see from the log which encoder their hardware ended up with. The
same interface runs on an RPi, a Jetson, or an x86 box — only the
throughput changes. A CUDA JPEG path (nvJPEG / GPUJPEG) is a documented
later optimization, not a v1 dependency — see ADR 0023.

All inputs are uint8 arrays, HW (mono) or HWC with C in {1, 3}, **RGB**
channel order (the capture path's native order). Color-order is the
historical bug class here: cv2 wants BGR, turbojpeg wants an explicit
pixel-format flag, PIL wants RGB — each branch below handles its own
conversion and the cross-backend parity test pins them to each other.
"""
from __future__ import annotations

import io
import logging
from typing import Any, Optional, Tuple

import numpy as np

_LOG = logging.getLogger(__name__)

# Resolved lazily on first encode: (name, handle). ``handle`` is the
# TurboJPEG instance / the cv2 module / None for PIL.
_BACKEND: Optional[Tuple[str, Any]] = None


def _resolve_backend() -> Tuple[str, Any]:
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    try:
        from turbojpeg import TurboJPEG  # type: ignore

        handle = TurboJPEG()  # raises OSError when libturbojpeg is absent
        _BACKEND = ("turbojpeg", handle)
    except Exception:
        try:
            import cv2  # type: ignore

            _BACKEND = ("cv2", cv2)
        except Exception:
            try:
                from PIL import Image  # noqa: F401

                _BACKEND = ("pil", None)
            except Exception:
                _BACKEND = ("none", None)
    _LOG.info("node JPEG encoder backend: %s", _BACKEND[0])
    return _BACKEND


def backend_name() -> str:
    """The resolved encoder backend ("turbojpeg" | "cv2" | "pil" | "none")."""
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
