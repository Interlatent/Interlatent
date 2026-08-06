"""Pin the video codec across lerobot's three encoder-config generations.

Both dataset writers (:mod:`lerobot_rebuild`, :mod:`lerobot_live`) need to
force **H.264** rather than take lerobot's ``libsvtav1`` default: this code
runs inside gVisor-sandboxed containers, where SVT-AV1 tries to set
worker-thread scheduling priority (``pthread_setschedparam``), gVisor
rejects it as EINVAL, and the encoder then stalls badly — an upload that
never finishes. libx264 has no such step. H.264 is also the platform
working codec (ADR 0021).

How that codec is requested has been renamed twice upstream, and
``LeRobotDataset.create()`` takes no ``**kwargs``, so passing the wrong one
is a hard ``TypeError``:

===========================  ====================  =========================
lerobot                      parameter             value
===========================  ====================  =========================
v0.4.4 – v0.5.1              ``vcodec``            the codec name
2026-05-14 (``bd9619df``)    ``camera_encoder``    ``VideoEncoderConfig``
2026-06-27 (``3dd19d04``),   ``rgb_encoder``       ``RGBEncoderConfig``
v0.6.x                       (+ ``depth_encoder``)
===========================  ====================  =========================

The engine used to pass ``vcodec=`` unconditionally, which meant every
lerobot from 2026-05-14 onward — including the ref ``docker/Dockerfile``
pins — raised ``TypeError`` inside ``build_from_source``. Both writers
catch broadly, so the failure surfaced as "rebuild failed; aborting
upload" and the staged episode was deleted. Detect the parameter instead
of assuming one.

Unknown future API: we log an ERROR and let lerobot pick its own codec.
Recording an AV1 shard is recoverable — the canonical is codec-agnostic
and any reader missing a decoder transcodes locally (ADR 0021) — whereas
raising here would discard the episode, which is not.
"""
from __future__ import annotations

import inspect
import logging
from typing import Any, Callable, Optional

_LOG = logging.getLogger(__name__)

# Emit the "unknown encoder API" error once per process, not once per
# episode: a box on an unsupported lerobot would otherwise log it every
# session for as long as it runs.
_WARNED = False


def _encoder_config(cls_name: str, vcodec: str) -> Any:
    """Instantiate one of lerobot's encoder dataclasses by name.

    ``lerobot.configs`` re-exports them; the submodule is the fallback for
    versions that don't.
    """
    last: Optional[Exception] = None
    for module in ("lerobot.configs", "lerobot.configs.video"):
        try:
            mod = __import__(module, fromlist=[cls_name])
            return getattr(mod, cls_name)(vcodec=vcodec)
        except (ImportError, AttributeError) as exc:  # noqa: PERF203
            last = exc
    raise ImportError(f"cannot import {cls_name} from lerobot.configs") from last


def video_encoder_kwargs(
    create_fn: Callable[..., Any],
    vcodec: Optional[str],
) -> dict[str, Any]:
    """Kwargs that ask ``create_fn`` for ``vcodec``, for this lerobot.

    ``create_fn`` is ``LeRobotDataset.create``; it is passed in rather than
    imported so the caller keeps ownership of the deferred lerobot import
    (and so tests can drive every generation without installing three
    lerobots).

    Returns ``{}`` when ``vcodec`` is falsy — meaning "no preference, take
    lerobot's default" — and also when no known parameter is present, after
    logging an ERROR.
    """
    if not vcodec:
        return {}

    try:
        params = inspect.signature(create_fn).parameters
    except (TypeError, ValueError):  # pragma: no cover — builtins/C funcs
        params = {}

    # Newest first: a version carrying both names is taking the newer one.
    if "rgb_encoder" in params:
        return {"rgb_encoder": _encoder_config("RGBEncoderConfig", vcodec)}
    if "camera_encoder" in params:
        return {"camera_encoder": _encoder_config("VideoEncoderConfig", vcodec)}
    if "vcodec" in params:
        return {"vcodec": vcodec}

    global _WARNED
    if not _WARNED:
        _WARNED = True
        _LOG.error(
            "This lerobot's LeRobotDataset.create() exposes no known video "
            "encoder parameter (looked for rgb_encoder / camera_encoder / "
            "vcodec); recording with its default codec instead of %s. Under "
            "gVisor the default (libsvtav1) can stall the encoder and hang "
            "the upload — pin a supported lerobot or teach "
            "storage/lerobot_codec.py the new parameter. Available: %s",
            vcodec, ", ".join(sorted(params)) or "<unknown>",
        )
    return {}


__all__ = ["video_encoder_kwargs"]
