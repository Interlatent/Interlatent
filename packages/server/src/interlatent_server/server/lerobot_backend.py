"""LeRobot policy adapter.

Single backend that loads any policy supported by lerobot via the
common `PreTrainedConfig.from_pretrained` + `make_policy` path. The
same backend serves SmolVLA, ACT, Diffusion Policy, Pi0, etc. — they
all expose `predict_action_chunk(batch)` and the same observation
dict convention.

To add a model family that LIVES OUTSIDE lerobot (e.g. OpenVLA via
HF transformers, an internal custom policy), drop a new backend file
next to this one and register it under a different name. Nothing in
this file is SmolVLA-specific.

Lazy import: lerobot is an optional engine dependency. The class is
defined unconditionally and registered with `policy_runtime`, but
lerobot is only imported inside `__init__`, so importing this module
on a machine without lerobot does not fail. You get the ImportError
when you actually try to open a session with backend="lerobot".

RTC in-painting:
    For flow-matching policies (SmolVLA / pi0 / pi0.5) we enable
    lerobot's Real-Time Chunking: each new chunk is generated as an
    in-painting problem so it stays continuous with the unexecuted
    tail of the previous chunk, removing the velocity jump at chunk
    boundaries. The backend caches the previous chunk's raw actions
    and feeds the overlapping tail as `prev_chunk_left_over` together
    with the client's measured `inference_delay`. RTC is enabled by
    default (see `_resolve_rtc_request`); set DRTC_DISABLE_RTC=1 to fall
    back to plain chunk concatenation. A legacy DRTC_RTC=0 is ignored.

    Non-flow-matching policies (ACT, etc.) don't support RTC and don't
    need it — their chunks are short enough that boundary
    discontinuities are usually fine. A legacy explicit-kwarg path
    (`inpainting_actions` / `prior_actions`) is kept for any custom
    policy that exposes one.
"""

from __future__ import annotations

import contextlib
import inspect
import logging
import os
import time
from typing import Any, Optional

import numpy as np

from .chunk_seam import crossfade_chunk
from .policy_runtime import register_backend

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chunk-size resolution
# ---------------------------------------------------------------------------
# The number of action steps a policy emits per forward — the chunk width
# the DRTC client must pace its schedule to. Getting it wrong (e.g. pacing
# to 32 against a model that emits 50) desyncs the controller's horizon /
# cooldown math AND records malformed datasets, so we resolve it from the
# model's OWN config rather than guessing.
#
# Preference order: predicted-chunk width (``chunk_size``) first, then the
# execution-horizon fallback (``n_action_steps``), then known aliases.
_CHUNK_SIZE_CONFIG_KEYS = ("chunk_size", "n_action_steps", "action_chunk_size")
_DEFAULT_CHUNK_SIZE = 32

# ---------------------------------------------------------------------------
# torch.compile mode resolution
# ---------------------------------------------------------------------------
# `DRTC_COMPILE_MODE` is the pod's single source of truth. Its default value,
# `auto`, resolves a concrete torch.compile mode from the loaded policy's
# family at session-open (see `_resolve_auto_compile_mode`).
#
# cfg.type strings (the lerobot policy type) known to be cudagraph-SAFE — i.e.
# their `sample_actions` / `forward` graphs can be captured/replayed as CUDA
# graphs without breaking. This set is EMPTY today: flow-matching policies
# (smolvla / pi0 / pi0.5), diffusion, and molmoact2 sample noise from an RNG
# inside the captured region (cudagraph replay would reuse stale noise), and
# ACT has its own capture issues — so every family currently resolves
# `auto -> "default"` (Inductor codegen, NO cudagraphs). Promote a family
# here ONLY after verifying its graph is stable-shape and side-effect-free
# under reduce-overhead's cudagraph trees on a real box. Cudagraphs buy little
# for these compute-bound (large-matmul) policies anyway, so the bar is high.
_CUDAGRAPH_SAFE_TYPES: frozenset = frozenset()


def _resolve_auto_compile_mode(cfg) -> str:
    """Resolve the `auto` compile mode from a policy's `cfg.type`.

    Allowlist-style: a cudagraph-safe family gets the cudagraph-on
    `reduce-overhead`; everything else — unsafe families, unrecognized
    `cfg.type`, or an unclassifiable local-path checkpoint — gets the
    cudagraph-free `default`. New/unknown families are safe-by-default.
    """
    ctype = (getattr(cfg, "type", None) or "").strip().lower()
    return "reduce-overhead" if ctype in _CUDAGRAPH_SAFE_TYPES else "default"


def _read_config_json(policy_uri: str) -> dict:
    """Read ``config.json`` from a checkpoint (local dir or HF repo).

    Pulls only the single config file (never the multi-GB weights), so it
    stays cheap at OpenSession. Returns ``{}`` on any failure. This mirrors
    ``molmoact2_backend._read_checkpoint_config_json`` — kept local to avoid
    a circular import (molmoact2_backend imports LeRobotBackend from here).
    """
    import json
    import os.path as _osp

    try:
        local = _osp.join(_osp.expanduser(policy_uri), "config.json")
        if _osp.isfile(local):
            with open(local) as fh:
                return json.load(fh)
        from huggingface_hub import hf_hub_download

        token = os.environ.get("HF_TOKEN") or os.environ.get("HF_ACCESS_TOKEN")
        path = hf_hub_download(policy_uri, "config.json", token=token)
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        log.warning("Could not read config.json for %s", policy_uri, exc_info=True)
        return {}


def _ensure_hf_auth() -> None:
    """Make sure the ambient HF token is visible to transformers / hub.

    Our own helpers accept the token under either ``HF_TOKEN`` or
    ``HF_ACCESS_TOKEN``, but lerobot's processor load reaches a bare
    ``AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")`` with no
    explicit ``token=`` — it relies on the *ambient* auth, which
    huggingface_hub/transformers read only from ``HF_TOKEN`` /
    ``HUGGINGFACE_HUB_TOKEN``. A box that set only ``HF_ACCESS_TOKEN`` would
    then hit an UNAUTHENTICATED 401 on that GATED repo, the whole processor
    pipeline falls back to identity, and the policy later crashes on a
    missing ``observation.language.tokens`` (pi0 / pi0.5 / SmolVLA tokenize
    in that step). Mirror whichever name is set into the canonical vars so
    the gated tokenizer download is authenticated.

    Note: this only fixes *which token is sent*. The token's account must
    still have ACCEPTED the gated repo's terms
    (https://huggingface.co/google/paligemma-3b-pt-224) — without that, the
    download 401s no matter what.
    """
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HF_ACCESS_TOKEN")
    )
    if not token:
        return
    for var in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        if not os.environ.get(var):
            os.environ[var] = token


def _resolve_chunk_size(requested: int, cfg: Any, policy_uri: str) -> int:
    """Resolve the chunk width, preferring the incoming model's own config.

    Precedence:
      1. an explicit non-zero ``requested`` (client override),
      2. the lerobot-parsed config attribute (``chunk_size`` then
         ``n_action_steps``),
      3. the same keys read DIRECTLY from the checkpoint's ``config.json`` —
         this catches policies whose config class doesn't surface the field
         as a python attribute under the name we expect, which is exactly
         the case that used to fall silently through to the hard default,
      4. the hard default 32, emitted as a WARNING. Pacing the client to 32
         against a model that emits a different width desyncs the controller
         and corrupts the recording, so this must never be silent.
    """
    if requested:
        return int(requested)

    # (2) lerobot's parsed config object — authoritative when it surfaces
    # the field. This is the fast, common path.
    for attr in ("chunk_size", "n_action_steps"):
        val = getattr(cfg, attr, None)
        if val:
            log.info("Resolved chunk_size=%d from cfg.%s for %s", int(val), attr, policy_uri)
            return int(val)

    # (3) Direct config.json read. We only get here when from_pretrained did
    # NOT expose a usable attribute, so re-read the raw file and look for the
    # value under any known key before defaulting.
    raw = _read_config_json(policy_uri)
    for key in _CHUNK_SIZE_CONFIG_KEYS:
        val = raw.get(key)
        if val:
            log.info(
                "Resolved chunk_size=%d from config.json[%r] for %s "
                "(lerobot cfg attribute was absent)", int(val), key, policy_uri,
            )
            return int(val)

    # (4) Genuine fallback — be loud so "in some cases it defaults" is
    # diagnosable rather than invisible.
    log.warning(
        "Could not resolve chunk_size from cfg or config.json for %s — "
        "defaulting to %d. The client will be paced to %d, which may not "
        "match the model's true chunk width. config.json keys present: %s",
        policy_uri, _DEFAULT_CHUNK_SIZE, _DEFAULT_CHUNK_SIZE,
        sorted(raw.keys()) if raw else "(config.json unreadable / empty)",
    )
    return _DEFAULT_CHUNK_SIZE


# ---------------------------------------------------------------------------
# Lerobot RTC device-bug patch
# ---------------------------------------------------------------------------
# ``RTCProcessor._add_leading_ones`` / ``_add_trailing_zeros`` /
# ``_linweights`` (lerobot ≤ 0.5.1) create CPU tensors via ``torch.ones`` /
# ``torch.zeros`` / ``torch.linspace`` without a ``device=`` arg, then
# ``torch.cat`` them with the weights tensor that lives on GPU. Torch
# 2.6+ rejects cross-device cat under cudagraph capture, which on the
# warmup path raises "skipping cudagraphs due to cpu device (cat)" and
# on the real Infer path trips a Dynamo SpeculationLog divergence on
# re-trace. The fix is trivial — match the weights tensor's device when
# building the prefix/trailing chunks. We monkey-patch at import time
# so it lands BEFORE any policy is loaded.
#
# Removable once lerobot lands the upstream fix (PR not filed yet at
# time of writing — same patch is the candidate). Track via grep
# for "device=weights.device" in lerobot/policies/rtc/modeling_rtc.py.
_RTC_PATCH_APPLIED = False


def _patch_lerobot_rtc_device_bug() -> None:
    """Force RTC's prefix/trailing weight tensors onto the same device
    as the working ``weights`` tensor. No-op when lerobot isn't
    installed or the RTCProcessor doesn't have the expected helpers
    (e.g. a future lerobot that already fixed this upstream).
    """
    global _RTC_PATCH_APPLIED
    if _RTC_PATCH_APPLIED:
        return
    try:
        from lerobot.policies.rtc.modeling_rtc import RTCProcessor
    except Exception:
        return  # lerobot not installed, or RTCProcessor moved — nothing to patch
    import torch as _torch

    if not all(
        hasattr(RTCProcessor, attr)
        for attr in ("_add_leading_ones", "_add_trailing_zeros")
    ):
        return  # Upstream renamed/removed these — punt

    def _add_leading_ones(self, weights, start, total):
        ones_len = min(start, total)
        if ones_len <= 0:
            return weights
        ones = _torch.ones(ones_len, device=weights.device, dtype=weights.dtype)
        return _torch.cat([ones, weights])

    def _add_trailing_zeros(self, weights, total, end):
        zeros_len = total - end
        if zeros_len <= 0:
            return weights
        zeros = _torch.zeros(zeros_len, device=weights.device, dtype=weights.dtype)
        return _torch.cat([weights, zeros])

    RTCProcessor._add_leading_ones = _add_leading_ones
    RTCProcessor._add_trailing_zeros = _add_trailing_zeros
    _RTC_PATCH_APPLIED = True
    log.info(
        "Patched lerobot RTCProcessor device bug "
        "(_add_leading_ones / _add_trailing_zeros now respect weights.device)"
    )


# Apply at import. Cheap: a single ImportError-guarded probe + two attr
# assignments. The function is idempotent so re-imports are safe.
_patch_lerobot_rtc_device_bug()


# ---------------------------------------------------------------------------
# Lerobot processor-step registry-rename compatibility
# ---------------------------------------------------------------------------
# lerobot renames processor-step *registry keys* across versions while the
# underlying class stays the same. A checkpoint saved under one lerobot
# serializes the registry key into its ``policy_preprocessor.json`` /
# ``policy_postprocessor.json``; loading it under a lerobot that uses a
# different key for the SAME class fails in ``make_pre_post_processors``
# with ``KeyError: Processor step '<name>' not found in registry``.
#
# Concrete case that bit us: the class ``RelativeActionsProcessorStep`` is
# registered as ``"delta_actions_processor"`` in our pinned ref
# (24017e96, ~v0.5.1) but as ``"relative_actions_processor"`` on lerobot
# main — and ``lerobot/pi05_base``'s preprocessor references the newer
# ``relative_actions_processor``. Same class, pure key rename.
#
# We bridge known renames by aliasing the registry dict: if exactly one
# side of a rename pair is registered, point the missing key at the same
# class. Bidirectional so it's correct whether the pod's lerobot is older
# or newer than the checkpoint. Done via direct ``_registry`` assignment
# (not ``register()``, which raises on duplicate and would also rewrite
# the class's ``_registry_name`` used for re-serialization).
_PROCESSOR_ALIAS_PAIRS = [
    # (older-ref key, newer-main key) — same RelativeActionsProcessorStep class
    ("delta_actions_processor", "relative_actions_processor"),
]
_PROCESSOR_ALIASES_APPLIED = False


def _patch_lerobot_processor_step_aliases() -> None:
    """Alias renamed processor-step registry keys so checkpoints saved
    under a different lerobot version still deserialize. No-op when
    lerobot isn't installed or neither key of a pair is present.
    """
    global _PROCESSOR_ALIASES_APPLIED
    if _PROCESSOR_ALIASES_APPLIED:
        return
    try:
        # Importing the package runs every step module's
        # ``@ProcessorStepRegistry.register(...)`` decorator, so the registry
        # is fully populated before we alias (importing only ``.pipeline``
        # would define the registry but leave the step keys unregistered).
        import lerobot.processor  # noqa: F401
        from lerobot.processor.pipeline import ProcessorStepRegistry
    except Exception:
        return  # lerobot not installed, or registry moved — nothing to patch

    reg = ProcessorStepRegistry._registry
    for a, b in _PROCESSOR_ALIAS_PAIRS:
        if a in reg and b not in reg:
            reg[b] = reg[a]
            log.info("Aliased lerobot processor step %r -> %r", b, a)
        elif b in reg and a not in reg:
            reg[a] = reg[b]
            log.info("Aliased lerobot processor step %r -> %r", a, b)
    _PROCESSOR_ALIASES_APPLIED = True


# Apply at import, after the registry module's own steps have registered
# (importing ProcessorStepRegistry triggers lerobot's step registrations).
_patch_lerobot_processor_step_aliases()


def _resolve_rtc_request() -> tuple[bool, str]:
    """Decide whether to ATTEMPT RTC in-painting, from the environment.

    RTC seam-smoothing is ON by default. Background: provisioned pods used
    to inject ``DRTC_RTC=0`` to dodge a torch-2.6 cudagraph/Dynamo bug in
    lerobot's RTC path. That bug no longer applies — cudagraphs are off
    (``DRTC_COMPILE_MODE=auto`` resolves to ``default`` for every family;
    see ``_resolve_auto_compile_mode``) and the cross-device ``cat`` is
    monkey-patched (``_patch_lerobot_rtc_device_bug``). Leaving RTC off is a
    SAFETY regression for flow-matching policies: a cold re-plan can sample
    a different action mode, and the client's last-write-wins overwrite then
    snaps the robot toward that other mode (observed as a violent jerk).

    Precedence:
      - ``DRTC_DISABLE_RTC`` truthy -> the real off switch (nothing injects
        it), for debugging / a known-bad policy.
      - a legacy ``DRTC_RTC`` set to a falsey value is now IGNORED (with a
        warning) rather than honored: a pod's provision-time env persists
        across restart (RunPod stop+start keeps the env), so honoring the
        stale ``DRTC_RTC=0`` would keep RTC off on the live pod even after a
        restart picks up this code. Ignoring it lets a plain restart
        re-enable seam smoothing without recreating the pod or redeploying
        the backend.
      - otherwise enabled.

    Enabling is safe-by-construction: the session-open warmup self-test
    disables RTC if the in-painting forward throws, and ``forward`` drops to
    plain chunking if RTC fails mid-session — so the worst case is the
    current (RTC-off) behavior, never a hard crash.
    """
    disable = os.environ.get("DRTC_DISABLE_RTC", "").strip().lower()
    if disable in ("1", "true", "on", "yes"):
        return False, "DRTC_DISABLE_RTC set"
    legacy = os.environ.get("DRTC_RTC")
    if legacy is not None and legacy.strip().lower() in ("0", "false", "off", ""):
        return True, (
            f"ignoring deprecated DRTC_RTC={legacy!r} — RTC is on by default; "
            "use DRTC_DISABLE_RTC=1 to disable"
        )
    return True, "default"


def _resolve_crossfade() -> tuple[bool, int]:
    """Whether to seam-cross-fade non-RTC policies, and the ramp width.

    RTC-incompatible policies (ACT and other single-shot decoders — no
    iterative denoiser to in-paint into) can't use RTC, so a fresh cold
    chunk hard-overwrites the imminent steps via the client's
    last-write-wins merge, snapping the robot between two plans. The
    cross-fade blends a new chunk into the previous chunk's overlapping
    tail in robot-action space, giving a continuous seam for ANY policy.
    Only applied when RTC is NOT active, so flow policies are untouched.

    Returns ``(enabled, ramp_steps)``. ``ramp_steps == 0`` means "auto":
    derive the ramp width from the client's measured ``inference_delay``
    per call. Off switch: ``DRTC_DISABLE_CROSSFADE=1``. Override the ramp
    width with ``DRTC_CROSSFADE_STEPS=<n>``.
    """
    if os.environ.get("DRTC_DISABLE_CROSSFADE", "").strip().lower() in (
        "1", "true", "on", "yes"
    ):
        return False, 0
    raw = os.environ.get("DRTC_CROSSFADE_STEPS", "").strip()
    steps = 0
    if raw:
        try:
            steps = max(0, int(raw))
        except ValueError:
            steps = 0
    return True, steps


def _resolve_images_to_device(default: bool) -> bool:
    """Whether ``_to_batch`` uploads camera frames to the GPU as uint8.

    The CPU path costs a full-resolution numpy transpose + float32 cast
    (a 4x memory inflation) per camera per forward, and then hands the
    inflated fp32 tensor to the processor pipeline, which copies it over
    PCIe and immediately resizes it DOWN to the model's native input.
    Uploading the uint8 frame first and doing the transpose / scale on
    the GPU moves that work to the device and cuts the transfer 4x.

    Safe only when the pipeline's image steps are torch ops that accept
    CUDA tensors. That holds for the processors built by
    ``make_pre_post_processors`` (we pass a ``device_processor`` override,
    so everything downstream of it already runs on device), which is why
    the generic backend defaults ON. MolmoAct2 builds its own pipeline
    whose image steps have not been verified against CUDA input, so it
    defaults OFF — see :class:`MolmoAct2Backend`.

    ``DRTC_IMAGES_TO_DEVICE=0|1`` overrides the per-backend default so a
    bad interaction can be switched off (or a fix switched on) on a live
    pod without a redeploy.
    """
    raw = os.environ.get("DRTC_IMAGES_TO_DEVICE", "").strip().lower()
    if raw in ("1", "true", "on", "yes"):
        return True
    if raw in ("0", "false", "off", "no"):
        return False
    return default


# Detailed per-forward latency lines logged at the start of a runtime's
# life. `_warmup` itself goes through `forward`, so #1 (and #2 when RTC is
# on) are synthetic warmup passes carrying the torch.compile cost — the
# real Infers start after those. At ~5 Hz this window is ~2s of session
# start, which is emphatically NOT steady state; see
# `_resolve_latency_sampling` for how the steady state gets measured.
_LAT_DETAIL_FORWARDS = 12


def _resolve_latency_sampling() -> tuple[int, int]:
    """How to measure forward latency once the session-start window closes.

    Honest split timing costs two full `cuda.synchronize()` calls, which
    stall the CPU. Paying that on every forward forever is real overhead;
    paying it never leaves us blind to the steady state — which is the
    only regime that matters for control, and the one the first-12 window
    cannot show (it is warmup + the first ~2s).

    The compromise is SAMPLING: sync-and-time one forward in
    ``sample_every``, and emit a p50/p95 summary once ``summary_every``
    samples have accumulated. At the defaults (1-in-50, summarize every
    10 samples) 2% of forwards pay a sync and a summary lands every ~100s
    at 5 Hz — steady-state visibility for ~nothing.

    Returns ``(sample_every, summary_every)``.
    ``DRTC_LATENCY_SAMPLE_EVERY=0`` disables steady-state sampling (and
    with it the syncs) entirely.
    """
    def _int_env(name: str, default: int) -> int:
        raw = os.environ.get(name, "").strip()
        if not raw:
            return default
        try:
            return max(0, int(raw))
        except ValueError:
            return default

    sample_every = _int_env("DRTC_LATENCY_SAMPLE_EVERY", 50)
    summary_every = _int_env("DRTC_LATENCY_SUMMARY_EVERY", 10) or 1
    return sample_every, summary_every


@register_backend("lerobot")
class LeRobotBackend:
    """Wraps any lerobot policy. The Interlatent DRTC server treats
    every policy uniformly through this adapter."""

    # Whether camera frames go to the GPU as uint8 for on-device
    # transpose/scale. Subclasses whose processor pipeline is CPU-side
    # override this to False. See `_resolve_images_to_device`.
    _images_to_device_default: bool = True

    def _init_runtime_common(
        self, device: Optional[str], dtype: str, default_task: str
    ) -> None:
        """Runtime setup shared by every lerobot-family backend (the
        generic loader and :class:`MolmoAct2Backend`): import torch,
        resolve the device, set dtype + default task. The policy-specific
        load is left to the caller.
        """
        # Heavy import inside the method so this module imports cleanly on
        # environments without torch installed.
        import torch

        self._torch = torch
        # The DRTC server (PolicyRuntime.load) does not plumb `device`
        # through, so it arrives as None — auto-detect, otherwise a GPU
        # container silently runs the entire policy on CPU.
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)
        self._dtype = getattr(torch, dtype)
        self._images_to_device = _resolve_images_to_device(
            type(self)._images_to_device_default
        )
        # Latency instrumentation state. `_lat_i` counts every forward for
        # the life of the runtime (which the PolicyRuntime cache makes
        # longer than one session), `_lat_window` accumulates sampled
        # splits until there are enough to summarize.
        self._lat_sample_every, self._lat_summary_every = _resolve_latency_sampling()
        self._lat_i = 0
        self._lat_window: list[tuple[float, float, float, float, float]] = []
        log.info(
            "%s device=%s images_to_device=%s latency_sampling=1/%s",
            type(self).__name__, self._device, self._images_to_device,
            self._lat_sample_every or "off",
        )
        self._default_task = default_task

    def __init__(
        self,
        chunk_size: int = 0,           # 0 -> use the policy's native value
        action_dim: int = 0,           # 0 -> use the policy's native value
        *,
        policy_uri: str,
        device: Optional[str] = None,   # None -> auto-detect cuda/cpu
        dtype: str = "float32",
        default_task: str = "",
        # Per-session OpenSession.metadata, forwarded verbatim by the
        # transport. MolmoAct2's released-checkpoint path reads camera
        # image keys / norm_tag / inference_action_mode out of it; the
        # generic lerobot path reads ``image_keys`` to reconcile the env's
        # cameras against the policy's own config (see
        # _reconcile_session_cameras) and otherwise ignores it.
        session_metadata: Optional[dict] = None,
        **_: Any,
    ) -> None:
        if not policy_uri:
            raise ValueError("LeRobotBackend requires policy_uri")

        # Shared runtime setup (torch import, device/dtype/default_task).
        # MolmoAct2Backend reuses this too — see molmoact2_backend.py.
        self._init_runtime_common(device, dtype, default_task)
        torch = self._torch

        # Heavy imports happen inside __init__ so this module imports
        # cleanly on environments without lerobot.
        # Order matters: importing the factory first registers every
        # policy's choice class with draccus, so from_pretrained can
        # decode `type: act|smolvla|...` from the policy's config.json.
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors
        try:
            from lerobot.utils.import_utils import register_third_party_plugins
            register_third_party_plugins()
        except Exception:
            pass
        from lerobot.configs.policies import PreTrainedConfig

        cfg = PreTrainedConfig.from_pretrained(policy_uri)
        # `make_policy(cfg)` requires dataset metadata or a sim env
        # because it's also a training-time builder. For inference we
        # go through the policy class's own from_pretrained, which
        # loads weights + normalization buffers from the checkpoint
        # without needing either.
        policy_cls = get_policy_class(cfg.type)
        policy = policy_cls.from_pretrained(policy_uri).to(self._device).eval()

        # --- torch.compile mode -----------------------------------------
        # DRTC_COMPILE_MODE is the single source of truth for how we compile
        # `sample_actions` / `forward`, resolved per loaded policy family at
        # session-open. See `_apply_compile_mode` / `_resolve_auto_compile_mode`
        # for the precedence chain and the cudagraph-safety classification.
        self._apply_compile_mode(torch, policy, cfg)

        # --- RTC (Real-Time Chunking) -----------------------------------
        # RTC turns chunk generation into an in-painting problem: the new
        # chunk is generated to stay continuous with the still-unexecuted
        # tail of the previous one, eliminating the velocity jump at chunk
        # boundaries. lerobot gates RTC on `config.rtc_config`; a stock
        # checkpoint loads with it unset, so enable it here. RTC needs a
        # flow-matching policy (SmolVLA / pi0 / pi0.5) — for others the
        # enable below is a harmless no-op. Disable with DRTC_DISABLE_RTC=1.
        self._rtc_ok = False
        rtc_requested, rtc_reason = _resolve_rtc_request()
        # Remembered so the post-warmup summary can tell "RTC was asked for but
        # the self-test forward disabled it" (a real RTC-can't-run signal)
        # apart from "RTC was never requested".
        self._rtc_requested = rtc_requested
        log.info("RTC requested=%s (%s)", rtc_requested, rtc_reason)
        if rtc_requested:
            try:
                from lerobot.policies.rtc.configuration_rtc import RTCConfig
                if getattr(policy.config, "rtc_config", None) is None:
                    policy.config.rtc_config = RTCConfig(enabled=True)
                else:
                    policy.config.rtc_config.enabled = True
                init_rtc = getattr(policy, "init_rtc_processor", None)
                if init_rtc is not None:
                    init_rtc()
                rtc_enabled = getattr(policy, "_rtc_enabled", None)
                self._rtc_ok = bool(rtc_enabled()) if rtc_enabled else False
            except Exception:
                log.warning(
                    "Could not enable RTC; running without in-painting "
                    "(chunk boundaries may be discontinuous)", exc_info=True,
                )
        # Whether RTC actually engaged at init (the policy supports it and
        # init_rtc_processor succeeded), captured BEFORE the warmup self-test
        # may flip _rtc_ok off. Lets the post-warmup summary tell "policy
        # doesn't support RTC" (e.g. ACT) apart from "RTC engaged but the
        # self-test forward threw".
        self._rtc_enabled_at_init = self._rtc_ok
        log.info("RTC in-painting %s", "enabled" if self._rtc_ok else "disabled")
        # The policy's processor pipeline owns normalization, image
        # transforms, language tokenization, and device placement.
        # Passing `pretrained_path=` loads the *saved* processor config
        # from the checkpoint — no dataset stats or sim env needed
        # (this mirrors the proven lerobot async PolicyServer path).
        # Skipping it breaks VLAs outright: SmolVLA reads
        # `observation.language.tokens`, produced only by the tokenizer
        # step, and non-VLA policies would run unnormalized.
        # `padding="max_length"` on the tokenizer keeps
        # `observation.language.tokens` at a constant shape regardless
        # of the task string. SmolVLA's default is `pad_language_to=
        # "longest"`, which makes seq_len depend on the per-session
        # task — under torch.compile that triggers a full
        # Dynamo+Inductor recompile every time the task changes (a
        # multi-minute stall right when the robot connects). PI0 /
        # PI0.5 already hardcode `padding="max_length"` for the same
        # reason. The cost is attention over padded positions
        # (masked, ~no quality impact), which is dwarfed by the
        # avoided recompile.
        #
        # That override names a `tokenizer_processor` step, which only
        # tokenizing VLAs (SmolVLA / PI0) have. MolmoAct2 builds its own
        # image/video/action processor pipeline with no such step, so
        # passing the override raises a KeyError. Try with it first to
        # keep the SmolVLA recompile fix, then retry without it before
        # giving up on the identity fallback — which the comment above
        # notes breaks any VLA outright.
        # Use the resolved device (the `device` kwarg may be None when the
        # caller relies on auto-detect — _init_runtime_common resolved it).
        device_only = {"device_processor": {"device": str(self._device)}}
        override_attempts = (
            {**device_only, "tokenizer_processor": {"padding": "max_length"}},
            device_only,
        )
        # Tokenizing VLAs pull their tokenizer from a (gated) HF repo —
        # e.g. pi0/pi0.5's `tokenizer_processor` fetches
        # `google/paligemma-3b-pt-224`. Make sure the box's HF token is in
        # the canonical env var the hub download reads, or that fetch 401s
        # and the whole pipeline silently degrades to identity (no
        # tokenization -> a later `observation.language.tokens` KeyError).
        _ensure_hf_auth()
        self._pre = self._post = None
        for i, pre_overrides in enumerate(override_attempts):
            try:
                self._pre, self._post = make_pre_post_processors(
                    cfg,
                    pretrained_path=policy_uri,
                    preprocessor_overrides=pre_overrides,
                    postprocessor_overrides=device_only,
                )
                log.info(
                    "Loaded policy processor pipeline from %s (%s)",
                    policy_uri,
                    "tokenizer override" if i == 0 else "no tokenizer override",
                )
                break
            except Exception:
                log.warning(
                    "make_pre_post_processors attempt %d/%d failed for %s",
                    i + 1, len(override_attempts), policy_uri, exc_info=True,
                )
        if self._pre is None:
            log.warning(
                "Could not load processors from %s — falling back to "
                "identity. Normalization + language tokenization will be "
                "missing; VLA policies will fail at the first forward with a "
                "missing observation.language.tokens. The usual cause is a "
                "GATED tokenizer repo (pi0/pi0.5 fetch "
                "google/paligemma-3b-pt-224): set HF_TOKEN on the box AND "
                "accept the repo's terms at "
                "https://huggingface.co/google/paligemma-3b-pt-224. See the "
                "attempt tracebacks above for the real error.", policy_uri,
            )
            self._pre = lambda b: b
            self._post = lambda x: x

        self.policy = policy
        self.cfg = cfg
        self.chunk_size = _resolve_chunk_size(chunk_size, cfg, policy_uri)
        action_feature = cfg.output_features.get("action")
        self.action_dim = action_dim or (action_feature.shape[0] if action_feature else 6)

        # Pick which inference method to call. Newer policies expose
        # predict_action_chunk; older / non-chunking ones only have
        # select_action and we synthesize a chunk by calling it
        # repeatedly (each call rolls one action forward).
        self._predict_chunk = getattr(policy, "predict_action_chunk", None)
        self._select_action = getattr(policy, "select_action", None)
        if self._predict_chunk is None and self._select_action is None:
            raise RuntimeError(
                f"policy {type(policy).__name__} exposes neither "
                "predict_action_chunk nor select_action"
            )

        # Legacy fallback: some non-lerobot policies expose an explicit
        # named in-painting kwarg. lerobot's RTC kwargs (prev_chunk_left_over,
        # inference_delay) are absorbed by predict_action_chunk's **kwargs
        # so they never show up here — they're handled via self._rtc_ok.
        self._inpainting_kw: Optional[str] = None
        if self._predict_chunk is not None and not self._rtc_ok:
            sig = inspect.signature(self._predict_chunk)
            for cand in ("inpainting_actions", "prior_actions", "rtc_inpainting"):
                if cand in sig.parameters:
                    self._inpainting_kw = cand
                    break

        # RTC state: the raw (pre-postprocessor) actions of the most
        # recent chunk and the absolute step it started at. The next
        # forward feeds the still-unexecuted tail of this as RTC's
        # `prev_chunk_left_over`. Reset again after warmup so synthetic
        # warmup actions never leak into the first real chunk.
        self._last_raw: Optional[np.ndarray] = None
        self._last_start: int = 0

        # Cross-fade state (the non-RTC seam smoother): the most recent
        # POST-processed (robot-space) chunk + its absolute start step. For
        # policies that can't do RTC (ACT and other single-shot decoders), the
        # next chunk is blended against the overlapping tail of this one so the
        # client's last-write-wins overwrite doesn't snap the robot between two
        # cold plans. Robot-space (not raw) so continuity is enforced in the
        # joint units the robot actually executes, and so it survives any
        # delta->absolute step in the postprocessor. See `_crossfade_chunk`.
        self._crossfade_requested, self._crossfade_steps_cfg = _resolve_crossfade()
        self._last_processed: Optional[np.ndarray] = None
        self._last_processed_start: int = 0

        # Snapshot the keys the policy expects so we can build a clean
        # batch from whatever the client sent.
        self._expected_keys: tuple[str, ...] = tuple(cfg.input_features.keys())
        log.info(
            "LeRobotBackend loaded policy=%s chunk_size=%d action_dim=%d "
            "inpainting_kw=%s expected_keys=%s",
            policy_uri, self.chunk_size, self.action_dim,
            self._inpainting_kw, self._expected_keys,
        )

        # Reconcile the cameras the caller will stream against the cameras
        # the policy's own config declares — fail loudly on a real
        # disagreement before we compile against a spec the robot won't
        # satisfy. (No-op when the caller supplied no image_keys: the policy
        # is self-describing and _warmup builds the obs straight from
        # cfg.input_features.)
        self._reconcile_session_cameras(session_metadata)

        # Pay the torch.compile cost now, at session-open, with a
        # synthetic forward — so the first real Infer is already fast.
        self._warmup()

        # Definitive RTC verdict for this session. Distinguishes the cases so
        # logs answer "can RTC run here?" at a glance:
        #   - not requested -> intentionally off
        #   - requested but policy doesn't support RTC (e.g. ACT) -> expected,
        #     not a failure
        #   - requested, engaged, survived warmup -> RTC verified, seam on
        #   - requested, engaged, but warmup self-test disabled it -> RTC
        #     CANNOT run for this policy/torch combo; this is the signal we
        #     want when evaluating whether RTC is viable
        if not self._rtc_requested:
            log.info("RTC disabled by configuration for %s", policy_uri)
        elif not self._rtc_enabled_at_init:
            log.info(
                "RTC not supported by policy %s (family %r exposes no RTC "
                "processor) — plain chunking, as expected for non-flow policies",
                policy_uri, getattr(cfg, "type", "?"),
            )
        elif self._rtc_ok:
            log.info(
                "RTC self-test PASSED for %s — in-painting active (seam "
                "smoothing on)", policy_uri,
            )
        else:
            log.warning(
                "RTC self-test FAILED for %s — RTC engaged at init but a warmup "
                "forward disabled it, so this session runs plain chunking "
                "(discontinuous seams). RTC is NOT viable for this policy/torch "
                "combo as-is; see the warmup traceback above.",
                policy_uri,
            )

        # Discard warmup's raw actions: the first real chunk must be a
        # clean cold start, not in-painted/blended against synthetic data.
        self.reset_session_state()
        if self._crossfade_requested and not self._rtc_ok:
            log.info(
                "Seam cross-fade ENABLED for %s (non-RTC policy) — new chunks "
                "blend into the previous chunk's overlapping tail (ramp=%s)",
                policy_uri,
                self._crossfade_steps_cfg or "auto(inference_delay)",
            )

    # ------------------------------------------------------------------

    def reset_session_state(self) -> None:
        """Clear every trail that must not cross a session boundary.

        Backends are cached process-wide and reused across sessions, so this
        is what stops one episode's tail from stitching into the next one's
        first chunk. The cross-fade pair (``_last_processed*``) was ABSENT
        from the attribute-poke this replaces; it survived only because
        ``_crossfade_chunk`` guards on ``offset < 0``, which happens to hold
        when a new session numbers below the old session's high-water step —
        a coincidence, not a check.
        """
        self._last_raw = None
        self._last_start = 0
        self._last_processed = None
        self._last_processed_start = 0

    @staticmethod
    def _latency_gate(lat_i: int, sample_every: int) -> tuple[bool, bool]:
        """``(detail, sampled)`` for forward number ``lat_i`` (0-based).

        ``detail`` -> log a per-forward line (session start). ``sampled``
        -> feed the steady-state summary. Either one means the forward is
        timed and pays the cuda syncs; neither means it runs sync-free.
        The two are mutually exclusive, so ``lat_i=0`` (which is both in
        the detail window and divisible by ``sample_every``) isn't counted
        twice.
        """
        detail = lat_i < _LAT_DETAIL_FORWARDS
        sampled = (
            not detail and sample_every > 0 and lat_i % sample_every == 0
        )
        return detail, sampled

    def _log_latency_summary(self) -> None:
        """Emit p50/p95 over the sampled forwards, then reset the window.

        This is the steady-state number — the first-12 detail lines are
        warmup and session start. `n` is deliberately in the line: with a
        small window p95 is close to max, and a reader needs to know that
        before treating it as a tail estimate.
        """
        window = self._lat_window
        if not window:
            return
        self._lat_window = []
        arr = np.asarray(window, dtype=np.float64)  # (n, 5)
        p50 = np.percentile(arr, 50, axis=0)
        p95 = np.percentile(arr, 95, axis=0)
        log.info(
            "DRTC-DEBUG latency steady-state | n=%d sampled 1/%d | "
            "total p50=%.0fms p95=%.0fms | to_batch p50=%.0f | "
            "pre p50=%.0f p95=%.0f | fwd p50=%.0f p95=%.0f | post p50=%.0f "
            "| num_inf_steps=%s chunk=%d",
            len(window), self._lat_sample_every,
            p50[0], p95[0], p50[1], p50[2], p95[2], p50[3], p95[3], p50[4],
            getattr(self.cfg, "num_inference_steps", "?"),
            self.chunk_size,
        )

    def _images_on_device(self) -> bool:
        """Whether `_to_batch` uploads camera frames as uint8 and does the
        transpose/scale on the device.

        Requires both the per-backend/env opt-in and an actual CUDA
        device — on a CPU box the numpy path is equivalent and there is no
        transfer to shrink.
        """
        return bool(self._images_to_device and self._device.type == "cuda")

    def _to_batch(self, observation: dict | np.ndarray) -> dict:
        """Turn the decoded observation into the (B=1) batch dict the
        policy expects.

        We are forgiving about what the client sends:
          - dict keys are passed through as-is when they match the
            policy's `input_features` schema
          - missing image keys are silently skipped (some policies
            tolerate this; some don't — the policy will error if so)
          - the `task` key, if present as a 0-d numpy array of dtype
            str, becomes a Python list of length 1
        """
        torch = self._torch
        if not isinstance(observation, dict):
            # Single-array path: try to map to observation.state.
            observation = {"observation.state": np.asarray(observation)}

        batch: dict[str, Any] = {}
        for key, value in observation.items():
            if key == "task":
                task = value.item() if hasattr(value, "item") else str(value)
                batch["task"] = [task] if isinstance(task, str) else list(task)
                continue
            arr = np.asarray(value)
            if arr.dtype == np.uint8:
                # Image -> (B, C, H, W) float in [0,1]. lerobot's image
                # convention is CHW everywhere; the processor pipeline
                # resizes + normalizes from there, but expects CHW input
                # (feeding HWC makes it resize H/C as if they were H/W).
                if self._images_on_device():
                    # Upload the uint8 frame FIRST, then transpose + scale
                    # on the GPU: the wire-sized bytes cross PCIe instead
                    # of a 4x-larger fp32 copy, and the per-pixel work runs
                    # on the device. `np.ascontiguousarray` because the npz
                    # view may be non-contiguous, which from_numpy rejects.
                    t = torch.from_numpy(np.ascontiguousarray(arr))
                    t = t.to(self._device, non_blocking=True)
                    if t.ndim == 3 and t.shape[-1] in (1, 3):
                        t = t.permute(2, 0, 1)  # HWC -> CHW
                    # float32 (not self._dtype) to match the CPU path
                    # exactly — autocast casts the matmuls later. The
                    # permute leaves a non-contiguous view; convs want it
                    # contiguous, and on GPU that copy is ~free.
                    batch[key] = (t.float() / 255.0).contiguous().unsqueeze(0)
                else:
                    if arr.ndim == 3 and arr.shape[-1] in (1, 3):
                        arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
                    arr = arr.astype(np.float32) / 255.0
                    batch[key] = torch.from_numpy(arr).unsqueeze(0)
            else:
                batch[key] = (
                    torch.from_numpy(arr.astype(np.float32))
                    .to(self._device, dtype=self._dtype)
                    .unsqueeze(0)
                )

        if "task" not in batch and self._default_task:
            batch["task"] = [self._default_task]
        return batch

    @staticmethod
    def _resolve_compile_action(requested: str, cfg) -> str:
        """Resolve `DRTC_COMPILE_MODE` to a concrete action.

        Returns one of:
          - ``"keep"`` — leave the checkpoint's own compilation untouched,
          - ``"off"``  — uncompile (run eager),
          - a torch.compile mode string (``"default"`` / ``"reduce-overhead"``
            / ``"max-autotune"``) to (re)wrap with.

        Precedence: an explicit operator mode always wins; ``keep`` is the
        no-touch escape hatch; ``auto`` (the default) defers to the per-family
        classification in `_resolve_auto_compile_mode`.
        """
        requested = (requested or "").strip().lower() or "auto"
        if requested == "keep":
            return "keep"
        if requested in ("off", "none", "eager"):
            return "off"
        if requested == "auto":
            return _resolve_auto_compile_mode(cfg)
        return requested  # explicit: default | reduce-overhead | max-autotune

    @staticmethod
    def _apply_compile_mode(torch, policy, cfg) -> None:
        """Apply the resolved `DRTC_COMPILE_MODE` to the policy's model.

        Reads the env var (default ``auto``), resolves it via
        `_resolve_compile_action`, and (re)wraps ``model.sample_actions`` /
        ``model.forward`` accordingly. The compile is lazy — capture happens
        at the first forward (the synthetic `_warmup`, later in __init__) — so
        re-wrapping here is always safe.

        Safety guarantee: each function is first unwrapped to its true original
        (``_torchdynamo_orig_callable or fn``) before re-wrapping, so a
        checkpoint that self-compiled with cudagraphs (``compile_model=True``)
        is DOWNGRADED to the resolved mode under ``auto`` — an unsafe family
        never keeps cudagraphs. The ``or fn`` fallback is load-bearing; do not
        simplify it.
        """
        requested_raw = os.environ.get("DRTC_COMPILE_MODE", "auto")
        resolved = LeRobotBackend._resolve_compile_action(requested_raw, cfg)
        log.info(
            "torch.compile resolve: DRTC_COMPILE_MODE=%s type=%s -> %s",
            (requested_raw or "").strip().lower() or "auto",
            getattr(cfg, "type", None), resolved,
        )
        if resolved == "keep":
            return
        model = getattr(policy, "model", None)
        if model is None:
            log.info("torch.compile skipped: policy has no .model (eager)")
            return
        try:
            if resolved != "off":
                # SmolVLA gates TF32 behind its compile flag; mirror that.
                torch.set_float32_matmul_precision("high")
            for attr in ("sample_actions", "forward"):
                fn = getattr(model, attr, None)
                if fn is None:
                    continue  # e.g. ACT exposes no sample_actions
                orig = getattr(fn, "_torchdynamo_orig_callable", None) or fn
                setattr(
                    model, attr,
                    orig if resolved == "off"
                    else torch.compile(orig, mode=resolved),
                )
            if hasattr(policy.config, "compile_model"):
                policy.config.compile_model = (resolved != "off")
            log.info("torch.compile applied: mode=%s", resolved)
        except Exception:
            log.warning(
                "torch.compile resolution failed; running as-is", exc_info=True
            )

    def _visual_input_keys(self) -> list[str]:
        """The policy's declared camera keys — the VISUAL entries in its
        ``input_features`` (e.g. ``observation.images.top``)."""
        keys = [
            key
            for key, feat in self.cfg.input_features.items()
            if "VISUAL" in str(getattr(feat, "type", "")).upper()
        ]
        return sorted(keys)

    def _reconcile_session_cameras(self, session_metadata: Optional[dict]) -> None:
        """Fail loudly when the cameras the caller will stream disagree with
        the cameras the policy's own config declares.

        ``session_metadata["image_keys"]`` is the env/node's declared camera
        set (CSV of ``observation.images.<name>``), derived from
        ``Environment.camera_names`` (or the policy-config fallback) by the
        backend warmup-target. The policy's required cameras are the VISUAL
        entries in its ``input_features``. A mismatch means the policy would
        be fed the wrong observation keys at inference — a silent
        wrong-joint-target bug — so raise instead of compiling against a
        spec the robot won't satisfy.

        No-ops when the caller supplies no ``image_keys`` (the policy is
        self-describing; ``_warmup`` builds the obs from ``cfg.input_features``
        directly) or when the policy declares no cameras of its own
        (state-only policies, or MolmoAct2 whose ``input_features`` is
        synthesized from the same ``image_keys`` and so always agrees).
        """
        declared = [
            k.strip()
            for k in str((session_metadata or {}).get("image_keys", "")).split(",")
            if k.strip()
        ]
        if not declared:
            return
        policy_cams = self._visual_input_keys()
        if not policy_cams:
            return
        if set(declared) != set(policy_cams):
            raise RuntimeError(
                "Camera mismatch between the environment and the policy: the "
                f"env/node will stream {sorted(declared)} but policy "
                f"{getattr(self.cfg, 'type', '?')!r} declares {policy_cams} in "
                "its config. Fix the env's camera names to match the checkpoint "
                "(or attach the right checkpoint)."
            )

    def _warmup(self) -> None:
        """Run synthetic forwards so torch.compile / inductor codegen
        completes here (at session-open) instead of stalling the first
        real Infer by minutes.

        Synthetic shapes mirror what the client sends: HWC uint8 images
        and a flat float32 state vector, keyed to the policy schema, so
        the compiled graph is reused by real Infers.

        When RTC is enabled a second forward exercises the in-painting
        path (which compiles a distinct autograd graph). If that pass
        fails, RTC is disabled rather than letting every real Infer
        crash on the same error.
        """
        obs: dict[str, Any] = {}
        for key, feat in self.cfg.input_features.items():
            shape = tuple(int(d) for d in feat.shape)
            is_image = (
                "VISUAL" in str(getattr(feat, "type", "")).upper()
                and len(shape) == 3
            )
            if is_image:
                c, h, w = shape                       # policy schema is CHW
                obs[key] = np.zeros((h, w, c), dtype=np.uint8)  # client sends HWC
            else:
                obs[key] = np.zeros(shape or (1,), dtype=np.float32)
        obs["task"] = self._default_task or "warmup"

        t0 = time.perf_counter()
        try:
            self.forward(obs, None)
            if self._rtc_ok:
                # _last_raw is now populated by the first pass; this
                # second pass hits the RTC in-painting branch.
                try:
                    self.forward(obs, None, next_action_step=0, inference_delay=2)
                except Exception:
                    self._rtc_ok = False
                    log.warning(
                        "RTC warmup forward failed — disabling RTC for this "
                        "session, falling back to plain chunking", exc_info=True,
                    )
            log.info("Warmup/compile completed in %.1fs", time.perf_counter() - t0)
        except Exception:
            log.warning(
                "Warmup forward failed (non-fatal; the first real Infer "
                "will pay the compile cost instead)", exc_info=True,
            )

    def _to_chunk_np(self, x: Any) -> np.ndarray:
        """Tensor/array -> contiguous (<=chunk_size, action_dim) float32."""
        torch = self._torch
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().to(torch.float32).numpy()
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 3:
            x = x[0]  # drop batch dim
        if x.shape[0] > self.chunk_size:
            x = x[: self.chunk_size]
        return np.ascontiguousarray(x)

    def _rtc_leftover(self, next_action_step: int) -> Optional[np.ndarray]:
        """Unexecuted tail of the previous raw chunk that overlaps the
        new chunk starting at `next_action_step`. This is RTC's
        `prev_chunk_left_over`. None when there is no overlap (cold
        start, or the robot ran past the whole previous chunk)."""
        if self._last_raw is None:
            return None
        offset = int(next_action_step) - self._last_start
        if offset < 0 or offset >= len(self._last_raw):
            return None
        leftover = self._last_raw[offset:]
        return leftover if leftover.shape[0] > 0 else None

    def _crossfade_chunk(
        self,
        processed_np: np.ndarray,
        next_action_step: int,
        inference_delay: int,
    ) -> np.ndarray:
        """Blend a fresh chunk into the previous chunk's overlapping tail.

        The non-RTC seam smoother. Operates in robot-action space on the
        already-postprocessed chunk. Returns ``processed_np`` unchanged when
        there's no previous chunk to blend against, no overlap, or a shape
        mismatch — so it's always safe to call.

        Geometry (all in absolute action-step space):
          - previous chunk covers ``[ps, ps + P)`` (ps = _last_processed_start)
          - new chunk covers ``[ns, ns + N)`` (ns = next_action_step)
          - overlap is ``[ns, min(ns+N, ps+P))`` (the new chunk anchors at the
            client's cursor, so ns >= ps and the overlap starts at ns).

        Weighting: the blend weight on the PREVIOUS plan starts at 1.0 (full
        continuity) through the latency region — the first ~inference_delay
        steps, which the client will mostly drop as already-executed — then
        cosine-ramps to 0.0 over ``ramp`` steps, after which the chunk is
        fully the fresh plan. So the seam is C0-continuous with what the robot
        is executing and smoothly hands off to the new prediction, instead of
        a hard jump.
        """
        return crossfade_chunk(
            processed_np,
            self._last_processed,
            self._last_processed_start,
            next_action_step,
            inference_delay,
            ramp_steps=self._crossfade_steps_cfg,
        )

    def forward(
        self,
        observation: np.ndarray | dict,
        prior_actions: Optional[np.ndarray],
        *,
        next_action_step: int = 0,
        inference_delay: int = 0,
    ) -> np.ndarray:
        torch = self._torch
        _cuda = self._device.type == "cuda"

        # DRTC-DEBUG latency split: _to_batch (decode/HWC->CHW) and _pre
        # (Molmo image tiling + prompt tokenization, CPU) run OUTSIDE any
        # CUDA graph, every step. Time them apart from the GPU forward.
        #
        # Making `fwd` honest costs two full device syncs that stall the
        # CPU, so we only pay them on forwards we actually report: the
        # session-start detail window, plus a 1-in-N sample thereafter that
        # feeds the steady-state p50/p95 summary. Everything else runs
        # sync-free and untimed.
        _lat_i = getattr(self, "_lat_i", 0)
        self._lat_i = _lat_i + 1
        _detail, _sampled = self._latency_gate(
            _lat_i, getattr(self, "_lat_sample_every", 0)
        )
        _timing = _detail or _sampled
        _t0 = time.perf_counter()
        _raw_batch = self._to_batch(observation)
        _t_batch = time.perf_counter()
        batch = self._pre(_raw_batch)
        _t_pre = time.perf_counter()

        kwargs: dict[str, Any] = {}
        if self._rtc_ok:
            # RTC path: in-paint the new chunk against the still-pending
            # tail of the previous one. predict_action_chunk absorbs
            # these via **kwargs and ignores them when RTC is off.
            leftover = self._rtc_leftover(next_action_step)
            if leftover is not None:
                kwargs["prev_chunk_left_over"] = (
                    torch.from_numpy(leftover)
                    .to(self._device, dtype=self._dtype)
                    .unsqueeze(0)
                )
                kwargs["inference_delay"] = max(0, int(inference_delay))
        elif prior_actions is not None and self._inpainting_kw is not None:
            kwargs[self._inpainting_kw] = (
                torch.from_numpy(prior_actions).to(self._device, dtype=self._dtype).unsqueeze(0)
            )

        # bf16 autocast on CUDA ~halves SmolVLA forward latency. Weights
        # stay fp32; autocast casts the matmul/conv ops. Preprocessing
        # (normalization, tokenization) already ran outside this block.
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self._device.type == "cuda"
            else contextlib.nullcontext()
        )
        if _cuda and _timing:
            torch.cuda.synchronize()
        _t_fwd0 = time.perf_counter()
        with torch.no_grad(), autocast:
            if self._predict_chunk is not None:
                try:
                    raw = self._predict_chunk(batch, **kwargs)
                except Exception:
                    # A mid-session failure in the RTC in-painting path (e.g. a
                    # torch.compile / Dynamo divergence on a `prev_chunk_left_over`
                    # shape the warmup self-test didn't exercise) must NOT crash
                    # the robot's inference. Drop the RTC kwargs, disable RTC for
                    # the rest of the session, and retry as a plain cold chunk.
                    # Degrades to the (still-functional) non-RTC seam rather than
                    # failing the Infer outright.
                    rtc_kwargs = [
                        k for k in ("prev_chunk_left_over", "inference_delay")
                        if k in kwargs
                    ]
                    if rtc_kwargs:
                        log.warning(
                            "RTC in-painting forward failed mid-session — "
                            "disabling RTC and retrying without in-painting "
                            "(plain chunking for the rest of this session)",
                            exc_info=True,
                        )
                        self._rtc_ok = False
                        for k in rtc_kwargs:
                            kwargs.pop(k, None)
                        raw = self._predict_chunk(batch, **kwargs)
                    else:
                        raise
            else:
                # Fallback: synthesize a chunk via repeated select_action.
                step_actions = []
                for _ in range(self.chunk_size):
                    a = self._select_action(batch)
                    step_actions.append(a)
                raw = torch.stack(
                    [a if a.ndim > 1 else a.unsqueeze(0) for a in step_actions], dim=1
                )
        if _cuda and _timing:
            torch.cuda.synchronize()  # make the GPU forward time real, not async-queued
        _t_fwd = time.perf_counter()

        # `raw` is the policy's pre-postprocessor output — the space RTC
        # operates in, so that is what we cache for the next chunk's
        # in-painting. Snapshot it BEFORE `_post`, which may consume or
        # mutate the tensor. `processed` is unnormalized into robot units.
        raw_np = self._to_chunk_np(raw)
        processed_np = self._to_chunk_np(self._post(raw))
        self._last_raw = raw_np
        self._last_start = int(next_action_step)

        # Non-RTC seam smoothing: when real RTC isn't active (ACT and other
        # single-shot decoders), blend this chunk into the previous chunk's
        # overlapping tail so the client's last-write-wins overwrite can't snap
        # the robot between two cold plans. Flow policies using real RTC skip
        # this (RTC already makes the seam continuous in the denoiser). Cache
        # the BLENDED chunk so the next blend is continuous with what we sent.
        if self._crossfade_requested and not self._rtc_ok:
            processed_np = self._crossfade_chunk(
                processed_np, next_action_step, inference_delay
            )
        self._last_processed = processed_np
        self._last_processed_start = int(next_action_step)
        _t_post = time.perf_counter()

        # DRTC-DEBUG latency split. If fwd_ms dominates: VLM prefill / CUDA
        # graph (not) engaging. If pre_ms dominates: image tiling +
        # tokenization on CPU is the cost.
        #
        # Two regimes: a per-forward line for the session-start window
        # (warmup + compile + first Infers), then a rolling p50/p95 summary
        # built from 1-in-N samples, which is the only view of the steady
        # state — the regime that actually decides control quality.
        if _timing:
            _split = (
                (_t_post - _t0) * 1e3,        # total
                (_t_batch - _t0) * 1e3,       # to_batch
                (_t_pre - _t_batch) * 1e3,    # pre
                (_t_fwd - _t_fwd0) * 1e3,     # fwd
                (_t_post - _t_fwd) * 1e3,     # post
            )
            if _detail:
                log.info(
                    "DRTC-DEBUG latency #%d | total=%.0fms = to_batch=%.0f + "
                    "pre=%.0f + fwd=%.0f + post=%.0f | num_inf_steps=%s chunk=%d",
                    _lat_i + 1, *_split,
                    getattr(self.cfg, "num_inference_steps", "?"),
                    self.chunk_size,
                )
            else:
                self._lat_window.append(_split)
                if len(self._lat_window) >= self._lat_summary_every:
                    self._log_latency_summary()

        # DRTC-DEBUG: dump the first few chunks per session so we can tell a
        # latency problem ("compute is slow") apart from a correctness problem
        # ("first action is out of range -> arm slams"). raw_np is the policy's
        # pre-postprocessor output; processed_np is unnormalized robot units —
        # the values actually sent to the arm. Capped so it isn't log spam.
        _dbg_n = getattr(self, "_dbg_n", 0)
        if _dbg_n < 5:
            self._dbg_n = _dbg_n + 1
            try:
                # What the VLM actually received this step: the language
                # instruction (a VLA does nothing sane without it) and the
                # image tensors per camera key (so we can confirm the right
                # cameras, in the right order, at the expected resolution).
                _imgs = {
                    k: (tuple(v.shape), round(float(v.min()), 3), round(float(v.max()), 3))
                    for k, v in _raw_batch.items()
                    if hasattr(v, "shape") and getattr(v, "ndim", 0) >= 3
                }
                # Proprio state fed to the model — confirm it's populated and
                # in the trained range, not zeros/garbage (a black image OR a
                # zero state both make a VLA hallucinate a pose).
                _state = _raw_batch.get("observation.state")
                _state_str = (
                    np.array2string(
                        _state.detach().cpu().float().numpy().reshape(-1),
                        precision=3, max_line_width=240,
                    )
                    if hasattr(_state, "shape") else repr(_state)
                )
                log.info(
                    "DRTC-DEBUG vlm-input #%d | task=%r | state=%s | "
                    "image_tensors(shape,min,max)=%s",
                    self._dbg_n, _raw_batch.get("task"), _state_str, _imgs,
                )
                log.info(
                    "DRTC-DEBUG forward #%d step=%d | raw shape=%s min=%.4f "
                    "max=%.4f | processed shape=%s min=%.4f max=%.4f | "
                    "processed[0]=%s",
                    self._dbg_n, int(next_action_step),
                    raw_np.shape, float(raw_np.min()), float(raw_np.max()),
                    processed_np.shape, float(processed_np.min()),
                    float(processed_np.max()),
                    np.array2string(processed_np[0], precision=4, max_line_width=240),
                )
            except Exception:
                log.warning("DRTC-DEBUG forward dump failed", exc_info=True)

        return processed_np
