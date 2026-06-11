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
    default; set DRTC_RTC=0 to fall back to plain chunk concatenation.

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

from .policy_runtime import register_backend

log = logging.getLogger(__name__)


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


@register_backend("lerobot")
class LeRobotBackend:
    """Wraps any lerobot policy. The Interlatent DRTC server treats
    every policy uniformly through this adapter."""

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
        log.info("%s device=%s", type(self).__name__, self._device)
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
        # generic lerobot path ignores it.
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
        # SmolVLA compiles `sample_actions` / `forward` itself when the
        # checkpoint config has compile_model=True, using `compile_mode`
        # from config.json — typically "max-autotune". Most published
        # checkpoints ship with compile_model=False, so we force-enable
        # compile here for the serving path: eager SmolVLA on an A100 is
        # ~370ms vs ~100ms compiled. SmolVLA also gates
        # `torch.set_float32_matmul_precision("high")` (TF32) behind the
        # same flag, so we mirror that here.
        # On a persistent GPU box (cloud/serve_gpu.py pins
        # TORCHINDUCTOR_CACHE_DIR / TRITON_CACHE_DIR under $HOME) the
        # one-time max-autotune cost is paid at first session-open and
        # reused thereafter — there is no cold-start tax to amortise.
        # Override: DRTC_COMPILE_MODE = keep | off | default | max-autotune
        model = getattr(policy, "model", None)
        sample_actions_fn = getattr(model, "sample_actions", None) if model is not None else None
        already_compiled = (
            sample_actions_fn is not None
            and hasattr(sample_actions_fn, "_torchdynamo_orig_callable")
        )
        default_mode = getattr(policy.config, "compile_mode", None) or "max-autotune"
        if not already_compiled and model is not None:
            torch.set_float32_matmul_precision("high")
            try:
                for attr in ("sample_actions", "forward"):
                    fn = getattr(model, attr, None)
                    if fn is None:
                        continue
                    setattr(model, attr, torch.compile(fn, mode=default_mode))
                if hasattr(policy.config, "compile_model"):
                    policy.config.compile_model = True
                log.info(
                    "Force-enabled torch.compile (mode=%s) — checkpoint had "
                    "compile_model=False", default_mode,
                )
            except Exception:
                log.warning("Force-compile failed; running eager", exc_info=True)

        compile_mode = os.environ.get("DRTC_COMPILE_MODE", "keep").strip().lower()
        if compile_mode not in ("keep", ""):
            try:
                for attr in ("sample_actions", "forward"):
                    fn = getattr(model, attr, None) if model is not None else None
                    orig = getattr(fn, "_torchdynamo_orig_callable", None)
                    if orig is None:
                        continue  # not compiled — nothing to adjust
                    if compile_mode in ("off", "none", "eager"):
                        setattr(model, attr, orig)
                    else:
                        setattr(model, attr, torch.compile(orig, mode=compile_mode))
                log.info("torch.compile mode -> %s", compile_mode)
            except Exception:
                log.warning("Could not adjust torch.compile mode", exc_info=True)
        else:
            log.info("torch.compile mode -> keep (mode=%s)", default_mode)

        # --- RTC (Real-Time Chunking) -----------------------------------
        # RTC turns chunk generation into an in-painting problem: the new
        # chunk is generated to stay continuous with the still-unexecuted
        # tail of the previous one, eliminating the velocity jump at chunk
        # boundaries. lerobot gates RTC on `config.rtc_config`; a stock
        # checkpoint loads with it unset, so enable it here. RTC needs a
        # flow-matching policy (SmolVLA / pi0 / pi0.5) — for others the
        # enable below is a harmless no-op. Disable with DRTC_RTC=0.
        self._rtc_ok = False
        if os.environ.get("DRTC_RTC", "1").lower() not in ("0", "false", "off", ""):
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
                "missing; VLA policies will fail.", policy_uri,
            )
            self._pre = lambda b: b
            self._post = lambda x: x

        self.policy = policy
        self.cfg = cfg
        self.chunk_size = chunk_size or getattr(cfg, "chunk_size", None) \
            or getattr(cfg, "n_action_steps", None) or 32
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

        # Snapshot the keys the policy expects so we can build a clean
        # batch from whatever the client sent.
        self._expected_keys: tuple[str, ...] = tuple(cfg.input_features.keys())
        log.info(
            "LeRobotBackend loaded policy=%s chunk_size=%d action_dim=%d "
            "inpainting_kw=%s expected_keys=%s",
            policy_uri, self.chunk_size, self.action_dim,
            self._inpainting_kw, self._expected_keys,
        )

        # Pay the torch.compile cost now, at session-open, with a
        # synthetic forward — so the first real Infer is already fast.
        self._warmup()
        # Discard warmup's raw actions: the first real chunk must be a
        # clean cold start, not in-painted against synthetic data.
        self._last_raw = None
        self._last_start = 0

    # ------------------------------------------------------------------

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
        if _cuda:
            torch.cuda.synchronize()
        _t_fwd0 = time.perf_counter()
        with torch.no_grad(), autocast:
            if self._predict_chunk is not None:
                raw = self._predict_chunk(batch, **kwargs)
            else:
                # Fallback: synthesize a chunk via repeated select_action.
                step_actions = []
                for _ in range(self.chunk_size):
                    a = self._select_action(batch)
                    step_actions.append(a)
                raw = torch.stack(
                    [a if a.ndim > 1 else a.unsqueeze(0) for a in step_actions], dim=1
                )
        if _cuda:
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
        _t_post = time.perf_counter()

        # DRTC-DEBUG latency split — fires for the first ~12 forwards so the
        # steady state (after warmup #1's compile cost) is visible. If fwd_ms
        # dominates: VLM prefill / CUDA graph (not) engaging. If pre_ms
        # dominates: Molmo image tiling + tokenization on CPU is the cost.
        _dbg_t = getattr(self, "_dbg_t", 0)
        if _dbg_t < 12:
            self._dbg_t = _dbg_t + 1
            log.info(
                "DRTC-DEBUG latency #%d | total=%.0fms = to_batch=%.0f + "
                "pre=%.0f + fwd=%.0f + post=%.0f | num_inf_steps=%s chunk=%d",
                self._dbg_t,
                (_t_post - _t0) * 1e3,
                (_t_batch - _t0) * 1e3,
                (_t_pre - _t_batch) * 1e3,
                (_t_fwd - _t_fwd0) * 1e3,
                (_t_post - _t_fwd) * 1e3,
                getattr(self.cfg, "num_inference_steps", "?"),
                self.chunk_size,
            )

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
