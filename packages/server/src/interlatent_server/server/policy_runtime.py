"""Policy load / warm / forward with RTC in-painting and stubbed
activation-hook seams for v2.

This module deliberately does not import torch at module level so the
file can be imported in lightweight contexts (tests, type checking,
SDK-side tools that only need the registry interface). The actual
policy backend is loaded lazily inside `PolicyRuntime.load`.

v1 ships with a single backend (`echo`) that returns deterministic
synthetic action chunks. Real policy backends (LeRobot SmolVLA, SB3,
custom torch) plug in via `register_backend`.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

import numpy as np

from .schedule import InpaintingContext


# ----------------------------------------------------------------------
# Backend registry
# ----------------------------------------------------------------------


class PolicyBackend(Protocol):
    """Minimal interface every policy implementation satisfies."""

    chunk_size: int
    action_dim: int

    def forward(
        self,
        observation: np.ndarray | dict,
        prior_actions: Optional[np.ndarray],
        *,
        next_action_step: int = 0,
        inference_delay: int = 0,
    ) -> np.ndarray:
        """Return shape (chunk_size, action_dim) float32.

        `next_action_step` / `inference_delay` are used by RTC-capable
        backends (LeRobotBackend) to in-paint the new chunk against the
        overlapping tail of the previous one. Non-RTC backends ignore
        them and use `prior_actions` instead."""
        ...


_BACKENDS: dict[str, Callable[..., PolicyBackend]] = {}

# Process-wide cache of loaded PolicyRuntime instances keyed by
# (backend_name, policy_uri). See PolicyRuntime.load for rationale —
# in short, lerobot's torch.compile path is expensive enough that
# re-building the runtime on every OpenSession would stretch session
# open by ~6 minutes per pod restart. Lifetime is the process lifetime.
_RUNTIME_CACHE: dict[tuple[str, str], "PolicyRuntime"] = {}


def register_backend(name: str):
    def deco(fn):
        _BACKENDS[name] = fn
        return fn
    return deco


# ----------------------------------------------------------------------
# Activation-hook seam (v2)
# ----------------------------------------------------------------------


@dataclass
class ActivationHookCtx:
    """Stub. v1 leaves callbacks empty; v2 plugs capture in here.

    Kept in this module so the seam is at the exact point in the
    forward path where activations naturally exist, and so v2 does
    not need to restructure callers.
    """

    on_forward_start: list[Callable[[dict], None]] = field(default_factory=list)
    on_forward_end: list[Callable[[dict, np.ndarray], None]] = field(default_factory=list)

    def fire_start(self, ctx: dict) -> None:
        for cb in self.on_forward_start:
            cb(ctx)

    def fire_end(self, ctx: dict, actions: np.ndarray) -> None:
        for cb in self.on_forward_end:
            cb(ctx, actions)


# ----------------------------------------------------------------------
# Echo backend — v1 default
# ----------------------------------------------------------------------


@register_backend("echo")
class EchoBackend:
    """Deterministic synthetic policy used until real backends land.

    Produces a smooth sinusoidal trajectory in action space so that
    the DRTC in-painting / LWW merge can be exercised end-to-end
    without a real GPU model. Picks up the last prior action as the
    starting phase so stitched trajectories visibly stay continuous.
    """

    def __init__(self, chunk_size: int = 32, action_dim: int = 6, **_: Any) -> None:
        self.chunk_size = chunk_size
        self.action_dim = action_dim

    def forward(
        self,
        observation: np.ndarray | dict,
        prior_actions: Optional[np.ndarray],
        **_: Any,
    ) -> np.ndarray:
        t = np.arange(self.chunk_size, dtype=np.float32) / max(self.chunk_size - 1, 1)
        base = np.sin(2 * np.pi * t)[:, None]
        out = np.tile(base, (1, self.action_dim)).astype(np.float32)
        if prior_actions is not None and len(prior_actions) > 0:
            # Anchor the new chunk to the last prior action — crude
            # but enough to test that in-painting context is wired.
            out += prior_actions[-1][None, :]
        return out


# ----------------------------------------------------------------------
# Tiny torch backend — for local end-to-end tests with a real nn.Module
# ----------------------------------------------------------------------
#
# Stand-in for a real VLA (SmolVLA et al). Same call signature:
#     forward(obs_dict, prior_actions) -> (chunk_size, action_dim) f32
# Expects an observation dict with at least:
#     "observation.state": (state_dim,) float32
# Optional:
#     "observation.image": (H, W, 3) uint8        (ignored by the MLP,
#                                                  here only so the
#                                                  npz codec gets
#                                                  exercised end-to-end)
# Swapping this out for SmolVLA later means writing a peer class that
# calls `policy.predict_action_chunk(batch)` instead of our MLP — the
# rest of the DRTC plumbing is unchanged.


@register_backend("tiny_torch")
class TinyTorchBackend:
    def __init__(
        self,
        chunk_size: int = 16,
        action_dim: int = 6,
        *,
        state_dim: int = 8,
        hidden: int = 64,
        device: str = "cpu",
        seed: int = 0,
        **_: Any,
    ) -> None:
        import torch
        import torch.nn as nn

        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self._state_dim = state_dim
        self._device = torch.device(device)
        torch.manual_seed(seed)
        self._net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, chunk_size * action_dim),
        ).to(self._device).eval()

    def forward(
        self,
        observation: np.ndarray | dict,
        prior_actions: Optional[np.ndarray],
        **_: Any,
    ) -> np.ndarray:
        import torch

        if isinstance(observation, dict):
            state = observation.get("observation.state")
            if state is None:
                # Fall back: flatten whatever we got into a state vec.
                state = next(iter(observation.values()))
        else:
            state = observation
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.size < self._state_dim:
            state = np.pad(state, (0, self._state_dim - state.size))
        else:
            state = state[: self._state_dim]
        x = torch.from_numpy(state).to(self._device).unsqueeze(0)
        with torch.no_grad():
            y = self._net(x).view(self.chunk_size, self.action_dim)
        actions = y.cpu().numpy().astype(np.float32)
        if prior_actions is not None and len(prior_actions) > 0:
            # Anchor to last prior action so stitched trajectories
            # visibly stay continuous when the LWW merge runs.
            actions += prior_actions[-1][None, :]
        return actions


# ----------------------------------------------------------------------
# Runtime
# ----------------------------------------------------------------------


@dataclass
class PolicyRuntime:
    """Server-side policy holder. One instance per loaded model.

    Lifecycle:
        rt = PolicyRuntime.load(backend="echo", chunk_size=32, action_dim=6)
        actions = rt.forward(observation, inpainting_ctx)
    """

    backend: PolicyBackend
    chunk_size: int
    action_dim: int
    hooks: ActivationHookCtx = field(default_factory=ActivationHookCtx)

    @classmethod
    def load(
        cls,
        backend: str = "echo",
        *,
        chunk_size: int = 32,
        action_dim: int = 6,
        **backend_kwargs: Any,
    ) -> "PolicyRuntime":
        if backend not in _BACKENDS:
            raise ValueError(
                f"Unknown policy backend {backend!r}. "
                f"Registered: {sorted(_BACKENDS)}"
            )

        # Process-wide cache keyed by (backend, policy_uri). Without it
        # the pre-warm at startup (serve_gpu._warmup) builds + compiles
        # a LeRobotBackend, then drops the reference; the first real
        # OpenSession from the Pi builds + compiles ANOTHER one — even
        # though the inductor disk cache is warm, the policy reloads
        # weights (~30s for SmolVLM) and re-traces graphs, which can
        # take many minutes on top of the pre-warm cost the operator
        # already paid. Caching here turns OpenSession into an O(1)
        # dict lookup after the first compile.
        #
        # Echo / tiny_torch don't cache: they're per-test cheap
        # instances and tests sometimes need a fresh seed/config.
        policy_uri = str(backend_kwargs.get("policy_uri") or "")
        cacheable = backend not in ("echo", "tiny_torch") and bool(policy_uri)
        cache_key = (backend, policy_uri)
        if cacheable:
            cached = _RUNTIME_CACHE.get(cache_key)
            if cached is not None:
                # Reset session-local state on the backend so the new
                # session starts clean. LeRobotBackend's RTC trail
                # (_last_raw / _last_start) is the only such state at
                # time of writing; future backends should expose a
                # ``reset_session_state()`` method instead of relying
                # on this attribute-poke pattern.
                impl = cached.backend
                if hasattr(impl, "_last_raw"):
                    impl._last_raw = None
                if hasattr(impl, "_last_start"):
                    impl._last_start = 0
                # Override the per-session default_task if the caller
                # supplied a new one. Per-tick observations usually
                # carry their own ``task`` field, so this only matters
                # when the client omits it.
                if "default_task" in backend_kwargs and hasattr(impl, "_default_task"):
                    impl._default_task = backend_kwargs["default_task"]
                import logging as _logging
                _logging.getLogger(__name__).info(
                    "Reusing cached PolicyRuntime backend=%s policy_uri=%s "
                    "(skipping load + warmup)", backend, policy_uri,
                )
                return cached

        factory = _BACKENDS[backend]
        impl = factory(chunk_size=chunk_size, action_dim=action_dim, **backend_kwargs)
        rt = cls(backend=impl, chunk_size=impl.chunk_size, action_dim=impl.action_dim)
        if cacheable:
            _RUNTIME_CACHE[cache_key] = rt
        return rt

    def forward(
        self,
        observation: np.ndarray | dict,
        ctx: InpaintingContext,
        inference_delay: int = 0,
    ) -> np.ndarray:
        hook_ctx = {
            "next_action_step": ctx.next_action_step,
            "has_prior": ctx.prior_actions is not None,
            "inference_delay": inference_delay,
        }
        self.hooks.fire_start(hook_ctx)
        actions = self.backend.forward(
            observation,
            ctx.prior_actions,
            next_action_step=ctx.next_action_step,
            inference_delay=inference_delay,
        )
        if actions.dtype != np.float32:
            actions = actions.astype(np.float32)
        self.hooks.fire_end(hook_ctx, actions)
        return actions


# ----------------------------------------------------------------------
# Payload decoding
# ----------------------------------------------------------------------


def _is_jpeg_blob(arr: np.ndarray) -> bool:
    """Test if ``arr`` is a wire-format JPEG blob (1-D uint8 starting FF D8 FF)."""
    return bool(
        arr.dtype == np.uint8
        and arr.ndim == 1
        and arr.size >= 3
        and arr[0] == 0xFF
        and arr[1] == 0xD8
        and arr[2] == 0xFF
    )


def _maybe_jpeg(
    arr: np.ndarray,
    *,
    return_bytes: bool = False,
) -> "np.ndarray | tuple[np.ndarray, Optional[bytes]]":
    """Decode a JPEG blob to an HWC uint8 image; pass anything else
    through untouched.

    The SDK JPEG-compresses camera frames inside the npz payload to keep
    the upload small. A JPEG blob arrives as a 1-D uint8 array starting
    with the JPEG magic bytes FF D8 FF — distinct from a raw state
    vector (float32) or a raw image (>=2-D), so detection is unambiguous
    and an old SDK sending raw images still works.

    When ``return_bytes`` is set, returns ``(decoded_array, raw_jpeg_bytes_or_None)``
    so callers can both consume the pixels and persist the original wire
    bytes verbatim — no re-encode, and the JPEG is decoded exactly once
    (here, in the hot path the policy already pays for).
    """
    if _is_jpeg_blob(arr):
        from PIL import Image

        raw_bytes = arr.tobytes()
        with io.BytesIO(raw_bytes) as f:
            decoded = np.asarray(Image.open(f).convert("RGB"), dtype=np.uint8)
        if return_bytes:
            return decoded, raw_bytes
        return decoded
    if return_bytes:
        return arr, None
    return arr


def decode_payload(
    payload: bytes,
    codec: str,
    *,
    return_jpeg_bytes: bool = False,
) -> "np.ndarray | dict | tuple[np.ndarray | dict, dict[str, bytes]]":
    """Turn the opaque Observation.payload bytes into something the
    backend can consume. Kept here (not in transport) so backends can
    register custom codecs alongside themselves later.

    When ``return_jpeg_bytes`` is set, returns
    ``(decoded, {key: raw_jpeg_bytes})`` so the DRTC server-side recorder
    can persist the original wire bytes without a second encode. Non-JPEG
    keys are absent from the map (not present, not None — the caller
    treats absence as "this key has no JPEG to persist").

    The decode happens at most once per blob: the policy needs decoded
    pixels and the recorder needs the original bytes, but the bytes
    never go through PIL twice.
    """
    if codec == "raw_f32":
        decoded_scalar = np.frombuffer(payload, dtype=np.float32).copy()
        if return_jpeg_bytes:
            return decoded_scalar, {}
        return decoded_scalar

    if codec == "npz":
        with io.BytesIO(payload) as f:
            data = np.load(f, allow_pickle=False)
            files = list(data.files)
            if return_jpeg_bytes:
                jpeg_bytes: dict[str, bytes] = {}
                decoded: dict[str, np.ndarray] = {}
                for k in files:
                    out, raw = _maybe_jpeg(data[k], return_bytes=True)
                    decoded[k] = out
                    if raw is not None:
                        jpeg_bytes[k] = raw
                if len(decoded) == 1:
                    return decoded[files[0]], jpeg_bytes
                return decoded, jpeg_bytes
            decoded_arr = {k: _maybe_jpeg(data[k]) for k in files}
            if len(decoded_arr) == 1:
                return decoded_arr[files[0]]
            return decoded_arr

    raise ValueError(f"Unsupported payload codec: {codec}")
