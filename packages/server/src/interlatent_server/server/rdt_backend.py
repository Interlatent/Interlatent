"""RDT-1B inference backend (Tsinghua THU-ML, Robotics Diffusion Transformer).

RDT-1B (https://github.com/thu-ml/RoboticsDiffusionTransformer, MIT;
HF ``robotics-diffusion-transformer/rdt-1b``) is a ~1.2B-param diffusion
transformer that predicts the next 64 actions in a 128-dim *unified* action
space, conditioned on proprioceptive state, a stack of camera views, and a
language instruction. Unlike SpatialVLA it explicitly ingests the robot's
state vector, which makes it the closest architectural fit to a real
action-chunking control loop in this server.

It is NOT transformers-native: the published inference path is the repo's
own ``create_model(...)`` factory whose returned model exposes
``step(proprio, images, text_embeds)`` — see ``scripts/agilex_model.py`` in
the upstream repo. ``text_embeds`` are *precomputed* T5-v1.1-XXL embeddings,
so this backend also loads a T5 encoder to turn the instruction into them
(cached per task). Because the upstream code isn't a pip package, the
factory is imported lazily from a small set of known module paths; if the
RDT repo isn't importable you get a precise, actionable error at session
open — never at import time.

Deployment-specific seams (passed via OpenSession metadata, all optional):
  - ``rdt_config``      : path to the RDT model config yaml (else the repo
                          default ``configs/base.yaml`` is attempted).
  - ``t5_model``        : T5 encoder repo (default ``google/t5-v1_1-xxl``).
  - ``vision_encoder``  : SigLIP repo (default
                          ``google/siglip-so400m-patch14-384``).
  - ``camera_order``    : CSV of observation image keys mapped to RDT's
                          (exterior, right_wrist, left_wrist) slots.
  - ``action_indices``  : CSV of columns of the 128-dim unified output that
                          map to this robot's DoFs (else the first
                          ``action_dim`` columns, with a warning — the real
                          map is checkpoint/robot specific).

License: MIT (model + code). Verified against the model card / repo 2026-06.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from .policy_runtime import (
    as_action_chunk,
    read_checkpoint_config_json,
    register_backend,
    register_router,
)

log = logging.getLogger(__name__)

_DEFAULT_CHUNK = 64          # RDT predicts the next 64 actions
_UNIFIED_ACTION_DIM = 128    # RDT's unified action space width
_DEFAULT_T5 = "google/t5-v1_1-xxl"
_DEFAULT_VISION = "google/siglip-so400m-patch14-384"
# RDT conditions on 3 camera views x 2 timesteps = 6 image slots.
_NUM_VIEWS = 3


# ---------------------------------------------------------------------------
# Detection + routing
# ---------------------------------------------------------------------------
def is_rdt(policy_uri: str) -> bool:
    """True for an RDT / Robotics-Diffusion-Transformer checkpoint.

    Fast path matches the published repo names without a network call;
    otherwise read ``config.json`` and look for an ``rdt`` model_type."""
    if not policy_uri:
        return False
    u = policy_uri.lower()
    if (
        "robotics-diffusion-transformer" in u
        or "rdt-1b" in u
        or "/rdt-" in u
        or "rdt_" in u
        or u.endswith("/rdt")
    ):
        return True
    cfg = read_checkpoint_config_json(policy_uri)
    if not cfg or isinstance(cfg.get("type"), str):
        return False
    return str(cfg.get("model_type", "")).lower().startswith("rdt")


@register_router
def _route_rdt(backend: str, policy_uri: str) -> Optional[str]:
    if backend == "lerobot" and is_rdt(policy_uri):
        return "rdt"
    return None


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------
@register_backend("rdt")
class RDTBackend:
    """RDT-1B checkpoint loaded for DRTC inference.

    Implements the :class:`policy_runtime.PolicyBackend` protocol directly.
    No RTC in-painting (RDT exposes no in-painting hook here); chunks are
    served plain. A one-step image history is kept so each forward can fill
    RDT's previous-timestep image slots.
    """

    def __init__(
        self,
        chunk_size: int = 0,           # 0 -> RDT's native 64
        action_dim: int = 0,           # robot DoF count (required for slicing)
        *,
        policy_uri: str,
        device: Optional[str] = None,  # None -> auto-detect cuda/cpu
        dtype: str = "bfloat16",       # downgraded to float32 on CPU
        default_task: str = "",
        session_metadata: Optional[dict] = None,
        **_: Any,
    ) -> None:
        if not policy_uri:
            raise ValueError("RDTBackend requires policy_uri")
        meta = dict(session_metadata or {})
        self.chunk_size = int(chunk_size) or _DEFAULT_CHUNK
        if int(action_dim) <= 0:
            raise ValueError(
                "RDTBackend needs a positive action_dim (the robot's DoF count) "
                "to slice RDT's 128-dim unified action output."
            )
        self.action_dim = int(action_dim)
        self._default_task = default_task
        self._t5_model = (meta.get("t5_model") or _DEFAULT_T5).strip()
        self._vision_encoder = (meta.get("vision_encoder") or _DEFAULT_VISION).strip()
        self._rdt_config = (meta.get("rdt_config") or "").strip() or None
        self._camera_order = [
            k.strip() for k in str(meta.get("camera_order", "")).split(",") if k.strip()
        ]
        self._action_indices = self._parse_indices(meta.get("action_indices"))
        # One-step image history (RDT's previous-timestep slots).
        self._prev_views: Optional[list] = None
        # T5 embeddings are expensive; cache by instruction string.
        self._text_cache: dict[str, Any] = {}
        # Non-RTC: present so the runtime cache's session-reset poke is a no-op.
        self._last_raw = None
        self._last_start = 0
        self._load(policy_uri, device, dtype)

    @staticmethod
    def _parse_indices(raw: Any) -> Optional[list[int]]:
        if not raw:
            return None
        try:
            return [int(x) for x in str(raw).split(",") if x.strip() != ""]
        except ValueError:
            return None

    # ------------------------------------------------------------------
    def _load(self, policy_uri: str, device: Optional[str], dtype: str) -> None:
        import torch

        self._torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)
        if self._device.type != "cuda" and dtype in ("bfloat16", "float16"):
            dtype = "float32"
        self._dtype = getattr(torch, dtype)

        create_model = self._import_create_model()
        config = self._load_rdt_config()
        log.info(
            "Loading RDT policy_uri=%s device=%s dtype=%s t5=%s vision=%s",
            policy_uri, self._device, dtype, self._t5_model, self._vision_encoder,
        )
        # Signature per upstream scripts/agilex_model.py: create_model builds
        # the RoboticDiffusionTransformerModel and loads the HF weights.
        self._model = create_model(
            args=config,
            dtype=self._dtype,
            pretrained=policy_uri,
            pretrained_vision_encoder_name_or_path=self._vision_encoder,
            control_frequency=int(self._rdt_config and 0 or 25),
        )
        self._load_t5()

    def _import_create_model(self):
        """Import RDT's ``create_model`` factory from the vendored repo.

        Upstream is not a pip package, so try the module paths the repo
        exposes when it (or its ``scripts/``) is on PYTHONPATH."""
        last_err: Optional[Exception] = None
        for modname in (
            "scripts.agilex_model",
            "agilex_model",
            "rdt.scripts.agilex_model",
        ):
            try:
                mod = __import__(modname, fromlist=["create_model"])
                return getattr(mod, "create_model")
            except Exception as exc:  # ImportError or AttributeError
                last_err = exc
        raise ImportError(
            "Could not import RDT's create_model. RDT-1B is not a pip package; "
            "clone https://github.com/thu-ml/RoboticsDiffusionTransformer and put "
            "its repo root (which holds scripts/agilex_model.py) on PYTHONPATH, "
            "and install its pinned deps (diffusers==0.27.2, transformers==4.41.0, "
            "flash-attn). Last import error: " + repr(last_err)
        )

    def _load_rdt_config(self) -> dict:
        """Load the RDT model config (yaml). Caller-supplied path wins; else
        try the repo's default ``configs/base.yaml`` from PYTHONPATH."""
        import os

        import yaml  # PyYAML ships with the RDT repo deps

        path = self._rdt_config
        if not path:
            for base in ("configs/base.yaml", "RoboticsDiffusionTransformer/configs/base.yaml"):
                if os.path.isfile(base):
                    path = base
                    break
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(
                "RDT needs its model config yaml. Pass session metadata "
                "'rdt_config=/path/to/configs/base.yaml' (from the RDT repo)."
            )
        with open(path) as fh:
            return yaml.safe_load(fh)

    def _load_t5(self) -> None:
        """Load the T5 encoder used to precompute ``text_embeds``."""
        from transformers import AutoTokenizer, T5EncoderModel

        self._t5_tokenizer = AutoTokenizer.from_pretrained(self._t5_model)
        self._t5_encoder = (
            T5EncoderModel.from_pretrained(self._t5_model, torch_dtype=self._dtype)
            .eval()
            .to(self._device)
        )

    # ------------------------------------------------------------------
    def _text_embeds(self, instruction: str):
        cached = self._text_cache.get(instruction)
        if cached is not None:
            return cached
        torch = self._torch
        toks = self._t5_tokenizer(
            instruction, return_tensors="pt", padding=True, truncation=True
        ).to(self._device)
        with torch.no_grad():
            out = self._t5_encoder(**toks).last_hidden_state
        self._text_cache[instruction] = out
        return out

    def _views(self, observation: Any) -> list:
        """Pull the current-timestep camera frames in RDT's view order.

        Uses ``camera_order`` when given, else the image-shaped values in
        insertion order. Missing slots become None (RDT tolerates absent
        views)."""
        images: dict[str, np.ndarray] = {}
        if isinstance(observation, dict):
            for key, value in observation.items():
                if key == "task":
                    continue
                arr = np.asarray(value)
                if arr.dtype == np.uint8 and arr.ndim == 3 and arr.shape[-1] in (1, 3):
                    images[key] = arr
        order = self._camera_order or list(images.keys())
        views: list = [images.get(k) for k in order[:_NUM_VIEWS]]
        while len(views) < _NUM_VIEWS:
            views.append(None)
        return views

    def _proprio(self, observation: Any):
        torch = self._torch
        if isinstance(observation, dict):
            state = observation.get("observation.state")
            if state is None:
                state = next(
                    (v for k, v in observation.items() if k != "task"), None
                )
        else:
            state = observation
        arr = np.asarray(state, dtype=np.float32).reshape(-1)
        return torch.from_numpy(arr).to(self._device, dtype=self._dtype).unsqueeze(0)

    def _slice_unified(self, chunk: np.ndarray) -> np.ndarray:
        """Map RDT's 128-dim unified action columns to the robot's DoFs."""
        if self._action_indices:
            idx = [i for i in self._action_indices if 0 <= i < chunk.shape[1]]
            return chunk[:, idx]
        if chunk.shape[1] > self.action_dim:
            log.warning(
                "RDT emitted %d unified action dims; no action_indices given, "
                "slicing the first %d. Pass session metadata 'action_indices' "
                "for the correct robot DoF map.",
                chunk.shape[1], self.action_dim,
            )
            return chunk[:, : self.action_dim]
        return chunk

    # ------------------------------------------------------------------
    def forward(
        self,
        observation: np.ndarray | dict,
        prior_actions: Optional[np.ndarray],
        **_: Any,
    ) -> np.ndarray:
        task = self._default_task or "stop"
        if isinstance(observation, dict) and "task" in observation:
            t = observation["task"]
            task = t.item() if hasattr(t, "item") else str(t)

        cur_views = self._views(observation)
        prev_views = self._prev_views if self._prev_views is not None else cur_views
        # RDT's 6 image slots: previous timestep (3 views) then current (3).
        images = list(prev_views) + list(cur_views)
        self._prev_views = cur_views

        proprio = self._proprio(observation)
        text_embeds = self._text_embeds(task)

        with self._torch.no_grad():
            raw = self._model.step(proprio, images, text_embeds)
        chunk = as_action_chunk(raw, self.chunk_size)  # (<=64, 128) unified
        chunk = self._slice_unified(chunk)
        return as_action_chunk(chunk, self.chunk_size, self.action_dim)
