"""SpatialVLA inference backend (Shanghai AI Lab / ShanghaiTech / SJTU).

SpatialVLA (https://github.com/SpatialVLA/SpatialVLA, MIT) is a spatial
vision-language-action model built on PaliGemma2: SigLIP vision + 3D
positional encodings predicted from monocular depth, decoded
autoregressively into a short end-effector action chunk. Released
checkpoints (``IPEC-COMMUNITY/spatialvla-*``) are *transformers-native* —
loaded with ``AutoModel.from_pretrained(trust_remote_code=True)`` — so the
generic lerobot loader (which goes through ``PreTrainedConfig.from_pretrained``)
can't decode them. Like MolmoAct2, SpatialVLA therefore lives here as its
own registered backend and is routed transparently from the wire contract
``policy_backend="lerobot"`` via :func:`policy_runtime.resolve_backend`.

It consumes a single camera image + a natural-language instruction and
emits an action chunk (horizon ~4, 7-DoF end-effector delta + gripper). It
does NOT consume the robot's proprioceptive state — that matches the
published inference path; the state vector in the observation is ignored.

Inference API (verified against the model card, 2026-06):
    https://huggingface.co/IPEC-COMMUNITY/spatialvla-4b-mix-224-pt

    processor = AutoProcessor.from_pretrained(uri, trust_remote_code=True)
    model = AutoModel.from_pretrained(uri, trust_remote_code=True,
                                      torch_dtype=torch.bfloat16).eval().cuda()
    inputs = processor(images=[image], text=prompt, return_tensors="pt")
    generation = model.predict_action(inputs)
    actions = processor.decode_actions(generation, unnorm_key="bridge_orig/1.0.0")

Lazy import: like every real-policy backend here, ``transformers`` /
``torch`` are imported only inside ``__init__``, so importing this module
on a machine without them (CI, the SDK side) never fails — you get a clear
error only when you actually open a SpatialVLA session.
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

# SpatialVLA's released checkpoints predict a short end-effector chunk.
# Defaults used when the caller / metadata don't override them; the actual
# emitted chunk is re-fit to these in `forward`, so a mismatch never breaks
# the DRTC buffer contract.
_DEFAULT_CHUNK = 4
_DEFAULT_ACTION_DIM = 7
# The model card's example normalization key. Real deployments pass the key
# matching their robot/dataset via OpenSession metadata ("unnorm_key").
_DEFAULT_UNNORM_KEY = "bridge_orig/1.0.0"


# ---------------------------------------------------------------------------
# Detection + routing
# ---------------------------------------------------------------------------
def is_spatialvla(policy_uri: str) -> bool:
    """True for a SpatialVLA checkpoint.

    Fast path: the released repos are named ``*spatialvla*`` (e.g.
    ``IPEC-COMMUNITY/spatialvla-4b-mix-224-pt``), so a name match avoids any
    network call. Otherwise fall back to reading ``config.json`` and looking
    for SpatialVLA's ``model_type`` / architecture marker.
    """
    if not policy_uri:
        return False
    if "spatialvla" in policy_uri.lower():
        return True
    cfg = read_checkpoint_config_json(policy_uri)
    if not cfg:
        return False
    if isinstance(cfg.get("type"), str):
        return False  # a lerobot-saved checkpoint — generic path handles it
    model_type = str(cfg.get("model_type", "")).lower()
    archs = " ".join(cfg.get("architectures", [])).lower()
    return "spatialvla" in model_type or "spatialvla" in archs


@register_router
def _route_spatialvla(backend: str, policy_uri: str) -> Optional[str]:
    if backend == "lerobot" and is_spatialvla(policy_uri):
        return "spatialvla"
    return None


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------
@register_backend("spatialvla")
class SpatialVLABackend:
    """SpatialVLA checkpoint loaded for DRTC inference.

    Implements the :class:`policy_runtime.PolicyBackend` protocol directly
    (it is not a lerobot policy, so it does not subclass ``LeRobotBackend``).
    No RTC in-painting: the chunk is short and the model exposes no
    in-painting hook, so chunks are served plain — ``prior_actions`` /
    ``next_action_step`` / ``inference_delay`` are ignored.
    """

    def __init__(
        self,
        chunk_size: int = 0,           # 0 -> SpatialVLA's native horizon
        action_dim: int = 0,           # 0 -> SpatialVLA's native 7-DoF
        *,
        policy_uri: str,
        device: Optional[str] = None,  # None -> auto-detect cuda/cpu
        dtype: str = "bfloat16",       # downgraded to float32 on CPU
        default_task: str = "",
        session_metadata: Optional[dict] = None,
        **_: Any,
    ) -> None:
        if not policy_uri:
            raise ValueError("SpatialVLABackend requires policy_uri")
        meta = dict(session_metadata or {})
        self.chunk_size = int(chunk_size) or _DEFAULT_CHUNK
        self.action_dim = int(action_dim) or _DEFAULT_ACTION_DIM
        self._default_task = default_task
        self._unnorm_key = (meta.get("unnorm_key") or _DEFAULT_UNNORM_KEY).strip()
        # Optional: which camera key to feed when several are present.
        self._primary_camera = (meta.get("primary_camera") or "").strip() or None
        # Non-RTC backend — these exist only so the runtime cache's
        # session-reset poke (policy_runtime.load) is a no-op, not an error.
        self._last_raw = None
        self._last_start = 0
        self._load(policy_uri, device, dtype)

    # ------------------------------------------------------------------
    def _load(self, policy_uri: str, device: Optional[str], dtype: str) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        self._torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)
        # bfloat16 is the card default but only sane on GPU; CPU stays f32.
        if self._device.type != "cuda" and dtype in ("bfloat16", "float16"):
            dtype = "float32"
        self._dtype = getattr(torch, dtype)
        log.info(
            "Loading SpatialVLA policy_uri=%s device=%s dtype=%s unnorm_key=%s",
            policy_uri, self._device, dtype, self._unnorm_key,
        )
        self._processor = AutoProcessor.from_pretrained(
            policy_uri, trust_remote_code=True
        )
        self._model = (
            AutoModel.from_pretrained(
                policy_uri, trust_remote_code=True, torch_dtype=self._dtype
            )
            .eval()
            .to(self._device)
        )

    # ------------------------------------------------------------------
    def _select_image(self, observation: Any) -> np.ndarray:
        """Pick the camera frame to condition on.

        Accepts either a bare HWC uint8 array or the standard observation
        dict; prefers ``self._primary_camera`` when set, else the first
        image-shaped value. Raises if no image is present, since SpatialVLA
        is image-conditioned."""
        if isinstance(observation, dict):
            if self._primary_camera and self._primary_camera in observation:
                return np.asarray(observation[self._primary_camera])
            for key, value in observation.items():
                if key == "task":
                    continue
                arr = np.asarray(value)
                if arr.dtype == np.uint8 and arr.ndim == 3 and arr.shape[-1] in (1, 3):
                    return arr
            raise ValueError(
                "SpatialVLA needs a camera image in the observation "
                f"(HWC uint8); got keys {sorted(observation)}. Start the node "
                "with --camera <name>=<device>."
            )
        arr = np.asarray(observation)
        if arr.ndim == 3:
            return arr
        raise ValueError("SpatialVLA needs an HWC image observation")

    def _task_of(self, observation: Any) -> str:
        if isinstance(observation, dict) and "task" in observation:
            t = observation["task"]
            return t.item() if hasattr(t, "item") else str(t)
        return self._default_task or "stop"

    @staticmethod
    def _decoded_to_array(decoded: Any) -> Any:
        """``processor.decode_actions`` may return a numpy array or a dict
        ({"actions": ...}); normalize to the raw action array."""
        if isinstance(decoded, dict):
            for key in ("actions", "action", "action_pred"):
                if key in decoded:
                    return decoded[key]
            # single-entry dict -> its value
            return next(iter(decoded.values()))
        return decoded

    # ------------------------------------------------------------------
    def forward(
        self,
        observation: np.ndarray | dict,
        prior_actions: Optional[np.ndarray],
        **_: Any,
    ) -> np.ndarray:
        from PIL import Image

        image = self._select_image(observation)
        prompt = self._task_of(observation)
        pil = Image.fromarray(np.ascontiguousarray(image)).convert("RGB")

        inputs = self._processor(images=[pil], text=prompt, return_tensors="pt")
        if hasattr(inputs, "to"):
            inputs = inputs.to(self._device)

        with self._torch.no_grad():
            generation = self._model.predict_action(inputs)
        decoded = self._processor.decode_actions(
            generation, unnorm_key=self._unnorm_key
        )
        actions = self._decoded_to_array(decoded)
        return as_action_chunk(actions, self.chunk_size, self.action_dim)
