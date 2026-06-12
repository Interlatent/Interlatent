"""MolmoAct2 inference backend (decoupled from the generic lerobot path).

AllenAI's *released* MolmoAct2 checkpoints (``allenai/MolmoAct2-*``) are
transformers-native: their ``config.json`` carries ``model_type:
"molmoact2"`` and no lerobot ``type`` field, so the generic
``LeRobotBackend`` loader (which goes through ``PreTrainedConfig.from_pretrained``)
can't decode them. They also need a hand-assembled ``MolmoAct2Config`` built
from the checkpoint's ``norm_stats.json`` plus session-supplied camera keys.

Rather than special-casing all of that inside ``lerobot_backend.py``, the
MolmoAct2 path lives here as its own registered backend (``"molmoact2"``).
It subclasses :class:`LeRobotBackend` purely to reuse the proven inference
machinery — ``forward`` / ``_to_batch`` / ``_warmup`` / ``_to_chunk_np`` /
``decode_payload`` and the per-step DRTC-DEBUG logging — while owning only
the load path that actually differs. The generic backend has no knowledge of
MolmoAct2; routing happens at the dispatch seam via :func:`resolve_backend`,
so the wire contract (``policy_backend="lerobot"``) is unchanged.

NOTE: the SO100/SO101 calibration migration (lerobot PR #777) that makes the
arm track correctly lives on the *node* side
(``interlatent-sdk .../node/control.py``), not here — this backend only loads
and runs the policy.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from .lerobot_backend import LeRobotBackend
from .policy_runtime import register_backend

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Released-checkpoint detection + routing
# ---------------------------------------------------------------------------
def _read_checkpoint_config_json(policy_uri: str) -> dict:
    """Read just ``config.json`` from a checkpoint (local dir or HF repo).

    Cheap on purpose: for HF repos we pull the single file, never the
    multi-GB weights, so format detection at OpenSession stays fast.
    Returns ``{}`` on any failure (treated as "not molmoact2").
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
        return {}


def is_released_molmoact2(policy_uri: str) -> bool:
    """True for an AllenAI *released* MolmoAct2 checkpoint.

    The released format is transformers-native: ``config.json`` carries
    ``model_type: "molmoact2"`` and NO lerobot ``type`` field. A
    lerobot-*saved* molmoact2 checkpoint, by contrast, has ``type:
    "molmoact2"`` and loads fine through the generic path — so we must
    not claim it here.
    """
    if not policy_uri:
        return False
    cfg = _read_checkpoint_config_json(policy_uri)
    if not cfg:
        return False
    if isinstance(cfg.get("type"), str):
        return False  # lerobot-saved checkpoint — generic path handles it
    return cfg.get("model_type") == "molmoact2"


def resolve_backend(backend: str, policy_uri: str) -> str:
    """Transparently route released MolmoAct2 checkpoints to this backend.

    Callers ask for ``"lerobot"``; if the URI is a released MolmoAct2
    checkpoint we swap in ``"molmoact2"``. Every other (backend, uri)
    pair is returned unchanged. Keeping this here means neither the
    transport nor the generic backend needs to know the detection rule —
    they just call ``resolve_backend(...)`` before ``PolicyRuntime.load``.
    """
    if backend == "lerobot" and is_released_molmoact2(policy_uri):
        return "molmoact2"
    return backend


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------
@register_backend("molmoact2")
class MolmoAct2Backend(LeRobotBackend):
    """Released MolmoAct2 checkpoint loaded for DRTC inference.

    Reuses :class:`LeRobotBackend`'s runtime (forward / batch / warmup /
    RTC trail) and only overrides construction: the generic
    ``__init__`` goes through ``PreTrainedConfig.from_pretrained``, which
    can't decode a transformers-native MolmoAct2 repo, so we skip it and
    assemble a ``MolmoAct2Config`` by hand instead.
    """

    def __init__(
        self,
        chunk_size: int = 0,           # 0 -> use the checkpoint's native value
        action_dim: int = 0,           # 0 -> use the checkpoint's native value
        *,
        policy_uri: str,
        device: Optional[str] = None,  # None -> auto-detect cuda/cpu
        dtype: str = "float32",
        default_task: str = "",
        session_metadata: Optional[dict] = None,
        **_: Any,
    ) -> None:
        if not policy_uri:
            raise ValueError("MolmoAct2Backend requires policy_uri")
        # Shared runtime setup (torch / device / dtype / default_task);
        # the generic lerobot load is deliberately NOT called.
        self._init_runtime_common(device, dtype, default_task)
        self._init_molmoact2_released(
            policy_uri,
            str(self._device),
            dict(session_metadata or {}),
            chunk_size,
            action_dim,
        )

    # ------------------------------------------------------------------
    # MolmoAct2 released-checkpoint loader
    # ------------------------------------------------------------------
    def _init_molmoact2_released(
        self,
        policy_uri: str,
        device: str,
        meta: dict,
        chunk_size: int,
        action_dim: int,
    ) -> None:
        """Load an AllenAI *released* MolmoAct2 checkpoint for inference.

        The released repo is transformers-native and carries no lerobot
        I/O contract, so we assemble a ``MolmoAct2Config`` by hand:
          - normalization + chunk/setup/control metadata come from the
            checkpoint's ``norm_stats.json`` (selected by ``norm_tag``);
          - the camera image keys come from the session metadata (the
            node maps ``--camera name=device`` to
            ``observation.images.<name>``) — the released checkpoint's
            ``camera_keys`` are empty, so this is the one piece that
            must arrive from outside;
          - ``action_dim`` comes from the dashboard session (the robot's
            real action width), falling back to the checkpoint's
            ``action_stats`` dimension.

        Everything else (forward / _to_batch / _warmup / RTC trail) is
        inherited from LeRobotBackend unchanged.
        """
        import json
        import os.path as _osp

        from lerobot.configs import FeatureType, PolicyFeature
        from lerobot.policies.molmoact2.configuration_molmoact2 import (
            MolmoAct2Config,
            _resolve_checkpoint_location,
        )
        from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
        from lerobot.policies.molmoact2.processor_molmoact2 import (
            make_molmoact2_pre_post_processors,
        )

        # --- read norm_stats.json from the checkpoint -------------------
        ckpt_dir = _resolve_checkpoint_location(policy_uri)
        cfg_json = {}
        cfg_path = _osp.join(ckpt_dir, "config.json")
        if _osp.isfile(cfg_path):
            with open(cfg_path) as fh:
                cfg_json = json.load(fh)
        norm_filename = cfg_json.get("norm_stats_filename") or "norm_stats.json"
        norm_path = _osp.join(ckpt_dir, norm_filename)
        if not _osp.isfile(norm_path):
            raise FileNotFoundError(
                f"MolmoAct2 checkpoint {policy_uri!r} is missing {norm_filename!r} "
                f"(looked in {ckpt_dir}). This file holds the action normalization "
                "statistics and ships with the checkpoint repo — make sure you "
                "downloaded the full checkpoint, or point --policy at a directory "
                "that contains it."
            )
        with open(norm_path) as fh:
            norm = json.load(fh)
        by_tag = norm.get("metadata_by_tag") or {}
        if not isinstance(by_tag, dict) or not by_tag:
            raise RuntimeError(
                f"MolmoAct2 checkpoint {policy_uri!r} has no metadata_by_tag in "
                f"{norm_filename}; cannot resolve normalization."
            )
        tags = sorted(by_tag)
        norm_tag = (meta.get("norm_tag") or "").strip() or (
            tags[0] if len(tags) == 1 else ""
        )
        if not norm_tag:
            raise RuntimeError(
                f"MolmoAct2 checkpoint {policy_uri!r} exposes multiple norm tags "
                f"{tags}; pass norm_tag in the session metadata to pick one."
            )
        tag_meta = by_tag[norm_tag]

        def _stat_dim(stats: Any) -> int:
            if not isinstance(stats, dict):
                return 0
            return max(
                (len(v) for v in stats.values() if isinstance(v, list)),
                default=0,
            )

        state_dim = _stat_dim(tag_meta.get("state_stats"))
        resolved_action_dim = int(action_dim) or _stat_dim(tag_meta.get("action_stats"))
        if resolved_action_dim <= 0:
            raise RuntimeError(
                "MolmoAct2 needs a positive action_dim; none in the session and "
                "none derivable from the checkpoint's action_stats."
            )

        # Camera keys: node-supplied (CSV) wins; fall back to whatever the
        # norm metadata names (empty on released SO100/101 checkpoints).
        image_keys = [
            k.strip() for k in str(meta.get("image_keys", "")).split(",") if k.strip()
        ]
        if not image_keys:
            image_keys = [str(k) for k in (tag_meta.get("camera_keys") or [])]
        if not image_keys:
            raise RuntimeError(
                "MolmoAct2 requires at least one camera image key, but none were "
                "provided by the node (session metadata 'image_keys') and the "
                "checkpoint's norm_stats carries no camera_keys. Start the node "
                "with --camera <name>=<device>."
            )

        inference_action_mode = (
            meta.get("inference_action_mode") or "continuous"
        ).strip()

        # Flow-matching denoising steps used at inference. The lerobot
        # wrapper's `num_inference_steps` defaults to None, which falls
        # back to the backbone's `flow_matching_num_steps` — the
        # *training* default (typically ~10). Real-time control needs
        # fewer; 5 is the sweet spot for SO100/101 from MolmoAct2's
        # own demos (visible difference in action quality only below 3).
        # Overridable per-session via OpenSession metadata so operators
        # can A/B without redeploying the GPU.
        try:
            num_inference_steps = max(1, int(meta.get("num_inference_steps", "5")))
        except (TypeError, ValueError):
            num_inference_steps = 5

        # VISUAL features use IDENTITY normalization, so the (C,H,W) shape
        # is only a placeholder — MolmoAct2's image processor resizes.
        input_features: dict[str, Any] = {
            key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))
            for key in image_keys
        }
        input_features["observation.state"] = PolicyFeature(
            type=FeatureType.STATE, shape=(int(state_dim),)
        )
        output_features = {
            "action": PolicyFeature(
                type=FeatureType.ACTION, shape=(int(resolved_action_dim),)
            )
        }

        cfg = MolmoAct2Config(
            checkpoint_path=policy_uri,
            norm_tag=norm_tag,
            inference_action_mode=inference_action_mode,
            chunk_size=int(chunk_size) or 30,
            image_keys=image_keys,
            input_features=input_features,
            output_features=output_features,
            device=device,
            num_inference_steps=num_inference_steps,
        )
        log.info(
            "Loading released MolmoAct2 policy_uri=%s norm_tag=%s "
            "inference_action_mode=%s image_keys=%s state_dim=%d action_dim=%d "
            "num_inference_steps=%d",
            policy_uri, norm_tag, inference_action_mode, image_keys,
            state_dim, resolved_action_dim, num_inference_steps,
        )
        # __init__ applies norm_tag metadata (which can correct chunk_size /
        # n_action_steps), loads the HF weights, and validates features.
        policy = MolmoAct2Policy(cfg).to(self._device).eval()
        self._pre, self._post = make_molmoact2_pre_post_processors(cfg)

        # --- wire the attributes the inherited forward()/_warmup() use --
        self.policy = policy
        self.cfg = cfg
        # RTC and torch.compile are left OFF for MolmoAct2's first cut —
        # the HF transformer backbone hasn't been validated under either.
        # Plain chunking; revisit for latency once it runs.
        self._rtc_ok = False
        self._inpainting_kw = None
        self._last_raw = None
        self._last_start = 0
        self.chunk_size = int(getattr(cfg, "n_action_steps", 0)) or (
            int(chunk_size) or 30
        )
        self.action_dim = int(resolved_action_dim)
        self._predict_chunk = getattr(policy, "predict_action_chunk", None)
        self._select_action = getattr(policy, "select_action", None)
        self._expected_keys = tuple(cfg.input_features.keys())
        log.info(
            "MolmoAct2 loaded chunk_size=%d action_dim=%d expected_keys=%s",
            self.chunk_size, self.action_dim, self._expected_keys,
        )
        # DRTC-DEBUG: the knobs that decide whether actions come out in-range.
        # action_mode is what the checkpoint was trained with; inference mode
        # must be compatible. expected_max_action_dim is the padded width the
        # action expert emits (released checkpoints = 32) — if our action_dim
        # disagrees with how norm stats are aligned, denorm slices wrong and
        # the arm slams. model_dtype/cuda-graph confirm the fast path is on.
        log.info(
            "DRTC-DEBUG MolmoAct2 cfg | norm_tag=%s action_mode=%s "
            "inference_action_mode=%s num_inference_steps=%s "
            "expected_max_action_dim=%s n_action_steps=%s chunk_size=%s "
            "model_dtype=%s enable_inference_cuda_graph=%s "
            "normalize_gripper=%s normalize_language=%s setup_type=%r "
            "control_mode=%r",
            getattr(cfg, "norm_tag", "?"),
            getattr(cfg, "action_mode", "?"),
            getattr(cfg, "inference_action_mode", "?"),
            getattr(cfg, "num_inference_steps", "?"),
            getattr(cfg, "expected_max_action_dim", "?"),
            getattr(cfg, "n_action_steps", "?"),
            getattr(cfg, "chunk_size", "?"),
            getattr(cfg, "model_dtype", "?"),
            getattr(cfg, "enable_inference_cuda_graph", "?"),
            getattr(cfg, "normalize_gripper", "?"),
            getattr(cfg, "normalize_language", "?"),
            getattr(cfg, "setup_type", "?"),
            getattr(cfg, "control_mode", "?"),
        )
        self._warmup()
        self._last_raw = None
        self._last_start = 0
