"""DreamZero (world-action model) policy backend. See ADR 0037.

DreamZero jointly denoises future video and an action chunk. Serving it through
DRTC means reconciling four mismatches with every policy this server has run
before, and this module is where each one is absorbed so nothing upstream has
to know:

1. **It is stateful.** A rolling KV cache over ~6.6 s of video. The frames
   arrive inside the ``Infer`` payload under a ``ctx.`` namespace (the node
   ships its whole window every time), and we forward only what the sidecar
   has not already encoded.
2. **It is multi-process and multi-GPU.** The model lives in a ``torchrun``
   sidecar on its own torch/CUDA floor; ``forward()`` is a blocking local
   socket round-trip. See :mod:`dreamzero_sidecar`.
3. **It emits relative actions.** Everything else here returns absolute
   robot-space actions, so we re-anchor on the proprioceptive state captured
   with the observation. We do NOT re-scale them: the policy's own eval
   transform already applied the inverse normalization from
   ``experiment_cfg/metadata.json`` before returning. See the note where the
   de-normalization step used to be.
4. **Its chunk is expressed at the checkpoint's own control rate**, not the
   node's. A DROID-lineage chunk is 24 steps at 15 Hz; a 30 Hz node consuming
   it verbatim plays the trajectory at double speed.

Nothing in this file may hardcode DROID's numbers. The horizon, fps, action
dimension and camera slots all come from the sidecar's declared contract —
which the sidecar reads from ``experiment_cfg/metadata.json``, not
``config.json`` — because a fine-tune can differ from the weights it derives
from, which is the entire point of making fine-tunes selectable by
``policy_uri``.

There is no RTC in-painting here: DreamZero exposes nothing equivalent to
lerobot's ``prev_chunk_left_over``, so seams are smoothed with the shared
cross-fade in :mod:`chunk_seam` instead.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Optional

import numpy as np

from .chunk_seam import crossfade_chunk
from .dreamzero_sidecar import DreamZeroSidecar, SidecarError
from .policy_runtime import register_backend

log = logging.getLogger(__name__)

#: Keys carrying the world-model context window in the npz payload. Namespaced
#: by the node precisely so this is unambiguous: ``_to_batch`` treats *any*
#: uint8 array as a camera frame, so an un-namespaced context frame would be
#: fed to a lerobot policy as an extra observation image.
_CTX_META = "ctx.meta"
_CTX_FRAME = re.compile(r"^ctx\.f(\d+)\.(.+)$")

#: Fallback node control rate when the session metadata omits it. The node's
#: own default is 30 Hz.
_DEFAULT_NODE_FPS = 30.0


def is_dreamzero(policy_uri: str, config: Optional[dict] = None) -> bool:
    """Identify a DreamZero checkpoint from its config, not its URI.

    MolmoAct2 could get away with a URI substring because
    ``is_released_molmoact2`` only ever matched checkpoints *we* released.
    Here arbitrary user fine-tunes are the point, so a substring test both
    misses ``myorg/my-dreamzero-ft`` and false-positives on anything with
    "dream" in the name. The config is authoritative; the URI is only a hint
    used when no config could be fetched at all.

    Keyed on the Hydra ``_target_`` of the action head / backbone, because that
    is the only field that actually names the family. A real DreamZero-DROID
    config says ``model_type: "vla"`` and ``architectures: ["VLA"]`` — shared
    with GR00T N1.5, which is the same ``groot`` codebase — so neither can
    discriminate. The instantiation targets can:
    ``groot.vla.model.dreamzero.action_head...`` vs ``...model.n1_5...``.
    A fine-tune inherits its head class, so this survives one.
    """
    for block in ("action_head_cfg", "backbone_cfg"):
        cfg = (config or {}).get(block)
        if isinstance(cfg, dict) and "model.dreamzero" in str(cfg.get("_target_") or ""):
            return True
    if config:
        model_type = str(config.get("model_type") or config.get("type") or "").lower()
        if "dreamzero" in model_type or "world_action" in model_type:
            return True
    return False


def resolve_backend(backend: str, policy_uri: str, config: Optional[dict] = None) -> str:
    """Dispatch-seam arm, mirroring ``molmoact2_backend.resolve_backend``.

    Lets a session opened against a DreamZero checkpoint land on this backend
    without the wire contract having to name it.
    """
    if backend in ("", "lerobot") and is_dreamzero(policy_uri, config):
        return "dreamzero"
    return backend


def _resample(chunk: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    """Resample a chunk from the checkpoint's control rate to the node's.

    Linear interpolation in **absolute joint space**, applied after the
    relative-to-absolute conversion — interpolating normalized deltas would
    be interpolating in a space where the endpoints mean different things.

    Preserves wall-clock duration: a 24-step chunk at 15 Hz is 1.6 s of
    motion, and it stays 1.6 s at 30 Hz by becoming 48 steps. That duration
    is what the client's pacing and the sync-mode hold are both reasoning
    about.
    """
    if chunk.shape[0] < 2 or src_fps <= 0 or dst_fps <= 0:
        return chunk
    ratio = dst_fps / src_fps
    if abs(ratio - 1.0) < 1e-6:
        return chunk
    n_src = chunk.shape[0]
    n_dst = max(1, int(round(n_src * ratio)))
    src_idx = np.arange(n_src, dtype=np.float64)
    dst_idx = np.linspace(0.0, n_src - 1, n_dst, dtype=np.float64)
    out = np.empty((n_dst, chunk.shape[1]), dtype=np.float32)
    for j in range(chunk.shape[1]):
        out[:, j] = np.interp(dst_idx, src_idx, chunk[:, j].astype(np.float64))
    return out


@register_backend("dreamzero")
class DreamZeroBackend:
    """World-action model backend fronting a ``torchrun`` sidecar."""

    def __init__(
        self,
        chunk_size: int = 0,
        action_dim: int = 0,
        *,
        policy_uri: str,
        default_task: str = "",
        session_metadata: Optional[dict] = None,
        gpus: int = 0,
        **_: Any,
    ) -> None:
        md = session_metadata or {}
        self._policy_uri = policy_uri
        self._default_task = default_task

        # The node's control rate, so the resample target is the rate the
        # robot actually runs at rather than an assumption.
        try:
            self._node_fps = float(md.get("fps") or _DEFAULT_NODE_FPS)
        except (TypeError, ValueError):
            self._node_fps = _DEFAULT_NODE_FPS

        gpus = gpus or int(os.environ.get("DREAMZERO_GPUS", "2") or 2)
        self._sidecar = DreamZeroSidecar(
            model_path=self._resolve_model_path(policy_uri),
            gpus=gpus,
        )
        contract = self._sidecar.start()

        # Shape comes from the checkpoint, never from a constant here. An
        # explicit non-zero request from OpenSession still wins, matching how
        # LeRobotBackend resolves its own shape.
        self._src_fps = float(contract.get("fps") or 0.0)
        self._horizon = int(contract.get("horizon") or 0)
        self.action_dim = int(action_dim or contract.get("action_dim") or 0)
        self._camera_slots = list(contract.get("camera_slots") or [])

        # What we advertise is the *resampled* width, because that is what the
        # client will pace against — the OpenSessionResponse carries this
        # straight back to the node.
        native = np.zeros((max(self._horizon, 1), max(self.action_dim, 1)), np.float32)
        self.chunk_size = int(chunk_size or _resample(
            native, self._src_fps, self._node_fps
        ).shape[0])

        # Seam trail, in robot-action space. Same role as LeRobotBackend's
        # ``_last_processed`` pair.
        self._last_processed: Optional[np.ndarray] = None
        self._last_processed_start: int = 0
        self._last_ctx_seq: int = -1

        log.info(
            "DreamZeroBackend ready: uri=%s horizon=%d@%.1fHz -> chunk=%d@%.1fHz "
            "action_dim=%d cameras=%s",
            policy_uri, self._horizon, self._src_fps, self.chunk_size,
            self._node_fps, self.action_dim, self._camera_slots,
        )

    # -- construction helpers -------------------------------------------

    @staticmethod
    def _resolve_model_path(policy_uri: str) -> str:
        """Local path for the checkpoint, downloading from HF if needed.

        Kept eager rather than letting the sidecar fetch it: the download is
        ~46 GB and belongs on the box's cache volume under credentials this
        process already has, not in an environment we do not control.
        """
        if os.path.isdir(policy_uri):
            return policy_uri
        if policy_uri.startswith("s3://"):
            raise SidecarError(
                f"s3:// policy_uri is not loadable yet ({policy_uri}); this is "
                "ADR 0012's standing gap. Use a local path or an HF repo id."
            )
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise SidecarError(
                "huggingface_hub is required to fetch a DreamZero checkpoint"
            ) from exc
        log.info("Fetching DreamZero checkpoint %s (this is ~46 GB)", policy_uri)
        return snapshot_download(policy_uri)

    # NOTE: there is deliberately no de-normalization step here.
    #
    # An earlier draft loaded ``relative_stats_dreamzero.json`` and undid a
    # q01/q99 min-max normalization. No such file exists in the checkpoint,
    # and no such step is needed: a ``groot`` checkpoint keeps its statistics
    # in ``experiment_cfg/metadata.json``, and the policy's own eval transform
    # applies the inverse before returning. Upstream's serving path does
    # nothing to the returned actions but concatenate the sub-keys, so
    # re-applying it here would double-normalize a chunk that is already in
    # joint units — plausible-looking and wrong.
    #
    # The relative -> absolute anchoring below is a SEPARATE question and is
    # kept: the DROID checkpoints train on ``droid_relative``, and the action
    # statistics are delta-shaped (joint_position mean about 0, q01/q99 about
    # +/-0.5 rad against +/-2.4 rad limits). That is strong evidence, not
    # proof — the numeric oracle run against upstream's own server is what
    # settles sign and frame convention.

    # -- session lifecycle ----------------------------------------------

    def reset_session_state(self) -> None:
        """Drop the KV cache and the seam trail (ADR 0037 §5).

        The sidecar reset is the load-bearing half: this backend is cached
        process-wide by ``(backend, policy_uri)``, so without it session 2
        would continue dreaming from session 1's final frames — a wrong,
        entirely plausible-looking chunk with nothing logged.
        """
        self._last_processed = None
        self._last_processed_start = 0
        self._last_ctx_seq = -1
        try:
            self._sidecar.reset()
        except SidecarError:
            log.exception("Sidecar reset failed; the next session may carry stale context")

    def close(self) -> None:
        self._sidecar.stop()

    # -- context handling -----------------------------------------------

    @staticmethod
    def _unpack_context(observation: dict) -> Optional[dict]:
        """Pull the ``ctx.*`` window out of a decoded npz observation.

        Returns ``{"first_seq", "produced", "fps", "frames": [{cam: bytes}]}``
        or None. Frames stay JPEG-encoded all the way to the sidecar: decoding
        them here would spend CPU on the inference executor to hand the
        sidecar something it has to re-encode into latents anyway.
        """
        meta = observation.get(_CTX_META)
        if meta is None:
            return None
        meta = np.asarray(meta).reshape(-1)
        if meta.size < 4:
            return None
        first_seq, n_frames, produced, fps_milli = (int(x) for x in meta[:4])

        frames: list = [dict() for _ in range(n_frames)]
        for key, value in observation.items():
            m = _CTX_FRAME.match(key)
            if not m:
                continue
            idx = int(m.group(1))
            if 0 <= idx < n_frames:
                frames[idx][m.group(2)] = np.asarray(value, dtype=np.uint8).tobytes()
        return {
            "first_seq": first_seq,
            "produced": produced,
            "fps": fps_milli / 1000.0,
            "frames": frames,
        }

    def _forward_context(self, ctx: dict) -> None:
        """Send the frames the sidecar has not encoded yet.

        The node ships its whole window on every observation (which is what
        makes a dropped ``Infer`` self-healing), so the dedupe happens here:
        we forward only frames whose sequence is past the sidecar's high-water
        mark. Re-sending costs bandwidth, never correctness.
        """
        first = ctx["first_seq"]
        frames = ctx["frames"]
        start = max(0, self._last_ctx_seq + 1 - first)
        if start >= len(frames):
            return

        # produced > len(frames) means the ring overflowed between sends: the
        # window is a tail and some motion was never shown to the model. Say
        # so rather than let the sidecar assume contiguity.
        if ctx["produced"] > len(frames):
            log.warning(
                "context window is a tail: %d frames produced, %d retained — "
                "%d frames of motion were never shown to the model",
                ctx["produced"], len(frames), ctx["produced"] - len(frames),
            )

        blobs: list = []
        layout: list = []
        for i in range(start, len(frames)):
            cams = frames[i]
            # Order matters: the model was trained on a fixed view layout and
            # degrades silently if the slots are permuted. When the checkpoint
            # declares slots, honour that order; otherwise fall back to a
            # stable sort so at least the order is consistent across frames.
            names = (
                [c for c in self._camera_slots if c in cams]
                if self._camera_slots else sorted(cams)
            )
            layout.append({"seq": first + i, "cameras": names})
            blobs.extend(cams[n] for n in names)

        self._sidecar.request({"op": "context", "frames": layout}, blobs)
        self._last_ctx_seq = first + len(frames) - 1

    # -- forward ---------------------------------------------------------

    def forward(
        self,
        observation: "np.ndarray | dict",
        prior_actions: Optional[np.ndarray],
        *,
        next_action_step: int = 0,
        inference_delay: int = 0,
    ) -> np.ndarray:
        if not isinstance(observation, dict):
            raise SidecarError(
                "DreamZero requires the npz payload codec (it needs camera "
                "frames and a context window, not a flat state vector)."
            )

        ctx = self._unpack_context(observation)
        if ctx is None:
            # Not fatal: the very first observation of a session can legally
            # precede the ring's first sample. It IS fatal if it persists —
            # the video branch drives the action branch, so a permanently
            # context-free session produces confident nonsense.
            log.warning(
                "observation carried no ctx.* window — the world model is "
                "running without video context and its actions will degrade. "
                "Expected on the first tick only; if this repeats, the node is "
                "too old to build a context ring (ADR 0037)."
            )
        else:
            self._forward_context(ctx)

        state = observation.get("observation.state")
        if state is None:
            raise SidecarError("DreamZero requires observation.state (proprioception)")
        state = np.asarray(state, dtype=np.float32).reshape(-1)

        task = observation.get("task") or self._default_task
        if isinstance(task, (list, tuple, np.ndarray)):
            task = task[0] if len(task) else ""
        task = str(task)

        resp = self._sidecar.request(
            {"op": "infer", "task": task, "state": state.tolist()}
        )
        raw = self._decode_actions(resp)

        # Normalized relative -> joint-space relative -> absolute, anchored on
        # the state captured *with this observation*. In synchronous mode the
        # arm is stationary while inference runs, so that anchor is still valid
        # when the chunk lands — which is the second reason sync mode is a
        # correctness requirement and not just a latency workaround.
        rel = raw
        absolute = state[None, : rel.shape[1]] + rel

        chunk = _resample(absolute, self._src_fps, self._node_fps)
        chunk = crossfade_chunk(
            chunk,
            self._last_processed,
            self._last_processed_start,
            next_action_step,
            inference_delay,
        )
        self._last_processed = chunk
        self._last_processed_start = int(next_action_step)
        return chunk.astype(np.float32, copy=False)

    @staticmethod
    def _decode_actions(resp: dict) -> np.ndarray:
        blobs = resp.get("_blobs") or []
        if not blobs:
            raise SidecarError("Sidecar returned no action payload")
        shape = tuple(resp.get("shape") or ())
        arr = np.frombuffer(blobs[0], dtype=np.float32)
        if len(shape) == 2:
            arr = arr.reshape(shape)
        elif arr.ndim == 1:
            raise SidecarError(f"Sidecar action payload has no shape: {resp!r}")
        if not np.isfinite(arr).all():
            # The node rejects this too, but catching it here names the sidecar
            # as the culprit instead of leaving a bare "policy action rejected"
            # in the robot's log.
            raise SidecarError(
                "Sidecar returned a non-finite action chunk — the model "
                "diverged (check for a dtype/quantization regression)."
            )
        return arr

