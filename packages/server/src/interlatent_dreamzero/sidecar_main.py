"""DreamZero sidecar entrypoint. Runs under the *DreamZero* interpreter.

Launched by :mod:`interlatent.inference.server.dreamzero_sidecar` as::

    python -m torch.distributed.run --standalone --nproc_per_node=N \\
        -m interlatent_dreamzero.sidecar_main --socket <path> --model-path <dir>

**This module must not import anything from ``interlatent``.** It lives in a
different environment (torch 2.8 / CUDA 12.9) than the engine (torch 2.7.1 /
cuda 12.8), and the two share no dependency set. That is the whole reason the
sidecar exists. It is a separate top-level package for the same reason — it
also keeps clear of the ``interlatent`` module name that both the SDK and the
engine already claim.

Rank 0 owns the socket and the protocol; other ranks join the collective and
otherwise stay out of the way. State is a single rolling KV cache, because
``serve_gpu`` runs one session per box.

Protocol (mirrors the client)::

    [4B len][4B header_len][JSON header][blob]...

Operations:

* ``ready``   -> ``{ready: bool, contract: {...}}``
* ``context`` -> append frames to the KV cache; blobs are JPEGs, ordered by
                 the ``cameras`` list of each entry in ``frames``
* ``infer``   -> ``{shape: [H, D]}`` + one float32 blob of **normalized
                 relative** actions; de-normalization happens engine-side
* ``reset``   -> drop the KV cache
* ``state``   -> ``{last_seq: int}``, the dedupe high-water mark
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import socket
import struct
import sys
import traceback

logging.basicConfig(
    level=os.environ.get("DREAMZERO_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [dreamzero-sidecar] %(message)s",
)
log = logging.getLogger(__name__)

_LEN = struct.Struct(">I")
_MAX_FRAME_BYTES = 256 * 1024 * 1024


# ----------------------------------------------------------------------
# Model adapter — the ONLY place that touches the DreamZero API
# ----------------------------------------------------------------------


class DreamZeroModel:
    """Thin adapter over the upstream model.

    Everything DreamZero-specific is confined to this class so that an
    upstream refactor is a change in one file, not across the serving path.

    NOTE FOR THE FIRST BRING-UP: the four call sites below are the integration
    points against the real ``groot.vla`` API. They are written against the
    documented behaviour (33-frame / 5 FPS context, joint video+action
    denoising, normalized relative joint outputs) and must be checked against
    the actual signatures on the box — that check is verification step 1 in
    the plan, and it is deliberately the first thing done, before any of this
    is trusted.
    """

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self._model = None
        self._frames: list = []
        self._last_seq = -1
        self.contract: dict = {}

    def load(self) -> None:
        import torch  # noqa: F401  (ensures the distributed context is live)

        log.info("Loading DreamZero checkpoint from %s", self.model_path)
        # --- integration point 1: construct + shard the model -------------
        from groot.vla.model.dreamzero import DreamZeroPolicy  # type: ignore

        self._model = DreamZeroPolicy.from_pretrained(self.model_path)
        self._model.eval()

        self.contract = self._read_contract()
        log.info("Contract: %s", self.contract)

    def _read_contract(self) -> dict:
        """The checkpoint's own shape. Never hardcode DROID's numbers here.

        A fine-tune may declare a different horizon, control rate, action
        width or camera layout than the weights it derives from, and making
        fine-tunes selectable by ``policy_uri`` is the point of the feature.
        """
        cfg_path = os.path.join(self.model_path, "config.json")
        cfg = {}
        if os.path.isfile(cfg_path):
            with open(cfg_path) as fh:
                cfg = json.load(fh)

        stats = None
        stats_path = os.path.join(self.model_path, "relative_stats_dreamzero.json")
        if os.path.isfile(stats_path):
            with open(stats_path) as fh:
                stats = json.load(fh)

        # --- integration point 2: the model's declared shape --------------
        m = self._model
        return {
            "horizon": int(getattr(m, "action_horizon", 0) or cfg.get("action_horizon", 0)),
            "fps": float(getattr(m, "control_fps", 0) or cfg.get("control_fps", 0)),
            "action_dim": int(getattr(m, "action_dim", 0) or cfg.get("action_dim", 0)),
            "context_frames": int(getattr(m, "context_frames", 0) or cfg.get("context_frames", 33)),
            "camera_slots": list(getattr(m, "camera_slots", None) or cfg.get("camera_slots") or []),
            "relative_stats": stats,
        }

    def append_context(self, seq: int, cameras: list, blobs: list) -> None:
        """Encode one multi-view sample into the rolling context.

        ``cameras`` is ordered by the checkpoint's declared slot layout, set
        engine-side. Order is load-bearing and wrong order fails silently —
        the model was trained on a fixed view arrangement.
        """
        from PIL import Image

        views = [Image.open(io.BytesIO(b)).convert("RGB") for b in blobs]
        # --- integration point 3: VAE-encode and append to the KV cache ----
        self._model.append_observation(views, camera_names=cameras)
        self._frames.append(seq)
        self._last_seq = max(self._last_seq, seq)
        keep = self.contract.get("context_frames") or 33
        if len(self._frames) > keep:
            self._frames = self._frames[-keep:]

    def infer(self, task: str, state: list):
        """Return normalized relative actions, shape (H, D), float32.

        De-normalization is deliberately *not* done here: the engine owns the
        conversion to joint units so that the stats used are the ones the
        engine validated at load, rather than a second copy in a second
        environment that could drift.
        """
        import numpy as np
        import torch

        with torch.no_grad():
            # --- integration point 4: joint video+action denoising ---------
            out = self._model.predict_action(
                instruction=task,
                state=torch.tensor(state, dtype=torch.float32),
            )
        arr = out.detach().float().cpu().numpy() if hasattr(out, "detach") else np.asarray(out)
        return np.ascontiguousarray(arr.reshape(-1, self.contract["action_dim"]), dtype="float32")

    def reset(self) -> None:
        self._frames.clear()
        self._last_seq = -1
        if self._model is not None and hasattr(self._model, "reset"):
            self._model.reset()

    @property
    def last_seq(self) -> int:
        return self._last_seq


# ----------------------------------------------------------------------
# Framed socket server
# ----------------------------------------------------------------------


def _recv_exact(conn, n: int) -> bytes:
    chunks, got = [], 0
    while got < n:
        b = conn.recv(min(n - got, 1 << 20))
        if not b:
            raise ConnectionError("client closed mid-frame")
        chunks.append(b)
        got += len(b)
    return b"".join(chunks)


def _read_frame(conn):
    total = _LEN.unpack(_recv_exact(conn, _LEN.size))[0]
    if total > _MAX_FRAME_BYTES:
        raise ValueError(f"frame of {total} bytes exceeds cap")
    buf = _recv_exact(conn, total)
    head_len = _LEN.unpack(buf[: _LEN.size])[0]
    head = json.loads(buf[_LEN.size : _LEN.size + head_len].decode())
    rest = buf[_LEN.size + head_len :]
    blobs, off = [], 0
    for n in head.get("blobs", []):
        blobs.append(rest[off : off + n])
        off += n
    return head, blobs


def _send_frame(conn, head: dict, blobs=None) -> None:
    blobs = blobs or []
    head = dict(head)
    head["blobs"] = [len(b) for b in blobs]
    raw = json.dumps(head).encode()
    body = b"".join([_LEN.pack(len(raw)), raw, *blobs])
    conn.sendall(_LEN.pack(len(body)) + body)


def _handle(model: DreamZeroModel, loaded: dict, head: dict, blobs: list) -> tuple:
    op = head.get("op")
    if op == "ready":
        return {"ready": loaded["ok"], "contract": model.contract}, []
    if op == "state":
        return {"last_seq": model.last_seq}, []
    if op == "reset":
        model.reset()
        return {"ok": True}, []
    if op == "context":
        i = 0
        for entry in head.get("frames", []):
            cams = entry.get("cameras", [])
            model.append_context(int(entry["seq"]), cams, blobs[i : i + len(cams)])
            i += len(cams)
        return {"ok": True, "last_seq": model.last_seq}, []
    if op == "infer":
        arr = model.infer(head.get("task", ""), head.get("state") or [])
        return {"shape": list(arr.shape)}, [arr.tobytes()]
    return {"error": f"unknown op {op!r}"}, []


def serve(socket_path: str, model_path: str) -> int:
    rank = int(os.environ.get("RANK", "0"))
    model = DreamZeroModel(model_path)
    loaded = {"ok": False}

    if rank != 0:
        # Non-zero ranks join the collective through load() and then park;
        # only rank 0 talks to the engine.
        model.load()
        log.info("rank %d loaded; parking", rank)
        try:
            import torch.distributed as dist

            while True:
                dist.barrier()
        except Exception:
            return 0

    if os.path.exists(socket_path):
        os.unlink(socket_path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(socket_path)
    srv.listen(1)
    # Bind BEFORE loading weights: the client's socket-wait covers process
    # start only, and readiness is reported through the `ready` op instead —
    # a 46 GB load would otherwise look like a failed spawn.
    log.info("listening on %s; loading weights", socket_path)

    try:
        model.load()
        loaded["ok"] = True
    except Exception:
        log.error("model load FAILED:\n%s", traceback.format_exc())

    while True:
        conn, _ = srv.accept()
        log.info("engine connected")
        try:
            while True:
                head, blobs = _read_frame(conn)
                try:
                    resp, out = _handle(model, loaded, head, blobs)
                except Exception as exc:
                    log.error("op %r failed:\n%s", head.get("op"), traceback.format_exc())
                    resp, out = {"error": f"{type(exc).__name__}: {exc}"}, []
                _send_frame(conn, resp, out)
        except (ConnectionError, OSError, ValueError) as exc:
            log.info("engine disconnected (%s)", exc)
        finally:
            conn.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", required=True)
    ap.add_argument("--model-path", required=True)
    args = ap.parse_args()
    return serve(args.socket, args.model_path)


if __name__ == "__main__":
    sys.exit(main())
