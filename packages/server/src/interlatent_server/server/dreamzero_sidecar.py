"""Supervisor + client for the DreamZero sidecar process group (ADR 0037).

A world-action model cannot run inside ``serve_gpu``. It needs a
``torchrun --nproc_per_node=N`` process group on torch 2.8 / CUDA 12.9, while
the engine is pinned at torch 2.7.1 on cuda 12.8 (ADR 0011) — irreconcilable
in one interpreter. So it runs beside us, in its own environment, and we talk
to it over a Unix socket.

``serve_gpu`` keeps everything that matters: the gRPC front door, the session
registry, the recorder, the box-status reconcile loop, ADR 0035 auth. The
sidecar owns exactly one thing — weights on GPUs — and never speaks to the
network or to the backend.

**Why not their WebSocket server.** The DreamZero repo ships
``socket_test_optimized_AR.py``, but it is a test harness: it writes an MP4 and
dumps the input observations per request, which on a long session is both a
disk leak and a competitor for the CPU cores ``serve_gpu`` has deliberately
partitioned for recording. We ship our own entrypoint (``sidecar_main.py``)
that runs under *their* interpreter and serves the protocol below.

**Wire protocol.** Length-prefixed frames, stdlib only — no msgpack, no
protobuf, because the sidecar's environment is not ours to add dependencies
to::

    [4B big-endian total length][JSON header][blob][blob]...

The header carries a ``blobs`` array of byte lengths so the reader can split
the payload without scanning it. Binary rides beside the JSON rather than
inside it because base64 on ~1 MB of JPEG per request is pure waste.

Everything here is synchronous and single-threaded by construction: the
backend's ``forward()`` is already called on ``serve_gpu``'s single-worker
inference executor, so one request is in flight at a time.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import socket
import struct
import subprocess
import time
from typing import Any, Optional

log = logging.getLogger(__name__)

_LEN = struct.Struct(">I")

#: Cap on a single frame. A context window is ~1-2 MB; anything an order of
#: magnitude past that means the stream desynchronised, and we should fail
#: rather than try to allocate it.
_MAX_FRAME_BYTES = 256 * 1024 * 1024

#: How long to wait for the sidecar's socket to appear after spawn. Model load
#: is ~46 GB of bf16 plus compile; the socket binds *before* weights load, so
#: this only covers process start, not readiness.
_SOCKET_WAIT_S = float(os.environ.get("DREAMZERO_SOCKET_WAIT_S", "120"))

#: How long to wait for the sidecar to report itself loaded and warm.
_READY_TIMEOUT_S = float(os.environ.get("DREAMZERO_READY_TIMEOUT_S", "1800"))

#: Per-request ceiling. Generous: an unquantized 23B on 2xH100 is ~3 s per
#: chunk, and a cold first call is far worse. This exists to turn a wedged
#: sidecar into an error instead of a hung session, not to enforce latency.
_REQUEST_TIMEOUT_S = float(os.environ.get("DREAMZERO_REQUEST_TIMEOUT_S", "120"))


class SidecarError(RuntimeError):
    """The sidecar failed, died, or answered incoherently."""


def _default_command(*, socket_path: str, model_path: str, gpus: int) -> list[str]:
    """The command we spawn when the operator has not overridden it.

    Deliberately references our own entrypoint rather than anything in the
    DreamZero tree, so an upstream refactor of their test harness cannot break
    a running box.
    """
    python = os.environ.get("DREAMZERO_PYTHON", "python")
    return [
        python,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={gpus}",
        "-m",
        "interlatent_dreamzero.sidecar_main",
        "--socket",
        socket_path,
        "--model-path",
        model_path,
    ]


class DreamZeroSidecar:
    """Owns the sidecar process group and the socket to it.

    Lifetime is the backend's lifetime, which — because backends are cached
    process-wide by ``(backend, policy_uri)`` — means one sidecar per distinct
    checkpoint for as long as the box lives. Switching ``policy_uri`` builds a
    new backend and therefore a new sidecar; that is the ~46 GB reload called
    out in ADR 0037's consequences.
    """

    def __init__(
        self,
        *,
        model_path: str,
        socket_path: Optional[str] = None,
        gpus: int = 2,
        command: Optional[str] = None,
        env: Optional[dict] = None,
    ) -> None:
        self.model_path = model_path
        self.gpus = max(1, int(gpus))
        self.socket_path = socket_path or os.environ.get(
            "DREAMZERO_SOCKET", "/tmp/interlatent-dreamzero.sock"
        )
        self._command = command or os.environ.get("DREAMZERO_SIDECAR_CMD") or None
        self._env = env
        self._proc: Optional[subprocess.Popen] = None
        self._sock: Optional[socket.socket] = None
        self._owns_process = False

    # -- lifecycle ------------------------------------------------------

    def start(self) -> dict:
        """Spawn (or attach to) the sidecar and block until it is ready.

        Returns the sidecar's declared contract — the checkpoint's own action
        horizon, control fps, action dimension, camera slots and relative
        normalization stats. The backend takes its shape from this rather than
        hardcoding DROID's numbers, because a fine-tune may declare different
        ones (ADR 0037 §4).
        """
        if os.path.exists(self.socket_path) and self._probe():
            # An operator-managed sidecar is already serving. Attach rather
            # than spawn: on a BYO box the sidecar may well be run by hand or
            # by systemd, and double-loading 46 GB would OOM the machine.
            log.info("Attaching to existing DreamZero sidecar at %s", self.socket_path)
            self._owns_process = False
        else:
            self._spawn()
            self._owns_process = True

        self._connect()
        contract = self._await_ready()
        log.info(
            "DreamZero sidecar ready: horizon=%s fps=%s action_dim=%s cameras=%s",
            contract.get("horizon"), contract.get("fps"),
            contract.get("action_dim"), contract.get("camera_slots"),
        )
        return contract

    def _spawn(self) -> None:
        if self._command:
            argv = shlex.split(self._command)
        else:
            argv = _default_command(
                socket_path=self.socket_path,
                model_path=self.model_path,
                gpus=self.gpus,
            )
        # A stale socket file from a crashed predecessor makes bind() fail
        # inside the sidecar, which surfaces here as an opaque start timeout.
        try:
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)
        except OSError:
            log.warning("Could not remove stale socket %s", self.socket_path)

        env = dict(os.environ)
        if self._env:
            env.update(self._env)
        log.info("Spawning DreamZero sidecar: %s", " ".join(argv))
        try:
            self._proc = subprocess.Popen(argv, env=env)
        except FileNotFoundError as exc:
            raise SidecarError(
                f"Could not spawn the DreamZero sidecar ({exc}). Set "
                "DREAMZERO_PYTHON to the interpreter that has DreamZero "
                "installed, or DREAMZERO_SIDECAR_CMD to the full command."
            ) from exc

        deadline = time.monotonic() + _SOCKET_WAIT_S
        while time.monotonic() < deadline:
            if os.path.exists(self.socket_path):
                return
            rc = self._proc.poll()
            if rc is not None:
                raise SidecarError(
                    f"DreamZero sidecar exited with code {rc} before binding "
                    f"{self.socket_path}. Check the sidecar's own stderr — it "
                    "runs under a different interpreter, so its import errors "
                    "do not appear in this process's traceback."
                )
            time.sleep(0.25)
        self.stop()
        raise SidecarError(
            f"DreamZero sidecar did not bind {self.socket_path} within "
            f"{_SOCKET_WAIT_S:.0f}s."
        )

    def _probe(self) -> bool:
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(2.0)
            s.connect(self.socket_path)
            s.close()
            return True
        except OSError:
            return False

    def _connect(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(_REQUEST_TIMEOUT_S)
        sock.connect(self.socket_path)
        self._sock = sock

    def _await_ready(self) -> dict:
        """Poll ``ready`` until the sidecar reports weights loaded and warm."""
        deadline = time.monotonic() + _READY_TIMEOUT_S
        last: dict = {}
        while time.monotonic() < deadline:
            last = self.request({"op": "ready"})
            if last.get("ready"):
                return last.get("contract") or {}
            self._assert_alive()
            time.sleep(2.0)
        raise SidecarError(
            f"DreamZero sidecar was not ready within {_READY_TIMEOUT_S:.0f}s "
            f"(last status: {last!r}). Loading ~46 GB of weights plus compile "
            "is slow on a cold cache; raise DREAMZERO_READY_TIMEOUT_S if this "
            "is a first boot on an unmounted cache volume."
        )

    def _assert_alive(self) -> None:
        if self._owns_process and self._proc is not None:
            rc = self._proc.poll()
            if rc is not None:
                raise SidecarError(
                    f"DreamZero sidecar died with code {rc}. On a multi-rank "
                    "group, a crash in any rank takes the whole group down."
                )

    def stop(self) -> None:
        try:
            if self._sock is not None:
                self._sock.close()
        except OSError:
            pass
        self._sock = None
        if self._owns_process and self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                log.warning("Sidecar ignored SIGTERM; killing")
                self._proc.kill()
        self._proc = None

    # -- request/response ----------------------------------------------

    def request(self, header: dict, blobs: Optional[list] = None) -> dict:
        """Send one framed request, block for the response.

        The response's binary payload, when present, is attached to the
        returned dict under ``_blobs`` so callers get one object back.
        """
        if self._sock is None:
            raise SidecarError("Sidecar socket is not connected")
        blobs = blobs or []
        head = dict(header)
        head["blobs"] = [len(b) for b in blobs]
        raw = json.dumps(head).encode("utf-8")
        body = b"".join([_LEN.pack(len(raw)), raw, *blobs])
        try:
            self._sock.sendall(_LEN.pack(len(body)) + body)
            resp_head, resp_blobs = self._read_frame()
        except (OSError, struct.error) as exc:
            self._assert_alive()
            raise SidecarError(f"Sidecar transport failed: {exc}") from exc
        if resp_head.get("error"):
            raise SidecarError(f"Sidecar error: {resp_head['error']}")
        resp_head["_blobs"] = resp_blobs
        return resp_head

    def _read_frame(self) -> tuple:
        total = _LEN.unpack(self._recv_exact(_LEN.size))[0]
        if total > _MAX_FRAME_BYTES:
            raise SidecarError(
                f"Sidecar frame of {total} bytes exceeds the {_MAX_FRAME_BYTES} "
                "cap — the stream has desynchronised."
            )
        buf = self._recv_exact(total)
        head_len = _LEN.unpack(buf[: _LEN.size])[0]
        head = json.loads(buf[_LEN.size : _LEN.size + head_len].decode("utf-8"))
        rest = buf[_LEN.size + head_len :]
        out, off = [], 0
        for n in head.get("blobs", []):
            out.append(rest[off : off + n])
            off += n
        return head, out

    def _recv_exact(self, n: int) -> bytes:
        chunks, got = [], 0
        while got < n:
            b = self._sock.recv(min(n - got, 1 << 20))
            if not b:
                self._assert_alive()
                raise SidecarError("Sidecar closed the connection mid-frame")
            chunks.append(b)
            got += len(b)
        return b"".join(chunks)

    # -- operations -----------------------------------------------------

    def reset(self, session_id: str = "") -> None:
        """Drop the KV cache. Called at every session boundary (ADR 0037 §5)."""
        self.request({"op": "reset", "session_id": session_id})

    def last_seen_seq(self) -> int:
        """Highest context frame sequence the sidecar has already encoded.

        The node ships its whole context window on every send, so this is what
        keeps that from costing a full re-encode: we forward only the frames
        past this mark.
        """
        return int(self.request({"op": "state"}).get("last_seq", -1))
