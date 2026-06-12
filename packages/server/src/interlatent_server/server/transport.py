"""grpc.aio service implementation.

Wires the wire-format messages from `protocol/` to `PolicyRuntime` and
`ChunkBuffer`. Hosted by :mod:`interlatent.cloud.serve_gpu` in
production; :func:`serve_local` below is the same servicer behind a
plain insecure-port gRPC server for tests + smoke runs.

Sessions:
    OpenSession creates and registers a PolicyRuntime. Streaming Infer
    looks up the runtime by session_id, decodes the observation,
    reconstructs in-painting context from the chunk buffer, runs
    forward, stores the new raw actions, and returns an ActionChunk.

When the OpenSession metadata carries ``record=1`` we additionally
allocate a :class:`SessionRecorder` per session. Each Infer enqueues
one step (observation state + first action + raw JPEG bytes from the
npz payload) onto a bounded async queue; a background drain task
writes to local SSD. On CloseSession (or idle-GC eviction) the
recorder builds a LeRobot dataset and uploads it through the standard
inbox protocol, using the same ``x-api-key`` the gRPC client already
authenticated with.

API-key validation is not in this file. When the public-facing
endpoint needs it,
:func:`interlatent.inference.server.auth.wrap_servicer_with_auth`
wraps every RPC. The production ``serve_gpu`` entrypoint runs
unguarded on a private Tailscale network and skips that wrapper.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import time
import tempfile
import uuid
from concurrent.futures import Executor
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Optional


from ..protocol import messages_pb2 as pb
from ..protocol import messages_pb2_grpc as pb_grpc
from .chunk_buffer import ChunkBuffer, InMemoryChunkBuffer, StoredChunk
from .policy_runtime import PolicyRuntime, decode_payload
from .recorder import RecorderConfig, SessionRecorder
from .schedule import reconstruct

log = logging.getLogger(__name__)


# Steps of trailing context we ask the buffer for. Equal to one full
# chunk is enough for RTC in-painting in the reference impl.
DEFAULT_CONTEXT_STEPS = 32

# Seconds of inactivity after which the idle-GC closes a session and
# kicks off its upload. Generous — a normal session sees an Infer
# every ~33ms at 30 Hz, so any silence longer than this is an actual
# Pi disconnect rather than a stall. The server process is long-lived,
# so the GC just keeps stale sessions from leaking memory until the
# user notices.
DEFAULT_IDLE_TIMEOUT_S = float(os.environ.get("DRTC_IDLE_TIMEOUT_S", "60"))

# Period between idle-GC scans.
_IDLE_GC_PERIOD_S = 15.0


# ---------------------------------------------------------------------------
# Recorder metadata defaults — match the backend's auto-Model layer.
# ---------------------------------------------------------------------------


_DEFAULT_API_BASE = os.environ.get(
    "INTERLATENT_API_BASE", "https://interlatent.com/api/v1"
)


@dataclass
class SessionState:
    runtime: PolicyRuntime
    payload_codec: str
    chunk_size: int
    min_execution_horizon: int
    # Per-session recording state. ``recorder`` is None when the client
    # did NOT request recording at OpenSession (e.g. local smoke tests).
    recorder: Optional[SessionRecorder] = None
    last_infer: float = field(default_factory=lambda: time.monotonic())
    # Hold a strong reference to the in-flight upload task so the
    # asyncio GC cannot kill it mid-PUT once SessionState has been
    # evicted from ``_sessions``. The InferenceServicer keeps a second
    # ref in ``_lingering_uploads`` for the same reason.
    upload_task: Optional[asyncio.Task] = None


class InferenceServicer(pb_grpc.InferenceServiceServicer):
    """DRTC servicer. Single-process: PolicyRuntime + ChunkBuffer +
    SessionRecorder all live in-process for the lifetime of the
    server. The client treats OpenSession as idempotent in case the
    process restarts mid-session — a fresh OpenSession will be issued
    and inference resumes from the new state.
    """

    def __init__(
        self,
        chunk_buffer: Optional[ChunkBuffer] = None,
        *,
        context_steps: int = DEFAULT_CONTEXT_STEPS,
        idle_timeout_s: float = DEFAULT_IDLE_TIMEOUT_S,
        recorder_base_dir: Optional[Path] = None,
        inference_executor: Optional[Executor] = None,
    ) -> None:
        self._buf = chunk_buffer or InMemoryChunkBuffer()
        self._sessions: dict[str, SessionState] = {}
        self._next_step: dict[str, int] = {}
        self._context_steps = context_steps
        self._idle_timeout_s = float(idle_timeout_s)
        # Dedicated executor for the blocking ``policy.forward()`` call.
        # forward() is CPU-heavy (a VLA's image tiling + tokenization runs
        # ~1.3 s before the GPU step) and MUST NOT run on the asyncio event
        # loop, or it stalls RecordTick ingest — the 30 Hz full-res capture
        # stream that feeds recording. serve_gpu passes a single-worker,
        # core-pinned executor here so inference and recording stop fighting
        # for cores. When None (tests/local), forward() runs on the default
        # executor; pass a max_workers=1 executor in production to preserve
        # per-session ordering of the in-painting buffer.
        self._inference_executor = inference_executor
        # ``recorder_base_dir`` lets tests pin a tempdir; in production
        # the per-session subdir is created under the OS temp dir,
        # which on the production GPU host is fast local SSD.
        self._recorder_base_dir = Path(recorder_base_dir) if recorder_base_dir else None
        self._gc_task: Optional[asyncio.Task] = None
        # Upload tasks for sessions whose state was already evicted but
        # whose upload is still running. We keep strong refs here.
        self._lingering_uploads: set[asyncio.Task] = set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def ensure_gc_started(self) -> None:
        """Lazily spin up the idle-GC task on the current event loop.

        Called from each RPC entry so we don't need a separate startup
        hook on the ASGI side. The task is a no-op when no sessions
        are recording.
        """
        if self._gc_task is None or self._gc_task.done():
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return
            self._gc_task = loop.create_task(
                self._idle_gc_loop(), name="drtc-idle-gc",
            )

    # ------------------------------------------------------------------
    # Unary RPCs
    # ------------------------------------------------------------------

    async def OpenSession(
        self, request: pb.OpenSessionRequest, context
    ) -> pb.OpenSessionResponse:
        self.ensure_gc_started()

        session_id = str(uuid.uuid4())
        chunk_size = request.chunk_size or 32
        min_horizon = request.min_execution_horizon or max(chunk_size // 4, 1)
        backend = request.policy_backend or "echo"
        # Released AllenAI MolmoAct2 checkpoints are transformers-native and
        # need the dedicated backend (the generic lerobot loader can't decode
        # their config.json). Route them there transparently so the wire
        # contract (policy_backend="lerobot") stays unchanged.
        from .molmoact2_backend import resolve_backend
        backend = resolve_backend(backend, request.policy_uri)
        action_dim = request.action_dim or 6
        # Natural-language task instruction (e.g. SmolVLA "pick up the
        # red cube"). Set once at session-open via OpenSession.metadata;
        # backends that accept default_task (LeRobotBackend) wire it
        # into every batch. Other backends ignore the kwarg.
        md = dict(request.metadata) if request.metadata else {}
        default_task = md.get("task", "")
        runtime = PolicyRuntime.load(
            backend=backend,
            chunk_size=chunk_size,
            action_dim=action_dim,
            policy_uri=request.policy_uri,
            default_task=default_task,
            # Forward the whole OpenSession.metadata map so format-specific
            # backends can pull what they need (e.g. MolmoAct2 reads
            # image_keys / norm_tag / inference_action_mode). Backends that
            # don't care ignore it via **_.
            session_metadata=md,
        )

        state = SessionState(
            runtime=runtime,
            payload_codec=request.payload_codec or "raw_f32",
            chunk_size=chunk_size,
            min_execution_horizon=min_horizon,
        )

        # Optional per-session recorder — opt-in via metadata so legacy
        # callers (smoke tests, local-dev) pay nothing.
        recorder = self._maybe_build_recorder(
            session_id=session_id,
            request=request,
            metadata=md,
            context=context,
        )
        if recorder is not None:
            recorder.start()
            state.recorder = recorder

        self._sessions[session_id] = state
        self._next_step[session_id] = 0
        log.info(
            "OpenSession session_id=%s model_id=%s chunk_size=%d recording=%s",
            session_id, request.model_id, chunk_size,
            "yes" if recorder is not None else "no",
        )
        return pb.OpenSessionResponse(
            session_id=session_id,
            chunk_size=chunk_size,
            action_dim=runtime.action_dim,
        )

    async def CloseSession(
        self, request: pb.CloseSessionRequest, context
    ) -> pb.CloseSessionResponse:
        state = self._sessions.pop(request.session_id, None)
        self._next_step.pop(request.session_id, None)
        self._buf.drop(request.session_id)
        if state is not None and state.recorder is not None:
            self._spawn_upload(state)
        return pb.CloseSessionResponse()

    async def Infer(self, request: pb.Observation, context) -> pb.ActionChunk:
        self.ensure_gc_started()
        return await self._infer_one(request)

    async def RecordTick(
        self, request: pb.RecordTickRequest, context,
    ) -> pb.RecordTickResponse:
        """Per-control-tick capture from the Pi.

        Decoupled from Infer so the recorder gets EVERY tick (30 Hz),
        not just the ones where Infer happened to fire (~5 Hz). The Pi
        calls this from a background thread; the recorder enqueue is
        non-blocking, so this RPC returns immediately.
        """
        sess = self._sessions.get(request.session_id)
        if sess is None or sess.recorder is None:
            # Either the session is gone or recording wasn't opted into.
            # Silently ack rather than erroring — the Pi treats record
            # failures as best-effort.
            return pb.RecordTickResponse(ok=False)
        # ``jpegs`` is a map<string, bytes> on the wire — full LeRobot
        # feature names are reconstructed in the recorder ("overhead"
        # -> "observation.images.overhead").
        jpegs = {
            f"observation.images.{cam}": data
            for cam, data in request.jpegs.items()
        }
        state = list(request.observation_state) if request.observation_state else None
        sess.recorder.enqueue_nowait(
            step=int(request.step),
            observation_state=state,
            action=list(request.action),
            jpegs=jpegs,
            control_timestamp=int(request.control_timestamp),
            control_source=request.control_source or None,
        )
        return pb.RecordTickResponse(ok=True)

    # ------------------------------------------------------------------
    # Bidi stream
    # ------------------------------------------------------------------

    async def Stream(
        self, request_iterator: AsyncIterator[pb.Observation], context
    ) -> AsyncIterator[pb.ActionChunk]:
        self.ensure_gc_started()
        async for obs in request_iterator:
            yield await self._infer_one(obs)

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    async def _infer_one(self, obs: pb.Observation) -> pb.ActionChunk:
        sess = self._sessions.get(obs.session_id)
        if sess is None:
            raise RuntimeError(
                f"Unknown session {obs.session_id}. Call OpenSession first."
            )

        spans = [(s.start_step, s.end_step) for s in obs.scheduled_spans]
        ctx = reconstruct(
            self._buf,
            session_id=obs.session_id,
            next_action_step=obs.next_action_step,
            spans=spans,
            context_steps=self._context_steps,
        )

        codec = obs.payload_codec or sess.payload_codec
        # Recording is fed by RecordTick (one call per Pi control tick)
        # rather than piggybacking on the Infer path — we don't need the
        # JPEG tee here anymore. Keep the legacy single-return signature.
        decoded = decode_payload(obs.payload, codec)

        # Run the blocking forward (CPU preprocessing + GPU step) off the
        # event loop on the dedicated inference executor, so the loop stays
        # free to drain 30 Hz RecordTick ingest while a chunk is computing.
        # A single-worker executor (production) serializes forwards, which
        # preserves the in-painting buffer's per-session ordering.
        _compute_t0 = time.monotonic_ns()
        loop = asyncio.get_running_loop()
        actions = await loop.run_in_executor(
            self._inference_executor,
            functools.partial(
                sess.runtime.forward, decoded, ctx, inference_delay=obs.inference_delay
            ),
        )
        compute_ns = time.monotonic_ns() - _compute_t0

        # Persist for future in-painting.
        self._buf.append(
            obs.session_id,
            StoredChunk(
                start_step=obs.next_action_step,
                control_timestamp=obs.control_timestamp,
                actions=actions,
                created_at=time.time(),
            ),
        )

        # Build response.
        chunk = pb.ActionChunk(
            session_id=obs.session_id,
            control_timestamp=obs.control_timestamp,
            server_timestamp_ns=time.monotonic_ns(),
            server_compute_ns=compute_ns,
        )
        for i, vec in enumerate(actions):
            a = chunk.actions.add()
            a.action_step = obs.next_action_step + i
            a.control_timestamp = obs.control_timestamp
            a.vector.extend(float(x) for x in vec)

        # Recording is now fed by RecordTick from the Pi — every control
        # tick, not just every Infer. The Infer path no longer enqueues
        # rows; see :meth:`RecordTick`.
        sess.last_infer = time.monotonic()
        return chunk

    # ------------------------------------------------------------------
    # Recorder allocation
    # ------------------------------------------------------------------

    def _maybe_build_recorder(
        self,
        *,
        session_id: str,
        request: pb.OpenSessionRequest,
        metadata: dict[str, str],
        context,
    ) -> Optional[SessionRecorder]:
        """Return a recorder iff metadata opts in AND auth is present.

        Recording requires an API key (used for the post-rollout HTTP
        upload back to the Interlatent backend). When the gRPC client
        somehow reached us without ``x-api-key`` — i.e. when local-dev
        is running without the auth wrapper — we silently disable
        recording rather than crash inference.
        """
        if not _truthy(metadata.get("record")):
            return None

        api_key = _api_key_from_context(context)
        if not api_key:
            log.warning(
                "Session %s requested recording but no x-api-key present; "
                "disabling recorder. Pass the key via gRPC metadata so the "
                "recorder can authenticate the inbox upload.",
                session_id,
            )
            return None

        # Required-ish metadata. Reasonable defaults so a slightly-old
        # SDK still records something useful.
        env_slug = metadata.get("env_slug") or "default"
        task = metadata.get("task") or env_slug
        fps = _int_or(metadata.get("fps"), 30)
        # ``episode_id`` defaults to the server's session_id — the Pi
        # snippet pins it from InferenceSession.id so the dashboard
        # join works without any extra round-trip.
        episode_id = metadata.get("episode_id") or session_id
        model_id = metadata.get("model_id") or request.model_id or None
        # Layer string MUST match the Model row's ``layer`` field to
        # let the backend route the episode to the right Model. The
        # backend creates rows with layer = "inference:<policy_uri>",
        # so we derive the same here unless overridden.
        policy_uri = request.policy_uri or ""
        layer = metadata.get("layer") or f"inference:{policy_uri}"

        # Per-session working directory. Lives on the container's local
        # SSD; cleaned up by SessionRecorder.upload() on completion.
        base = self._recorder_base_dir or Path(tempfile.gettempdir())
        working_dir = base / f"drtc_recorder_{session_id}"

        config = RecorderConfig(
            episode_id=episode_id,
            env_slug=env_slug,
            model_id=model_id,
            task=task,
            fps=fps,
            policy_uri=policy_uri,
            layer=layer,
            api_key=api_key,
            api_base=metadata.get("api_base") or _DEFAULT_API_BASE,
        )
        return SessionRecorder(working_dir, config)

    # ------------------------------------------------------------------
    # Upload + idle GC
    # ------------------------------------------------------------------

    def _spawn_upload(self, state: SessionState) -> None:
        """Kick off the recorder's upload as a fire-and-forget task.

        We hold a strong reference (``self._lingering_uploads``) so
        asyncio's task GC cannot kill the upload between the gRPC
        reply going back and the actual S3 PUTs completing. Each task
        removes itself from the set on completion.
        """
        if state.recorder is None or state.upload_task is not None:
            return

        loop = asyncio.get_running_loop()
        task = loop.create_task(
            state.recorder.upload(),
            name=f"recorder-upload[{state.recorder.config.episode_id}]",
        )
        state.upload_task = task
        self._lingering_uploads.add(task)
        task.add_done_callback(self._lingering_uploads.discard)

    async def _idle_gc_loop(self) -> None:
        """Force-close sessions that have gone silent.

        Runs forever in the server process's lifetime. Safe to run
        with zero sessions — the inner loop just sleeps.
        """
        while True:
            try:
                await asyncio.sleep(_IDLE_GC_PERIOD_S)
                now = time.monotonic()
                stale: list[str] = []
                for sid, state in self._sessions.items():
                    if state.recorder is None:
                        continue
                    if (now - state.last_infer) > self._idle_timeout_s:
                        stale.append(sid)
                for sid in stale:
                    state = self._sessions.pop(sid, None)
                    if state is None:
                        continue
                    self._next_step.pop(sid, None)
                    self._buf.drop(sid)
                    if state.recorder is not None:
                        # Idle-GC means the Pi vanished without sending
                        # CloseSession — almost always a crash, network
                        # drop, or torch.compile-induced give-up. The
                        # capture is bound to an InferenceSession.id
                        # the user has already moved past, so uploading
                        # would just populate a stale dashboard row.
                        # Drop instead. CloseSession remains the sole
                        # upload trigger.
                        rec = state.recorder
                        task = asyncio.create_task(
                            rec.discard(),
                            name=f"recorder-discard[{rec.config.episode_id}]",
                        )

                        def _on_done(t: asyncio.Task, sid=sid, episode_id=rec.config.episode_id) -> None:
                            try:
                                dropped = t.result()
                            except Exception:
                                log.exception(
                                    "Idle-GC discard failed for session %s (episode_id=%s)",
                                    sid, episode_id,
                                )
                                return
                            log.info(
                                "Idle-GC dropped session %s (silent for >%.0fs, %d rows discarded, episode_id=%s)",
                                sid, self._idle_timeout_s, dropped, episode_id,
                            )

                        task.add_done_callback(_on_done)
            except asyncio.CancelledError:
                return
            except Exception:
                # Never let a transient error kill the GC loop.
                log.exception("Idle-GC loop iteration raised; continuing")


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in ("1", "true", "yes", "on")


def _int_or(value: Optional[str], default: int) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _api_key_from_context(context) -> str:
    """Pull ``x-api-key`` from the gRPC invocation metadata.

    The auth wrapper validates the key on the way in; we just read it
    back here so the recorder can present the same identity to the
    Interlatent backend on its HTTP upload calls.
    """
    if context is None:
        return ""
    try:
        md = dict(context.invocation_metadata() or [])
    except Exception:
        return ""
    return (md.get("x-api-key") or "").strip()


# ----------------------------------------------------------------------
# Bare-metal gRPC server (used in tests + local dev)
# ----------------------------------------------------------------------


async def serve_local(host: str = "0.0.0.0", port: int = 50051) -> None:
    """Run a plain gRPC server. Used by tests + smoke runs.

    Production serves through :mod:`interlatent.cloud.serve_gpu`,
    which wires the same servicer plus startup warmup + persistent
    torch.compile caches.
    """
    import grpc

    server = grpc.aio.server()
    pb_grpc.add_InferenceServiceServicer_to_server(InferenceServicer(), server)
    server.add_insecure_port(f"{host}:{port}")
    await server.start()
    log.info("DRTC gRPC server listening on %s:%d", host, port)
    await server.wait_for_termination()
