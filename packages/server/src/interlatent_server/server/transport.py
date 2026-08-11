"""grpc.aio service implementation.

Wires the wire-format messages from `protocol/` to `PolicyRuntime` and
`ChunkBuffer`. Hosted by :mod:`interlatent_server.serve_gpu` in
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
:func:`interlatent_server.server.auth.wrap_servicer_with_auth`
wraps every RPC. The ``serve_gpu`` entrypoint applies it by default on
a self-hosted (owner-key) box, with the owner-scoped check; a
dashboard-provisioned box runs unguarded (managed network posture).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import tempfile
import uuid
from concurrent.futures import Executor
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Optional


from ..box_status import report_status as _report_box_status
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


def _peek_policy_config(request) -> dict:
    """Best-effort ``config.json`` for a checkpoint, for backend dispatch.

    Pulls the single config file, never the weights. Returns ``{}`` on any
    failure, leaving the decision to the URI hint — a config we could not read
    must not silently route a session to the wrong backend.
    """
    uri = getattr(request, "policy_uri", "") or ""
    if not uri:
        return {}
    try:
        from .lerobot_backend import _read_config_json

        return _read_config_json(uri)
    except Exception:
        log.debug("policy config peek failed for %s", uri, exc_info=True)
        return {}


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
    # Client ran this session in sequential (request-response) chunking mode
    # (--synchronous). Purely informational server-side: RTC in-painting and
    # crossfade already self-disable when the client stops overlapping chunks, so
    # the server needs no behavior change — this is recorded for logging only.
    synchronous: bool = False
    # Per-session recording state. ``recorder`` is None when the client
    # did NOT request recording at OpenSession (e.g. local smoke tests).
    recorder: Optional[SessionRecorder] = None
    last_infer: float = field(default_factory=lambda: time.monotonic())
    # Hold a strong reference to the in-flight upload task so the
    # asyncio GC cannot kill it mid-PUT once SessionState has been
    # evicted from ``_sessions``. The InferenceServicer keeps a second
    # ref in ``_lingering_uploads`` for the same reason.
    upload_task: Optional[asyncio.Task] = None
    # Backend InferenceSession id this gRPC session serves (from OpenSession
    # metadata ``episode_id``) — the join key ``_by_episode`` indexes on so a
    # reconnecting node supersedes the live session. Plus the node's robot
    # kind and the api key (used for the recorder's HTTP upload).
    episode_id: str = ""
    robot_kind: str = ""
    api_key: str = ""


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
        warmup_warning: Optional[str] = None,
        live_encode: bool = False,
        record_backpressure: bool = False,
    ) -> None:
        self._buf = chunk_buffer or InMemoryChunkBuffer()
        self._sessions: dict[str, SessionState] = {}
        self._next_step: dict[str, int] = {}
        # Monotonic timestamp of the last RecordTick whose control_source
        # was "teleop" — i.e. the last tick a human actually drove. The
        # teleop-recording pod's idle timeout anchors on this (ADR 0020):
        # under QUIC the pod has no teleop relay, so operator engagement is
        # only visible through the tick stream. Initialized at construction
        # so the provisioning window counts against idle.
        self.last_engaged_at: float = time.monotonic()
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
        # ADR 0016: recorders on this servicer build their LeRobot dataset
        # live during the session (teleop-record pod sets this; serve_gpu
        # keeps the close-time rebuild).
        self._live_encode = bool(live_encode)
        # ADR 0023: on a recorder-only pod there is no inference latency to
        # protect, so RecordTicks may await the recorder queue (bounded)
        # instead of refusing ticks when the writer is briefly behind.
        # serve_gpu keeps the non-blocking enqueue — Infer shares its loop.
        self._record_backpressure = bool(record_backpressure)
        self._gc_task: Optional[asyncio.Task] = None
        # Upload tasks for sessions whose state was already evicted but
        # whose upload is still running. We keep strong refs here.
        self._lingering_uploads: set[asyncio.Task] = set()
        # episode_id (backend InferenceSession id) -> gRPC session_id, so a
        # reconnect can find (and supersede) the live session by backend id.
        self._by_episode: dict[str, str] = {}
        # Subset of ``_lingering_uploads`` that actually recorded steps —
        # i.e. the uploads that should surface as "uploading". A no-op
        # upload (recorder.no_steps) still runs for cleanup but never
        # counts toward the reported status (it would flash a spurious
        # ready→uploading flip). The status reconcile reads this set.
        self._reportable_uploads: set[asyncio.Task] = set()
        # Box self-report state. This servicer is the SOLE writer of the
        # box's activity status (running/ready/uploading): the idle-GC
        # loop derives the true state from live sessions + uploads and
        # reports it, reporting only on change. Removing the old
        # event-path reports (OpenSession/CloseSession/_spawn_upload)
        # leaves a single, serialized writer, which sidesteps the
        # fire-and-forget POST ordering race two concurrent reporters had.
        self._last_reported_status: Optional[str] = None
        # Non-fatal pre-warm warning (e.g. "load failed; first session
        # will compile"). Attached as ``status_detail`` to every status
        # report until the first real session loads successfully and
        # clears it. None when warmup was clean.
        self._warmup_warning: Optional[str] = warmup_warning

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
        # 0 == "unspecified": let the runtime resolve the model's NATIVE
        # chunk_size / action_dim from its config. A hard default here (the old
        # `or 32` / `or 6`) is truthy and short-circuits the backend's
        # `chunk_size or cfg.chunk_size` resolution, so it would shadow the
        # model's real shape — and, via the (backend, uri) runtime cache, a
        # pre-warmed 32/6 runtime would then be handed to every later session
        # regardless of what it requested. Pass the hint through verbatim and
        # trust runtime.chunk_size / runtime.action_dim below.
        requested_chunk = request.chunk_size or 0
        backend = request.policy_backend or "echo"
        # Released AllenAI MolmoAct2 checkpoints are transformers-native and
        # need the dedicated backend (the generic lerobot loader can't decode
        # their config.json). Route them there transparently so the wire
        # contract (policy_backend="lerobot") stays unchanged.
        from .molmoact2_backend import resolve_backend
        backend = resolve_backend(backend, request.policy_uri)
        # World-action models (DreamZero) are neither lerobot- nor
        # transformers-native. Unlike the arm above this keys on the checkpoint
        # config, not the URI: user fine-tunes are the point here, and a
        # substring test would miss "myorg/my-dreamzero-ft" and fire on
        # anything with "dream" in the name. See ADR 0037 (platform repo).
        from .dreamzero_backend import resolve_backend as _resolve_wm
        backend = _resolve_wm(backend, request.policy_uri, _peek_policy_config(request))
        requested_action_dim = request.action_dim or 0
        # Natural-language task instruction (e.g. SmolVLA "pick up the
        # red cube"). Set once at session-open via OpenSession.metadata;
        # backends that accept default_task (LeRobotBackend) wire it
        # into every batch. Other backends ignore the kwarg.
        md = dict(request.metadata) if request.metadata else {}
        default_task = md.get("task", "")
        # Sequential (request-response) chunking flag from the client. Informational
        # only — the geometry (no overlapping chunks) already disengages RTC and
        # crossfade; we record + log it so a session's cadence is visible pod-side.
        synchronous = str(md.get("synchronous", "")).strip().lower() in (
            "1", "true", "yes", "on",
        )
        runtime = PolicyRuntime.load(
            backend=backend,
            chunk_size=requested_chunk,
            action_dim=requested_action_dim,
            policy_uri=request.policy_uri,
            default_task=default_task,
            # Forward the whole OpenSession.metadata map so format-specific
            # backends can pull what they need (e.g. MolmoAct2 reads
            # image_keys / norm_tag / inference_action_mode). Backends that
            # don't care ignore it via **_.
            session_metadata=md,
        )
        # A real session loaded the policy successfully — clear any
        # pre-warm warning so subsequent status reports drop the
        # degraded ``status_detail``. (A failed load raises above and
        # leaves the warning in place.)
        self._warmup_warning = None

        # The runtime is authoritative. A cache hit (e.g. the startup pre-warm)
        # or a cfg-bearing backend can resolve a chunk_size that differs from the
        # hint we passed — report what it ACTUALLY produces so the client paces
        # its schedule to the real chunk width. Previously this returned the
        # request value (e.g. 100) while the cached runtime generated 32, which
        # silently desynced the controller's horizon/cooldown math.
        chunk_size = runtime.chunk_size
        min_horizon = request.min_execution_horizon or max(chunk_size // 4, 1)

        state = SessionState(
            runtime=runtime,
            payload_codec=request.payload_codec or "raw_f32",
            chunk_size=chunk_size,
            min_execution_horizon=min_horizon,
            synchronous=synchronous,
        )

        # Session identity index. The episode_id is the backend
        # InferenceSession id (same value the recorder pins); a reconnecting
        # node re-opens with the same episode_id, so the index overwrites —
        # newest gRPC session wins.
        state.episode_id = md.get("episode_id") or session_id
        state.robot_kind = (md.get("robot_kind") or "").strip()
        state.api_key = _api_key_from_context(context)

        # ADR 0024: a reconnect for an episode we're already recording
        # supersedes the old gRPC session immediately — and hands its
        # recorder over so the capture CONTINUES instead of a second
        # recorder later producing a duplicate same-UUID upload.
        inherited = self._supersede_episode_session(state.episode_id, session_id)

        # Optional per-session recorder — opt-in via metadata so legacy
        # callers (smoke tests, local-dev) pay nothing.
        recorder = inherited
        if recorder is None:
            recorder = self._maybe_build_recorder(
                session_id=session_id,
                request=request,
                metadata=md,
                context=context,
            )
        if recorder is not None:
            recorder.start()  # idempotent for an inherited (running) recorder
            state.recorder = recorder

        self._sessions[session_id] = state
        self._next_step[session_id] = 0
        self._by_episode[state.episode_id] = session_id
        log.info(
            "OpenSession session_id=%s model_id=%s chunk_size=%d recording=%s "
            "chunking=%s",
            session_id, request.model_id, chunk_size,
            "yes" if recorder is not None else "no",
            "sequential(sync)" if synchronous else "overlapping(async)",
        )
        # NOTE: status is NOT reported here. The idle-GC loop is the sole
        # writer of running/ready/uploading — its first pass (which runs
        # right after this RPC returns and registers the session) reports
        # "running". See ``_reconcile_status``.
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
        if state is not None:
            self._drop_episode_index(state.episode_id, request.session_id)
        if state is not None and state.recorder is not None:
            self._spawn_upload(state)
        # Status is NOT reported here; the idle-GC reconcile is the sole
        # writer and will report ready/uploading on its next pass (within
        # _IDLE_GC_PERIOD_S). The accepted cost is up to one GC period of
        # latency on running→ready, in exchange for a single serialized
        # reporter with no fire-and-forget ordering race.
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
        # ``ok`` is honest: False when the recorder refused (queue full /
        # step cap / closed), so a spooling client knows to retry.
        return pb.RecordTickResponse(ok=self._enqueue_tick(sess, request))

    async def RecordTicks(
        self, request: pb.RecordTicksRequest, context,
    ) -> pb.RecordTicksResponse:
        """Batched RecordTick — many ticks in one RPC.

        The Pi's recorder drain coalesces queued ticks so the remote link
        stops being the capture bottleneck (a single unary RecordTick per
        tick can't keep up with the 30 Hz control loop). Each contained
        tick carries its own ``step``, so batching does not affect the
        server-side ordering.

        ``accepted`` is a PREFIX count (ADR 0023): ticks are processed in
        order and processing stops at the first one the recorder refuses
        (queue full / step cap / closed). A client that deletes spooled
        ticks on ack may therefore delete exactly the first ``accepted``
        ticks of the batch and must retry the rest — an over-count here
        would convert transient backpressure into permanent frame loss.
        Wire-compatible: old clients ignored partial counts anyway.
        """
        sess = self._sessions.get(
            request.ticks[0].session_id if request.ticks else "",
        )
        if sess is None or sess.recorder is None:
            return pb.RecordTicksResponse(accepted=0)
        accepted = 0
        for tick in request.ticks:
            if self._record_backpressure:
                ok = await self._enqueue_tick_blocking(sess, tick)
            else:
                ok = self._enqueue_tick(sess, tick)
            if not ok:
                break
            accepted += 1
        return pb.RecordTicksResponse(accepted=accepted)

    def _prepare_tick(
        self, sess: "SessionState", tick: pb.RecordTickRequest,
    ) -> dict:
        """Side effects + kwargs shared by both enqueue variants.

        Reconstructs full LeRobot feature names ("overhead" ->
        "observation.images.overhead") and touches session liveness.
        Returns the recorder enqueue kwargs. Deliberately runs even for a
        tick that ends up refused: liveness reflects "the node is alive and
        sending", while the ack reflects "the tick is durably queued" — the
        client retries refused ticks, at which point these side effects are
        idempotent (monotonic liveness).
        """
        jpegs = {
            f"observation.images.{cam}": data
            for cam, data in tick.jpegs.items()
        }
        state = list(tick.observation_state) if tick.observation_state else None
        # RecordTick counts as session activity — a teleop-only session
        # (no policy, so no Infer) must not be reaped by the idle GC.
        sess.last_infer = time.monotonic()
        if tick.control_source in ("teleop", "intervention"):
            self.last_engaged_at = time.monotonic()
        return {
            "step": int(tick.step),
            "observation_state": state,
            "action": list(tick.action),
            "jpegs": jpegs,
            "control_timestamp": int(tick.control_timestamp),
            "control_source": tick.control_source or None,
        }

    def _enqueue_tick(
        self, sess: "SessionState", tick: pb.RecordTickRequest,
    ) -> bool:
        """Non-blocking tick enqueue. True iff the recorder accepted it."""
        return sess.recorder.enqueue_nowait(**self._prepare_tick(sess, tick))

    async def _enqueue_tick_blocking(
        self, sess: "SessionState", tick: pb.RecordTickRequest,
    ) -> bool:
        """Backpressure variant (recorder-only pods): awaits queue space,
        bounded by the recorder's enqueue timeout."""
        return await sess.recorder.enqueue(**self._prepare_tick(sess, tick))

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

        # Run decode AND the blocking forward off the event loop on the
        # dedicated inference executor, so the loop stays free to drain 30 Hz
        # RecordTick ingest while a chunk is computing. A single-worker
        # executor (production) serializes forwards, which preserves the
        # in-painting buffer's per-session ordering.
        #
        # `decode_payload` belongs in here with the forward: it PIL-decodes
        # every camera JPEG at full resolution, which is exactly the
        # CPU-heavy, GIL-holding work this executor exists to keep off the
        # loop. Running it inline stalled RecordTick ingest for the decode's
        # duration on every Infer — the recorder drops ticks it never sees.
        def _decode_and_forward():
            # Recording is fed by RecordTick (one call per Pi control tick)
            # rather than piggybacking on the Infer path — we don't need the
            # JPEG tee here anymore. Keep the legacy single-return signature.
            decoded = decode_payload(obs.payload, codec)
            return sess.runtime.forward(
                decoded, ctx, inference_delay=obs.inference_delay
            )

        # compute_ns now covers decode + forward — the whole server-side
        # cost the client's inference_delay actually pays for.
        _compute_t0 = time.monotonic_ns()
        loop = asyncio.get_running_loop()
        actions = await loop.run_in_executor(
            self._inference_executor, _decode_and_forward
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
    # Episode index cleanup
    # ------------------------------------------------------------------

    def _drop_episode_index(self, episode_id: str, session_id: str) -> None:
        """Remove the episode index entry iff it still points at this
        session (a reconnect may have already overwritten it)."""
        if episode_id and self._by_episode.get(episode_id) == session_id:
            self._by_episode.pop(episode_id, None)

    def _supersede_episode_session(
        self, episode_id: str, new_session_id: str,
    ) -> Optional[SessionRecorder]:
        """Retire the previous gRPC session for ``episode_id`` (ADR 0024).

        A node that reconnects mid-recording re-opens with the same
        episode_id. Before this existed, the abandoned session lingered in
        ``_sessions`` holding its recorder until the idle-GC or the
        recorder pod's stop-drain force-closed it — UPLOADING a second
        inbox session under the same episode UUID, which the merge then
        rejects as a duplicate. Retire the old session immediately
        instead: drop it from every index, detach its recorder, and hand a
        still-open recorder back so the new session continues the same
        capture — one upload at close, every acked tick preserved (the
        node deleted them from its spool on our ack; dropping them here
        would violate the ADR 0023 lossless contract).

        Returns the recorder to inherit, or None (no prior session, no
        recorder, or the recorder already began closing/uploading — in
        that last case the new session records separately and the
        merge-side quarantine contains the resulting duplicate).
        """
        old_sid = self._by_episode.get(episode_id)
        if not old_sid or old_sid == new_session_id:
            return None
        old = self._sessions.pop(old_sid, None)
        self._next_step.pop(old_sid, None)
        self._buf.drop(old_sid)
        self._drop_episode_index(episode_id, old_sid)
        if old is None:
            return None
        recorder, old.recorder = old.recorder, None  # old session must never upload
        if recorder is None:
            log.info(
                "episode %s: session %s superseded by reconnect %s "
                "(no recorder attached)",
                episode_id, old_sid, new_session_id,
            )
            return None
        if not recorder.reusable:
            log.warning(
                "episode %s: session %s superseded by reconnect %s but its "
                "recorder already began closing/uploading — the new session "
                "records separately; expect a same-UUID inbox session that "
                "the merge will quarantine",
                episode_id, old_sid, new_session_id,
            )
            return None
        log.warning(
            "episode %s: session %s superseded by reconnect %s — recorder "
            "continues (%d steps captured so far)",
            episode_id, old_sid, new_session_id, recorder.step_count,
        )
        return recorder

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
            task_id=metadata.get("task_id") or None,
            fps=fps,
            policy_uri=policy_uri,
            layer=layer,
            api_key=api_key,
            api_base=metadata.get("api_base") or _DEFAULT_API_BASE,
            live_encode=self._live_encode,
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

        # Surface this upload as "uploading" via the status reconcile, but
        # ONLY when it actually recorded steps. A no-op upload still runs
        # (for finalize() + working-dir cleanup) but must not flash a
        # ready→uploading flip. The reconcile reads ``_reportable_uploads``;
        # the box returns to ready once this set drains.
        if not state.recorder.no_steps:
            self._reportable_uploads.add(task)
            task.add_done_callback(self._reportable_uploads.discard)

    def _reconcile_status(self) -> None:
        """Derive the box's true activity state and self-report it.

        This is the SOLE writer of the box's activity status. The true
        state is read off live in-process state:

          - ``uploading`` if any step-bearing upload is still in flight
            (``_reportable_uploads`` — mirrors the no-op filter
            ``_spawn_upload`` applies, so empty flushes never surface);
          - else ``running`` if any session is live;
          - else ``ready``.

        Reports only on a change from the last reported value, so a steady
        state doesn't spam the backend. The non-fatal ``_warmup_warning``
        rides along as ``status_detail`` until a real session clears it.
        No-op without box identity (``report_status`` handles that).
        """
        if self._reportable_uploads:
            target = "uploading"
        elif self._sessions:
            target = "running"
        else:
            target = "ready"
        if target == self._last_reported_status:
            return
        self._last_reported_status = target
        _report_box_status(target, detail=self._warmup_warning)

    async def _idle_gc_loop(self) -> None:
        """Evict silent sessions and reconcile the box's reported status.

        Runs for the server process's lifetime. Reconciles FIRST, then
        sleeps: the task is created at the first RPC (``ensure_gc_started``)
        but only runs once that RPC returns, so on the first session this
        reports ``running`` essentially immediately rather than waiting a
        full ``_IDLE_GC_PERIOD_S``. Safe with zero sessions — it just
        reports ``ready`` (on change) and sleeps.
        """
        while True:
            try:
                now = time.monotonic()
                stale = [
                    sid for sid, state in self._sessions.items()
                    if (now - state.last_infer) > self._idle_timeout_s
                ]
                for sid in stale:
                    state = self._sessions.pop(sid, None)
                    if state is None:
                        continue
                    self._next_step.pop(sid, None)
                    self._buf.drop(sid)
                    if state.recorder is None:
                        # Non-recording session (e.g. a smoke/local run, or
                        # a client that dropped without CloseSession). Just
                        # drop it — there is nothing to flush.
                        log.info(
                            "Idle-GC dropped non-recording session %s "
                            "(silent for >%.0fs)", sid, self._idle_timeout_s,
                        )
                        continue
                    rec = state.recorder
                    if rec.no_steps:
                        # Nothing was captured — discard rather than upload an
                        # empty dataset. Cleans up the working dir; never
                        # surfaces an "uploading" status.
                        task = asyncio.create_task(
                            rec.discard(),
                            name=f"recorder-discard[{rec.config.episode_id}]",
                        )

                        def _on_discard(t: asyncio.Task, sid=sid, episode_id=rec.config.episode_id) -> None:
                            try:
                                t.result()
                            except Exception:
                                log.exception(
                                    "Idle-GC discard failed for empty session %s (episode_id=%s)",
                                    sid, episode_id,
                                )
                                return
                            log.info(
                                "Idle-GC dropped empty session %s (silent for >%.0fs, episode_id=%s)",
                                sid, self._idle_timeout_s, episode_id,
                            )

                        task.add_done_callback(_on_discard)
                        continue
                    # The session recorded steps but went silent without a
                    # CloseSession — almost always a dashboard stop that never
                    # reached the box (the node daemon exited without closing),
                    # a crash, or a network drop. The capture is already bound
                    # to an InferenceSession.id whose dashboard row exists, so
                    # UPLOAD fills that row instead of leaving it permanently
                    # empty. This is the durable safety net for any missed
                    # CloseSession — dropping here silently loses the episode.
                    log.info(
                        "Idle-GC uploading orphaned session %s (silent for "
                        ">%.0fs, %d steps, episode_id=%s)",
                        sid, self._idle_timeout_s, rec.step_count,
                        rec.config.episode_id,
                    )
                    self._spawn_upload(state)
                # Sole-writer status reconcile: report running/ready/uploading
                # derived from the now-current session + upload state. This is
                # how a stale "running" gets corrected back to "ready" after a
                # client drops without CloseSession.
                self._reconcile_status()
            except asyncio.CancelledError:
                return
            except Exception:
                # Never let a transient error kill the GC loop.
                log.exception("Idle-GC loop iteration raised; continuing")
            await asyncio.sleep(_IDLE_GC_PERIOD_S)


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

    Production serves through :mod:`interlatent_server.serve_gpu`,
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
