"""Server-side DRTC episode recorder.

Every Infer call already lands on the GPU server with the full
observation (state + camera JPEGs) decoded for the policy forward pass.
The recorder pulls that data off the wire ONCE — JPEG bytes are taken
verbatim from the npz, never re-encoded — and persists per-step rows
plus per-frame JPEGs into a per-session working directory on local
SSD.

On :meth:`CloseSession` the recorder is finalised and a
:class:`LeRobotRebuilder` build is kicked off (against the local
on-disk staging) followed by the standard inbox-upload protocol used
by the SDK today: ``POST /api/v1/episodes`` → ``POST
.../upload-urls`` → presigned ``PUT`` → ``POST .../upload-complete``.

This file lives on the engine side only — the SDK never imports it.
The auth header reused for backend HTTP calls is the same
``x-api-key`` value the Pi already presents on gRPC metadata, so we
inherit per-user ownership scoping for free.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence, Tuple

from ..protocol import messages_pb2 as pb  # noqa: F401  (type-only ref in docstrings)
from ...storage.lerobot_rebuild import LeRobotRebuilder, StepRow
from ...storage.lerobot_live import (
    FpsDivergenceError,
    LiveEncodeError,
    LiveEpisodeBuilder,
)

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Tunables
# ----------------------------------------------------------------------

# Bounded backlog between Infer (producer) and the writer task (consumer).
# At 30 Hz with ~1ms writes the queue never fills; cap exists only to
# bound RAM under pathological producer/consumer skew.
_QUEUE_MAXSIZE = 64

# Hard cap on number of recorded steps per session. At 30 Hz this is
# ~60 min — a generous upper bound that protects local SSD if a
# session never gets a CloseSession (Pi crash, network drop).
_MAX_STEPS_PER_SESSION = 108_000

# Bound on the awaitable ``enqueue`` variant: if the writer can't take a
# tick within this window the disk writer is wedged, not merely behind —
# refuse the tick (the client keeps it spooled and retries) instead of
# stalling the gRPC handler indefinitely.
_ENQUEUE_TIMEOUT_S = 5.0

# Upload defaults — match the SDK's behaviour.
_UPLOAD_BATCH_SIZE = 100
_PUT_TIMEOUT = 300.0
_HTTP_TIMEOUT = 60.0


_IMAGE_KEY_PREFIX = "observation.images."


def _live_encode_enabled() -> bool:
    """ADR 0016 kill-switch: live encode is on unless INTERLATENT_LIVE_ENCODE
    is set to an explicit off value."""
    return os.environ.get("INTERLATENT_LIVE_ENCODE", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


def _warm_lerobot_import() -> None:
    """Pre-import lerobot (torch et al, seconds) off the hot path so the
    first live add_step's LeRobotDataset.create doesn't stall the drain
    queue behind the import."""
    try:
        import lerobot.datasets.lerobot_dataset  # noqa: F401
    except Exception:
        # Missing/broken lerobot surfaces properly at create time.
        pass


def _short_cam(key: str) -> str:
    """``observation.images.<cam>`` -> ``<cam>``; anything else -> as-is."""
    if key.startswith(_IMAGE_KEY_PREFIX):
        return key[len(_IMAGE_KEY_PREFIX):]
    return key


def _api_v1_root(api_base: str) -> str:
    """Normalize an Interlatent API base to its ``/api/v1`` root.

    ``api_base`` reaches us via ``INTERLATENT_API_BASE`` (and OpenSession
    metadata), but two conventions collide in the same image:
    ``serve_gpu`` and the whole node SDK treat it as the **bare origin**
    (``https://interlatent.com``) and append ``/api/v1/...`` per call,
    while older recorder code assumed it already ended in ``/api/v1``.
    A box configured for the warmup-target fetch carries the bare origin,
    so the recorder posted to ``/episodes`` and got 405. Accept either
    form and always return the ``/api/v1`` root so the routes resolve.
    """
    base = api_base.rstrip("/")
    if base.endswith("/api/v1"):
        return base
    return f"{base}/api/v1"


# ----------------------------------------------------------------------
# Recorder
# ----------------------------------------------------------------------


@dataclass
class RecorderConfig:
    """Static per-session config, set once at :meth:`OpenSession`.

    ``layer`` defaults to ``"inference:<policy_uri>"`` to match the
    Model row auto-created by the backend in
    ``routers/inference.py:create_session``. Override via
    OpenSession metadata if the backend's convention ever changes.
    """

    episode_id: str
    env_slug: str
    model_id: Optional[str]
    task: str
    fps: int
    policy_uri: str
    layer: str
    api_key: str
    api_base: str
    sdk_version: str = "drtc-server"
    # ADR 0016: build the LeRobot dataset live during the session (video
    # streaming-encoded as frames arrive) instead of rebuilding everything
    # at CloseSession. Set by the teleop-record pod only; serve_gpu keeps
    # the close-time rebuild (its measured fps genuinely diverges).
    live_encode: bool = False
    # Durable Task link (backend ``tasks`` row id), echoed back to
    # POST /episodes so the catalog row attributes without a lookup.
    task_id: Optional[str] = None


class SessionRecorder:
    """Per-session episode recorder.

    Lifecycle:
        rec = SessionRecorder(working_dir, RecorderConfig(...))
        rec.start()                             # spins up the drain task
        rec.enqueue_nowait(...)                  # called once per Infer
        ...
        await rec.upload()                       # finalises + uploads + GCs

    The recorder is **single-session, single-owner**: ``enqueue_nowait``
    is safe to call from the gRPC event loop thread, the drain task is
    a single background asyncio task on the same loop, and ``upload()``
    is called exactly once (further calls are no-ops).
    """

    def __init__(
        self,
        working_dir: Path,
        config: RecorderConfig,
        *,
        max_steps: int = _MAX_STEPS_PER_SESSION,
    ) -> None:
        self.config = config
        self.working_dir = Path(working_dir)
        self.frames_dir = self.working_dir / "frames"
        self.steps_path = self.working_dir / "steps.jsonl"
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        # Create the JSONL file up-front so the path exists even if
        # the session closes with zero steps recorded.
        self.steps_path.touch()

        self._queue: asyncio.Queue = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)
        self._drain_task: Optional[asyncio.Task] = None
        self._closed = False
        self._uploaded = False
        self._max_steps = int(max_steps)
        self._step_count = 0
        self._dropped = 0
        self._first_warn_emitted = False
        self._cap_warn_emitted = False
        # ADR 0024: steps already staged, so a spool retry whose ack was
        # lost in a reconnect (the recorder is inherited across gRPC
        # sessions of one episode) is re-acked without re-staging.
        self._steps_seen: set[int] = set()
        self._dedup_acked = 0
        self._dedup_warn_emitted = False
        self._cameras: list[str] = []  # first-seen order; one entry per cam
        self._cam_set: set[str] = set()
        # Track first/last control timestamps (monotonic ns) so we can compute
        # the actual capture rate at upload time. The config-declared fps is
        # only the requested control rate — the real rate is throttled by
        # inference latency, so encoding the video at config.fps makes
        # playback look much faster than the original rollout.
        self._first_ts_ns: Optional[int] = None
        self._last_ts_ns: Optional[int] = None
        # Steps enqueued with a human-driven control_source ("teleop", or
        # "intervention" for a mid-rollout override of a running policy,
        # ADR 0034) — a human drove the robot for at least part of this
        # episode. Aggregated here (the pod is the authority on what it
        # executed) and reported episode-level as ``has_teleop`` on
        # upload-complete, which the backend stores as the Episode badge
        # (see CONTEXT.md "Episode collection origin").
        self._teleop_steps = 0
        # ADR 0016 live-encode path. The builder is fed from _write_one
        # (executor, serialized by the drain loop); the JPEG + JSONL
        # staging above keeps being written regardless — it is the
        # fallback for every live-path failure. _live_dead flips on the
        # first failure and the session silently reverts to the
        # rebuild-at-close behavior.
        self._live: Optional[LiveEpisodeBuilder] = None
        self._live_dead = False
        if config.live_encode and _live_encode_enabled():
            self._live = LiveEpisodeBuilder(
                self.working_dir / "live" / "v3",
                fps=config.fps,
                task=config.task,
                env_slug=config.env_slug,
                episode_id=config.episode_id,
            )

    # ------------------------------------------------------------------
    # Hot-path enqueue + drain
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spin up the background writer task on the current event loop."""
        if self._drain_task is not None:
            return
        if self._live is not None:
            # Warm the lerobot import so the first frame's dataset create
            # is subsecond instead of stalling behind a torch import.
            asyncio.get_running_loop().run_in_executor(None, _warm_lerobot_import)
        self._drain_task = asyncio.create_task(
            self._drain_loop(), name=f"recorder-drain[{self.config.episode_id}]"
        )

    def _build_item(
        self,
        *,
        step: int,
        observation_state: Optional[Sequence[float]],
        action: Sequence[float],
        jpegs: dict[str, bytes],
        control_timestamp: int,
        control_source: Optional[str] = None,
    ) -> Optional[dict]:
        """Admission checks + queue-item construction shared by both
        enqueue variants. Returns None when the recorder can accept no
        more steps (closed, or at the per-session step cap)."""
        if self._closed:
            return None

        if self._step_count >= self._max_steps:
            if not self._cap_warn_emitted:
                log.warning(
                    "SessionRecorder %s hit step cap (%d); stopping recording",
                    self.config.episode_id, self._max_steps,
                )
                self._cap_warn_emitted = True
            return None

        # Note new cameras in first-appearance order so the rebuilder
        # gets a stable schema across the whole episode.
        for key in jpegs:
            cam = _short_cam(key)
            if cam not in self._cam_set:
                self._cam_set.add(cam)
                self._cameras.append(cam)

        return {
            "step": int(step),
            "observation": _to_list(observation_state),
            "action": _to_list(action),
            "control_timestamp": int(control_timestamp),
            "jpegs": jpegs,  # already bytes; not serialized through JSONL
            "control_source": (control_source or None),
        }

    def _note_enqueued(self, item: dict) -> None:
        """Bookkeeping after an item actually made it onto the queue."""
        self._step_count += 1
        self._steps_seen.add(item["step"])
        if item["control_source"] in ("teleop", "intervention"):
            self._teleop_steps += 1
        ts = item["control_timestamp"]
        if self._first_ts_ns is None:
            self._first_ts_ns = ts
        self._last_ts_ns = ts

    def _already_recorded(self, step: int) -> bool:
        """ADR 0024 dedupe: True when ``step`` is already staged.

        Happens when a node reconnect replays spooled ticks whose ack was
        lost in flight — the tick IS durably staged here, so the retry is
        acked (True from enqueue) without staging it twice. A node whose
        step counter restarted mid-episode also lands here; its colliding
        ticks are dropped-with-ack, which is why the first hit logs loudly.
        """
        if step not in self._steps_seen:
            return False
        self._dedup_acked += 1
        if not self._dedup_warn_emitted:
            log.warning(
                "SessionRecorder %s: step %d already staged — acking the "
                "retry without re-staging (reconnect replay; further "
                "dedupes suppressed)",
                self.config.episode_id, step,
            )
            self._dedup_warn_emitted = True
        return True

    def _note_refused(self, step: int, why: str) -> None:
        self._dropped += 1
        if not self._first_warn_emitted:
            log.warning(
                "SessionRecorder %s %s — refusing step %d "
                "(writer falling behind; further refusals suppressed)",
                self.config.episode_id, why, step,
            )
            self._first_warn_emitted = True

    def enqueue_nowait(
        self,
        *,
        step: int,
        observation_state: Optional[Sequence[float]],
        action: Sequence[float],
        jpegs: dict[str, bytes],
        control_timestamp: int,
        control_source: Optional[str] = None,
    ) -> bool:
        """Push one step of data onto the writer queue.

        Constant-time and non-blocking. On a full queue the row is
        REFUSED rather than blocking — inference latency is the
        invariant we protect. Returns True iff the step was accepted
        onto the queue; a False return MUST be reported to the client
        (RecordTicks accepted-prefix) so it can retry from its spool
        instead of treating the tick as durably recorded.

        ``observation_state`` is the 1-D state vector (or None if the
        policy got only images — rare). ``action`` is the executed
        action for this tick (we record the first row of the chunk the
        backend just returned — see ``transport._infer_one`` for the
        rationale).
        """
        if not self._closed and self._already_recorded(int(step)):
            return True
        item = self._build_item(
            step=step, observation_state=observation_state, action=action,
            jpegs=jpegs, control_timestamp=control_timestamp,
            control_source=control_source,
        )
        if item is None:
            return False
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            self._note_refused(step, "queue full")
            return False
        self._note_enqueued(item)
        return True

    async def enqueue(
        self,
        *,
        step: int,
        observation_state: Optional[Sequence[float]],
        action: Sequence[float],
        jpegs: dict[str, bytes],
        control_timestamp: int,
        control_source: Optional[str] = None,
        timeout: float = _ENQUEUE_TIMEOUT_S,
    ) -> bool:
        """Awaitable enqueue: real backpressure instead of refusal.

        Used by the recorder-only pod path (no inference latency to
        protect), so a briefly-behind disk writer slows the RecordTicks
        stream via gRPC flow control instead of refusing ticks. Bounded
        by ``timeout`` — a wedged writer degrades to a refused tick, not
        a stuck handler. Returns True iff the step was accepted.
        """
        if not self._closed and self._already_recorded(int(step)):
            return True
        item = self._build_item(
            step=step, observation_state=observation_state, action=action,
            jpegs=jpegs, control_timestamp=control_timestamp,
            control_source=control_source,
        )
        if item is None:
            return False
        try:
            await asyncio.wait_for(self._queue.put(item), timeout)
        except asyncio.TimeoutError:
            self._note_refused(step, f"writer stalled >{timeout:.0f}s")
            return False
        self._note_enqueued(item)
        return True

    async def _drain_loop(self) -> None:
        """Consume the queue and write rows + JPEGs to disk off-loop.

        Disk writes happen on the default ThreadPoolExecutor so the
        event loop never blocks on I/O. JSONL is opened append-mode
        per batch — one flush per loop iteration keeps the on-disk
        copy nearly current while a sustained 30-Hz stream is
        recording.
        """
        loop = asyncio.get_running_loop()
        try:
            while True:
                item = await self._queue.get()
                if item is None:
                    return  # close sentinel
                try:
                    await loop.run_in_executor(None, self._write_one, item)
                except Exception:
                    log.exception(
                        "SessionRecorder %s: writer task error (continuing)",
                        self.config.episode_id,
                    )
        except asyncio.CancelledError:
            return

    def _write_one(self, item: dict) -> None:
        """Blocking writer (runs in the executor)."""
        step = item["step"]
        jpegs: dict[str, bytes] = item.pop("jpegs", {})

        # Frames: one file per cam, named ``<cam_short>/<step:08d>.jpg``.
        # The rebuilder doesn't care about subdirs — it consumes whatever
        # iter_frames yields — so a per-cam subdir keeps the listing fast.
        for key, raw in jpegs.items():
            cam = _short_cam(key)
            cam_dir = self.frames_dir / cam
            cam_dir.mkdir(parents=True, exist_ok=True)
            path = cam_dir / f"{step:08d}.jpg"
            # Write to a tmp + rename so a partial write never leaves
            # a corrupt file the rebuilder would later try to decode.
            tmp = path.with_suffix(".jpg.part")
            tmp.write_bytes(raw)
            tmp.replace(path)

        # Step row: append JSON-line. Use a list of floats so the file
        # round-trips through json.loads without numpy types.
        row = {
            "step": step,
            "observation": item["observation"],
            "action": item["action"],
            "control_timestamp": item["control_timestamp"],
        }
        cs = item.get("control_source")
        if cs:
            row["control_source"] = cs
        with self.steps_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row))
            fh.write("\n")

        # ADR 0016: feed the live dataset AFTER the staging writes so the
        # fallback is always at least as complete as the live output.
        if self._live is not None and not self._live_dead:
            try:
                self._live.add_step(
                    step=step,
                    observation=item["observation"],
                    action=item["action"],
                    jpegs=jpegs,
                    control_source=cs,
                )
            except Exception as exc:
                self._live_dead = True
                if isinstance(exc, LiveEncodeError):
                    log.warning(
                        "SessionRecorder %s: live encode disabled (%s); "
                        "will rebuild from staging at close",
                        self.config.episode_id, exc,
                    )
                else:
                    log.warning(
                        "SessionRecorder %s: live encode failed; "
                        "will rebuild from staging at close",
                        self.config.episode_id, exc_info=True,
                    )
                try:
                    self._live.abort()
                except Exception:
                    pass

    def _measured_fps(self) -> Optional[float]:
        """Compute the actual capture rate from recorded control timestamps.

        Returns None when fewer than 2 steps were captured or the elapsed
        time is degenerate (e.g. identical timestamps). Otherwise returns
        ``(num_intervals) / (elapsed_seconds)`` — i.e. the average frames
        per second the policy actually produced.
        """
        if (
            self._first_ts_ns is None
            or self._last_ts_ns is None
            or self._step_count < 2
        ):
            return None
        elapsed_s = (self._last_ts_ns - self._first_ts_ns) / 1e9
        if elapsed_s <= 0:
            return None
        return (self._step_count - 1) / elapsed_s

    async def finalize(self) -> None:
        """Stop accepting new rows and wait for the writer to drain."""
        if self._closed:
            return
        self._closed = True
        # Sentinel makes the drain task return cleanly.
        await self._queue.put(None)
        if self._drain_task is not None:
            try:
                await self._drain_task
            except Exception:
                log.exception(
                    "SessionRecorder %s: drain task raised on finalise",
                    self.config.episode_id,
                )
            self._drain_task = None

    async def discard(self) -> int:
        """Drop captured data without uploading.

        Used by the idle-GC: if the Pi disappears without sending
        CloseSession, the partial capture is bound to an
        InferenceSession.id the user has already moved past, so
        uploading would just populate a stale dashboard row. Cancels
        the drain task, removes the working dir, and reports the
        number of rows that would have been uploaded.

        Idempotent against re-entry via the same ``_uploaded`` flag
        ``upload()`` uses — exactly one of upload/discard ever runs.
        """
        if self._uploaded:
            return 0
        self._uploaded = True
        await self.finalize()
        if self._live is not None:
            try:
                await asyncio.get_running_loop().run_in_executor(
                    None, self._live.abort,
                )
            except Exception:
                pass
        dropped = self._step_count
        self._cleanup_working_dir()
        return dropped

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------

    async def upload(self) -> None:
        """Build the LeRobot dataset and ship it to the backend.

        Steps mirror the SDK's :class:`Interlatent.upload`:

        1. Finalise the local writer (sentinel + await drain task).
        2. Build a LeRobot v3.0 dataset on disk via :class:`LeRobotRebuilder`.
        3. ``POST /api/v1/episodes`` to register the episode.
        4. Walk dataset files; ``POST .../upload-urls`` in batches.
        5. Presigned ``PUT`` each file (in a thread pool).
        6. ``POST .../upload-complete`` to enqueue the merge + analysis.
        7. ``shutil.rmtree`` the working dir.

        Idempotent against a re-entry: the ``self._uploaded`` flag short-
        circuits any double call from a tardy CloseSession arriving
        right after the idle-GC has already fired.
        """
        if self._uploaded:
            return
        self._uploaded = True

        await self.finalize()

        if self._step_count == 0:
            log.info(
                "SessionRecorder %s: zero steps recorded; skipping upload",
                self.config.episode_id,
            )
            self._cleanup_working_dir()
            return

        loop = asyncio.get_running_loop()

        # Measured fps wins over the requested config.fps — see _first_ts_ns
        # comment for why. Fall back to config.fps when we don't have enough
        # samples (<2 steps or degenerate timestamps).
        measured_fps = self._measured_fps()
        if measured_fps is not None:
            log.info(
                "SessionRecorder %s: measured fps=%.2f (config.fps=%d)",
                self.config.episode_id, measured_fps, self.config.fps,
            )

        # ADR 0016 fast path: the live builder already holds a fully
        # encoded dataset; finalize seals + verifies it. ANY failure —
        # divergent fps, frame-count mismatch, encoder trouble — falls
        # through to the rebuild lane below, which is byte-equivalent to
        # the pre-ADR behavior.
        dataset_root: Optional[Path] = None
        if self._live is not None and not self._live_dead:
            try:
                dataset_root = await loop.run_in_executor(
                    None, self._live.finalize, measured_fps,
                )
                log.info(
                    "SessionRecorder %s: live-encoded dataset ready "
                    "(%d rows); skipping close-time rebuild",
                    self.config.episode_id, self._live.rows,
                )
            except FpsDivergenceError as exc:
                log.info(
                    "SessionRecorder %s: %s; rebuilding at the measured rate",
                    self.config.episode_id, exc,
                )
                await loop.run_in_executor(None, self._live.abort)
            except Exception:
                log.warning(
                    "SessionRecorder %s: live finalize failed; "
                    "rebuilding from staging",
                    self.config.episode_id, exc_info=True,
                )
                await loop.run_in_executor(None, self._live.abort)

        if dataset_root is None:
            # Build the LeRobot dataset under a unique subdir;
            # LeRobotDataset.create requires its root NOT to pre-exist.
            dataset_parent = self.working_dir / "dataset"
            dataset_parent.mkdir(parents=True, exist_ok=True)
            dataset_root = dataset_parent / "v3"

            effective_fps = (
                measured_fps if measured_fps is not None else self.config.fps
            )
            rebuilder = LeRobotRebuilder(
                root=dataset_root,
                fps=int(round(effective_fps)) if effective_fps >= 1 else 1,
                task=self.config.task,
                env_slug=self.config.env_slug,
                # This recorder always runs inside Modal's gVisor-sandboxed
                # container, where lerobot's default libsvtav1 encoder stalls
                # badly (it can't set worker-thread priority there) — see
                # LeRobotRebuilder's vcodec docstring.
                vcodec="h264",
            )

            source = RecorderStepSource(self)
            try:
                _root, episode_uuids = await loop.run_in_executor(
                    None, rebuilder.build_from_source, source
                )
            except Exception:
                log.exception(
                    "SessionRecorder %s: LeRobot rebuild failed; aborting upload",
                    self.config.episode_id,
                )
                self._cleanup_working_dir()
                return

            if not episode_uuids:
                log.warning(
                    "SessionRecorder %s: rebuilder produced zero episodes",
                    self.config.episode_id,
                )
                self._cleanup_working_dir()
                return

        # The rebuilder uses the StepSource's episode IDs and we feed it
        # our session_id, so both lanes upload under config.episode_id.
        episode_id = self.config.episode_id

        try:
            await self._post_episodes_create(episode_id)
            await self._upload_dataset_dir(episode_id, dataset_root)
            await self._post_upload_complete(episode_id)
            if self._dropped:
                log.warning(
                    "SessionRecorder %s: uploaded with %d dropped frames",
                    self.config.episode_id, self._dropped,
                )
            else:
                log.info(
                    "SessionRecorder %s: upload complete (%d steps, %d cams)",
                    self.config.episode_id, self._step_count, len(self._cameras),
                )
            if self._dedup_acked:
                log.info(
                    "SessionRecorder %s: %d retried tick(s) deduped across "
                    "reconnects (ADR 0024)",
                    self.config.episode_id, self._dedup_acked,
                )
        except Exception:
            log.exception(
                "SessionRecorder %s: backend upload failed",
                self.config.episode_id,
            )
        finally:
            self._cleanup_working_dir()

    # ------------------------------------------------------------------
    # Upload internals
    # ------------------------------------------------------------------

    def _build_headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.config.api_key,
            "Accept": "application/json",
        }

    async def _post_episodes_create(self, episode_id: str) -> None:
        import httpx

        body = {
            "episode_id": episode_id,
            "environment": self.config.env_slug,
            "layer": self.config.layer,
            "model_id": self.config.model_id,
            "tags": {
                "source": "drtc-server",
                "policy_uri": self.config.policy_uri,
            },
            "sdk_version": self.config.sdk_version,
            "model_framework": "drtc",
        }
        if self.config.task_id:
            body["task_id"] = self.config.task_id
        url = f"{_api_v1_root(self.config.api_base)}/episodes"
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            r = await client.post(url, json=body, headers=self._build_headers())
            if r.status_code == 409:
                # Pre-existing episode row — tolerate (idle-GC may have
                # fired and committed the upload before a tardy
                # CloseSession arrived).
                log.info(
                    "SessionRecorder %s: episode row already existed (409)",
                    self.config.episode_id,
                )
                return
            r.raise_for_status()

    async def _upload_dataset_dir(self, episode_id: str, root: Path) -> None:
        """Walk ``root``, mint presigned URLs in batches, ``PUT`` each file."""
        import httpx

        # Collect manifest entries (matches SDK ``_inbox/<uuid>/<rel_path>``).
        session_uuid = uuid.uuid4().hex
        manifest: list[dict[str, Any]] = []
        for f in sorted(root.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(root).as_posix()
            manifest.append({
                "key": f"_inbox/{session_uuid}/{rel}",
                "local": str(f),
                "size": f.stat().st_size,
            })
        if not manifest:
            log.warning(
                "SessionRecorder %s: dataset root contained zero files",
                self.config.episode_id,
            )
            return

        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            headers = self._build_headers()

            for i in range(0, len(manifest), _UPLOAD_BATCH_SIZE):
                batch = manifest[i : i + _UPLOAD_BATCH_SIZE]
                url = (
                    f"{_api_v1_root(self.config.api_base)}/episodes/"
                    f"{episode_id}/upload-urls"
                )
                req_body = {
                    "files": [{"key": e["key"], "size": e["size"]} for e in batch],
                }
                r = await client.post(url, json=req_body, headers=headers)
                r.raise_for_status()
                presigned: dict[str, str] = r.json().get("presigned_urls", {})

                # PUT each file. Concurrency is bounded by the asyncio
                # gather; httpx pools the connections.
                async def _put(entry: dict[str, Any]) -> None:
                    put_url = presigned.get(entry["key"])
                    if not put_url:
                        raise RuntimeError(
                            f"Backend did not return a presigned URL for {entry['key']}"
                        )
                    # Read+upload with a fresh client per call would be
                    # wasteful; httpx.AsyncClient is fine to share across
                    # PUTs but S3 doesn't accept the same auth headers,
                    # so we open a per-PUT call with no global headers.
                    async with httpx.AsyncClient(timeout=_PUT_TIMEOUT) as put_client:
                        with open(entry["local"], "rb") as fh:
                            data = fh.read()
                        resp = await put_client.put(put_url, content=data)
                        if not (200 <= resp.status_code < 300):
                            raise RuntimeError(
                                f"S3 PUT {entry['key']} -> "
                                f"{resp.status_code} {resp.text[:160]}"
                            )

                await asyncio.gather(*(_put(e) for e in batch))

    async def _post_upload_complete(self, episode_id: str) -> None:
        import httpx

        url = (
            f"{_api_v1_root(self.config.api_base)}/episodes/"
            f"{episode_id}/upload-complete"
        )
        body = {
            "manifest": None,
            # Pod-reported collection origin: ≥1 executed step came from a
            # human over the teleop relay. The backend stores it as the
            # episode's has_teleop badge; the SDK upload path never sets it.
            "has_teleop": self._teleop_steps > 0,
        }
        if self._teleop_steps:
            log.info(
                "SessionRecorder %s: reporting has_teleop=True "
                "(%d/%d teleop steps)",
                self.config.episode_id, self._teleop_steps, self._step_count,
            )
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            r = await client.post(url, json=body, headers=self._build_headers())
            r.raise_for_status()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _cleanup_working_dir(self) -> None:
        try:
            shutil.rmtree(self.working_dir, ignore_errors=True)
        except Exception:
            log.warning(
                "SessionRecorder %s: cleanup of %s raised",
                self.config.episode_id, self.working_dir, exc_info=True,
            )

    # ------------------------------------------------------------------
    # Status — useful from the idle-GC + tests
    # ------------------------------------------------------------------

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def no_steps(self) -> bool:
        """True when nothing was recorded, so the upload would be a no-op.

        Lets callers skip the ``uploading`` status transition entirely
        rather than flashing ``uploading`` then ``ready`` for an empty
        session.
        """
        return self._step_count == 0

    @property
    def dropped(self) -> int:
        return self._dropped

    @property
    def reusable(self) -> bool:
        """True while the recorder can still accept a reconnect handoff
        (ADR 0024): not finalised and no upload/discard begun."""
        return not self._closed and not self._uploaded

    @property
    def cameras(self) -> list[str]:
        return list(self._cameras)


# ----------------------------------------------------------------------
# StepSource adapter — reads the recorder's on-disk staging.
# ----------------------------------------------------------------------


class RecorderStepSource:
    """Exposes a :class:`SessionRecorder`'s on-disk layout as a :class:`StepSource`.

    One recorder = one episode (the DRTC session ID). The adapter reads
    ``steps.jsonl`` once into memory; rolls of even an hour at 30 Hz
    are well under a megabyte of step rows, so this is cheap.

    Frame iteration scans ``frames/<cam>/<step>.jpg``. The rebuilder
    sorts by step internally; ``cameras_for_episode`` returns the
    first-seen order recorded during writes (stable across the build).
    """

    def __init__(self, recorder: SessionRecorder) -> None:
        self._recorder = recorder
        self._episode_id = recorder.config.episode_id
        self._cameras: list[str] = list(recorder.cameras)
        self._rows: list[StepRow] = []
        self._load_steps()

    # StepSource protocol ----------------------------------------------

    def episode_ids(self) -> list[str]:
        return [self._episode_id] if self._rows else []

    def iter_steps(self, episode_id: str) -> Iterable[StepRow]:
        if episode_id != self._episode_id:
            return iter(())
        return iter(self._rows)

    def cameras_for_episode(self, episode_id: str) -> list[Optional[str]]:
        if episode_id != self._episode_id:
            return []
        # The rebuilder accepts ``Optional[str]`` to support unnamed
        # single-cam setups; the server always knows the cam name
        # (from the ``observation.images.<name>`` schema), so every
        # entry is a real string.
        return [c for c in self._cameras]

    def iter_frames(
        self, episode_id: str
    ) -> Iterator[Tuple[int, Optional[str], Path]]:
        if episode_id != self._episode_id:
            return iter(())
        return self._walk_frames()

    # Internals --------------------------------------------------------

    def _load_steps(self) -> None:
        path = self._recorder.steps_path
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    log.warning(
                        "RecorderStepSource %s: skipping malformed JSONL line",
                        self._episode_id,
                    )
                    continue
                self._rows.append(StepRow(
                    episode_id=self._episode_id,
                    step=int(obj.get("step") or 0),
                    observation=list(obj.get("observation") or []),
                    action=list(obj.get("action") or []),
                    # reward/done/truncated/metrics/failure_type have no
                    # source in real-robot inference; leave defaults.
                    control_source=obj.get("control_source") or None,
                ))
        self._rows.sort(key=lambda r: r.step)

    def _walk_frames(self) -> Iterator[Tuple[int, Optional[str], Path]]:
        frames_dir = self._recorder.frames_dir
        if not frames_dir.is_dir():
            return
        # Per-cam subdirs were created at write time. Walking them
        # gives us (step, cam, path) without filename parsing.
        for cam_dir in sorted(frames_dir.iterdir()):
            if not cam_dir.is_dir():
                continue
            cam = cam_dir.name
            for f in sorted(cam_dir.iterdir()):
                if not f.is_file() or f.suffix != ".jpg":
                    continue
                try:
                    step = int(f.stem)
                except ValueError:
                    continue
                yield step, cam, f


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _to_list(value: Any) -> list[float]:
    """Convert an ndarray / sequence / scalar to a plain list[float].

    Used for JSON-serialisable step rows. ``None`` becomes ``[]``.
    """
    if value is None:
        return []
    # Fast-path numpy without importing it at module load (this module
    # is hot-imported by the gRPC servicer at OpenSession time).
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return [float(x) for x in value.reshape(-1).tolist()]
    except ImportError:
        pass
    try:
        return [float(x) for x in value]
    except TypeError:
        try:
            return [float(value)]
        except (TypeError, ValueError):
            return []


__all__ = [
    "RecorderConfig",
    "SessionRecorder",
    "RecorderStepSource",
]
