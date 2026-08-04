"""Live (during-session) LeRobot episode builder for the DRTC recorder.

ADR 0016: teleop recordings encode video live during the session. This
module owns the live path — a single-episode LeRobot v3.0 dataset built
incrementally as steps arrive, with per-camera video encoded in real
time by lerobot's ``StreamingVideoEncoder`` (threads + PyAV, no PNG
round-trip). At CloseSession only ``save_episode()`` (near-instant),
a frame-count verify, and the standard post-edits remain.

Failure philosophy (one recovery lane): ANY problem — encoder crash,
schema surprise, frame-count mismatch, measured-fps divergence — raises
:class:`LiveEncodeError`; the caller (``SessionRecorder``) then discards
the live output and rebuilds from its JPEG + JSONL staging, which is
byte-equivalent to the pre-ADR-0016 behavior. Corrupt or misaligned
data can never ship; every failure mode costs a slower close only.

The artifact contract matches ``LeRobotRebuilder.build_from_source``
for the recorder's single-episode case: same features (via
``_discover_features``), same frame semantics (via
``build_frame_from_row``), same post-edits (uuid injection, info.json
stamp, control_source int→string).
"""

from __future__ import annotations

import io
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Optional

from .lerobot_codec import video_encoder_kwargs
from .lerobot_rebuild import (
    CONTROL_SOURCE_ID_TO_NAME,
    LeRobotRebuilder,
    StepRow,
    _discover_features,
    build_frame_from_row,
)

log = logging.getLogger(__name__)


# Encoder queue depth per camera. 4 s of buffer against a 30 Hz feed —
# overflow should be unreachable on an 8-core pod; if it happens anyway,
# lerobot drops frames and the close-time verify catches the mismatch.
_ENCODER_QUEUE_MAXSIZE = 120

# Steps buffered while waiting for the first camera frame before the
# live path gives up (the fallback rebuild handles camera-less or
# late-camera sessions correctly).
_MAX_PENDING_NO_CAMERA = 256

# Default relative fps-divergence tolerance. A real control loop never
# averages the declared rate exactly (a healthy 30 Hz node measures
# ~29.4), and the original round-to-integer trigger sent every such
# session down the slow rebuild lane — a multi-minute close for a
# ~2% playback-speed error the merge doesn't preserve anyway (the
# canonical keeps its own declared fps). Within this fraction the live
# output ships as-is; beyond it (a genuinely throttled loop) the
# rebuild lane still encodes at the measured rate.
_FPS_REL_TOLERANCE = 0.05


def _fps_tolerance() -> float:
    """Relative divergence tolerance from INTERLATENT_LIVE_FPS_TOL
    (fraction of the declared fps; default 0.05)."""
    try:
        tol = float(os.environ.get("INTERLATENT_LIVE_FPS_TOL", "") or _FPS_REL_TOLERANCE)
    except (TypeError, ValueError):
        tol = _FPS_REL_TOLERANCE
    return max(0.0, tol)


def fps_diverges(measured_fps: float, declared_fps: int) -> bool:
    """True when the measured capture rate is far enough from the declared
    rate that the live output should be discarded for the rebuild lane."""
    if declared_fps <= 0:
        return False
    return abs(measured_fps - declared_fps) > _fps_tolerance() * declared_fps


class LiveEncodeError(RuntimeError):
    """The live path cannot (or should not) produce this episode.

    Deliberate signal, not a bug marker — the caller falls back to the
    rebuild-from-staging path and the episode survives.
    """


class FpsDivergenceError(LiveEncodeError):
    """Measured capture rate diverges from the declared fps beyond the
    relative tolerance (``fps_diverges``).

    The live dataset's timestamps and video PTS were stamped at the
    declared rate during the session; the rebuild path encodes at the
    measured rate natively, so divergent sessions take that lane.
    """


def _short_cam(key: str) -> str:
    prefix = "observation.images."
    return key[len(prefix):] if key.startswith(prefix) else key


class LiveEpisodeBuilder:
    """Incrementally build one episode's LeRobot dataset on local SSD.

    Single-owner, single-thread-at-a-time: ``add_step`` calls are
    serialized by the recorder's drain task (each runs to completion in
    the executor before the next is submitted), and ``finalize``/
    ``abort`` run only after the drain task has stopped. No locking.

    Construction is cheap; lerobot imports and ``LeRobotDataset.create``
    are deferred to the first step that carries a camera frame (that's
    when the image shape — and therefore the feature schema — is known).
    """

    def __init__(
        self,
        root: Path | str,
        *,
        fps: int,
        task: str,
        env_slug: str,
        episode_id: str,
        repo_id: Optional[str] = None,
    ) -> None:
        # Must NOT exist yet — LeRobotDataset.create mkdirs with
        # exist_ok=False.
        self.root = Path(root)
        self.fps = int(fps)
        self.task = task or env_slug or "rollout"
        self.env_slug = env_slug or "unknown"
        self.episode_id = episode_id
        self.repo_id = repo_id or f"interlatent/{(env_slug or 'session').strip('/')}"

        self._dataset: Any = None
        self._features: Optional[dict] = None
        self._cameras: list[str] = []
        self._rows = 0
        # Steps seen before any camera frame arrived; replayed (with
        # zero images) once the schema locks on the first camera step.
        self._pending: list[StepRow] = []

    # ------------------------------------------------------------------
    # Hot path (executor thread, serialized)
    # ------------------------------------------------------------------

    def add_step(
        self,
        *,
        step: int,
        observation: Any,
        action: Any,
        jpegs: dict[str, bytes],
        control_source: Optional[str],
    ) -> None:
        """Feed one recorded step. Blocking; raises LiveEncodeError (or
        any lerobot error) to signal the caller to kill the live path."""
        row = StepRow(
            episode_id=self.episode_id,
            step=int(step),
            observation=list(observation or []),
            action=list(action or []),
            control_source=control_source or None,
        )
        images = self._decode_jpegs(jpegs)

        if self._dataset is None:
            if not images:
                self._pending.append(row)
                if len(self._pending) > _MAX_PENDING_NO_CAMERA:
                    raise LiveEncodeError(
                        f"no camera frame within the first "
                        f"{_MAX_PENDING_NO_CAMERA} steps"
                    )
                return
            self._create_dataset(first_row=self._pending[0] if self._pending else row,
                                 first_images=images)
            for early in self._pending:
                self._add_frame(early, {})
            self._pending.clear()

        unknown = [c for c in images if c not in self._cameras]
        if unknown:
            # Schema is locked at create time; a camera appearing
            # mid-session is a shape the rebuild path handles and this
            # one deliberately doesn't.
            raise LiveEncodeError(f"new camera(s) mid-session: {unknown}")

        self._add_frame(row, images)

    # ------------------------------------------------------------------
    # Close path
    # ------------------------------------------------------------------

    def finalize(self, measured_fps: Optional[float]) -> Path:
        """Seal the dataset, verify video/parquet alignment, post-edit.

        Returns the dataset root, upload-ready. Raises
        :class:`LiveEncodeError` when the rebuild path should take over.
        """
        if self._dataset is None:
            raise LiveEncodeError("live dataset was never created")
        if self._rows == 0:
            raise LiveEncodeError("zero rows fed")

        if measured_fps is not None and measured_fps >= 1:
            if fps_diverges(measured_fps, self.fps):
                raise FpsDivergenceError(
                    f"measured fps {measured_fps:.2f} diverges from declared "
                    f"{self.fps} by more than {_fps_tolerance():.0%}"
                )
            if measured_fps != self.fps:
                log.info(
                    "live encode: measured fps %.2f within %.0f%% of declared "
                    "%d — keeping live output (episode=%s)",
                    measured_fps, _fps_tolerance() * 100, self.fps,
                    self.episode_id,
                )

        dataset, features = self._dataset, self._features or {}
        try:
            # numpy-2.x scalarization — same dance as the rebuilder.
            LeRobotRebuilder._scalarize_singleton_columns(dataset, features)
            dataset.save_episode()
        finally:
            try:
                dataset.finalize()
            except Exception:
                log.exception("live dataset finalize() raised")

        self._verify_frame_counts(features)

        # Post-edits — reuse the rebuilder's helpers. Instantiating it
        # here never calls build_from_source, so the root-must-not-exist
        # rule doesn't apply; __init__ only records fields.
        toolbox = LeRobotRebuilder(
            root=self.root, fps=self.fps, task=self.task, env_slug=self.env_slug,
        )
        toolbox._inject_episode_uuids(self.root, [self.episode_id])
        toolbox._stamp_info_json(self.root, [])
        toolbox._convert_int_column_to_string(
            self.root,
            col_name="annotation.interlatent.control_source",
            id_to_name=CONTROL_SOURCE_ID_TO_NAME,
            nullable=False,
        )
        return self.root

    def abort(self) -> None:
        """Stop encoder threads and remove any partial output. Idempotent,
        best-effort — called on any live-path failure and on discard."""
        dataset, self._dataset = self._dataset, None
        self._pending.clear()
        if dataset is not None:
            try:
                # Cancels the streaming encoder + drops the buffer.
                dataset.clear_episode_buffer(delete_images=True)
            except Exception:
                pass
            try:
                dataset.finalize()
            except Exception:
                pass
        shutil.rmtree(self.root, ignore_errors=True)

    @property
    def rows(self) -> int:
        return self._rows

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _decode_jpegs(self, jpegs: dict[str, bytes]) -> dict[str, Any]:
        if not jpegs:
            return {}
        import numpy as np
        from PIL import Image

        out: dict[str, Any] = {}
        for key, raw in jpegs.items():
            if not raw:
                continue
            with Image.open(io.BytesIO(raw)) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                out[_short_cam(key)] = np.asarray(img, dtype=np.uint8)
        return out

    def _create_dataset(self, *, first_row: StepRow, first_images: dict[str, Any]) -> None:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise LiveEncodeError(f"lerobot unavailable: {exc}") from exc

        cameras = list(first_images.keys())
        shape = next(iter(first_images.values())).shape
        image_shape = (int(shape[0]), int(shape[1]))

        # control_source is always declared on the live path: every
        # recorder row on the teleop pod carries one, and a stable schema
        # keeps live-built and rebuilt datasets mergeable.
        features = _discover_features(
            first_row=first_row,
            cameras=cameras,
            image_shape=image_shape,
            has_failure_types=False,
            has_control_source=True,
            metric_names=[],
        )

        self.root.parent.mkdir(parents=True, exist_ok=True)
        self._dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            fps=self.fps,
            features=features,
            root=str(self.root),
            robot_type=self.env_slug or "custom",
            use_videos=bool(cameras),
            # gVisor: SVT-AV1 stalls (thread-priority EINVAL); libx264
            # doesn't. Same rationale as the rebuild path's vcodec — and the
            # same per-version parameter detection, since a mismatch here
            # kills the live lane AND then the rebuild lane it falls back to.
            **video_encoder_kwargs(LeRobotDataset.create, "h264"),
            streaming_encoding=True,
            encoder_queue_maxsize=_ENCODER_QUEUE_MAXSIZE,
        )
        self._features = features
        self._cameras = cameras
        log.info(
            "live encode: dataset created (episode=%s cams=%s shape=%s fps=%d)",
            self.episode_id, cameras, image_shape, self.fps,
        )

    def _add_frame(self, row: StepRow, images: dict[str, Any]) -> None:
        import numpy as np

        frame = build_frame_from_row(
            row=row,
            step_images=images,
            features=self._features or {},
            cameras=list(self._cameras),
            failure_type_to_id={},
            metric_names=[],
            np=np,
        )
        frame["task"] = self.task
        self._dataset.add_frame(frame)
        self._rows += 1

    def _verify_frame_counts(self, features: dict) -> None:
        """Every camera's mp4 frame count must equal the parquet row count.

        A streaming-encoder queue drop (or any other desync) shifts all
        later frames relative to their rows — silent label corruption.
        Probe is metadata-only (mp4 stts via PyAV), no decode.
        """
        video_keys = [k for k, ft in features.items() if ft.get("dtype") == "video"]
        if not video_keys:
            return
        try:
            import av
        except ImportError as exc:
            raise LiveEncodeError(f"pyav unavailable; cannot verify: {exc}") from exc

        for key in video_keys:
            vdir = self.root / "videos" / key
            files = sorted(vdir.rglob("*.mp4")) if vdir.is_dir() else []
            if not files:
                raise LiveEncodeError(f"no video files for {key}")
            n = 0
            for f in files:
                with av.open(str(f)) as container:
                    stream = container.streams.video[0]
                    count = int(stream.frames or 0)
                    if count <= 0:
                        count = sum(1 for pkt in container.demux(stream) if pkt.size)
                    n += count
            if n != self._rows:
                raise LiveEncodeError(
                    f"frame-count mismatch for {key}: video={n} rows={self._rows}"
                )


__all__ = [
    "LiveEpisodeBuilder",
    "LiveEncodeError",
    "FpsDivergenceError",
    "fps_diverges",
]
