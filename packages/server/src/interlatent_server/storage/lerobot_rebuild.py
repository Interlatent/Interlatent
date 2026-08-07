"""Build a LeRobot v3.0 dataset from a generic :class:`StepSource`.

The rebuilder is intentionally I/O-agnostic: it consumes the
:class:`StepSource` protocol below, which the server-side DRTC
recorder implements against its own storage layout (JSONL + JPEG
staging dir). Collection is streaming-first (ADR 0022) — the robot
node never builds a dataset, so there is no client-side implementor
of this protocol.

Interlatent's closed `interlatent-engine` carries a copy of this
module at ``interlatent/storage/lerobot_rebuild.py`` for hosted GPU
boxes. Per ADR 0035 the two must not drift; the copy goes away once
the hosted image builds from this package.

Activations are NOT part of the rebuilt dataset. The SAE / latent
interpretability path is no longer supported through this rebuilder;
if it returns, it will be a separate sidecar pipeline rather than a
column on the LeRobot table.
"""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Protocol, Sequence, Tuple

from .lerobot_codec import video_encoder_kwargs

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Source protocol — implemented by SDK and server adapters.
# ---------------------------------------------------------------------------


@dataclass
class StepRow:
    """One env step worth of scalar data, as the rebuilder needs it.

    Field semantics match the LeRobot column they back:

    - ``observation`` / ``action`` flatten to ``(D,)`` float32 columns
    - ``reward`` / ``done`` / ``truncated`` go into ``next.reward`` /
      ``next.done`` / ``annotation.interlatent.truncated``
    - ``metrics`` is exploded to one ``annotation.interlatent.metrics.<name>``
      column per name across the whole source
    - ``failure_type`` lands as a nullable ``pa.string()`` column at
      ``annotation.interlatent.failure_type``. Per ``CONTEXT.md``, the
      rule name is stored directly — no int64 catalog map. The column
      is written as int64 during ``LeRobotDataset.add_frame()`` (lerobot
      0.5.x's add_frame validation is happier with scalar numerics) and
      converted to ``pa.string()`` in a post-edit, mirroring the
      ``interlatent.episode_uuid`` injection pattern.

    Server-side recorders that have no reward/done/etc. simply leave
    them at their defaults — the corresponding columns end up zero/false
    and stay schema-consistent.
    """

    episode_id: str
    step: int
    observation: Sequence[float] = field(default_factory=list)
    action: Sequence[float] = field(default_factory=list)
    reward: float = 0.0
    done: bool = False
    truncated: bool = False
    metrics: Mapping[str, float] = field(default_factory=dict)
    failure_type: Optional[str] = None
    # "policy" for steps driven by the inference policy, "intervention"
    # for steps a human drove while overriding a running policy (the
    # DAgger correction label, ADR 0034), "teleop" for human steps in a
    # policy-less TeleopRecording, "hold" for no-motion ticks. Defaults
    # to None when the source doesn't distinguish — the rebuilder treats
    # that as "policy" so legacy datasets stay schema-consistent.
    control_source: Optional[str] = None


class StepSource(Protocol):
    """Read interface the rebuilder consumes.

    Implementations:

    - ``interlatent_server.server.recorder.RecorderStepSource``
      (the GPU-side DRTC recorder — the only one today)

    All methods MUST be safe to call multiple times and MUST drain any
    pending background writes before returning — the rebuilder reads
    in episode-major order and expects a stable snapshot.
    """

    def episode_ids(self) -> list[str]:
        """Episodes in first-appearance order."""
        ...

    def iter_steps(self, episode_id: str) -> Iterable[StepRow]:
        """Per-step rows in ascending ``step`` order for one episode."""
        ...

    def cameras_for_episode(self, episode_id: str) -> list[Optional[str]]:
        """Camera names present for an episode (``None`` = unnamed single cam)."""
        ...

    def iter_frames(
        self, episode_id: str
    ) -> Iterable[Tuple[int, Optional[str], Path]]:
        """``(step, camera_name_or_None, path_to_image_file)`` triples."""
        ...


# ---------------------------------------------------------------------------
# control_source staging ids
# ---------------------------------------------------------------------------

# Write-time staging ids for ``annotation.interlatent.control_source``.
# lerobot 0.5.x's add_frame validation rejects pa.string(), so the column
# is staged as int64 and converted to string in a post-edit. Keep in sync
# with CONTEXT.md's four-value contract
# {"policy","teleop","hold","intervention"} — "hold" marks disengaged,
# no-motion ticks and must NOT collapse into "policy"; "intervention"
# (ADR 0034) marks a human override of a running policy and is the DAgger
# correction label the training stack weights on, so mislabeling it is
# silent data corruption.
CONTROL_SOURCE_TO_ID = {"policy": 0, "teleop": 1, "hold": 2, "intervention": 3}
CONTROL_SOURCE_ID_TO_NAME = {v: k for k, v in CONTROL_SOURCE_TO_ID.items()}


# ---------------------------------------------------------------------------
# Feature discovery
# ---------------------------------------------------------------------------


def _discover_features(
    *,
    first_row: StepRow,
    cameras: list[Optional[str]],
    image_shape: Optional[Tuple[int, int]],
    has_failure_types: bool,
    has_control_source: bool,
    metric_names: list[str],
) -> dict:
    """Build the LeRobot ``features`` dict from the first step + frame.

    Sized from the first row's observation / action and the first frame's
    HxW. Other rows are zero-padded or truncated to match at write time.
    """
    obs = list(first_row.observation or [])
    action = list(first_row.action or [])

    features: dict[str, dict] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (max(1, len(obs)),),
            "names": None,
        },
        "action": {
            "dtype": "float32",
            "shape": (max(1, len(action)),),
            "names": None,
        },
        "next.reward": {
            "dtype": "float32",
            "shape": (1,),
            "names": None,
        },
        "next.done": {
            "dtype": "bool",
            "shape": (1,),
            "names": None,
        },
        "annotation.interlatent.truncated": {
            "dtype": "bool",
            "shape": (1,),
            "names": None,
        },
    }

    if has_failure_types:
        features["annotation.interlatent.failure_type"] = {
            "dtype": "int64",
            "shape": (1,),
            "names": None,
        }

    if has_control_source:
        # Same int64-staging-then-string-postedit dance as failure_type:
        # lerobot 0.5.x's add_frame validation rejects pa.string() at
        # write time, so we stage as int64 (CONTROL_SOURCE_TO_ID) and
        # convert the column to pa.string() once the parquet is closed.
        features["annotation.interlatent.control_source"] = {
            "dtype": "int64",
            "shape": (1,),
            "names": None,
        }

    for name in metric_names:
        features[f"annotation.interlatent.metrics.{name}"] = {
            "dtype": "float32",
            "shape": (1,),
            "names": None,
        }

    if cameras and image_shape is not None:
        H, W = image_shape
        for cam in cameras:
            key = f"observation.images.{cam}" if cam else "observation.images.default"
            features[key] = {
                "dtype": "video",
                "shape": (H, W, 3),
                "names": {"height": H, "width": W, "channels": 3},
            }

    return features


# ---------------------------------------------------------------------------
# Shared row → LeRobot frame builder
# ---------------------------------------------------------------------------


def build_frame_from_row(
    *,
    row: StepRow,
    step_images: Mapping[Optional[str], Any],
    features: dict,
    cameras: list[Optional[str]],
    failure_type_to_id: Mapping[str, int],
    metric_names: list[str],
    np,
) -> dict[str, Any]:
    """Turn one :class:`StepRow` + decoded RGB arrays into an
    ``add_frame``-ready dict.

    Shared by the rebuild path (images loaded from staged files) and the
    live-encode path (images decoded from in-flight JPEG bytes). Vectors
    are zero-padded/truncated to the feature dims; a camera missing on
    this step gets a zero frame; images are grayscale-expanded and
    zero-padded to the declared HxW.
    """
    obs_dim = features["observation.state"]["shape"][0]
    action_dim = features["action"]["shape"][0]

    obs_vec = np.asarray(list(row.observation or []), dtype=np.float32)
    if obs_vec.shape[0] != obs_dim:
        buf = np.zeros(obs_dim, dtype=np.float32)
        buf[: min(obs_vec.shape[0], obs_dim)] = obs_vec[:obs_dim]
        obs_vec = buf

    action_vec = np.asarray(list(row.action or []), dtype=np.float32)
    if action_vec.shape[0] != action_dim:
        buf = np.zeros(action_dim, dtype=np.float32)
        buf[: min(action_vec.shape[0], action_dim)] = action_vec[:action_dim]
        action_vec = buf

    frame: dict[str, Any] = {
        "observation.state": obs_vec,
        "action": action_vec,
        "next.reward": np.array([float(row.reward)], dtype=np.float32),
        "next.done": np.array([bool(row.done)], dtype=bool),
        "annotation.interlatent.truncated": np.array(
            [bool(row.truncated)], dtype=bool,
        ),
    }

    if "annotation.interlatent.failure_type" in features:
        ft = row.failure_type
        ft_id = failure_type_to_id.get(ft, 0) if ft else 0
        frame["annotation.interlatent.failure_type"] = np.array([ft_id], dtype=np.int64)

    if "annotation.interlatent.control_source" in features:
        # None (legacy rows) defaults to "policy".
        label = row.control_source or "policy"
        cs_id = CONTROL_SOURCE_TO_ID.get(label)
        if cs_id is None:
            # Version-skew tripwire (ADR 0034): an SDK newer than this engine
            # emitted a label we don't know. Falling back to "policy" is the
            # worst possible mislabel for a human-driven frame, so make the
            # skew loud instead of silent.
            _LOG.warning(
                "Unknown control_source %r (SDK newer than engine?) — "
                "recording as 'policy'. Update the engine deployment: this "
                "mislabels human frames and corrupts intervention-weighted "
                "training.",
                label,
            )
            cs_id = 0
        frame["annotation.interlatent.control_source"] = np.array([cs_id], dtype=np.int64)

    metrics = row.metrics or {}
    for name in metric_names:
        val = metrics.get(name)
        try:
            v = 0.0 if val is None else float(val)
        except (TypeError, ValueError):
            v = 0.0
        frame[f"annotation.interlatent.metrics.{name}"] = np.array([v], dtype=np.float32)

    if cameras:
        for cam in cameras:
            feat_key = f"observation.images.{cam}" if cam else "observation.images.default"
            if feat_key not in features:
                continue
            spec = features[feat_key]
            H, W = spec["shape"][:2]
            img_arr = step_images.get(cam)
            if img_arr is not None:
                img_arr = np.asarray(img_arr, dtype=np.uint8)
                if img_arr.ndim == 2:
                    img_arr = np.stack([img_arr] * 3, axis=-1).astype(np.uint8)
                if img_arr.shape[:2] != (H, W):
                    # Resize would be ideal but introduces a Pillow
                    # filter dependency; for now zero-pad.
                    buf = np.zeros((H, W, 3), dtype=np.uint8)
                    h2 = min(H, img_arr.shape[0])
                    w2 = min(W, img_arr.shape[1])
                    buf[:h2, :w2] = img_arr[:h2, :w2, :3]
                    img_arr = buf
                frame[feat_key] = img_arr
            else:
                frame[feat_key] = np.zeros((H, W, 3), dtype=np.uint8)

    return frame


# ---------------------------------------------------------------------------
# Rebuilder
# ---------------------------------------------------------------------------


class LeRobotRebuilder:
    """Build a LeRobot v3.0 dataset on disk from a :class:`StepSource`.

    Construction is cheap — heavy imports (lerobot, PIL, pyarrow) are
    deferred to :meth:`build_from_source`. Callers do not import this
    class directly; on the SDK it is invoked by
    :class:`interlatent.Interlatent.upload`, on the engine it is
    invoked by the DRTC server recorder.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        fps: int,
        task: str,
        env_slug: str,
        repo_id: Optional[str] = None,
        vcodec: Optional[str] = None,
        force_control_source: bool = False,
        measured_fps: Optional[float] = None,
    ) -> None:
        # ``root`` must NOT yet exist. ``LeRobotDataset.create()`` calls
        # ``root.mkdir(exist_ok=False)`` internally and will raise
        # ``FileExistsError`` otherwise.
        self.root = Path(root)
        self.fps = int(fps)
        self.task = task or env_slug or "rollout"
        self.env_slug = env_slug or "unknown"
        # ``repo_id`` is required by LeRobotDataset.create even when we
        # never push to the Hub. Synthesize a stable per-env identifier.
        self.repo_id = repo_id or f"interlatent/{(env_slug or 'session').strip('/')}"
        # When True, always emit the ``control_source`` column even if no
        # teleop occurred this session. Merge-on-stop sinks aggregate sessions
        # into one dataset and lerobot's ``aggregate_datasets`` rejects
        # mismatched ``features``, so a schema that varies with whether a human
        # happened to intervene makes the merge fail on the second session.
        self.force_control_source = bool(force_control_source)
        # The real capture rate, stamped into info["interlatent"] for reference
        # when the dataset ``fps`` is pinned to the declared rate so merges stay
        # consistent. None = not recorded.
        self.measured_fps = measured_fps
        # None keeps lerobot's own default ("libsvtav1"). The DRTC recorder
        # (Modal's gVisor-sandboxed teleop-record pod) passes "h264": SVT-AV1
        # tries to set worker-thread scheduling priority
        # (pthread_setschedparam), which gVisor rejects as EINVAL — logged as
        # "Failed to set thread priority: Invalid argument" — and the encoder
        # then stalls badly (an upload that never finishes) instead of just
        # running unprioritized. libx264 has no such step.
        #
        # How the request reaches lerobot depends on its version (the
        # parameter was renamed twice); ``lerobot_codec`` resolves that.
        self.vcodec = vcodec

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_from_source(self, source: StepSource) -> Tuple[Path, list[str]]:
        """Read ``source`` and produce a LeRobot v3.0 dataset on disk.

        Returns ``(root, episode_uuids)`` where ``episode_uuids[i]`` is
        the source-provided UUID for LeRobot ``episode_index = i``.

        Raises :class:`RuntimeError` if ``lerobot`` or ``Pillow`` are
        not installed (both required at build time).
        """
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise RuntimeError(
                "lerobot is required to rebuild a dataset. "
                "Install with: pip install 'interlatent-server[lerobot]'"
            ) from exc

        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "Pillow is required to rebuild a dataset. "
                "Install with: pip install Pillow"
            ) from exc

        import numpy as np

        # 1. Collect episodes + step rows up-front. Per-episode iteration
        #    is light (a tens-of-thousands rows per episode at the very
        #    upper end) and we need to know totals to discover features.
        episode_uuids = list(source.episode_ids())
        if not episode_uuids:
            return self.root, []

        rows_by_episode: dict[str, list[StepRow]] = {}
        for eid in episode_uuids:
            ep_rows = list(source.iter_steps(eid))
            ep_rows.sort(key=lambda r: r.step)
            rows_by_episode[eid] = ep_rows

        # Drop empty episodes — they would produce zero-frame LeRobot
        # episodes which the writer rejects.
        episode_uuids = [eid for eid in episode_uuids if rows_by_episode[eid]]
        if not episode_uuids:
            return self.root, []

        # 2. Schema discovery: cameras + image shape from the first
        #    episode that has frames; failure-type catalog + metric
        #    catalog from all rows in first-appearance order.
        cameras: list[Optional[str]] = []
        image_shape: Optional[Tuple[int, int]] = None
        for eid in episode_uuids:
            ep_cams = list(source.cameras_for_episode(eid))
            if not ep_cams:
                continue
            cameras = ep_cams
            for _step, _cam, frame_path in source.iter_frames(eid):
                with Image.open(frame_path) as img:
                    arr = np.asarray(img)
                if arr.ndim == 3 and arr.shape[2] == 3:
                    image_shape = (arr.shape[0], arr.shape[1])
                    break
                if arr.ndim == 2:
                    image_shape = (arr.shape[0], arr.shape[1])
                    break
            if image_shape is not None:
                break

        failure_types: list[str] = []
        metric_names: list[str] = []
        has_control_source = bool(self.force_control_source)
        for eid in episode_uuids:
            for row in rows_by_episode[eid]:
                ft = row.failure_type
                if ft and ft not in failure_types:
                    failure_types.append(ft)
                for name in row.metrics:
                    if name not in metric_names:
                        metric_names.append(name)
                if row.control_source:
                    has_control_source = True
        # 0 is reserved for "no failure" so live ids start at 1.
        failure_type_to_id = {ft: i + 1 for i, ft in enumerate(failure_types)}

        first_row = rows_by_episode[episode_uuids[0]][0]
        features = _discover_features(
            first_row=first_row,
            cameras=cameras,
            image_shape=image_shape,
            has_failure_types=bool(failure_type_to_id),
            has_control_source=has_control_source,
            metric_names=metric_names,
        )

        # 3. Open the LeRobot writer. ``LeRobotDataset.create``
        #    materializes meta/info.json and the parquet writers.
        dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            fps=self.fps,
            features=features,
            root=str(self.root),
            robot_type=self.env_slug or "custom",
            use_videos=bool(cameras and image_shape is not None),
            # The codec parameter has been renamed twice upstream and
            # create() takes no **kwargs, so the right one is detected
            # per-version — see storage/lerobot_codec.py.
            **video_encoder_kwargs(LeRobotDataset.create, self.vcodec),
        )

        # 4. Pre-index frames per (episode, step) so the inner loop is
        #    O(steps) instead of O(steps * frames).
        frames_by_episode: dict[str, dict[int, dict[Optional[str], Path]]] = {}
        for eid in episode_uuids:
            ep_frames: dict[int, dict[Optional[str], Path]] = {}
            for step, cam, path in source.iter_frames(eid):
                ep_frames.setdefault(step, {})[cam] = path
            frames_by_episode[eid] = ep_frames

        # 5. Stream rows into the dataset. ``finalize()`` MUST run even
        #    on error or the parquet files end up with truncated footers.
        try:
            for eid in episode_uuids:
                ep_rows = rows_by_episode.get(eid) or []
                if not ep_rows:
                    continue
                ep_frames = frames_by_episode.get(eid, {})
                for row in ep_rows:
                    frame = self._build_frame(
                        row=row,
                        ep_frames=ep_frames,
                        features=features,
                        cameras=cameras,
                        failure_type_to_id=failure_type_to_id,
                        metric_names=metric_names,
                        np=np,
                        Image=Image,
                    )
                    # add_frame() expects ``task`` to live inside the
                    # frame dict (it does ``frame.pop("task")``
                    # internally) — it is not a kwarg.
                    frame["task"] = self.task
                    dataset.add_frame(frame)
                # numpy 2.x compat: lerobot's
                # ``get_hf_features_from_features`` maps every shape-(1,)
                # feature to ``datasets.Value`` (scalar), but its own
                # ``validate_frame`` requires ``np.ndarray`` of shape (1,).
                # On numpy 2.x HF then calls ``float(np.array([0.0]))``
                # during encode and raises
                # "only 0-dimensional arrays can be converted to Python
                # scalars". Squeeze each shape-(1,) buffer entry to a
                # Python scalar before save_episode so HF gets what it
                # expects.
                self._scalarize_singleton_columns(dataset, features)
                dataset.save_episode()
        finally:
            try:
                dataset.finalize()
            except Exception:
                _LOG.exception("LeRobotDataset.finalize() raised")

        # 6. Post-edit meta/episodes/.../parquet for the UUID column.
        self._inject_episode_uuids(self.root, episode_uuids)

        # 7. Stamp meta/info.json with the custom interlatent block.
        self._stamp_info_json(self.root, metric_names)

        # 8. Convert annotation.interlatent.failure_type from int64 catalog
        #    IDs to nullable pa.string() per CONTEXT.md. The int64 path
        #    above is just a write-time staging convenience.
        if failure_type_to_id:
            id_to_name = {idx: name for name, idx in failure_type_to_id.items()}
            self._convert_failure_type_to_string(self.root, id_to_name)

        # 9. Convert annotation.interlatent.control_source from int64 to
        #    pa.string() ("policy"/"teleop"). Same staging trick as
        #    failure_type.
        if has_control_source:
            self._convert_int_column_to_string(
                self.root,
                col_name="annotation.interlatent.control_source",
                id_to_name=CONTROL_SOURCE_ID_TO_NAME,
                nullable=False,
            )

        return self.root, episode_uuids

    def cleanup(self) -> None:
        """Remove the on-disk dataset directory.

        Also reaps the parent if the caller nested us inside a fresh
        tempdir (best-effort — silently ignores if the parent is shared).
        """
        if self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)
        parent = self.root.parent
        try:
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _scalarize_singleton_columns(dataset, features: dict) -> None:
        """Convert episode-buffer entries for shape-(1,) features from
        ``np.array([x])`` to plain scalars in place.

        See call site for the numpy-2.x rationale.
        """
        buf = getattr(dataset, "episode_buffer", None)
        if not isinstance(buf, dict):
            return
        for key, ft in features.items():
            if ft.get("shape") != (1,):
                continue
            col = buf.get(key)
            if not isinstance(col, list):
                continue
            for i, v in enumerate(col):
                if hasattr(v, "item") and hasattr(v, "shape") and v.shape == (1,):
                    try:
                        col[i] = v.item()
                    except Exception:
                        pass

    def _build_frame(
        self,
        *,
        row: StepRow,
        ep_frames: dict[int, dict[Optional[str], Path]],
        features: dict,
        cameras: list[Optional[str]],
        failure_type_to_id: dict[str, int],
        metric_names: list[str],
        np,
        Image,
    ) -> dict[str, Any]:
        """Load this step's images from disk, then delegate to the shared
        row→frame builder (also used by the live-encode path, which holds
        decoded arrays instead of paths — see ADR 0016)."""
        step_images: dict[Optional[str], Any] = {}
        for cam, path in (ep_frames.get(row.step, {}) or {}).items():
            with Image.open(path) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                step_images[cam] = np.asarray(img, dtype=np.uint8)
        return build_frame_from_row(
            row=row,
            step_images=step_images,
            features=features,
            cameras=cameras,
            failure_type_to_id=failure_type_to_id,
            metric_names=metric_names,
            np=np,
        )

    def _inject_episode_uuids(self, root: Path, episode_uuids: list[str]) -> None:
        """Append an ``interlatent.episode_uuid`` column to ``meta/episodes/...``.

        LeRobot keys episodes by integer ``episode_index``; the dashboard
        keys by source-supplied UUID. Carrying both lets the backend
        merge worker (and any downstream analysis pipeline) keep the
        join.
        """
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            _LOG.warning("pyarrow not installed; skipping episode-uuid post-edit")
            return

        episodes_dir = root / "meta" / "episodes"
        if not episodes_dir.exists():
            _LOG.warning(
                "meta/episodes/ not found in dataset at %s — skipping uuid post-edit",
                root,
            )
            return

        for parquet_path in sorted(episodes_dir.rglob("*.parquet")):
            table = pq.read_table(parquet_path)
            if "episode_index" not in table.column_names:
                continue
            ep_idx = table.column("episode_index").to_pylist()
            uuids = [
                episode_uuids[i] if 0 <= i < len(episode_uuids) else ""
                for i in ep_idx
            ]
            uuid_col = pa.array(uuids, type=pa.string())
            new_table = table.append_column("interlatent.episode_uuid", uuid_col)
            pq.write_table(new_table, parquet_path)

    def _stamp_info_json(
        self,
        root: Path,
        metric_names: list[str],
    ) -> None:
        info_path = root / "meta" / "info.json"
        if not info_path.exists():
            return
        with open(info_path, "r") as fh:
            info = json.load(fh)

        info["interlatent"] = {
            "environment_slug": self.env_slug,
            "task": self.task,
            "metric_names": list(metric_names),
        }
        if self.measured_fps is not None:
            # The dataset's ``fps`` is pinned to the declared rate so sessions
            # stay mergeable; the rate actually captured is preserved here.
            info["interlatent"]["measured_fps"] = float(self.measured_fps)

        with open(info_path, "w") as fh:
            json.dump(info, fh, indent=2)

    def _convert_failure_type_to_string(
        self,
        root: Path,
        id_to_name: dict[int, str],
    ) -> None:
        """Rewrite ``annotation.interlatent.failure_type`` from int64 → nullable string.

        Thin wrapper around :meth:`_convert_int_column_to_string` that
        preserves the "0 = no failure → null" sentinel semantics specific
        to this column.
        """
        self._convert_int_column_to_string(
            root,
            col_name="annotation.interlatent.failure_type",
            id_to_name=id_to_name,
            nullable=True,
        )

    def _convert_int_column_to_string(
        self,
        root: Path,
        *,
        col_name: str,
        id_to_name: dict[int, str],
        nullable: bool,
    ) -> None:
        """Rewrite an int64-staged annotation column to ``pa.string()``.

        lerobot 0.5.x's ``add_frame`` validation is happier with scalar
        numerics than with strings, so string columns are staged as int64
        catalog IDs during write and converted in a post-edit pass.
        ``nullable=True`` maps id ``0`` to ``None`` (the "no value"
        sentinel for failure_type); ``nullable=False`` keeps every value
        as a string.
        """
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            _LOG.warning(
                "pyarrow not installed; skipping %s int64→string conversion",
                col_name,
            )
            return

        data_dir = root / "data"
        if not data_dir.exists():
            return

        for parquet_path in sorted(data_dir.rglob("*.parquet")):
            table = pq.read_table(parquet_path)
            if col_name not in table.column_names:
                continue
            raw = table.column(col_name).to_pylist()
            # Column may be either scalar ints or length-1 list-encoded ints.
            converted: list[Optional[str]] = []
            for v in raw:
                if isinstance(v, list):
                    v = v[0] if v else None
                if v is None:
                    converted.append(None if nullable else id_to_name.get(0))
                    continue
                iv = int(v)
                if nullable and iv == 0:
                    converted.append(None)
                else:
                    converted.append(id_to_name.get(iv))
            new_col = pa.array(converted, type=pa.string())
            idx = table.column_names.index(col_name)
            table = table.set_column(idx, col_name, new_col)
            pq.write_table(table, parquet_path)

        # Update info.json features dict so loaders see the new dtype.
        info_path = root / "meta" / "info.json"
        if info_path.exists():
            with open(info_path, "r") as fh:
                info = json.load(fh)
            features = info.get("features") or {}
            if col_name in features:
                features[col_name] = {
                    "dtype": "string",
                    "shape": (1,),
                    "names": None,
                }
                info["features"] = features
                with open(info_path, "w") as fh:
                    json.dump(info, fh, indent=2)


__all__ = ["StepRow", "StepSource", "LeRobotRebuilder"]
