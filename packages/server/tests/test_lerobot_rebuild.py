"""Tests for the LeRobot v3 dataset writer (``storage/lerobot_rebuild.py``).

This module turns the recorder's staging into the thing that gets merged
into an environment's canonical dataset. The merge is append-only and there
is no un-merge, so a schema or label mistake here is permanent.

``lerobot`` itself is not installed in CI (it drags torch), so
``LeRobotDataset`` is stubbed — the rebuilder holds it behind a single
deferred import, and everything worth testing is on this side of that seam:
feature discovery, the row→frame conversion, and the three parquet
post-edits that give the dataset its Interlatent-specific columns.

The label mapping gets the most attention. ``CONTROL_SOURCE_TO_ID`` is a
four-value contract with CONTEXT.md, and two of its values are load-bearing:
``hold`` must not collapse into ``policy`` (it marks disengaged, no-motion
ticks) and ``intervention`` is the ADR 0034 (platform repo) DAgger correction label the
training stack upweights. Mislabeling either is silent data corruption that
no downstream job can detect.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402

from interlatent_server.storage import lerobot_rebuild as lr  # noqa: E402
from interlatent_server.storage.lerobot_rebuild import (  # noqa: E402
    CONTROL_SOURCE_ID_TO_NAME,
    CONTROL_SOURCE_TO_ID,
    LeRobotRebuilder,
    StepRow,
    build_frame_from_row,
)

pq = pytest.importorskip("pyarrow.parquet")
pa = pytest.importorskip("pyarrow")


# ----------------------------------------------------------------------
# The control_source contract
# ----------------------------------------------------------------------


def test_control_source_ids_match_the_context_md_contract() -> None:
    """Four values, stable ids, and a round-trippable inverse. The ids are
    written into parquet, so renumbering them silently rewrites history."""
    assert CONTROL_SOURCE_TO_ID == {
        "policy": 0, "teleop": 1, "hold": 2, "intervention": 3,
    }
    assert CONTROL_SOURCE_ID_TO_NAME == {
        0: "policy", 1: "teleop", 2: "hold", 3: "intervention",
    }


# ----------------------------------------------------------------------
# Feature discovery
# ----------------------------------------------------------------------


def _discover(**over) -> dict:
    kwargs = dict(
        first_row=StepRow(episode_id="e", step=0,
                          observation=[0.0] * 7, action=[0.0] * 6),
        cameras=[],
        image_shape=None,
        has_failure_types=False,
        has_control_source=False,
        metric_names=[],
    )
    kwargs.update(over)
    return lr._discover_features(**kwargs)


def test_discover_features_sizes_vectors_from_the_first_row() -> None:
    f = _discover()
    assert f["observation.state"]["shape"] == (7,)
    assert f["action"]["shape"] == (6,)
    assert f["observation.state"]["dtype"] == "float32"
    # The always-present annotation columns.
    assert f["next.reward"]["dtype"] == "float32"
    assert f["next.done"]["dtype"] == "bool"
    assert f["annotation.interlatent.truncated"]["dtype"] == "bool"


def test_discover_features_never_emits_a_zero_width_vector() -> None:
    """An image-only policy has no state vector; shape (0,) would make
    LeRobotDataset.create reject the schema."""
    f = _discover(first_row=StepRow(episode_id="e", step=0,
                                    observation=[], action=[]))
    assert f["observation.state"]["shape"] == (1,)
    assert f["action"]["shape"] == (1,)


def test_discover_features_adds_optional_columns_only_when_present() -> None:
    bare = _discover()
    assert "annotation.interlatent.failure_type" not in bare
    assert "annotation.interlatent.control_source" not in bare

    rich = _discover(
        has_failure_types=True,
        has_control_source=True,
        metric_names=["grasp_score", "drift"],
    )
    # Both string columns are STAGED as int64 — lerobot 0.5.x's add_frame
    # validation rejects pa.string() — and converted in a post-edit.
    assert rich["annotation.interlatent.failure_type"]["dtype"] == "int64"
    assert rich["annotation.interlatent.control_source"]["dtype"] == "int64"
    assert rich["annotation.interlatent.metrics.grasp_score"]["dtype"] == "float32"
    assert rich["annotation.interlatent.metrics.drift"]["shape"] == (1,)


def test_discover_features_names_camera_columns() -> None:
    f = _discover(cameras=["wrist", "top"], image_shape=(48, 64))
    assert f["observation.images.wrist"]["shape"] == (48, 64, 3)
    assert f["observation.images.wrist"]["dtype"] == "video"
    assert f["observation.images.wrist"]["names"] == {
        "height": 48, "width": 64, "channels": 3,
    }
    assert "observation.images.top" in f

    # An unnamed single camera gets the `default` key.
    unnamed = _discover(cameras=[None], image_shape=(8, 8))
    assert "observation.images.default" in unnamed


def test_discover_features_skips_cameras_without_a_known_image_shape() -> None:
    """No decodable frame means no HxW; declaring a video column anyway
    would produce a dataset whose video features never get written."""
    assert not [
        k for k in _discover(cameras=["wrist"], image_shape=None)
        if k.startswith("observation.images.")
    ]


# ----------------------------------------------------------------------
# Row -> frame
# ----------------------------------------------------------------------


def _frame(row: StepRow, features: dict, **over):
    kwargs = dict(
        row=row,
        step_images={},
        features=features,
        cameras=[],
        failure_type_to_id={},
        metric_names=[],
        np=np,
    )
    kwargs.update(over)
    return build_frame_from_row(**kwargs)


def test_build_frame_pads_and_truncates_vectors_to_the_declared_width() -> None:
    """Later rows must match the first row's width or the parquet schema
    breaks mid-episode. Short rows zero-pad; long rows truncate."""
    features = _discover()  # obs 7, action 6

    short = _frame(StepRow(episode_id="e", step=0,
                           observation=[1.0, 2.0], action=[3.0]), features)
    assert short["observation.state"].tolist() == [1, 2, 0, 0, 0, 0, 0]
    assert short["action"].tolist() == [3, 0, 0, 0, 0, 0]

    long = _frame(StepRow(episode_id="e", step=1,
                          observation=list(range(9)), action=list(range(8))),
                  features)
    assert long["observation.state"].tolist() == [0, 1, 2, 3, 4, 5, 6]
    assert long["action"].tolist() == [0, 1, 2, 3, 4, 5]
    assert long["observation.state"].dtype == np.float32


def test_build_frame_carries_reward_done_truncated() -> None:
    f = _frame(
        StepRow(episode_id="e", step=0, observation=[0.0] * 7, action=[0.0] * 6,
                reward=0.75, done=True, truncated=True),
        _discover(),
    )
    assert f["next.reward"].tolist() == [pytest.approx(0.75)]
    assert f["next.done"].tolist() == [True]
    assert f["annotation.interlatent.truncated"].tolist() == [True]


@pytest.mark.parametrize(
    "label,expected",
    [("policy", 0), ("teleop", 1), ("hold", 2), ("intervention", 3), (None, 0)],
)
def test_build_frame_maps_every_control_source_label(label, expected: int) -> None:
    """``hold`` must stay distinct from ``policy`` and ``intervention`` must
    survive as itself — both are relied on downstream."""
    features = _discover(has_control_source=True)
    row = StepRow(episode_id="e", step=0, observation=[0.0] * 7,
                  action=[0.0] * 6, control_source=label)
    f = _frame(row, features)
    assert f["annotation.interlatent.control_source"].tolist() == [expected]


def test_build_frame_warns_loudly_on_an_unknown_control_source(caplog) -> None:
    """Version skew (SDK newer than the engine) falls back to 'policy',
    which is the worst possible mislabel for a human frame — it must warn."""
    features = _discover(has_control_source=True)
    row = StepRow(episode_id="e", step=0, observation=[0.0] * 7,
                  action=[0.0] * 6, control_source="telepathy")
    with caplog.at_level("WARNING"):
        f = _frame(row, features)
    assert f["annotation.interlatent.control_source"].tolist() == [0]
    assert "telepathy" in caplog.text


def test_build_frame_maps_failure_types_through_the_catalog() -> None:
    features = _discover(has_failure_types=True)
    catalog = {"dropped": 1, "collision": 2}
    for label, expected in (("dropped", 1), ("collision", 2), (None, 0)):
        f = _frame(
            StepRow(episode_id="e", step=0, observation=[0.0] * 7,
                    action=[0.0] * 6, failure_type=label),
            features, failure_type_to_id=catalog,
        )
        assert f["annotation.interlatent.failure_type"].tolist() == [expected]


def test_build_frame_defaults_missing_and_unparseable_metrics_to_zero() -> None:
    features = _discover(metric_names=["a", "b", "c"])
    f = _frame(
        StepRow(episode_id="e", step=0, observation=[0.0] * 7, action=[0.0] * 6,
                metrics={"a": 1.5, "c": "not-a-number"}),
        features, metric_names=["a", "b", "c"],
    )
    assert f["annotation.interlatent.metrics.a"].tolist() == [pytest.approx(1.5)]
    assert f["annotation.interlatent.metrics.b"].tolist() == [0.0]
    assert f["annotation.interlatent.metrics.c"].tolist() == [0.0]


def test_build_frame_substitutes_a_black_frame_for_a_missing_camera() -> None:
    """A dropped frame on one camera must not desync the columns."""
    features = _discover(cameras=["wrist", "top"], image_shape=(4, 6))
    img = np.full((4, 6, 3), 200, dtype=np.uint8)
    f = _frame(
        StepRow(episode_id="e", step=0, observation=[0.0] * 7, action=[0.0] * 6),
        features, cameras=["wrist", "top"], step_images={"wrist": img},
    )
    assert f["observation.images.wrist"].shape == (4, 6, 3)
    assert int(f["observation.images.wrist"][0, 0, 0]) == 200
    assert f["observation.images.top"].shape == (4, 6, 3)
    assert not f["observation.images.top"].any()


def test_build_frame_expands_grayscale_and_fits_odd_sizes() -> None:
    features = _discover(cameras=["wrist"], image_shape=(4, 4))

    gray = _frame(
        StepRow(episode_id="e", step=0, observation=[0.0] * 7, action=[0.0] * 6),
        features, cameras=["wrist"],
        step_images={"wrist": np.full((4, 4), 128, dtype=np.uint8)},
    )["observation.images.wrist"]
    assert gray.shape == (4, 4, 3)
    assert (gray == 128).all()

    # Oversized frames are cropped, undersized ones zero-padded — either way
    # the array matches the declared shape.
    big = _frame(
        StepRow(episode_id="e", step=0, observation=[0.0] * 7, action=[0.0] * 6),
        features, cameras=["wrist"],
        step_images={"wrist": np.full((9, 9, 3), 7, dtype=np.uint8)},
    )["observation.images.wrist"]
    assert big.shape == (4, 4, 3)

    small = _frame(
        StepRow(episode_id="e", step=0, observation=[0.0] * 7, action=[0.0] * 6),
        features, cameras=["wrist"],
        step_images={"wrist": np.full((2, 2, 3), 9, dtype=np.uint8)},
    )["observation.images.wrist"]
    assert small.shape == (4, 4, 3)
    assert int(small[0, 0, 0]) == 9
    assert int(small[3, 3, 0]) == 0


# ----------------------------------------------------------------------
# Constructor + cleanup
# ----------------------------------------------------------------------


def test_constructor_synthesizes_a_repo_id_and_fills_blank_fields(
    tmp_path: Path,
) -> None:
    r = LeRobotRebuilder(tmp_path / "ds", fps=30, task="", env_slug="pick-cube")
    assert r.repo_id == "interlatent/pick-cube"
    assert r.task == "pick-cube"  # falls back to the env slug, never empty

    blank = LeRobotRebuilder(tmp_path / "ds2", fps=30, task="", env_slug="")
    assert blank.task == "rollout"
    assert blank.env_slug == "unknown"
    assert blank.repo_id == "interlatent/session"

    explicit = LeRobotRebuilder(tmp_path / "ds3", fps=10, task="t",
                                env_slug="e", repo_id="me/mine")
    assert explicit.repo_id == "me/mine"


def test_cleanup_removes_the_tree_and_an_emptied_parent(tmp_path: Path) -> None:
    root = tmp_path / "parent" / "v3"
    (root / "data").mkdir(parents=True)
    (root / "data" / "x.parquet").write_bytes(b"PAR1")
    r = LeRobotRebuilder(root, fps=30, task="t", env_slug="e")
    r.cleanup()
    assert not root.exists()
    assert not root.parent.exists()


def test_cleanup_keeps_a_parent_that_still_holds_other_data(tmp_path: Path) -> None:
    root = tmp_path / "shared" / "v3"
    root.mkdir(parents=True)
    (root.parent / "sibling.txt").write_text("keep me")
    LeRobotRebuilder(root, fps=30, task="t", env_slug="e").cleanup()
    assert not root.exists()
    assert (root.parent / "sibling.txt").exists()


def test_cleanup_is_safe_when_nothing_was_ever_built(tmp_path: Path) -> None:
    LeRobotRebuilder(tmp_path / "never" / "v3", fps=30, task="t",
                     env_slug="e").cleanup()


# ----------------------------------------------------------------------
# Parquet post-edits
# ----------------------------------------------------------------------


def _write_info(root: Path, features: dict | None = None) -> Path:
    (root / "meta").mkdir(parents=True, exist_ok=True)
    p = root / "meta" / "info.json"
    p.write_text(json.dumps({"codebase_version": "v3.0",
                             "features": features or {}}))
    return p


def test_stamp_info_json_adds_the_interlatent_block(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    _write_info(root)
    r = LeRobotRebuilder(root, fps=30, task="pick", env_slug="pick-cube")
    r._stamp_info_json(root, ["grasp_score"])
    info = json.loads((root / "meta" / "info.json").read_text())
    assert info["interlatent"] == {
        "environment_slug": "pick-cube",
        "task": "pick",
        "metric_names": ["grasp_score"],
    }
    assert info["codebase_version"] == "v3.0"  # untouched


def test_stamp_info_json_is_a_noop_without_an_info_file(tmp_path: Path) -> None:
    r = LeRobotRebuilder(tmp_path / "ds", fps=30, task="t", env_slug="e")
    r._stamp_info_json(tmp_path / "ds", [])  # must not raise


def test_inject_episode_uuids_joins_index_to_uuid(tmp_path: Path) -> None:
    """LeRobot keys episodes by int index; the coordinator keys by UUID. The
    merge worker needs both to keep the join."""
    root = tmp_path / "ds"
    ep_dir = root / "meta" / "episodes" / "chunk-000"
    ep_dir.mkdir(parents=True)
    pq.write_table(
        pa.table({"episode_index": [0, 1, 2], "length": [10, 20, 30]}),
        ep_dir / "file-000.parquet",
    )
    r = LeRobotRebuilder(root, fps=30, task="t", env_slug="e")
    r._inject_episode_uuids(root, ["uuid-a", "uuid-b", "uuid-c"])

    t = pq.read_table(ep_dir / "file-000.parquet")
    assert t.column("interlatent.episode_uuid").to_pylist() == [
        "uuid-a", "uuid-b", "uuid-c",
    ]
    assert t.column("length").to_pylist() == [10, 20, 30]


def test_inject_episode_uuids_blanks_an_out_of_range_index(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    ep_dir = root / "meta" / "episodes"
    ep_dir.mkdir(parents=True)
    pq.write_table(pa.table({"episode_index": [0, 5]}), ep_dir / "e.parquet")
    LeRobotRebuilder(root, fps=30, task="t", env_slug="e")._inject_episode_uuids(
        root, ["only-one"]
    )
    assert pq.read_table(ep_dir / "e.parquet").column(
        "interlatent.episode_uuid"
    ).to_pylist() == ["only-one", ""]


def test_inject_episode_uuids_is_a_noop_without_meta_episodes(tmp_path: Path) -> None:
    LeRobotRebuilder(tmp_path / "ds", fps=30, task="t",
                     env_slug="e")._inject_episode_uuids(tmp_path / "ds", ["a"])


def test_control_source_column_converts_to_strings(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    data = root / "data" / "chunk-000"
    data.mkdir(parents=True)
    col = "annotation.interlatent.control_source"
    pq.write_table(
        pa.table({col: [0, 1, 2, 3], "frame_index": [0, 1, 2, 3]}),
        data / "file-000.parquet",
    )
    _write_info(root, {col: {"dtype": "int64", "shape": [1], "names": None}})

    LeRobotRebuilder(root, fps=30, task="t", env_slug="e")._convert_int_column_to_string(
        root, col_name=col, id_to_name=CONTROL_SOURCE_ID_TO_NAME, nullable=False,
    )

    t = pq.read_table(data / "file-000.parquet")
    assert t.column(col).to_pylist() == ["policy", "teleop", "hold", "intervention"]
    assert t.schema.field(col).type == pa.string()
    assert t.column("frame_index").to_pylist() == [0, 1, 2, 3]
    # Loaders read dtypes off info.json, so it has to move with the column.
    assert json.loads((root / "meta" / "info.json").read_text())[
        "features"][col]["dtype"] == "string"


def test_failure_type_column_maps_the_zero_sentinel_to_null(tmp_path: Path) -> None:
    """0 means 'no failure' for failure_type — it must become NULL, not the
    string name of catalog entry 0."""
    root = tmp_path / "ds"
    data = root / "data"
    data.mkdir(parents=True)
    col = "annotation.interlatent.failure_type"
    pq.write_table(pa.table({col: [0, 1, 2, 0]}), data / "f.parquet")

    LeRobotRebuilder(root, fps=30, task="t", env_slug="e")\
        ._convert_failure_type_to_string(root, {1: "dropped", 2: "collision"})

    assert pq.read_table(data / "f.parquet").column(col).to_pylist() == [
        None, "dropped", "collision", None,
    ]


def test_conversion_handles_list_encoded_and_null_values(tmp_path: Path) -> None:
    """lerobot may write shape-(1,) columns as length-1 lists depending on
    version; both encodings have to convert."""
    root = tmp_path / "ds"
    data = root / "data"
    data.mkdir(parents=True)
    col = "annotation.interlatent.control_source"
    pq.write_table(
        pa.table({col: pa.array([[1], [3], [], None], type=pa.list_(pa.int64()))}),
        data / "f.parquet",
    )
    LeRobotRebuilder(root, fps=30, task="t", env_slug="e")._convert_int_column_to_string(
        root, col_name=col, id_to_name=CONTROL_SOURCE_ID_TO_NAME, nullable=False,
    )
    assert pq.read_table(data / "f.parquet").column(col).to_pylist() == [
        "teleop", "intervention", "policy", "policy",
    ]


def test_conversion_skips_parquet_without_the_column(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    data = root / "data"
    data.mkdir(parents=True)
    pq.write_table(pa.table({"frame_index": [0, 1]}), data / "f.parquet")
    LeRobotRebuilder(root, fps=30, task="t", env_slug="e")._convert_int_column_to_string(
        root, col_name="annotation.interlatent.control_source",
        id_to_name=CONTROL_SOURCE_ID_TO_NAME, nullable=False,
    )
    assert pq.read_table(data / "f.parquet").column_names == ["frame_index"]


# ----------------------------------------------------------------------
# _scalarize_singleton_columns (numpy 2.x compat)
# ----------------------------------------------------------------------


def test_scalarize_unwraps_only_shape_one_columns() -> None:
    """HF's encoder calls float() on shape-(1,) buffers and numpy 2.x raises
    'only 0-dimensional arrays can be converted'. Unwrap before save."""
    features = _discover()  # observation.state (7,), next.reward (1,)

    class _DS:
        episode_buffer = {
            "next.reward": [np.array([1.5], dtype=np.float32)],
            "next.done": [np.array([True])],
            "observation.state": [np.zeros(7, dtype=np.float32)],
        }

    ds = _DS()
    LeRobotRebuilder._scalarize_singleton_columns(ds, features)
    assert ds.episode_buffer["next.reward"] == [pytest.approx(1.5)]
    assert ds.episode_buffer["next.done"] == [True]
    # Multi-element vectors are left alone.
    assert isinstance(ds.episode_buffer["observation.state"][0], np.ndarray)


def test_scalarize_tolerates_a_dataset_without_a_buffer() -> None:
    class _DS:
        episode_buffer = None

    LeRobotRebuilder._scalarize_singleton_columns(_DS(), _discover())


# ----------------------------------------------------------------------
# build_from_source, against a stubbed LeRobotDataset
# ----------------------------------------------------------------------


class _StubSource:
    def __init__(self, rows_by_ep, cameras=(), frames=None) -> None:
        self._rows = rows_by_ep
        self._cameras = list(cameras)
        self._frames = frames or {}

    def episode_ids(self):
        return list(self._rows)

    def iter_steps(self, eid):
        return list(self._rows.get(eid, []))

    def cameras_for_episode(self, eid):
        return list(self._cameras)

    def iter_frames(self, eid):
        return list(self._frames.get(eid, []))


class _StubDataset:
    """Minimal stand-in for LeRobotDataset that writes a v3-shaped tree.

    Only what the rebuilder touches: the create kwargs, the frame stream,
    and enough parquet on disk for the post-edit passes to bite.
    """

    created: "_StubDataset | None" = None

    def __init__(self, root: Path, features: dict, kwargs: dict) -> None:
        self.root = Path(root)
        self.features = features
        self.create_kwargs = kwargs
        self.frames: list[dict] = []
        self.episodes: list[list[dict]] = []
        self.episode_buffer: dict[str, list] = {}
        self.finalized = 0

    # The signature mirrors the generation docker/Dockerfile pins (post
    # bd9619df): a `camera_encoder` config, NOT the `vcodec` string the
    # rebuilder used to pass unconditionally. Keeping this honest is the
    # whole point — a stub taking **kwargs accepts the broken call and
    # certifies it as correct.
    @classmethod
    def create(cls, *, repo_id, fps, features, root, robot_type,
               use_videos, camera_encoder=None, **extra):
        self = cls(Path(root), features, dict(
            repo_id=repo_id, fps=fps, robot_type=robot_type,
            use_videos=use_videos, camera_encoder=camera_encoder, **extra,
        ))
        (self.root / "meta").mkdir(parents=True, exist_ok=False)
        (self.root / "meta" / "info.json").write_text(json.dumps({
            "fps": fps,
            "features": {k: dict(v, shape=list(v["shape"]))
                         for k, v in features.items()},
        }))
        _StubDataset.created = self
        return self

    def add_frame(self, frame: dict) -> None:
        self.frames.append(frame)
        for k, v in frame.items():
            self.episode_buffer.setdefault(k, []).append(v)

    def save_episode(self) -> None:
        self.episodes.append(list(self.episode_buffer.get("action", [])))
        idx = len(self.episodes) - 1
        n = len(self.episode_buffer.get("action", []))

        data_dir = self.root / "data" / "chunk-000"
        data_dir.mkdir(parents=True, exist_ok=True)
        cols: dict = {"episode_index": [idx] * n, "frame_index": list(range(n))}
        for col in ("annotation.interlatent.control_source",
                    "annotation.interlatent.failure_type"):
            if col in self.features:
                cols[col] = [int(v) for v in self.episode_buffer.get(col, [])]
        pq.write_table(pa.table(cols), data_dir / f"file-{idx:03d}.parquet")

        meta_dir = self.root / "meta" / "episodes" / "chunk-000"
        meta_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table({"episode_index": [idx], "length": [n]}),
            meta_dir / f"file-{idx:03d}.parquet",
        )
        self.episode_buffer = {}

    def finalize(self) -> None:
        self.finalized += 1


def _install_lerobot(monkeypatch) -> None:
    import types
    from dataclasses import dataclass

    _StubDataset.created = None
    pkg = types.ModuleType("lerobot")
    datasets = types.ModuleType("lerobot.datasets")
    mod = types.ModuleType("lerobot.datasets.lerobot_dataset")
    mod.LeRobotDataset = _StubDataset
    pkg.datasets = datasets
    datasets.lerobot_dataset = mod
    # The codec shim builds one of these to name the encoder (see
    # storage/lerobot_codec.py).
    configs = types.ModuleType("lerobot.configs")

    @dataclass
    class VideoEncoderConfig:
        vcodec: str

    configs.VideoEncoderConfig = VideoEncoderConfig
    configs.RGBEncoderConfig = VideoEncoderConfig
    pkg.configs = configs
    monkeypatch.setitem(sys.modules, "lerobot", pkg)
    monkeypatch.setitem(sys.modules, "lerobot.datasets", datasets)
    monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", mod)
    monkeypatch.setitem(sys.modules, "lerobot.configs", configs)


def _jpeg(path: Path, size=(6, 4), color=(10, 20, 30)) -> Path:
    Image = pytest.importorskip("PIL.Image")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


def test_build_from_source_writes_a_dataset_and_all_three_post_edits(
    monkeypatch, tmp_path: Path
) -> None:
    _install_lerobot(monkeypatch)
    frames_dir = tmp_path / "frames"
    rows = {
        "uuid-a": [
            StepRow(episode_id="uuid-a", step=0, observation=[1.0, 2.0],
                    action=[3.0], control_source="policy",
                    metrics={"grasp": 0.5}),
            StepRow(episode_id="uuid-a", step=1, observation=[1.0, 2.0],
                    action=[3.0], control_source="intervention",
                    failure_type="dropped", reward=1.0, done=True),
        ],
        "uuid-b": [
            StepRow(episode_id="uuid-b", step=0, observation=[9.0, 9.0],
                    action=[9.0], control_source="hold"),
        ],
    }
    frames = {
        "uuid-a": [
            (0, "wrist", _jpeg(frames_dir / "wrist" / "0.jpg")),
            (1, "wrist", _jpeg(frames_dir / "wrist" / "1.jpg")),
        ],
        "uuid-b": [(0, "wrist", _jpeg(frames_dir / "wrist" / "b0.jpg"))],
    }
    src = _StubSource(rows, cameras=["wrist"], frames=frames)

    root = tmp_path / "ds" / "v3"
    r = LeRobotRebuilder(root, fps=10, task="pick", env_slug="pick-cube",
                         vcodec="h264")
    out_root, uuids = r.build_from_source(src)

    assert out_root == root
    assert uuids == ["uuid-a", "uuid-b"]

    ds = _StubDataset.created
    assert ds.create_kwargs["repo_id"] == "interlatent/pick-cube"
    assert ds.create_kwargs["fps"] == 10
    assert ds.create_kwargs["robot_type"] == "pick-cube"
    # Cameras + a decodable frame => video features are on.
    assert ds.create_kwargs["use_videos"] is True
    # gVisor stalls on libsvtav1, so the recorder's h264 must reach lerobot —
    # through whichever parameter this lerobot generation exposes.
    assert ds.create_kwargs["camera_encoder"].vcodec == "h264"
    assert ds.finalized == 1

    # Image shape discovered from the first decodable frame (PIL: W x H).
    assert ds.features["observation.images.wrist"]["shape"] == (4, 6, 3)
    assert ds.features["observation.state"]["shape"] == (2,)
    assert "annotation.interlatent.metrics.grasp" in ds.features

    # Every frame carried the task string (add_frame pops it internally).
    assert {f["task"] for f in ds.frames} == {"pick"}
    assert len(ds.frames) == 3

    # Post-edit 1: the UUID join column.
    ep_uuids: list[str] = []
    for p in sorted((root / "meta" / "episodes").rglob("*.parquet")):
        ep_uuids += pq.read_table(p).column("interlatent.episode_uuid").to_pylist()
    assert ep_uuids == ["uuid-a", "uuid-b"]

    # Post-edit 2: the interlatent info.json block.
    info = json.loads((root / "meta" / "info.json").read_text())
    assert info["interlatent"]["environment_slug"] == "pick-cube"
    assert info["interlatent"]["metric_names"] == ["grasp"]

    # Post-edit 3: int64 staging -> strings, with the labels intact.
    labels: list = []
    failures: list = []
    for p in sorted((root / "data").rglob("*.parquet")):
        t = pq.read_table(p)
        labels += t.column("annotation.interlatent.control_source").to_pylist()
        failures += t.column("annotation.interlatent.failure_type").to_pylist()
    assert labels == ["policy", "intervention", "hold"]
    assert failures == [None, "dropped", None]


def test_build_from_source_without_cameras_disables_video(
    monkeypatch, tmp_path: Path
) -> None:
    _install_lerobot(monkeypatch)
    src = _StubSource({"e": [StepRow(episode_id="e", step=0,
                                     observation=[0.0], action=[0.0])]})
    root = tmp_path / "ds" / "v3"
    LeRobotRebuilder(root, fps=30, task="t", env_slug="e").build_from_source(src)
    ds = _StubDataset.created
    assert ds.create_kwargs["use_videos"] is False
    # vcodec=None means "no preference": no encoder kwarg is forwarded at all.
    assert ds.create_kwargs["camera_encoder"] is None
    assert not [k for k in ds.features if k.startswith("observation.images.")]
    # No labels anywhere => no optional columns at all.
    assert "annotation.interlatent.control_source" not in ds.features
    assert "annotation.interlatent.failure_type" not in ds.features


def test_build_from_source_sorts_rows_and_drops_empty_episodes(
    monkeypatch, tmp_path: Path
) -> None:
    """An out-of-order source must still write ascending steps, and an
    episode with no rows must not reach the writer (it rejects zero frames)."""
    _install_lerobot(monkeypatch)
    src = _StubSource({
        "empty": [],
        "e": [
            StepRow(episode_id="e", step=2, observation=[2.0], action=[2.0]),
            StepRow(episode_id="e", step=0, observation=[0.0], action=[0.0]),
            StepRow(episode_id="e", step=1, observation=[1.0], action=[1.0]),
        ],
    })
    root = tmp_path / "ds" / "v3"
    _out, uuids = LeRobotRebuilder(
        root, fps=30, task="t", env_slug="e"
    ).build_from_source(src)
    assert uuids == ["e"]
    ds = _StubDataset.created
    assert [float(f["action"][0]) for f in ds.frames] == [0.0, 1.0, 2.0]


def test_build_from_source_returns_empty_for_a_source_with_no_rows(
    monkeypatch, tmp_path: Path
) -> None:
    _install_lerobot(monkeypatch)
    root = tmp_path / "ds" / "v3"
    r = LeRobotRebuilder(root, fps=30, task="t", env_slug="e")

    assert r.build_from_source(_StubSource({})) == (root, [])
    assert r.build_from_source(_StubSource({"e": []})) == (root, [])
    # Nothing was created on disk, so a later real build can still run.
    assert _StubDataset.created is None
    assert not root.exists()


def test_build_from_source_finalizes_even_when_a_frame_write_explodes(
    monkeypatch, tmp_path: Path
) -> None:
    """Skipping finalize() leaves truncated parquet footers behind — an
    unreadable dataset instead of a short one."""
    _install_lerobot(monkeypatch)

    def boom(self, frame):
        raise RuntimeError("writer died")

    monkeypatch.setattr(_StubDataset, "add_frame", boom)
    src = _StubSource({"e": [StepRow(episode_id="e", step=0,
                                     observation=[0.0], action=[0.0])]})
    root = tmp_path / "ds" / "v3"
    r = LeRobotRebuilder(root, fps=30, task="t", env_slug="e")
    with pytest.raises(RuntimeError, match="writer died"):
        r.build_from_source(src)
    assert _StubDataset.created.finalized == 1


def test_build_from_source_reports_a_missing_lerobot_as_an_install_hint(
    monkeypatch, tmp_path: Path
) -> None:
    # None out the submodules too, not just the package: on a machine that
    # HAS lerobot, `interlatent_server.server` already imported
    # lerobot.datasets.lerobot_dataset (lerobot_backend applies an RTC
    # patch at import time), so blanking only the top-level name leaves the
    # cached submodule importable and this test would pass vacuously.
    for name in ("lerobot", "lerobot.datasets", "lerobot.datasets.lerobot_dataset"):
        monkeypatch.setitem(sys.modules, name, None)
    src = _StubSource({"e": [StepRow(episode_id="e", step=0,
                                     observation=[0.0], action=[0.0])]})
    with pytest.raises(RuntimeError, match="pip install"):
        LeRobotRebuilder(tmp_path / "ds", fps=30, task="t",
                         env_slug="e").build_from_source(src)
