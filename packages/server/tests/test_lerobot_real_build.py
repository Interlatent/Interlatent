"""Build a real LeRobot v3 dataset with the real lerobot, and read it back.

The rest of the rebuilder suite stubs ``LeRobotDataset`` — lerobot drags
torch, so CI's base job cannot have it. That stub is why a real bug
shipped: the writers passed ``vcodec=`` to ``create()``, a stub taking
``**kwargs`` accepted it happily, and every lerobot from 2026-05-14
onward — including the ref ``docker/Dockerfile`` pins — raised
``TypeError``. The writers catch broadly, so it surfaced as "rebuild
failed; aborting upload" and the staged episode was deleted.

No stub can catch that. These tests call the real thing.

Requires ``lerobot[dataset]`` and an ffmpeg with an H.264 encoder. Opt-in
via ``pytest --real-lerobot``, in its own process — see ``conftest.py``.
CI runs it in the ``server-lerobot`` job.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

lerobot = pytest.importorskip("lerobot", reason="needs the [lerobot] extra")
pytest.importorskip("lerobot.datasets.lerobot_dataset",
                    reason="needs lerobot[dataset] (datasets + av)")
av = pytest.importorskip("av", reason="needs lerobot[dataset]")
np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")
pq = pytest.importorskip("pyarrow.parquet")

from interlatent_server.storage.lerobot_rebuild import (  # noqa: E402
    LeRobotRebuilder,
    StepRow,
)

N_STEPS = 8
TASK = "pick up the cube"
ENV = "pick-cube"


class _Source:
    """Minimal StepSource with one camera and a mix of annotations."""

    def __init__(self, tmp: Path) -> None:
        self.dir = tmp / "frames" / "wrist"
        self.dir.mkdir(parents=True)
        for i in range(N_STEPS):
            Image.new("RGB", (64, 48), (i * 20 % 256, 40, 90)).save(
                self.dir / f"{i:08d}.jpg"
            )

    def episode_ids(self):
        return ["ep-uuid-aaa"]

    def iter_steps(self, eid):
        for i in range(N_STEPS):
            yield StepRow(
                episode_id=eid, step=i,
                observation=[0.1 * i, 0.2, 0.3],
                action=[0.5 * i, 1.0],
                reward=i / N_STEPS,
                done=(i == N_STEPS - 1),
                control_source=("intervention" if i % 4 == 0 else "policy"),
                failure_type=("dropped" if i == 5 else None),
                metrics={"grasp": 0.5 * i},
            )

    def cameras_for_episode(self, eid):
        return ["wrist"]

    def iter_frames(self, eid):
        for i in range(N_STEPS):
            yield i, "wrist", self.dir / f"{i:08d}.jpg"


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """One real build, shared by the assertions below (encoding is slow)."""
    tmp = tmp_path_factory.mktemp("real-build")
    root = tmp / "ds" / "v3"
    rebuilder = LeRobotRebuilder(
        root, fps=10, task=TASK, env_slug=ENV,
        # Exactly what SessionRecorder.upload() passes.
        vcodec="h264",
    )
    out_root, uuids = rebuilder.build_from_source(_Source(tmp))
    return out_root, uuids


def test_the_recorders_own_call_does_not_raise(built) -> None:
    """The regression: `vcodec="h264"` used to be a hard TypeError on any
    lerobot from 2026-05-14 on, and the recorder swallowed it into a
    discarded episode."""
    root, uuids = built
    assert uuids == ["ep-uuid-aaa"]
    assert (root / "meta" / "info.json").is_file()
    assert list((root / "data").rglob("*.parquet"))


def test_the_video_is_actually_h264(built) -> None:
    """Not just "the call worked" — the codec has to have taken effect.

    Silently falling back to lerobot's libsvtav1 default is the ADR 0021
    violation, and under gVisor SVT-AV1 stalls the encoder outright. A
    shim that resolved the wrong parameter would pass the test above and
    fail this one.
    """
    root, _uuids = built
    videos = list(root.rglob("*.mp4"))
    assert videos, "no video was written"
    for path in videos:
        with av.open(str(path)) as container:
            codecs = [s.codec_context.name for s in container.streams.video]
        assert codecs, f"{path.name} has no video stream"
        for name in codecs:
            assert "264" in name, f"{path.name} encoded as {name}, expected H.264"


def _open_stock(root: Path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(repo_id=f"interlatent/{ENV}", root=str(root))


def test_stock_lerobot_can_load_it_back(built) -> None:
    """The canonical's contract is the stock layout, loadable by stock
    lerobot — so read it with lerobot's own reader, not ours."""
    root, _uuids = built
    ds = _open_stock(root)
    assert ds.num_frames == N_STEPS
    assert ds.num_episodes == 1
    assert ds.fps == 10
    assert "observation.images.wrist" in ds.features
    assert "annotation.interlatent.control_source" in ds.features


def test_the_video_decodes_back_to_pixels(built) -> None:
    """Indexing the dataset pulls a frame through lerobot's video decoder.

    Skipped when the installed decoder backend can't load — torchcodec
    ships against a specific ffmpeg ABI and a host mismatch is an
    environment problem, not a defect in what we wrote. The layout
    assertions above still run in that case.
    """
    root, _uuids = built
    ds = _open_stock(root)
    try:
        item = ds[0]
    except Exception as exc:  # noqa: BLE001 — backend-specific error types
        if "torchcodec" in str(exc) or "libav" in str(exc):
            pytest.skip(f"no working video decoder on this host: {exc}"[:200])
        raise
    assert item["observation.state"].shape[-1] == 3
    assert item["action"].shape[-1] == 2
    assert tuple(item["observation.images.wrist"].shape) == (3, 48, 64)


def test_the_interlatent_annotations_survive_a_real_build(built) -> None:
    """The int64-staging-then-string-postedit dance has to hold against a
    real parquet writer, not just our own fixtures."""
    root, _uuids = built

    labels, failures = [], []
    for path in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(path)
        labels += table.column("annotation.interlatent.control_source").to_pylist()
        failures += table.column("annotation.interlatent.failure_type").to_pylist()

    assert labels == [
        "intervention", "policy", "policy", "policy",
        "intervention", "policy", "policy", "policy",
    ]
    assert failures[5] == "dropped"
    assert failures[0] is None

    uuids = []
    for path in sorted((root / "meta" / "episodes").rglob("*.parquet")):
        uuids += pq.read_table(path).column("interlatent.episode_uuid").to_pylist()
    assert uuids == ["ep-uuid-aaa"]

    info = json.loads((root / "meta" / "info.json").read_text())
    assert info["interlatent"]["environment_slug"] == ENV
    assert info["interlatent"]["metric_names"] == ["grasp"]
    # lerobot ignores unknown info blocks rather than rejecting the dataset.
    assert info["codebase_version"].startswith("v3")


def test_no_codec_preference_still_builds(tmp_path: Path) -> None:
    """The other lane: vcodec=None must leave lerobot's default alone."""
    root = tmp_path / "ds" / "v3"
    out, uuids = LeRobotRebuilder(
        root, fps=10, task=TASK, env_slug=ENV, vcodec=None,
    ).build_from_source(_Source(tmp_path))
    assert uuids == ["ep-uuid-aaa"]
    assert list(out.rglob("*.mp4"))
