"""One real rollout through the whole DRTC stack, in one process.

    connect_drtc()  ->  grpc.aio  ->  InferenceServicer (echo backend)
                                       -> SessionRecorder (real staging)
                                       -> LeRobotRebuilder (real lerobot)
                                       -> LocalDirSink (real directory)

Nothing here is stubbed at all: real protobuf over a real localhost
socket, the real client scheduler and tick spool, the real recorder, the
real dataset build, and a real destination on disk. No GPU, no robot, no
control plane — and, since ADR 0039, no credential either: the rollout
below passes no API key, because recording no longer takes one.

This exists because the unit suites' seams hid a data-loss bug. The
recorder passed a codec argument that current lerobot rejects; the stub
accepted it, so the suites were green while a real close-session
discarded the episode — rebuild raises, ``upload()`` catches broadly,
staging is deleted, and the robot side sees a clean ``close()``. The
assertion that catches that class of bug is the one below: after a
rollout, a complete dataset must exist at the destination.

Skipped unless BOTH dists and lerobot are importable; CI runs it in the
``server-lerobot`` job.
"""
from __future__ import annotations

import io
import sys
import threading
import time
import uuid
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
# The SDK half of the loop. Present in a dev checkout; the CI job installs it.
SDK_SRC = Path(__file__).resolve().parents[3] / "sdk" / "src"
if SDK_SRC.is_dir() and str(SDK_SRC) not in sys.path:
    sys.path.insert(0, str(SDK_SRC))

grpc = pytest.importorskip("grpc", reason="needs grpcio")
np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")
pytest.importorskip("lerobot.datasets.lerobot_dataset",
                    reason="needs lerobot[dataset]")
pytest.importorskip("interlatent.inference.integration.connect",
                    reason="needs the interlatent SDK installed")

STEPS = 24
FPS = 30.0
CAM = "wrist"          # bare camera name: what a robot adapter reports
STATE_DIM = 7
ACTION_DIM = 6


# ------------------------------------------------------------------ grpc server
def _start_grpc(recorder_dir: Path):
    import asyncio

    from interlatent_server.protocol import messages_pb2_grpc as pb_grpc
    from interlatent_server.server.transport import InferenceServicer

    box: dict = {}
    ready = threading.Event()

    async def _run():
        server = grpc.aio.server()
        pb_grpc.add_InferenceServiceServicer_to_server(
            InferenceServicer(recorder_base_dir=recorder_dir), server,
        )
        box["port"] = server.add_insecure_port("127.0.0.1:0")
        box["server"] = server
        await server.start()
        ready.set()
        await server.wait_for_termination()

    loop = asyncio.new_event_loop()

    def _thread():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_run())

    threading.Thread(target=_thread, daemon=True).start()
    assert ready.wait(60), "gRPC server did not start"
    box["loop"] = loop
    return box


def _jpeg(i: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), ((i * 9) % 256, 60, 120)).save(buf, format="JPEG")
    return buf.getvalue()


def _npz(i: int, jpg: bytes) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **{
        "observation.state": np.arange(STATE_DIM, dtype=np.float32) + i * 0.01,
        # The inference payload uses the fully-qualified policy-schema key.
        f"observation.images.{CAM}": np.frombuffer(jpg, dtype=np.uint8),
    })
    return buf.getvalue()


# ------------------------------------------------------------------------ test
@pytest.fixture(scope="module")
def rollout(tmp_path_factory):
    """Run one recorded rollout and return (dataset_dir, episode_id, stats)."""
    from interlatent.inference.integration.connect import connect_drtc

    tmp = tmp_path_factory.mktemp("drtc-e2e")
    dataset = tmp / "dataset"
    grpc_box = _start_grpc(tmp / "recordings")
    episode_id = f"e2e-{uuid.uuid4().hex[:8]}"

    client = connect_drtc(
        # No api_key: recording is account-free (ADR 0039), and a rollout
        # that quietly depended on one would hide that regressing.
        environment="pick-cube",
        policy_uri="",
        policy_backend="echo",
        server_address=f"127.0.0.1:{grpc_box['port']}",
        chunk_size=32,
        action_dim=ACTION_DIM,
        fps=FPS,
        task="pick up the cube",
        payload_codec="npz",
        record=True,
        episode_id=episode_id,
        # The destination, exactly as a coordinator stamps it onto the
        # session's `recording` block (ADR 0002).
        metadata={"output_dir": str(dataset)},
    )

    stats = {"actions": 0, "ticks": 0, "refused": 0, "first_action_step": None,
             "session_id": client.session_id, "action_dim": client.action_dim}
    period = 1.0 / FPS
    next_t = time.monotonic()
    for i in range(STEPS):
        jpg = _jpeg(i)
        action = client.step(_npz(i, jpg), codec="npz")
        if action is not None:
            stats["actions"] += 1
            if stats["first_action_step"] is None:
                stats["first_action_step"] = i
            ok = client.record_tick(
                step=i,
                observation_state=[float(i), 0.0],
                action=[float(x) for x in action],
                # Bare camera name — the server namespaces it. Sending the
                # qualified key here double-prefixes the video column.
                jpegs={CAM: jpg},
                control_timestamp_ns=time.monotonic_ns(),
                control_source=("intervention" if i % 8 == 0 else "policy"),
            )
            stats["ticks" if ok else "refused"] += 1
        next_t += period
        time.sleep(max(0.0, next_t - time.monotonic()))

    client.close()

    # Publish is a fire-and-forget task on the server's loop. Generous, but
    # bounded: a stalled publish should fail the run, not hang it.
    deadline = time.time() + 120
    while time.time() < deadline:
        if (dataset / "meta" / "info.json").exists():
            break
        time.sleep(0.25)

    return dataset, episode_id, stats


def test_the_client_streams_actions_at_the_control_rate(rollout) -> None:
    """`step()` returns None while the first chunk is in flight, then
    streams — the Tier 1 contract, never previously exercised."""
    _dataset, _eid, stats = rollout
    assert stats["session_id"]
    assert stats["action_dim"] == ACTION_DIM
    assert stats["first_action_step"] is not None
    # A couple of leading Nones are expected; a rollout of Nones is not.
    assert stats["first_action_step"] <= 3
    assert stats["actions"] >= STEPS - 4


def test_every_captured_tick_is_journalled(rollout) -> None:
    """A refused tick means the spool hard-stopped (ADR 0023). On a healthy
    loopback link there is no reason for one."""
    _dataset, _eid, stats = rollout
    assert stats["refused"] == 0
    assert stats["ticks"] == stats["actions"]


def _files(dataset: Path) -> list[str]:
    return sorted(
        p.relative_to(dataset).as_posix()
        for p in dataset.rglob("*") if p.is_file()
    )


def test_the_episode_reaches_the_destination(rollout) -> None:
    """The regression that motivated this file: a rebuild failure is caught
    and turned into a discarded episode, so the ONLY way to know the
    recording survived is that a dataset exists at the destination
    afterwards.

    (The name is pinned by the CI check in ``.github/workflows/ci.yml`` that
    greps for it — rename both together.)
    """
    dataset, _episode_id, _stats = rollout
    assert (dataset / "meta" / "info.json").is_file(), _files(dataset)


def test_the_uploaded_dataset_is_a_complete_lerobot_tree(rollout) -> None:
    dataset, _eid, _stats = rollout
    names = _files(dataset)
    assert any(n.endswith("meta/info.json") for n in names), names
    assert any(n.startswith("data/") and n.endswith(".parquet") for n in names), names
    assert any(n.endswith(".mp4") for n in names), names
    assert all((dataset / n).stat().st_size > 0 for n in names), names


def test_the_camera_key_is_namespaced_exactly_once(rollout) -> None:
    """The node sends a bare camera name and the server adds the
    ``observation.images.`` prefix. Double-prefixing still records happily
    but yields a video column no policy's schema matches."""
    dataset, _eid, _stats = rollout
    videos = [n for n in _files(dataset) if n.endswith(".mp4")]
    assert videos
    for path in videos:
        assert f"videos/observation.images.{CAM}/" in path, path
        assert "observation.images.observation.images." not in path, path


def test_human_driven_steps_survive_into_the_dataset(rollout) -> None:
    """ADR 0034 (platform repo) over the real wire: intervention ticks must
    land as the per-step ``control_source`` label training reads. The column
    is present for every session (the local sink normalizes for merge), so a
    pure-policy rollout would still have it — with no intervention rows."""
    pq = pytest.importorskip("pyarrow.parquet")

    col = "annotation.interlatent.control_source"
    dataset, _eid, _stats = rollout
    parquets = list(dataset.rglob("*.parquet"))
    assert parquets
    sources: list[str] = []
    for p in parquets:
        table = pq.read_table(p)
        if col in table.column_names:
            sources += [str(v) for v in table.column(col).to_pylist()]
    assert sources, f"{col} missing from {[p.name for p in parquets]}"
    assert "intervention" in sources, sorted(set(sources))
