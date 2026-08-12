"""Tests for the server-side DRTC episode recorder (``server/recorder.py``).

The recorder is the whole of streaming-first collection (ADR 0022) on the
server side: every tick the robot node sends lands here, and what it stages
to local SSD is the only thing that survives into the canonical dataset. It
shipped with no tests at all — the closest thing to coverage was the import
walk in ``test_import_surface.py``.

Everything below runs with no GPU, no lerobot, and no network: the drain
loop is a real asyncio task over a real temp dir, the LeRobot build is a
stub (``LeRobotRebuilder`` is a seam the recorder holds by name), and the
backend HTTP calls go through a fake ``httpx.AsyncClient`` that records
what was sent.

What is deliberately asserted, because getting it wrong is silent data loss:

- a refused tick returns **False** — the node keeps it spooled and retries
  (ADR 0023). A recorder that returned True on a full queue would drop
  ticks the node then deletes.
- a replayed tick is deduped and acked **True** (ADR 0024, platform repo), because a
  reconnect re-sends ticks whose ack was lost.
- ``control_source`` in {teleop, intervention} drives ``has_teleop`` on
  upload-complete — the Episode badge the dashboard shows and the DAgger
  label (ADR 0034) training weights on.
- the encode rate comes from the **measured** control timestamps, not the
  requested ``config.fps``; using config.fps makes playback run fast.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from interlatent_server.server import recorder as rec_mod  # noqa: E402
from interlatent_server.server.recorder import (  # noqa: E402
    RecorderConfig,
    RecorderStepSource,
    SessionRecorder,
)


# ----------------------------------------------------------------------
# Fixtures / helpers
# ----------------------------------------------------------------------


def _config(**over) -> RecorderConfig:
    base = dict(
        episode_id="ep-1",
        env_slug="pick-cube",
        model_id=None,
        task="pick up the cube",
        fps=30,
        policy_uri="lerobot/smolvla_base",
        layer="inference:lerobot/smolvla_base",
        api_key="ilat_test",
        api_base="https://interlatent.test",
    )
    base.update(over)
    return RecorderConfig(**base)


def _rec(tmp_path: Path, **cfg_over) -> SessionRecorder:
    return SessionRecorder(tmp_path / "work", _config(**cfg_over))


async def _staged(rec: SessionRecorder, n: int, timeout: float = 5.0) -> None:
    """Wait until ``n`` step rows have actually hit disk.

    The drain task writes in an executor, so enqueue returning is not the
    same as staged. Tests that flip recorder state mid-session need the
    earlier ticks to have landed first.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if len([x for x in rec.steps_path.read_text().splitlines() if x]) >= n:
            return
        await asyncio.sleep(0.005)
    raise AssertionError(f"only {rec.steps_path.read_text()!r} staged, wanted {n}")


def _tick(rec: SessionRecorder, step: int, *, ts: int | None = None, **over) -> bool:
    kwargs = dict(
        step=step,
        observation_state=[0.1 * step, 0.2],
        action=[float(step), 1.0],
        jpegs={"observation.images.wrist": b"\xff\xd8jpeg"},
        control_timestamp=step * 33_333_333 if ts is None else ts,
    )
    kwargs.update(over)
    return rec.enqueue_nowait(**kwargs)


# ----------------------------------------------------------------------
# Module helpers
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "key,expected",
    [
        ("observation.images.wrist", "wrist"),
        ("observation.images.top", "top"),
        ("wrist", "wrist"),
        ("observation.state", "observation.state"),
    ],
)
def test_short_cam_strips_only_the_image_prefix(key: str, expected: str) -> None:
    assert rec_mod._short_cam(key) == expected


@pytest.mark.parametrize(
    "given,expected",
    [
        # Bare origin — what serve_gpu and the whole node SDK pass.
        ("https://interlatent.com", "https://interlatent.com/api/v1"),
        ("https://interlatent.com/", "https://interlatent.com/api/v1"),
        # Already-rooted — what older recorder config passed. Must not double.
        ("https://interlatent.com/api/v1", "https://interlatent.com/api/v1"),
        ("https://interlatent.com/api/v1/", "https://interlatent.com/api/v1"),
    ],
)
def test_api_v1_root_accepts_both_conventions(given: str, expected: str) -> None:
    """A bare origin used to POST to /episodes and get 405 back."""
    assert rec_mod._api_v1_root(given) == expected


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "OFF", " False "])
def test_live_encode_kill_switch_off_values(monkeypatch, value: str) -> None:
    monkeypatch.setenv("INTERLATENT_LIVE_ENCODE", value)
    assert rec_mod._live_encode_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "anything"])
def test_live_encode_on_by_default_and_for_other_values(monkeypatch, value: str) -> None:
    monkeypatch.setenv("INTERLATENT_LIVE_ENCODE", value)
    assert rec_mod._live_encode_enabled() is True
    monkeypatch.delenv("INTERLATENT_LIVE_ENCODE")
    assert rec_mod._live_encode_enabled() is True


def test_to_list_handles_none_sequences_scalars_and_junk() -> None:
    assert rec_mod._to_list(None) == []
    assert rec_mod._to_list([1, 2.5]) == [1.0, 2.5]
    assert rec_mod._to_list((3,)) == [3.0]
    assert rec_mod._to_list(4.0) == [4.0]
    assert rec_mod._to_list(object()) == []


def test_to_list_flattens_numpy_arrays() -> None:
    np = pytest.importorskip("numpy")
    out = rec_mod._to_list(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    assert out == [1.0, 2.0, 3.0, 4.0]
    assert all(isinstance(v, float) for v in out)


# ----------------------------------------------------------------------
# Construction + admission control
# ----------------------------------------------------------------------


def test_init_creates_staging_layout_including_empty_jsonl(tmp_path: Path) -> None:
    """steps.jsonl must exist even for a session that records nothing —
    RecorderStepSource reads it unconditionally at close."""
    rec = _rec(tmp_path)
    assert rec.working_dir.is_dir()
    assert rec.frames_dir.is_dir()
    assert rec.steps_path.is_file()
    assert rec.steps_path.read_text() == ""
    assert rec.no_steps is True
    assert rec.reusable is True


def test_enqueue_records_cameras_in_first_seen_order(tmp_path: Path) -> None:
    rec = _rec(tmp_path)
    _tick(rec, 0, jpegs={"observation.images.wrist": b"a"})
    _tick(rec, 1, jpegs={
        "observation.images.wrist": b"a", "observation.images.top": b"b",
    })
    _tick(rec, 2, jpegs={"observation.images.top": b"b"})
    # Stable schema for the whole episode, in first-appearance order.
    assert rec.cameras == ["wrist", "top"]


def test_enqueue_refuses_past_the_step_cap(tmp_path: Path) -> None:
    rec = SessionRecorder(tmp_path / "work", _config(), max_steps=3)
    assert [_tick(rec, i) for i in range(3)] == [True, True, True]
    assert _tick(rec, 3) is False
    assert rec.step_count == 3


def test_enqueue_refuses_when_the_queue_is_full(monkeypatch, tmp_path: Path) -> None:
    """A full queue must return False, not silently drop: the False is what
    tells the node to keep the tick spooled (ADR 0023)."""
    monkeypatch.setattr(rec_mod, "_QUEUE_MAXSIZE", 2)
    rec = _rec(tmp_path)  # no start() — nothing drains the queue
    assert _tick(rec, 0) is True
    assert _tick(rec, 1) is True
    assert _tick(rec, 2) is False
    assert rec.dropped == 1
    assert rec.step_count == 2


def test_enqueue_after_close_is_refused(tmp_path: Path) -> None:
    async def main() -> None:
        rec = _rec(tmp_path)
        rec.start()
        assert _tick(rec, 0) is True
        await rec.finalize()
        assert rec.reusable is False
        assert _tick(rec, 1) is False
        # Even a step already staged is refused once closed — the recorder
        # must not ack a tick it can no longer durably write.
        assert _tick(rec, 0) is False

    asyncio.run(main())


def test_replayed_step_is_deduped_and_acked(tmp_path: Path) -> None:
    """ADR 0024: a reconnect replays ticks whose ack was lost. They are
    already durably staged, so ack them — but do NOT stage them twice."""
    rec = _rec(tmp_path)
    assert _tick(rec, 7) is True
    assert _tick(rec, 7) is True  # acked...
    assert rec.step_count == 1  # ...but not re-staged
    assert rec._dedup_acked == 1


def test_await_enqueue_applies_backpressure_then_gives_up(
    monkeypatch, tmp_path: Path
) -> None:
    """The recorder-only pod path awaits instead of refusing, but a wedged
    writer must degrade to a refused tick rather than a stuck gRPC handler."""
    monkeypatch.setattr(rec_mod, "_QUEUE_MAXSIZE", 1)

    async def main() -> None:
        rec = _rec(tmp_path)  # no drain task: the writer is "wedged"
        ok = await rec.enqueue(
            step=0, observation_state=[0.0], action=[0.0], jpegs={},
            control_timestamp=0, timeout=0.05,
        )
        assert ok is True
        blocked = await rec.enqueue(
            step=1, observation_state=[0.0], action=[0.0], jpegs={},
            control_timestamp=1, timeout=0.05,
        )
        assert blocked is False
        assert rec.dropped == 1
        # Dedupe applies on the awaitable variant too.
        assert await rec.enqueue(
            step=0, observation_state=[0.0], action=[0.0], jpegs={},
            control_timestamp=0, timeout=0.05,
        ) is True
        assert rec.step_count == 1

    asyncio.run(main())


def test_teleop_and_intervention_steps_are_counted_as_human(tmp_path: Path) -> None:
    """``intervention`` is the ADR 0034 DAgger label — it must count as
    human-driven exactly like ``teleop``. ``hold`` and ``policy`` must not."""
    rec = _rec(tmp_path)
    _tick(rec, 0, control_source="policy")
    _tick(rec, 1, control_source="teleop")
    _tick(rec, 2, control_source="intervention")
    _tick(rec, 3, control_source="hold")
    _tick(rec, 4, control_source=None)
    assert rec._teleop_steps == 2


# ----------------------------------------------------------------------
# Measured fps
# ----------------------------------------------------------------------


def test_measured_fps_uses_real_control_timestamps(tmp_path: Path) -> None:
    """Encoding at config.fps when inference throttled the loop makes
    playback run fast; the measured rate is the truth."""
    rec = _rec(tmp_path, fps=30)
    # 11 ticks, 100ms apart -> 10 intervals over 1.0s -> 10 fps.
    for i in range(11):
        _tick(rec, i, ts=i * 100_000_000)
    assert rec._measured_fps() == pytest.approx(10.0)


def test_measured_fps_is_none_when_undeterminable(tmp_path: Path) -> None:
    empty = _rec(tmp_path / "a")
    assert empty._measured_fps() is None

    one = _rec(tmp_path / "b")
    _tick(one, 0, ts=1_000)
    assert one._measured_fps() is None  # a single sample has no interval

    degenerate = _rec(tmp_path / "c")
    _tick(degenerate, 0, ts=5_000)
    _tick(degenerate, 1, ts=5_000)  # identical clock -> no elapsed time
    assert degenerate._measured_fps() is None


# ----------------------------------------------------------------------
# Drain loop / on-disk staging
# ----------------------------------------------------------------------


def test_drain_writes_frames_and_jsonl(tmp_path: Path) -> None:
    async def main() -> SessionRecorder:
        rec = _rec(tmp_path)
        rec.start()
        _tick(rec, 0, jpegs={
            "observation.images.wrist": b"WRIST0",
            "observation.images.top": b"TOP0",
        }, control_source="policy")
        _tick(rec, 1, jpegs={"observation.images.wrist": b"WRIST1"},
              control_source="intervention")
        await rec.finalize()
        return rec

    rec = asyncio.run(main())

    assert (rec.frames_dir / "wrist" / "00000000.jpg").read_bytes() == b"WRIST0"
    assert (rec.frames_dir / "wrist" / "00000001.jpg").read_bytes() == b"WRIST1"
    assert (rec.frames_dir / "top" / "00000000.jpg").read_bytes() == b"TOP0"
    # No .part files survive: the writer renames into place so the rebuilder
    # never sees a half-written JPEG.
    assert not list(rec.frames_dir.rglob("*.part"))

    rows = [json.loads(x) for x in rec.steps_path.read_text().splitlines() if x]
    assert [r["step"] for r in rows] == [0, 1]
    assert rows[0]["action"] == [0.0, 1.0]
    assert rows[0]["control_source"] == "policy"
    assert rows[1]["control_source"] == "intervention"


def test_drain_omits_control_source_when_unset(tmp_path: Path) -> None:
    async def main() -> SessionRecorder:
        rec = _rec(tmp_path)
        rec.start()
        _tick(rec, 0, control_source=None)
        await rec.finalize()
        return rec

    rec = asyncio.run(main())
    row = json.loads(rec.steps_path.read_text().splitlines()[0])
    assert "control_source" not in row


def test_finalize_is_idempotent_and_survives_a_failing_write(
    tmp_path: Path, monkeypatch
) -> None:
    """A writer exception must not kill the drain task — the rest of the
    session still gets staged."""
    rec = _rec(tmp_path)
    real_write = rec._write_one
    calls = {"n": 0}

    def flaky(item):
        calls["n"] += 1
        if item["step"] == 0:
            raise OSError("disk hiccup")
        return real_write(item)

    monkeypatch.setattr(rec, "_write_one", flaky)

    async def main() -> None:
        rec.start()
        _tick(rec, 0)
        _tick(rec, 1)
        await rec.finalize()
        await rec.finalize()  # idempotent

    asyncio.run(main())
    assert calls["n"] == 2
    rows = [json.loads(x) for x in rec.steps_path.read_text().splitlines() if x]
    assert [r["step"] for r in rows] == [1]


# ----------------------------------------------------------------------
# discard()
# ----------------------------------------------------------------------


def test_discard_drops_everything_and_blocks_a_later_upload(tmp_path: Path) -> None:
    """The idle-GC discards a capture whose session the user moved past.
    Exactly one of upload/discard may ever run."""
    uploaded = {"n": 0}

    async def main() -> SessionRecorder:
        rec = _rec(tmp_path)
        rec.start()
        _tick(rec, 0)
        _tick(rec, 1)
        rec._post_episodes_create = lambda *a, **k: uploaded.update(n=1)  # type: ignore[assignment]
        dropped = await rec.discard()
        assert dropped == 2
        assert not rec.working_dir.exists()
        assert await rec.discard() == 0  # idempotent
        await rec.upload()  # must be a no-op after discard
        return rec

    asyncio.run(main())
    assert uploaded["n"] == 0


# ----------------------------------------------------------------------
# RecorderStepSource
# ----------------------------------------------------------------------


def test_step_source_reads_sorts_and_tolerates_malformed_rows(tmp_path: Path) -> None:
    rec = _rec(tmp_path)
    rec.steps_path.write_text(
        json.dumps({"step": 2, "observation": [1.0], "action": [2.0]}) + "\n"
        + "{not json\n"
        + "\n"
        + json.dumps({
            "step": 0, "observation": [3.0], "action": [4.0],
            "control_source": "teleop",
        }) + "\n"
    )
    rec._cameras = ["wrist"]

    src = RecorderStepSource(rec)
    assert src.episode_ids() == ["ep-1"]
    rows = list(src.iter_steps("ep-1"))
    assert [r.step for r in rows] == [0, 2]  # sorted, malformed line skipped
    assert rows[0].control_source == "teleop"
    assert rows[1].control_source is None
    assert src.cameras_for_episode("ep-1") == ["wrist"]

    # A foreign episode id yields nothing rather than another episode's rows.
    assert list(src.iter_steps("other")) == []
    assert src.cameras_for_episode("other") == []
    assert list(src.iter_frames("other")) == []


def test_step_source_reports_no_episodes_when_nothing_was_staged(
    tmp_path: Path,
) -> None:
    """An empty episode would make LeRobotDataset reject a zero-frame write,
    so the source must not claim the episode exists."""
    src = RecorderStepSource(_rec(tmp_path))
    assert src.episode_ids() == []


def test_step_source_walks_frames_and_skips_non_frames(tmp_path: Path) -> None:
    rec = _rec(tmp_path)
    for cam, steps in (("wrist", (0, 1)), ("top", (0,))):
        (rec.frames_dir / cam).mkdir(parents=True, exist_ok=True)
        for s in steps:
            (rec.frames_dir / cam / f"{s:08d}.jpg").write_bytes(b"x")
    # Noise the walker must ignore: a stray file, a non-.jpg, a bad stem.
    (rec.frames_dir / "loose.jpg").write_bytes(b"x")
    (rec.frames_dir / "wrist" / "notes.txt").write_text("x")
    (rec.frames_dir / "wrist" / "banana.jpg").write_bytes(b"x")

    rec.steps_path.write_text(
        json.dumps({"step": 0, "observation": [], "action": []}) + "\n"
    )
    got = sorted(
        (step, cam) for step, cam, _p in RecorderStepSource(rec).iter_frames("ep-1")
    )
    assert got == [(0, "top"), (0, "wrist"), (1, "wrist")]


# ----------------------------------------------------------------------
# upload()
# ----------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status_code: int = 200, body: dict | None = None) -> None:
        self.status_code = status_code
        self._body = body or {}
        self.text = json.dumps(self._body)

    def json(self) -> dict:
        return self._body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeHttp:
    """Stands in for httpx.AsyncClient, recording every call.

    ``presign`` decides which keys get a presigned URL back, so the
    "backend forgot a file" path is reachable.
    """

    def __init__(self, *, create_status: int = 200, presign: bool = True) -> None:
        self.posts: list[tuple[str, dict, dict]] = []
        self.puts: list[tuple[str, bytes]] = []
        self.create_status = create_status
        self.presign = presign

    def client_factory(self, **_kw):
        outer = self

        class _Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def post(self, url, json=None, headers=None):
                outer.posts.append((url, json or {}, headers or {}))
                if url.endswith("/upload-urls"):
                    keys = [f["key"] for f in (json or {}).get("files", [])]
                    urls = (
                        {k: f"https://s3.test/{k}" for k in keys}
                        if outer.presign else {}
                    )
                    return _FakeResponse(200, {"presigned_urls": urls})
                if url.endswith("/episodes"):
                    return _FakeResponse(outer.create_status)
                return _FakeResponse(200)

            async def put(self, url, content=None):
                outer.puts.append((url, content))
                return _FakeResponse(200)

        return _Client()


class _StubRebuilder:
    """Stands in for LeRobotRebuilder: writes a plausible v3 tree."""

    last: "_StubRebuilder | None" = None

    def __init__(
        self, *, root, fps, task, env_slug, vcodec=None,
        force_control_source=False, measured_fps=None,
    ) -> None:
        self.root = Path(root)
        self.fps = fps
        self.task = task
        self.env_slug = env_slug
        self.vcodec = vcodec
        self.force_control_source = force_control_source
        self.measured_fps = measured_fps
        self.built_rows: list = []
        type(self).last = self

    def build_from_source(self, source):
        self.built_rows = list(source.iter_steps(source.episode_ids()[0]))
        (self.root / "meta").mkdir(parents=True, exist_ok=True)
        (self.root / "data").mkdir(parents=True, exist_ok=True)
        (self.root / "meta" / "info.json").write_text("{}")
        (self.root / "data" / "chunk-000.parquet").write_bytes(b"PAR1")
        return self.root, list(source.episode_ids())


def _install_stubs(monkeypatch, http: _FakeHttp) -> None:
    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", http.client_factory)
    monkeypatch.setattr(rec_mod, "LeRobotRebuilder", _StubRebuilder)


def test_upload_of_an_empty_session_skips_the_backend_entirely(
    monkeypatch, tmp_path: Path
) -> None:
    http = _FakeHttp()
    _install_stubs(monkeypatch, http)

    async def main() -> SessionRecorder:
        rec = _rec(tmp_path)
        rec.start()
        await rec.upload()
        return rec

    rec = asyncio.run(main())
    assert http.posts == []
    assert not rec.working_dir.exists()


def test_upload_posts_creates_puts_and_completes(monkeypatch, tmp_path: Path) -> None:
    http = _FakeHttp()
    _install_stubs(monkeypatch, http)

    async def main() -> SessionRecorder:
        rec = _rec(tmp_path, task_id="task-42", model_id="m-9")
        rec.start()
        for i in range(4):
            _tick(rec, i, ts=i * 100_000_000, control_source="intervention")
        await rec.upload()
        await rec.upload()  # idempotent: a tardy CloseSession after idle-GC
        return rec

    rec = asyncio.run(main())

    urls = [u for u, _b, _h in http.posts]
    root = "https://interlatent.test/api/v1"
    assert urls == [
        f"{root}/episodes",
        f"{root}/episodes/ep-1/upload-urls",
        f"{root}/episodes/ep-1/upload-complete",
    ]

    create_body = http.posts[0][1]
    assert create_body["episode_id"] == "ep-1"
    assert create_body["environment"] == "pick-cube"
    assert create_body["task_id"] == "task-42"
    assert create_body["model_id"] == "m-9"
    assert create_body["model_framework"] == "drtc"
    assert create_body["tags"]["policy_uri"] == "lerobot/smolvla_base"
    assert http.posts[0][2]["x-api-key"] == "ilat_test"

    # Every dataset file is PUT, namespaced under a single _inbox session.
    assert len(http.puts) == 2
    prefixes = {u.split("/_inbox/")[1].split("/")[0] for u, _c in http.puts}
    assert len(prefixes) == 1
    assert {u.rsplit("/", 1)[-1] for u, _c in http.puts} == {
        "info.json", "chunk-000.parquet",
    }

    # Human-driven steps were reported to the backend as the episode badge.
    assert http.posts[2][1] == {"manifest": None, "has_teleop": True}

    # The rebuilder was handed the MEASURED rate (4 ticks 100ms apart -> 10),
    # not the requested 30, and the h264 codec (gVisor stalls on libsvtav1).
    assert _StubRebuilder.last.fps == 10
    assert _StubRebuilder.last.vcodec == "h264"
    assert [r.step for r in _StubRebuilder.last.built_rows] == [0, 1, 2, 3]

    assert not rec.working_dir.exists()


def test_upload_reports_has_teleop_false_for_a_pure_policy_rollout(
    monkeypatch, tmp_path: Path
) -> None:
    http = _FakeHttp()
    _install_stubs(monkeypatch, http)

    async def main() -> None:
        rec = _rec(tmp_path)
        rec.start()
        for i in range(2):
            _tick(rec, i, control_source="policy")
        await rec.upload()

    asyncio.run(main())
    assert http.posts[-1][1]["has_teleop"] is False


def test_upload_tolerates_a_409_on_episode_create(monkeypatch, tmp_path: Path) -> None:
    """The idle-GC may already have created the row; a 409 is not an error."""
    http = _FakeHttp(create_status=409)
    _install_stubs(monkeypatch, http)

    async def main() -> None:
        rec = _rec(tmp_path)
        rec.start()
        _tick(rec, 0)
        _tick(rec, 1)
        await rec.upload()

    asyncio.run(main())
    assert [u for u, _b, _h in http.posts][-1].endswith("/upload-complete")
    assert http.puts


def test_upload_does_not_complete_when_a_presigned_url_is_missing(
    monkeypatch, tmp_path: Path
) -> None:
    """A missing URL means that file never reached S3 — completing anyway
    would enqueue a merge against an incomplete inbox session."""
    http = _FakeHttp(presign=False)
    _install_stubs(monkeypatch, http)

    async def main() -> SessionRecorder:
        rec = _rec(tmp_path)
        rec.start()
        _tick(rec, 0)
        _tick(rec, 1)
        await rec.upload()
        return rec

    rec = asyncio.run(main())
    assert http.puts == []
    assert not any(u.endswith("/upload-complete") for u, _b, _h in http.posts)
    # The working dir is still reaped so a failed upload can't wedge the disk.
    assert not rec.working_dir.exists()


def test_upload_aborts_when_the_rebuild_fails(monkeypatch, tmp_path: Path) -> None:
    http = _FakeHttp()
    _install_stubs(monkeypatch, http)

    class _Boom(_StubRebuilder):
        def build_from_source(self, source):
            raise RuntimeError("lerobot exploded")

    monkeypatch.setattr(rec_mod, "LeRobotRebuilder", _Boom)

    async def main() -> SessionRecorder:
        rec = _rec(tmp_path)
        rec.start()
        _tick(rec, 0)
        _tick(rec, 1)
        await rec.upload()
        return rec

    rec = asyncio.run(main())
    assert http.posts == []
    assert not rec.working_dir.exists()


def test_upload_aborts_when_the_rebuild_produced_no_episodes(
    monkeypatch, tmp_path: Path
) -> None:
    http = _FakeHttp()
    _install_stubs(monkeypatch, http)

    class _Empty(_StubRebuilder):
        def build_from_source(self, source):
            super().build_from_source(source)
            return self.root, []

    monkeypatch.setattr(rec_mod, "LeRobotRebuilder", _Empty)

    async def main() -> None:
        rec = _rec(tmp_path)
        rec.start()
        _tick(rec, 0)
        _tick(rec, 1)
        await rec.upload()

    asyncio.run(main())
    assert http.posts == []


# ----------------------------------------------------------------------
# ADR 0016 live-encode lane
#
# The live builder encodes video as frames arrive so CloseSession doesn't
# pay a full rebuild. It is a pure optimisation: EVERY failure has to fall
# back to the staging rebuild, silently and completely. JPEG + JSONL
# staging is written regardless, which is what makes that fallback safe —
# these tests pin that invariant.
# ----------------------------------------------------------------------


class _StubLive:
    """Stands in for LiveEpisodeBuilder."""

    last: "_StubLive | None" = None

    def __init__(self, root, *, fps, task, env_slug, episode_id) -> None:
        self.root = Path(root)
        self.fps = fps
        self.task = task
        self.env_slug = env_slug
        self.episode_id = episode_id
        self.steps: list[dict] = []
        self.aborted = 0
        self.finalize_calls: list = []
        self.add_step_error: Exception | None = None
        self.finalize_error: Exception | None = None
        _StubLive.last = self

    @property
    def rows(self) -> int:
        return len(self.steps)

    def add_step(self, *, step, observation, action, jpegs, control_source) -> None:
        if self.add_step_error is not None:
            raise self.add_step_error
        self.steps.append({"step": step, "control_source": control_source})

    def finalize(self, measured_fps):
        self.finalize_calls.append(measured_fps)
        if self.finalize_error is not None:
            raise self.finalize_error
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "info.json").write_text("{}")
        return self.root

    def abort(self) -> None:
        self.aborted += 1


@pytest.fixture
def live(monkeypatch):
    _StubLive.last = None
    monkeypatch.setattr(rec_mod, "LiveEpisodeBuilder", _StubLive)
    # start() warms the lerobot import in an executor; keep it out of tests.
    monkeypatch.setattr(rec_mod, "_warm_lerobot_import", lambda: None)
    return _StubLive


def test_live_builder_is_only_created_when_configured_and_not_killed(
    live, monkeypatch, tmp_path: Path
) -> None:
    assert _rec(tmp_path / "a")._live is None  # off by default (serve_gpu)
    assert _rec(tmp_path / "b", live_encode=True)._live is not None

    monkeypatch.setenv("INTERLATENT_LIVE_ENCODE", "0")
    assert _rec(tmp_path / "c", live_encode=True)._live is None


def test_live_builder_is_fed_after_staging(live, tmp_path: Path) -> None:
    async def main() -> SessionRecorder:
        rec = _rec(tmp_path, live_encode=True)
        rec.start()
        _tick(rec, 0, control_source="policy")
        _tick(rec, 1, control_source="intervention")
        await rec.finalize()
        return rec

    rec = asyncio.run(main())
    assert [s["step"] for s in _StubLive.last.steps] == [0, 1]
    assert _StubLive.last.steps[1]["control_source"] == "intervention"
    # ...and the staging fallback is complete regardless.
    assert len(rec.steps_path.read_text().splitlines()) == 2


def test_a_live_add_step_failure_reverts_to_staging_without_losing_steps(
    live, tmp_path: Path
) -> None:
    async def main() -> SessionRecorder:
        rec = _rec(tmp_path, live_encode=True)
        rec.start()
        _tick(rec, 0)
        await _staged(rec, 1)  # step 0 reaches the live builder healthy
        _StubLive.last.add_step_error = rec_mod.LiveEncodeError("encoder gone")
        _tick(rec, 1)
        _tick(rec, 2)
        await rec.finalize()
        return rec

    rec = asyncio.run(main())
    assert rec._live_dead is True
    assert _StubLive.last.aborted == 1  # aborted once, not once per later step
    assert len(_StubLive.last.steps) == 1
    # Every step still reached staging, so the close-time rebuild is whole.
    rows = [json.loads(x) for x in rec.steps_path.read_text().splitlines() if x]
    assert [r["step"] for r in rows] == [0, 1, 2]


def test_upload_uses_the_live_dataset_and_skips_the_rebuild(
    live, monkeypatch, tmp_path: Path
) -> None:
    http = _FakeHttp()
    _install_stubs(monkeypatch, http)
    _StubRebuilder.last = None

    async def main() -> None:
        rec = _rec(tmp_path, live_encode=True)
        rec.start()
        for i in range(4):
            _tick(rec, i, ts=i * 100_000_000)
        await rec.upload()

    asyncio.run(main())
    # Sealed at the measured rate, and no rebuilder was ever constructed.
    assert _StubLive.last.finalize_calls == [pytest.approx(10.0)]
    assert _StubRebuilder.last is None
    assert [u for u, _b, _h in http.posts][-1].endswith("/upload-complete")
    assert http.puts  # the live dataset's files were uploaded


@pytest.mark.parametrize(
    "err",
    [
        rec_mod.FpsDivergenceError("measured rate diverged"),
        RuntimeError("encoder blew up at seal time"),
    ],
)
def test_a_failed_live_finalize_falls_back_to_the_staging_rebuild(
    live, monkeypatch, tmp_path: Path, err: Exception
) -> None:
    http = _FakeHttp()
    _install_stubs(monkeypatch, http)
    _StubRebuilder.last = None

    async def main() -> None:
        rec = _rec(tmp_path, live_encode=True)
        rec.start()
        for i in range(3):
            _tick(rec, i, ts=i * 100_000_000)
        _StubLive.last.finalize_error = err
        await rec.upload()

    asyncio.run(main())
    assert _StubLive.last.aborted == 1
    # The rebuild lane ran instead, off the staged rows — byte-equivalent to
    # the pre-ADR-0016 behaviour.
    assert _StubRebuilder.last is not None
    assert [r.step for r in _StubRebuilder.last.built_rows] == [0, 1, 2]
    assert [u for u, _b, _h in http.posts][-1].endswith("/upload-complete")


def test_discard_aborts_the_live_builder(live, tmp_path: Path) -> None:
    async def main() -> None:
        rec = _rec(tmp_path, live_encode=True)
        rec.start()
        _tick(rec, 0)
        assert await rec.discard() == 1

    asyncio.run(main())
    assert _StubLive.last.aborted == 1


# ----------------------------------------------------------------------
# Publish failure must not delete a built dataset
# ----------------------------------------------------------------------


def test_failed_publish_root_honours_the_env_override(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("INTERLATENT_FAILED_PUBLISH_DIR", str(tmp_path / "parked"))
    assert rec_mod._failed_publish_root() == tmp_path / "parked"


def test_failed_publish_root_defaults_under_the_interlatent_home(monkeypatch) -> None:
    monkeypatch.delenv("INTERLATENT_FAILED_PUBLISH_DIR", raising=False)
    monkeypatch.setattr(rec_mod.Path, "home", staticmethod(lambda: Path("/home/pilot")))
    assert rec_mod._failed_publish_root() == Path("/home/pilot/.interlatent/failed-publish")


def test_publish_failure_keeps_the_dataset_instead_of_deleting_it(
    monkeypatch, tmp_path: Path
) -> None:
    """The regression this whole fix exists for.

    The dataset is fully built; only shipping it failed. Before the fix the
    ``finally: self._cleanup_working_dir()`` rmtree'd it and the episode was
    gone for good.
    """
    parked = tmp_path / "parked"
    monkeypatch.setenv("INTERLATENT_FAILED_PUBLISH_DIR", str(parked))
    http = _FakeHttp(create_status=500)
    _install_stubs(monkeypatch, http)

    async def main() -> SessionRecorder:
        rec = _rec(tmp_path)
        rec.start()
        for i in range(3):
            _tick(rec, i, ts=i * 100_000_000)
        await rec.upload()
        return rec

    rec = asyncio.run(main())

    # Working dir is still cleaned up — we did not simply stop tidying.
    assert not rec.working_dir.exists()
    # ...but the built dataset survived, intact, outside it.
    kept = parked / "ep-1"
    assert kept.is_dir()
    assert (kept / "meta" / "info.json").exists()
    assert (kept / "data" / "chunk-000.parquet").read_bytes() == b"PAR1"


def test_quarantine_disambiguates_a_repeated_episode_id(
    monkeypatch, tmp_path: Path
) -> None:
    parked = tmp_path / "parked"
    monkeypatch.setenv("INTERLATENT_FAILED_PUBLISH_DIR", str(parked))
    rec = _rec(tmp_path)

    for _ in range(2):
        built = tmp_path / "build"
        built.mkdir()
        (built / "info.json").write_text("{}")
        rec._quarantine_dataset(built)

    assert sorted(p.name for p in parked.iterdir()) == ["ep-1", "ep-1.1"]


@pytest.mark.parametrize("root", [None, "missing"])
def test_quarantine_is_a_noop_without_a_dataset(
    monkeypatch, tmp_path: Path, root
) -> None:
    parked = tmp_path / "parked"
    monkeypatch.setenv("INTERLATENT_FAILED_PUBLISH_DIR", str(parked))
    rec = _rec(tmp_path)
    rec._quarantine_dataset(None if root is None else tmp_path / root)
    assert not parked.exists()


def test_a_failing_custom_sink_quarantines_too(monkeypatch, tmp_path: Path) -> None:
    """Not just the hosted inbox. With local and S3 destinations, a publish
    failure becomes routine — a wrong bucket, an expired key — so the
    quarantine has to cover whatever sink is configured, not one code path."""
    parked = tmp_path / "parked"
    monkeypatch.setenv("INTERLATENT_FAILED_PUBLISH_DIR", str(parked))
    _install_stubs(monkeypatch, _FakeHttp())

    class _ExplodingSink:
        def requires_api_key(self):
            return False

        def normalize_for_merge(self):
            return True

        async def publish(self, **_kw):
            raise RuntimeError("bucket does not exist")

    async def main() -> SessionRecorder:
        rec = _rec(tmp_path, sink=_ExplodingSink())
        rec.start()
        for i in range(3):
            _tick(rec, i, ts=i * 100_000_000)
        await rec.upload()
        return rec

    rec = asyncio.run(main())
    assert not rec.working_dir.exists()
    assert (parked / "ep-1" / "meta" / "info.json").exists()


def test_the_sink_drives_merge_normalization(monkeypatch, tmp_path: Path) -> None:
    """A merge-on-stop sink aggregates sessions into one dataset, and
    lerobot's aggregate_datasets rejects mismatched `features` — so the
    control_source column must not depend on whether a human intervened."""
    _install_stubs(monkeypatch, _FakeHttp())

    class _MergingSink:
        def requires_api_key(self):
            return False

        def normalize_for_merge(self):
            return True

        async def publish(self, **_kw):
            return None

    async def main() -> None:
        rec = _rec(tmp_path, sink=_MergingSink())
        rec.start()
        for i in range(3):
            _tick(rec, i, ts=i * 100_000_000)
        await rec.upload()

    asyncio.run(main())
    assert _StubRebuilder.last.force_control_source is True
    assert _StubRebuilder.last.measured_fps is not None
