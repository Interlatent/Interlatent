"""Dataset publish sinks: metadata selection, the recorder gate, and the
local merge-on-stop accumulation behaviour.

The merge tests need lerobot (+ Pillow) installed and ``importorskip`` out
otherwise — mirroring how the rest of the suite stays GPU/robot-free. The
metadata-selection and recorder-gate tests need neither.
"""
import pytest

from interlatent_server.server.sinks import (
    BackendInboxSink,
    LocalDirSink,
    S3Sink,
    sink_from_metadata,
)


# ----------------------------------------------------------------------
# sink selection from OpenSession metadata (no heavy deps)
# ----------------------------------------------------------------------


def test_sink_from_metadata_none_when_unset():
    assert sink_from_metadata({}) is None
    assert sink_from_metadata({"record": "1"}) is None


def test_sink_from_metadata_local_dir():
    sink = sink_from_metadata({"output_dir": "/data/run"})
    assert isinstance(sink, LocalDirSink)
    assert sink.requires_api_key() is False
    assert sink.normalize_for_merge() is True


def test_sink_from_metadata_s3():
    sink = sink_from_metadata({
        "s3_uri": "s3://bucket/some/prefix",
        "s3_endpoint_url": "http://localhost:9000",
        "s3_access_key": "ak",
        "s3_secret_key": "sk",
        "s3_region": "us-east-1",
    })
    assert isinstance(sink, S3Sink)
    assert sink.bucket == "bucket"
    assert sink.prefix == "some/prefix"
    assert sink.endpoint_url == "http://localhost:9000"
    assert sink.requires_api_key() is False


def test_backend_inbox_sink_requires_key_and_no_merge():
    sink = BackendInboxSink()
    assert sink.requires_api_key() is True
    assert sink.normalize_for_merge() is False


# ----------------------------------------------------------------------
# recorder gate: local/S3 sinks record without an API key; inbox doesn't
# ----------------------------------------------------------------------


def test_recorder_gate(tmp_path):
    pytest.importorskip("grpc")
    import interlatent_server.protocol.messages_pb2 as pb
    from interlatent_server.server.transport import InferenceServicer

    svc = InferenceServicer(recorder_base_dir=tmp_path)
    req = pb.OpenSessionRequest(policy_uri="lerobot/x")

    # Local sink requested via metadata, no x-api-key (context=None) -> records.
    rec = svc._maybe_build_recorder(
        session_id="s1",
        request=req,
        metadata={"record": "1", "output_dir": str(tmp_path / "ds")},
        context=None,
    )
    assert rec is not None
    assert isinstance(rec.config.sink, LocalDirSink)

    # Hosted inbox (no sink configured), no x-api-key -> recording disabled.
    rec2 = svc._maybe_build_recorder(
        session_id="s2", request=req, metadata={"record": "1"}, context=None
    )
    assert rec2 is None

    # Serve-level default sink is used when metadata doesn't specify one.
    svc_default = InferenceServicer(
        recorder_base_dir=tmp_path, dataset_sink=LocalDirSink(str(tmp_path / "def"))
    )
    rec3 = svc_default._maybe_build_recorder(
        session_id="s3", request=req, metadata={"record": "1"}, context=None
    )
    assert rec3 is not None
    assert isinstance(rec3.config.sink, LocalDirSink)


# ----------------------------------------------------------------------
# merge-on-stop: accumulate into one flat dataset; schema stays stable
# ----------------------------------------------------------------------


def _build_dataset(root, episode_id, *, n_steps=4, action_dim=6, teleop=False):
    """Build a tiny LeRobot v3 dataset via the real rebuilder."""
    from interlatent_server.storage.lerobot_rebuild import LeRobotRebuilder, StepRow

    class _MemSource:
        def episode_ids(self):
            return [episode_id]

        def iter_steps(self, eid):
            for i in range(n_steps):
                yield StepRow(
                    episode_id=episode_id,
                    step=i,
                    observation=[0.0] * 6,
                    action=[1.0] * action_dim,
                    control_source=("teleop" if (teleop and i == 0) else None),
                )

        def cameras_for_episode(self, eid):
            return []

        def iter_frames(self, eid):
            return iter(())

    rb = LeRobotRebuilder(
        root=root,
        fps=30,
        task="probe",
        env_slug="probe",
        force_control_source=True,  # stable schema across sessions
        measured_fps=12.3,
    )
    rb.build_from_source(_MemSource())


def test_localdir_accumulates_and_schema_stable(tmp_path):
    pytest.importorskip("lerobot")
    pytest.importorskip("PIL")
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    from interlatent_server.server.sinks import merge_local_dataset

    dest = tmp_path / "canonical"

    # First episode (pure policy) -> dest created by move.
    a = tmp_path / "a"
    _build_dataset(a, "epA", teleop=False)
    merge_local_dataset(dest, a, "epA")
    assert (dest / "meta" / "info.json").exists()
    assert not a.exists()  # moved into place

    # Second episode WITH teleop -> merge must succeed (force_control_source
    # keeps the schema identical) and accumulate to 2 episodes.
    b = tmp_path / "b"
    _build_dataset(b, "epB", teleop=True)
    merge_local_dataset(dest, b, "epB")

    meta = LeRobotDatasetMetadata("interlatent/canonical", root=str(dest))
    assert meta.total_episodes == 2
    # No leftover temp dirs from the swap.
    assert not [p for p in dest.parent.iterdir() if p.name.startswith((".aggr_", ".bak_"))]


def test_localdir_mismatch_writes_sibling_not_dropped(tmp_path):
    pytest.importorskip("lerobot")
    pytest.importorskip("PIL")
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    from interlatent_server.server.sinks import merge_local_dataset

    dest = tmp_path / "canonical"
    a = tmp_path / "a"
    _build_dataset(a, "epA", action_dim=6)
    merge_local_dataset(dest, a, "epA")

    # Incompatible schema (different action dim) -> validate_all_metadata
    # rejects the merge. The episode must be preserved as a sibling, and the
    # canonical dataset left intact.
    b = tmp_path / "b"
    _build_dataset(b, "epB", action_dim=8)
    merge_local_dataset(dest, b, "epB")

    sibling = dest.parent / "canonical__epB"
    assert sibling.exists() and (sibling / "meta" / "info.json").exists()
    meta = LeRobotDatasetMetadata("interlatent/canonical", root=str(dest))
    assert meta.total_episodes == 1  # canonical untouched
