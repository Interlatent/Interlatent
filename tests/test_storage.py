"""SDK SQLite staging backend."""
import json

from interlatent._schema import ActivationEvent
from interlatent._storage import CollectionSQLiteBackend


def make_backend(tmp_path):
    return CollectionSQLiteBackend(str(tmp_path / "staging.db"))


def ev(episode_id="ep1", step=0, layer="l0", tensor=(1.0, 2.0), context=None):
    return ActivationEvent(
        episode_id=episode_id,
        step=step,
        layer=layer,
        tensor=list(tensor),
        context=context or {},
    )


def read_all(backend):
    cur = backend._conn.cursor()
    cur.execute("SELECT * FROM activations ORDER BY episode_id, step")
    return cur.fetchall()


def test_write_and_read_roundtrip(tmp_path):
    b = make_backend(tmp_path)
    b.write_events([ev(step=0, context={"a": 1}), ev(step=1)])
    rows = read_all(b)
    assert len(rows) == 2
    assert json.loads(rows[0]["tensor"]) == [1.0, 2.0]
    assert json.loads(rows[0]["context"]) == {"a": 1}
    b.close()


def test_write_events_merges_batch_samples(tmp_path):
    b = make_backend(tmp_path)
    # Same (episode, step, layer): full tensor wins over scalar sample.
    b.write_events([
        ev(step=0, tensor=(5.0,)),
        ev(step=0, tensor=(1.0, 2.0, 3.0)),
    ])
    rows = read_all(b)
    assert len(rows) == 1
    assert json.loads(rows[0]["tensor"]) == [1.0, 2.0, 3.0]
    b.close()


def test_update_step_contexts_targets_correct_rows(tmp_path):
    b = make_backend(tmp_path)
    b.write_events([ev(episode_id="ep1", step=0), ev(episode_id="ep1", step=1)])
    b.update_step_contexts(contexts={1: {"episode_id": "ep1", "reward": 3.5}})
    rows = {r["step"]: json.loads(r["context"]) for r in read_all(b)}
    assert rows[1] == {"episode_id": "ep1", "reward": 3.5}
    assert rows[0] == {}  # untouched
    b.close()


def test_update_step_contexts_unknown_episode_is_noop(tmp_path):
    b = make_backend(tmp_path)
    b.write_events([ev(episode_id="ep1", step=0)])
    b.update_step_contexts(contexts={0: {"episode_id": "other", "x": 1}})
    rows = read_all(b)
    assert json.loads(rows[0]["context"]) == {}
    b.close()


def test_flush_checkpoints_wal(tmp_path):
    b = make_backend(tmp_path)
    b.write_events([ev()])
    b.flush()  # must not raise
    b.close()
