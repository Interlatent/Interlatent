"""Server-side chunk buffer + in-painting context reconstruction."""
import numpy as np

from interlatent_server.server.chunk_buffer import (
    InMemoryChunkBuffer,
    StoredChunk,
    gc_inmemory,
)
from interlatent_server.server.schedule import reconstruct


def chunk(start: int, n: int, dim: int = 2, ts: int = 0, created_at: float = 0.0) -> StoredChunk:
    actions = np.arange(start, start + n, dtype=np.float32)[:, None].repeat(dim, axis=1)
    return StoredChunk(start_step=start, control_timestamp=ts, actions=actions, created_at=created_at)


def test_lookup_within_one_chunk():
    buf = InMemoryChunkBuffer()
    buf.append("s", chunk(0, 10))
    out = buf.lookup_steps("s", 3, 7)
    assert out is not None
    np.testing.assert_array_equal(out[:, 0], np.arange(3, 8, dtype=np.float32))


def test_lookup_stitches_across_chunks():
    buf = InMemoryChunkBuffer()
    buf.append("s", chunk(0, 5))
    buf.append("s", chunk(5, 5))
    out = buf.lookup_steps("s", 2, 8)
    assert out is not None
    np.testing.assert_array_equal(out[:, 0], np.arange(2, 9, dtype=np.float32))


def test_lookup_overlapping_chunks_prefers_later_append():
    buf = InMemoryChunkBuffer()
    buf.append("s", chunk(0, 8))
    newer = chunk(4, 8)
    newer.actions += 100.0  # overwritten values
    buf.append("s", newer)
    out = buf.lookup_steps("s", 0, 11)
    assert out is not None
    # steps 4..7 must come from the later chunk (walk order: later wins)
    assert out[4, 0] == 104.0


def test_lookup_gap_returns_none():
    buf = InMemoryChunkBuffer()
    buf.append("s", chunk(0, 4))
    buf.append("s", chunk(8, 4))  # steps 4..7 missing
    assert buf.lookup_steps("s", 0, 11) is None


def test_lookup_unknown_session_and_bad_range():
    buf = InMemoryChunkBuffer()
    assert buf.lookup_steps("nope", 0, 3) is None
    buf.append("s", chunk(0, 4))
    assert buf.lookup_steps("s", 3, 1) is None  # end < start


def test_ring_evicts_old_chunks():
    buf = InMemoryChunkBuffer(max_chunks_per_session=2)
    buf.append("s", chunk(0, 4))
    buf.append("s", chunk(4, 4))
    buf.append("s", chunk(8, 4))  # evicts chunk(0, 4)
    assert buf.lookup_steps("s", 0, 3) is None
    assert buf.lookup_steps("s", 4, 11) is not None


def test_drop_and_gc():
    buf = InMemoryChunkBuffer()
    buf.append("a", chunk(0, 4, created_at=0.0))  # ancient
    buf.append("b", chunk(0, 4, created_at=2**62 / 1e9))  # far future, never stale
    gc_inmemory(buf, max_age_s=3600.0)
    assert buf.recent("a") == []
    assert buf.recent("b") != []
    buf.drop("b")
    assert buf.recent("b") == []


# --- reconstruct() ------------------------------------------------------


def test_reconstruct_with_full_client_coverage():
    buf = InMemoryChunkBuffer()
    buf.append("s", chunk(0, 16))
    ctx = reconstruct(buf, "s", next_action_step=10, spans=[(0, 15)], context_steps=4)
    assert ctx.next_action_step == 10
    assert ctx.prior_actions is not None
    np.testing.assert_array_equal(ctx.prior_actions[:, 0], np.arange(6, 10, dtype=np.float32))


def test_reconstruct_without_client_coverage():
    buf = InMemoryChunkBuffer()
    buf.append("s", chunk(0, 16))
    # Client only claims steps 0..4; tail 6..9 not covered -> no in-painting.
    ctx = reconstruct(buf, "s", next_action_step=10, spans=[(0, 4)], context_steps=4)
    assert ctx.prior_actions is None


def test_reconstruct_cold_start():
    buf = InMemoryChunkBuffer()
    ctx = reconstruct(buf, "s", next_action_step=0, spans=[], context_steps=8)
    assert ctx.prior_actions is None
    assert ctx.next_action_step == 0
