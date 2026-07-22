"""Write-through tick spool (inference/client/spool.py, ADR 0023).

Covers the delete-after-ack contract, crash hygiene (.part leftovers,
index rebuild), the hard-stop hysteresis, and orphan discovery/GC.
"""
from __future__ import annotations

import json

from interlatent.inference.client.spool import (
    TickSpool,
    disk_pressure,
    gc_orphans,
    orphan_sessions,
)


def _spool(tmp_path, session="sess-1", **kw) -> TickSpool:
    return TickSpool(session, server_address="host:50051", root=tmp_path, **kw)


def test_append_peek_ack_roundtrip(tmp_path):
    sp = _spool(tmp_path)
    seqs = [sp.append(f"tick-{i}".encode()) for i in range(5)]
    assert seqs == [0, 1, 2, 3, 4]
    assert sp.pending_count == 5

    batch = sp.peek_batch(3, max_bytes=1 << 20)
    assert [s for s, _ in batch] == [0, 1, 2]
    assert batch[0][1] == b"tick-0"

    sp.ack(2)
    assert sp.pending_count == 2
    # Acked files are gone from disk; the rest remain.
    assert not (sp.dir / "00000000.tick").exists()
    assert (sp.dir / "00000003.tick").exists()
    # Next peek starts at the first unacked seq.
    assert [s for s, _ in sp.peek_batch(10, 1 << 20)] == [3, 4]


def test_peek_batch_respects_byte_cap_but_returns_one(tmp_path):
    sp = _spool(tmp_path)
    sp.append(b"x" * 1000)
    sp.append(b"y" * 1000)
    sp.append(b"z" * 1000)
    # Cap smaller than one tick: the lone oversized tick still goes solo.
    assert len(sp.peek_batch(10, max_bytes=100)) == 1
    assert len(sp.peek_batch(10, max_bytes=2100)) == 2


def test_crash_rebuild_and_part_cleanup(tmp_path):
    sp = _spool(tmp_path)
    sp.append(b"a")
    sp.append(b"b")
    sp.ack(0)
    # Simulate a torn write from a crash.
    (sp.dir / "00000099.tick.part").write_bytes(b"torn")

    resumed = _spool(tmp_path)  # same session id → same dir
    assert resumed.pending_count == 1
    assert [s for s, _ in resumed.peek_batch(10, 1 << 20)] == [1]
    assert not list(resumed.dir.glob("*.part"))
    # New appends continue after the highest journaled seq.
    assert resumed.append(b"c") == 2


def test_blocked_hysteresis(tmp_path, monkeypatch):
    monkeypatch.setenv("INTERLATENT_SPOOL_MAX_MB", "0.001")   # ~1049 bytes
    monkeypatch.setenv("INTERLATENT_SPOOL_MIN_FREE_MB", "0")
    sp = _spool(tmp_path, session="sess-hyst")
    assert sp.blocked is False
    sp.append(b"x" * 600)
    assert sp.blocked is False
    sp.append(b"y" * 600)          # 1200 > cap
    assert sp.blocked is True
    sp.ack(0)                       # 600 pending — above 0.8*cap? 0.8*1048=839 → below
    assert sp.blocked is False      # drained under the resume line
    sp.append(b"z" * 200)           # 800 < 839: hysteresis, but was unblocked
    assert sp.blocked is False


def test_blocked_stays_until_resume_threshold(tmp_path, monkeypatch):
    monkeypatch.setenv("INTERLATENT_SPOOL_MAX_MB", "0.001")
    monkeypatch.setenv("INTERLATENT_SPOOL_MIN_FREE_MB", "0")
    sp = _spool(tmp_path, session="sess-hyst2")
    a = sp.append(b"x" * 500)
    b = sp.append(b"y" * 500)
    sp.append(b"z" * 500)          # 1500 > cap → blocked
    assert sp.blocked is True
    sp.ack(a)                       # 1000 still ≥ 0.8*1048? 838.8 → yes → stays blocked
    assert sp.blocked is True
    sp.ack(b)                       # 500 < 838.8 → resumes
    assert sp.blocked is False


def test_dispose(tmp_path):
    sp = _spool(tmp_path)
    sp.append(b"a")
    sp.dispose()
    assert not sp.dir.exists()


def test_orphans_and_gc(tmp_path):
    sp = _spool(tmp_path, session="dead-1")
    sp.append(b"a" * 100)
    sp.append(b"b" * 100)

    orphans = orphan_sessions(tmp_path)
    assert len(orphans) == 1
    assert orphans[0]["session_id"] == "dead-1"
    assert orphans[0]["pending_count"] == 2
    assert orphans[0]["pending_bytes"] == 200
    assert orphans[0]["meta"]["server_address"] == "host:50051"

    # Young orphan survives GC; aging it past retention removes it.
    assert gc_orphans(tmp_path) == 0
    meta = sp.dir / "meta.json"
    m = json.loads(meta.read_text())
    m["created_at"] = 1.0
    meta.write_text(json.dumps(m))
    assert gc_orphans(tmp_path) == 1
    assert orphan_sessions(tmp_path) == []


def test_disk_pressure_backlog(tmp_path, monkeypatch):
    monkeypatch.setenv("INTERLATENT_SPOOL_MAX_MB", "0.001")
    monkeypatch.setenv("INTERLATENT_SPOOL_MIN_FREE_MB", "0")
    assert disk_pressure(tmp_path) is None
    sp = _spool(tmp_path, session="fat")
    sp.append(b"x" * 2000)
    reason = disk_pressure(tmp_path)
    assert reason is not None and "backlog" in reason
