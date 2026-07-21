"""Controller ↔ spool integration (ADR 0023): delete-after-ack with
honest accepted-prefix semantics, retry-not-drop on failure, and the
capture hard-stop when the spool is full.

Drives the sender machinery synchronously (no background thread) against
a fake gRPC stub, so the ack/retry logic is tested without a server.
"""
from __future__ import annotations

import grpc
import pytest

from interlatent.inference.client.controller import DRTCClient, DRTCConfig
from interlatent.inference.client.spool import TickSpool
from interlatent.inference.protocol import messages_pb2 as pb


class _FakeRpcError(grpc.RpcError):
    def __init__(self, code):
        self._code = code

    def code(self):
        return self._code


class _Stub:
    """RecordTicks stub scripted by a list of behaviors:
    int n  -> respond accepted=n
    "all"  -> accept the whole batch
    "err"  -> raise UNAVAILABLE
    """

    def __init__(self, script):
        self.script = list(script)
        self.batches: list[int] = []

    def RecordTicks(self, req, metadata=None, timeout=None):
        self.batches.append(len(req.ticks))
        beh = self.script.pop(0) if self.script else "all"
        if beh == "err":
            raise _FakeRpcError(grpc.StatusCode.UNAVAILABLE)
        n = len(req.ticks) if beh == "all" else min(int(beh), len(req.ticks))
        return pb.RecordTicksResponse(accepted=n)


def _client(tmp_path, script) -> tuple[DRTCClient, _Stub]:
    cli = DRTCClient(DRTCConfig(server_address="stub:0", model_id="env"))
    cli.session_id = "sess-t"
    cli._stub = _Stub(script)
    cli._spool = TickSpool("sess-t", root=tmp_path)
    cli._rec_pace_bps = 0.0  # no pacing sleeps in tests
    return cli, cli._stub


def _tick(cli, step, jpeg_size: int = 10):
    return cli.record_tick(
        step=step, observation_state=[0.1], action=[0.5],
        jpegs={"cam": b"\xff\xd8\xff" + bytes(jpeg_size)},
        control_timestamp_ns=step * 1000,
    )


def test_full_accept_acks_and_clears_spool(tmp_path):
    cli, stub = _client(tmp_path, script=["all"])
    for i in range(4):
        assert _tick(cli, i) is True
    assert cli._spool.pending_count == 4
    cli._ship_available()
    assert cli._spool.pending_count == 0
    assert cli._rec_sent == 4


def test_partial_accept_keeps_suffix_spooled(tmp_path):
    cli, stub = _client(tmp_path, script=[2, "all"])
    for i in range(4):
        _tick(cli, i)
    cli._ship_available()          # first RPC: prefix 2 acked, backoff, return
    assert cli._rec_sent == 2
    assert cli._spool.pending_count == 2
    cli._ship_available()          # retry ships the remainder
    assert cli._rec_sent == 4
    assert cli._spool.pending_count == 0
    # The retried batch contained exactly the unacked suffix.
    assert stub.batches == [4, 2]


def test_rpc_failure_drops_nothing(tmp_path):
    cli, stub = _client(tmp_path, script=["err", "err", "all"])
    for i in range(3):
        _tick(cli, i)
    cli._ship_available()
    cli._ship_available()
    assert cli._rec_sent == 0
    assert cli._spool.pending_count == 3   # nothing lost across failures
    cli._ship_available()
    assert cli._rec_sent == 3
    assert cli._spool.pending_count == 0


def test_spool_full_hard_stops_capture(tmp_path, monkeypatch):
    monkeypatch.setenv("INTERLATENT_SPOOL_MAX_MB", "0.001")  # ~1 KiB cap
    monkeypatch.setenv("INTERLATENT_SPOOL_MIN_FREE_MB", "0")
    cli, stub = _client(tmp_path, script=["all"])
    assert _tick(cli, 0, jpeg_size=2000) is True   # one 2 KB tick > cap
    assert cli.recording_blocked is True
    assert _tick(cli, 1, jpeg_size=2000) is False  # refused, never dropped
    assert cli._rec_refused == 1
    assert cli._rec_captured == 1
    cli._ship_available()                  # drain → auto-resume
    assert cli.recording_blocked is False
    assert _tick(cli, 2) is True


def test_unary_fallback_acks_per_tick(tmp_path):
    class _UnaryStub(_Stub):
        def __init__(self):
            super().__init__([])
            self.unary = 0

        def RecordTicks(self, req, metadata=None, timeout=None):
            raise _FakeRpcError(grpc.StatusCode.UNIMPLEMENTED)

        def RecordTick(self, req, metadata=None, timeout=None):
            self.unary += 1
            return pb.RecordTickResponse(ok=True)

    cli, _ = _client(tmp_path, script=[])
    stub = _UnaryStub()
    cli._stub = stub
    for i in range(3):
        _tick(cli, i)
    cli._ship_available()
    assert stub.unary == 3
    assert cli._rec_sent == 3
    assert cli._spool.pending_count == 0
    assert cli._rec_batch_unsupported is True


# ---------------------------------------------------------------------------
# Dynamic close-drain ceiling (_rec_drain_ceiling_s + _drain_recorder)
# ---------------------------------------------------------------------------

from interlatent.inference.client import controller as _ctrl  # noqa: E402


def test_drain_ceiling_scales_with_pending(monkeypatch):
    monkeypatch.delenv("INTERLATENT_REC_DRAIN_CEILING_S", raising=False)
    # Small backlog: the historical 600s floor holds.
    assert _ctrl._rec_drain_ceiling_s(0) == 600.0
    assert _ctrl._rec_drain_ceiling_s(10 * 1024 * 1024) == 600.0
    # 3 GiB at the assumed 250 KiB/s floor.
    three_gib = 3 * 2**30
    assert _ctrl._rec_drain_ceiling_s(three_gib) == pytest.approx(
        three_gib / (250 * 1024)
    )
    # Env override wins regardless of pending; garbage falls to the formula.
    monkeypatch.setenv("INTERLATENT_REC_DRAIN_CEILING_S", "60")
    assert _ctrl._rec_drain_ceiling_s(three_gib) == 60.0
    monkeypatch.setenv("INTERLATENT_REC_DRAIN_CEILING_S", "banana")
    assert _ctrl._rec_drain_ceiling_s(0) == 600.0


def _start_sender(cli):
    import threading

    cli._rec_thread = threading.Thread(target=cli._rec_loop, daemon=True)
    cli._rec_thread.start()


def test_drain_scaling_log_and_full_drain(tmp_path, monkeypatch, caplog):
    import logging

    # Make a tiny spool exceed the base ceiling so the scaling INFO fires.
    monkeypatch.setattr(_ctrl, "_REC_DRAIN_ASSUMED_MIN_BPS", 1)
    cli, stub = _client(tmp_path, script=["all"])
    for i in range(3):
        _tick(cli, i)
    _start_sender(cli)
    with caplog.at_level(logging.INFO, logger=_ctrl.log.name):
        cli._drain_recorder()
    assert any("ceiling scaled to" in r.getMessage() for r in caplog.records)
    assert cli._rec_unsent_retained == 0
    assert not cli._spool.dir.exists()  # fully drained -> disposed


def test_drain_stall_retains_tail_with_loud_warning(tmp_path, monkeypatch, caplog):
    import logging

    monkeypatch.setattr(_ctrl, "_REC_DRAIN_STALL_S", 0.2)
    cli, stub = _client(tmp_path, script=["err"] * 100)
    for i in range(3):
        _tick(cli, i)
    _start_sender(cli)
    with caplog.at_level(logging.WARNING, logger=_ctrl.log.name):
        cli._drain_recorder()
        cli._log_recording_summary()
    assert cli._rec_unsent_retained == 3
    assert cli._spool.dir.exists()  # tail retained, not disposed
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "stalled" in text
    # The summary names the spool path and the GC consequence.
    assert str(cli._spool.dir) in text
    assert "GC" in text or "re-assigned" in text


def test_drain_env_ceiling_bounds_slow_link(tmp_path, monkeypatch, caplog):
    import logging
    import time as _time

    # Stall detector effectively off; a slow-but-alive stub acks 1 tick
    # per RPC with a delay, so only the (forced tiny) ceiling can end it.
    monkeypatch.setattr(_ctrl, "_REC_DRAIN_STALL_S", 30.0)
    monkeypatch.setenv("INTERLATENT_REC_DRAIN_CEILING_S", "0.4")
    cli, stub = _client(tmp_path, script=[1] * 100)
    real_record_ticks = stub.RecordTicks

    def slow(req, metadata=None, timeout=None):
        _time.sleep(0.15)
        return real_record_ticks(req, metadata=metadata, timeout=timeout)

    stub.RecordTicks = slow
    for i in range(30):
        _tick(cli, i)
    _start_sender(cli)
    with caplog.at_level(logging.WARNING, logger=_ctrl.log.name):
        cli._drain_recorder()
    assert cli._rec_unsent_retained > 0
    assert any("0s ceiling" in r.getMessage() or "ceiling" in r.getMessage()
               for r in caplog.records)
