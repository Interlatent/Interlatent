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
