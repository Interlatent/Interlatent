"""QUIC child-process isolation for the node teleop channel.

Covers the three offline-testable layers of the design (ADR 0017 amendment):
the ``_quic_ipc`` loopback framing, ``QuicTeleopChannel``'s child supervision
and datagram handling (with a fake child played by the test over a real
loopback UDP socket), and a real ``-m interlatent.node.teleop._quic_proc``
subprocess smoke test (hello heartbeat + stdin-EOF exit). No aioquic and no
network needed; the live relay path is signed off on-robot.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from interlatent.node.teleop import _quic_ipc
from interlatent.node.teleop import quic_channel as qc
from interlatent.node.teleop.quic_channel import QuicTeleopChannel

_SRC_DIR = Path(__file__).resolve().parent.parent / "packages" / "sdk" / "src"


# ---------------------------------------------------------------------------
# _quic_ipc framing
# ---------------------------------------------------------------------------

def test_data_roundtrip():
    payload = b'{"mode": "targets"}'
    kind, out = _quic_ipc.parse(_quic_ipc.encode_data(payload))
    assert kind == _quic_ipc.TYPE_DATA
    assert out == payload


def test_ctrl_roundtrip():
    obj = {"t": "hello", "cookie": "abc", "pid": 7}
    kind, out = _quic_ipc.parse(_quic_ipc.encode_ctrl(obj))
    assert kind == _quic_ipc.TYPE_CTRL
    assert _quic_ipc.parse_ctrl(out) == obj


def test_parse_empty_is_none():
    assert _quic_ipc.parse(b"") is None


def test_parse_unknown_type_returned_for_caller_to_ignore():
    kind, payload = _quic_ipc.parse(bytes((0x7F,)) + b"junk")
    assert kind == 0x7F
    assert payload == b"junk"


def test_parse_ctrl_garbage_is_none():
    assert _quic_ipc.parse_ctrl(b"\xff\xfe not json") is None
    assert _quic_ipc.parse_ctrl(b'"a bare string"') is None


# ---------------------------------------------------------------------------
# Parent supervision + datagram handling (fake child)
# ---------------------------------------------------------------------------

class FakeStdin:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakePopen:
    """Stands in for the child process; the test itself plays the child over
    UDP using the cookie/port captured from the spawn env."""

    def __init__(self, argv, stdin=None, stdout=None, stderr=None, env=None):
        self.argv = argv
        self.env = dict(env or {})
        self.pid = 4242
        self.returncode = None
        self.stdin = FakeStdin()

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9


class ChildSim:
    """The test's end of the loopback pipe: a UDP socket that speaks the
    _quic_ipc protocol at the parent, as the real child would."""

    def __init__(self, spawn_env: dict) -> None:
        self.cookie = spawn_env[_quic_ipc.ENV_COOKIE]
        self.parent_addr = ("127.0.0.1", int(spawn_env[_quic_ipc.ENV_PARENT_PORT]))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.settimeout(1.0)

    def hello(self, cookie: str | None = None) -> None:
        self.sock.sendto(
            _quic_ipc.encode_ctrl(
                {"t": "hello", "cookie": cookie or self.cookie, "pid": 1}
            ),
            self.parent_addr,
        )

    def ctrl(self, obj: dict) -> None:
        self.sock.sendto(_quic_ipc.encode_ctrl(obj), self.parent_addr)

    def targets(self, seq: int) -> None:
        frame = {"mode": "targets", "engaged": True, "seq": seq,
                 "joint_targets": [0.1, 0.2, float(seq)]}
        self.sock.sendto(
            _quic_ipc.encode_data(json.dumps(frame).encode()), self.parent_addr
        )

    def recv_data(self):
        kind, payload = _quic_ipc.parse(self.sock.recvfrom(4096)[0])
        assert kind == _quic_ipc.TYPE_DATA
        return json.loads(payload.decode())

    def close(self) -> None:
        self.sock.close()


def _wait_for(pred, timeout: float = 3.0, what: str = "condition"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return
        time.sleep(0.01)
    pytest.fail(f"timed out waiting for {what}")


@pytest.fixture
def channel(monkeypatch):
    """A started channel whose child is a FakePopen; yields (channel, sim)."""
    spawned: list[FakePopen] = []

    def fake_popen(argv, **kwargs):
        proc = FakePopen(argv, **kwargs)
        spawned.append(proc)
        return proc

    monkeypatch.setattr(qc.subprocess, "Popen", fake_popen)
    chan = QuicTeleopChannel(
        session_id="sess-quic-test", api_base="http://api.example",
        api_key="ilat_test",
    )
    chan.start()
    assert spawned, "start() must spawn the child"
    sim = ChildSim(spawned[0].env)
    yield chan, sim, spawned
    sim.close()
    chan.stop()


def test_spawn_env_contract(channel):
    _, _, spawned = channel
    env = spawned[0].env
    assert spawned[0].argv[-1] == "interlatent.node.teleop._quic_proc"
    assert env[_quic_ipc.ENV_API_BASE] == "http://api.example"
    assert env[_quic_ipc.ENV_API_KEY] == "ilat_test"
    assert env[_quic_ipc.ENV_SESSION_ID] == "sess-quic-test"
    assert env[_quic_ipc.ENV_TOKEN_PATH].endswith("/teleop-token")
    assert len(env[_quic_ipc.ENV_COOKIE]) == 32


def test_hello_connected_targets_flow(channel):
    chan, sim, _ = channel
    assert not chan.connected
    sim.hello()
    sim.ctrl({"t": "connected"})
    _wait_for(lambda: chan.connected, what="connected")

    sim.targets(seq=5)
    _wait_for(lambda: chan.latest_frame() is not None, what="target frame")
    frame = chan.latest_frame()
    assert frame.mode == "targets" and frame.seq == 5


def test_seq_dedupe_latest_wins(channel):
    chan, sim, _ = channel
    sim.hello()
    sim.ctrl({"t": "connected"})
    _wait_for(lambda: chan.connected, what="connected")

    sim.targets(seq=5)
    _wait_for(lambda: chan._latest is not None and chan._latest.seq == 5,
              what="seq 5")
    sim.targets(seq=4)  # late/older duplicate must not clobber
    sim.targets(seq=5)
    time.sleep(0.1)
    assert chan._latest.seq == 5
    sim.targets(seq=6)
    _wait_for(lambda: chan._latest.seq == 6, what="seq 6")


def test_send_state_duplicated_only_while_connected(channel):
    chan, sim, _ = channel
    sim.hello()
    time.sleep(0.1)  # let the hello pin the child addr

    # Not connected yet: nothing must arrive.
    chan.send_state([1.0, 2.0])
    with pytest.raises(socket.timeout):
        sim.sock.recvfrom(4096)

    sim.ctrl({"t": "connected"})
    _wait_for(lambda: chan.connected, what="connected")
    chan._last_state_sent_at = 0.0  # defeat the 15 Hz pacing gate
    chan.note_applied(41)
    chan.send_state([1.0, 2.0])
    msgs = [sim.recv_data() for _ in range(2)]  # _STATE_DUP copies
    assert all(m["type"] == "state" for m in msgs)
    assert msgs[0] == msgs[1]
    assert msgs[0]["qpos"] == [1.0, 2.0]
    assert msgs[0]["applied_seq"] == 41


def test_stray_sender_ignored(channel):
    chan, sim, _ = channel
    sim.hello()
    sim.ctrl({"t": "connected"})
    _wait_for(lambda: chan.connected, what="connected")

    stray = ChildSim({  # fresh socket aimed at the same parent, wrong cookie
        _quic_ipc.ENV_COOKIE: "0" * 32,
        _quic_ipc.ENV_PARENT_PORT: str(sim.parent_addr[1]),
    })
    try:
        stray.hello()  # wrong cookie → must not re-pin the child addr
        stray.targets(seq=99)
        stray.ctrl({"t": "disconnected", "reason": "spoof"})
        time.sleep(0.2)
        assert chan.connected  # spoofed disconnect ignored
        assert chan._latest is None or chan._latest.seq != 99
    finally:
        stray.close()


def test_disconnected_clears_state(channel):
    chan, sim, _ = channel
    sim.hello()
    sim.ctrl({"t": "connected"})
    _wait_for(lambda: chan.connected, what="connected")
    sim.targets(seq=7)
    _wait_for(lambda: chan._latest is not None, what="target frame")

    sim.ctrl({"t": "disconnected", "reason": "relay gone"})
    _wait_for(lambda: not chan.connected, what="disconnected")
    assert chan.latest_frame() is None
    assert chan._latest is None  # stale engaged frame must not survive


def test_child_exit_respawns_with_backoff_reset_on_hello(channel):
    chan, sim, spawned = channel
    sim.hello()
    _wait_for(lambda: chan._child_addr is not None, what="hello processed")

    chan._spawn_backoff = 0.05  # keep the test fast
    spawned[0].returncode = 1  # child "crashes"
    _wait_for(lambda: len(spawned) >= 2, what="respawn")
    assert not chan.connected
    assert chan._latest is None
    assert chan._spawn_backoff == pytest.approx(0.1)  # doubled after exit

    # A hello from the (new) child proves it runs → backoff resets.
    sim2 = ChildSim(spawned[1].env)
    try:
        sim2.hello()
        _wait_for(lambda: chan._spawn_backoff == qc._RECONNECT_INITIAL_S,
                  what="backoff reset")
    finally:
        sim2.close()


def test_stop_closes_stdin_and_joins(channel):
    chan, sim, spawned = channel
    chan.stop()
    assert spawned[0].stdin.closed
    assert chan._thread is None
    assert chan._proc is None
    assert chan._sock is None
    chan.stop()  # idempotent (daemon calls it from two paths)


# ---------------------------------------------------------------------------
# Real subprocess smoke (offline-safe: mint fails forever, that's fine)
# ---------------------------------------------------------------------------

def test_quic_proc_hello_and_stdin_eof_exit():
    parent = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    parent.bind(("127.0.0.1", 0))
    parent.settimeout(10.0)
    cookie = "c0ffee" * 5 + "aa"
    env = {
        **os.environ,
        _quic_ipc.ENV_PARENT_PORT: str(parent.getsockname()[1]),
        _quic_ipc.ENV_COOKIE: cookie,
        _quic_ipc.ENV_API_BASE: "http://127.0.0.1:9",  # mint fails: unroutable
        _quic_ipc.ENV_API_KEY: "ilat_test",
        _quic_ipc.ENV_SESSION_ID: "sess-smoke",
        _quic_ipc.ENV_TOKEN_PATH: "/api/v1/inference/sessions/sess-smoke/teleop-token",
        "PYTHONPATH": str(_SRC_DIR) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "interlatent.node.teleop._quic_proc"],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        env=env,
    )
    try:
        kind, payload = _quic_ipc.parse(parent.recvfrom(4096)[0])
        assert kind == _quic_ipc.TYPE_CTRL
        msg = _quic_ipc.parse_ctrl(payload)
        assert msg["t"] == "hello"
        assert msg["cookie"] == cookie
        assert msg["pid"] == proc.pid

        proc.stdin.close()  # EOF → child must exit promptly
        assert proc.wait(timeout=5.0) == 0
    finally:
        parent.close()
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5.0)
        else:
            proc.stderr.read()  # drain
        proc.stderr.close()
