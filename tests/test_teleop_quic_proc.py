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
import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from interlatent.node.teleop import _quic_ipc
from interlatent.node.teleop import _quic_proc  # safe: its aioquic import is lazy
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


def test_video_roundtrip():
    wire = b"\x00\x0f" + b'{"type":"video"}' + b"\xff\xd8jpegbytes"
    kind, payload = _quic_ipc.parse(_quic_ipc.encode_video("wrist_cam", wire))
    assert kind == _quic_ipc.TYPE_VIDEO
    assert _quic_ipc.parse_video(payload) == ("wrist_cam", wire)


def test_parse_video_garbage_is_none():
    assert _quic_ipc.parse_video(b"") is None
    assert _quic_ipc.parse_video(b"\x05ab") is None  # truncated cam name
    assert _quic_ipc.parse_video(b"\x00rest") is None  # empty cam name
    assert _quic_ipc.parse_video(b"\x03cam") is None  # no wire bytes
    assert _quic_ipc.parse_video(b"\x02\xff\xfexx") is None  # bad utf-8 cam


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
# Video tee: parent gating + framing (preview_due / send_preview)
# ---------------------------------------------------------------------------

def _make_viewer_present(chan, sim, seq: int = 1):
    """Connect and deliver one target datagram so preview presence is live."""
    sim.hello()
    sim.ctrl({"t": "connected"})
    _wait_for(lambda: chan.connected, what="connected")
    sim.targets(seq=seq)
    _wait_for(lambda: chan._last_rx_at > 0.0, what="presence stamp")


def test_preview_due_gating(channel):
    chan, sim, _ = channel
    assert not chan.preview_due()  # not connected yet
    _make_viewer_present(chan, sim)
    assert chan.preview_due()

    # Steady state: a send advances the credit deadline by exactly one
    # period, so the next tick is not due until the period elapses.
    now = time.monotonic()
    chan._next_preview_due = now
    chan.send_preview({"cam": b"\xff\xd8jpeg"}, ts_ns=1_000_000_000)
    assert chan._next_preview_due == pytest.approx(now + chan._preview_period_s)
    assert not chan.preview_due()
    # ...until the deadline passes.
    chan._next_preview_due = 0.0
    assert chan.preview_due()

    # No viewer (no recent target datagrams) → no encode cost.
    chan._last_rx_at = time.monotonic() - 60.0
    assert not chan.preview_due()


def test_preview_deadline_credit_beats_tick_quantization():
    # A 46 ms control tick sampling a 50 ms preview period: naive
    # "now + period" anchoring sends every OTHER tick (~10.9 fps); the
    # credit deadline must hold the configured 20 fps average.
    period, tick = 0.05, 0.046
    next_due, sends, t = 0.0, 0, 0.0
    while t < 10.0:
        if t >= next_due:
            sends += 1
            next_due = qc._advance_preview_deadline(next_due, t, period)
        t += tick
    assert 195 <= sends <= 202


def test_preview_deadline_no_burst_after_gap():
    period = 0.05
    # Deadline long stale (viewer was absent): the send must not schedule
    # the next deadline in the past (catch-up burst)...
    nd = qc._advance_preview_deadline(0.0, now=500.0, period_s=period)
    assert nd == 500.0
    # ...and the following send resumes the normal cadence.
    nd = qc._advance_preview_deadline(nd, now=500.001, period_s=period)
    assert nd == pytest.approx(500.05)


def test_preview_kill_switch(monkeypatch):
    monkeypatch.setenv("INTERLATENT_QUIC_VIDEO", "0")
    spawned: list[FakePopen] = []
    monkeypatch.setattr(
        qc.subprocess, "Popen",
        lambda argv, **kw: spawned.append(FakePopen(argv, **kw)) or spawned[-1],
    )
    chan = QuicTeleopChannel(
        session_id="sess-kill", api_base="http://api.example", api_key="ilat_test",
    )
    chan.start()
    sim = ChildSim(spawned[0].env)
    try:
        _make_viewer_present(chan, sim)
        assert not chan.preview_due()  # switch wins over everything else
    finally:
        sim.close()
        chan.stop()


def test_send_preview_framing(channel):
    chan, sim, _ = channel
    _make_viewer_present(chan, sim)

    jpeg = b"\xff\xd8" + b"j" * 64
    chan.send_preview({"wrist": jpeg}, ts_ns=1_234_567_890)

    # The channel must emit exactly one TYPE_VIDEO datagram for the camera.
    raw = sim.sock.recvfrom(65536)[0]
    kind, payload = _quic_ipc.parse(raw)
    assert kind == _quic_ipc.TYPE_VIDEO
    cam, wire = _quic_ipc.parse_video(payload)
    assert cam == "wrist"
    # Wire layout: uint16-BE header length + JSON header + JPEG verbatim.
    hlen = int.from_bytes(wire[:2], "big")
    header = json.loads(wire[2:2 + hlen].decode("utf-8"))
    assert header["type"] == "video"
    assert header["cam"] == "wrist"
    assert header["seq"] == chan._preview_seq
    assert header["ts_ms"] == 1_234_567_890 // 1_000_000
    assert wire[2 + hlen:] == jpeg


# ---------------------------------------------------------------------------
# Video tee: child governor + TYPE_VIDEO dispatch (aioquic-free)
# ---------------------------------------------------------------------------

class FakeClock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def _governor(finished: set, resets: list, clock: FakeClock):
    return _quic_proc._VideoGovernor(
        now=clock,
        is_finished=lambda sid: sid in finished,
        reset=resets.append,
    )


def test_governor_per_cam_and_global_caps():
    finished: set = set()
    resets: list = []
    gov = _governor(finished, resets, FakeClock())

    assert gov.admit("a")
    gov.note_open(1, "a")
    assert gov.admit("a")
    gov.note_open(2, "a")
    assert not gov.admit("a")  # per-cam cap (2) hit
    assert gov.admit("b")  # other camera unaffected
    gov.note_open(3, "b")

    # Fill to the global cap (6) across cameras.
    for sid, cam in ((4, "b"), (5, "c"), (6, "c")):
        assert gov.admit(cam)
        gov.note_open(sid, cam)
    assert not gov.admit("d")  # global cap hit
    assert gov.dropped_cap == 2

    # Finished streams free their slots.
    finished.update({1, 2})
    assert gov.admit("a")
    assert gov.finished == 2
    assert not resets


def test_governor_ttl_resets_stale_streams():
    finished: set = set()
    resets: list = []
    clock = FakeClock()
    gov = _governor(finished, resets, clock)

    assert gov.admit("a")
    gov.note_open(1, "a")
    clock.t += _quic_proc._VIDEO_STREAM_TTL_S + 0.1
    assert gov.admit("a")  # sweep resets the stale stream, freeing the slot
    assert resets == [1]
    assert gov.reset_ttl == 1


def test_parent_link_video_dispatch():
    link = _quic_proc._ParentLink()
    got: list = []
    link.set_video_sender(lambda cam, wire: got.append((cam, wire)))
    link.datagram_received(
        _quic_ipc.encode_video("cam0", b"WIREBYTES"), ("127.0.0.1", 1)
    )
    assert got == [("cam0", b"WIREBYTES")]
    assert link.rx_video_from_parent == 1

    # No sender (no session up) → dropped silently, not counted.
    link.set_video_sender(None)
    link.datagram_received(
        _quic_ipc.encode_video("cam0", b"MORE"), ("127.0.0.1", 1)
    )
    assert got == [("cam0", b"WIREBYTES")]
    assert link.rx_video_from_parent == 1

    # Garbage payload with a sender set → ignored.
    link.set_video_sender(lambda cam, wire: got.append((cam, wire)))
    link.datagram_received(bytes((_quic_ipc.TYPE_VIDEO,)) + b"\x00", ("127.0.0.1", 1))
    assert got == [("cam0", b"WIREBYTES")]


# ---------------------------------------------------------------------------
# Node-sourced kinematic_spec: framing, request detection, dispatch, serving
# ---------------------------------------------------------------------------

def test_spec_ipc_roundtrip():
    wire = b"\x00\x0f" + b'{"type":"spec"}' + b"specbody"
    kind, payload = _quic_ipc.parse(_quic_ipc.encode_spec(wire))
    assert kind == _quic_ipc.TYPE_SPEC
    assert payload == wire
    assert _quic_ipc.TYPE_SPEC not in (
        _quic_ipc.TYPE_DATA, _quic_ipc.TYPE_CTRL, _quic_ipc.TYPE_VIDEO
    )


def test_frame_spec_wire_envelope():
    spec = {"version": 1, "chains": {"left": {"damping": {"lam_pos": 0.05}}}}
    wire = qc.frame_spec_wire(spec, "nori")
    hlen = int.from_bytes(wire[:2], "big")
    header = json.loads(wire[2:2 + hlen].decode("utf-8"))
    body = json.loads(wire[2 + hlen:].decode("utf-8"))
    assert header == {"type": "spec", "robot_kind": "nori"}
    assert body == spec  # the browser rebuilds its solver from this verbatim


def test_is_request_spec_discriminates():
    assert qc.is_request_spec(b'{"type":"request_spec"}')
    assert qc.is_request_spec(b'{"type":"request_spec","n":3}')
    # A target frame must not be mistaken for a request.
    assert not qc.is_request_spec(
        b'{"mode":"targets","seq":5,"joint_targets":[1,2]}'
    )
    assert not qc.is_request_spec(b'{"mode":"pose","ee_pos":[0,0,0]}')
    # Cheap substring guard: no marker → False, never a parse/raise.
    assert not qc.is_request_spec(b"{bad json, no marker")
    assert not qc.is_request_spec(b"\xff\xfe binary")


def test_parent_link_spec_dispatch():
    link = _quic_proc._ParentLink()
    got: list = []
    link.set_spec_sender(got.append)
    link.datagram_received(_quic_ipc.encode_spec(b"SPECWIRE"), ("127.0.0.1", 1))
    assert got == [b"SPECWIRE"]
    # No sender (no session up) → dropped silently, exactly like video.
    link.set_spec_sender(None)
    link.datagram_received(_quic_ipc.encode_spec(b"MORE"), ("127.0.0.1", 1))
    assert got == [b"SPECWIRE"]


def test_load_spec_wire_none_without_kind():
    chan = QuicTeleopChannel(
        session_id="s", api_base="http://x", api_key="ilat_k"
    )
    assert chan._load_spec_wire() is None


def test_load_spec_wire_none_for_uninstalled_kind():
    # Best-effort: an uninstalled kind logs and returns None (browser then
    # falls back to the platform HTTP spec endpoint) — never raises.
    chan = QuicTeleopChannel(
        session_id="s", api_base="http://x", api_key="ilat_k",
        robot_kind="definitely-not-installed",
    )
    assert chan._load_spec_wire() is None


def _send_request_spec(sim: ChildSim) -> None:
    sim.sock.sendto(
        _quic_ipc.encode_data(json.dumps({"type": "request_spec"}).encode()),
        sim.parent_addr,
    )


def test_request_spec_served_when_loaded(channel, caplog):
    chan, sim, _ = channel
    wire = qc.frame_spec_wire({"version": 1, "chains": {}}, "nori")
    chan._spec_wire = wire
    sim.hello()
    sim.ctrl({"t": "connected"})
    _wait_for(lambda: chan.connected, what="connected")

    with caplog.at_level(logging.INFO):
        _send_request_spec(sim)
        raw = sim.sock.recvfrom(65536)[0]  # the answer, back to the child
        kind, payload = _quic_ipc.parse(raw)
        assert kind == _quic_ipc.TYPE_SPEC
        assert payload == wire
        # The handshake is observable in the node log (once per session).
        _wait_for(lambda: "served kinematic_spec" in caplog.text,
                  what="serve log line")
    # A request must never be decoded as a target frame.
    assert chan.latest_frame() is None


def test_request_spec_throttled(channel):
    chan, sim, _ = channel
    chan._spec_wire = qc.frame_spec_wire({"version": 1}, "nori")
    sim.hello()
    sim.ctrl({"t": "connected"})
    _wait_for(lambda: chan.connected, what="connected")

    _send_request_spec(sim)
    first = _quic_ipc.parse(sim.sock.recvfrom(65536)[0])
    assert first[0] == _quic_ipc.TYPE_SPEC

    # A retry inside the throttle window opens no second stream.
    _send_request_spec(sim)
    sim.sock.settimeout(0.4)
    with pytest.raises(socket.timeout):
        sim.sock.recvfrom(65536)


def test_request_spec_ignored_without_local_spec(channel, caplog):
    chan, sim, _ = channel
    assert chan._spec_wire is None  # nothing loaded
    sim.hello()
    sim.ctrl({"t": "connected"})
    _wait_for(lambda: chan.connected, what="connected")

    with caplog.at_level(logging.INFO):
        _send_request_spec(sim)
        # No spec to serve → no answer, and the request is not mistaken for a
        # target frame. There is no fallback source, so this is fatal for the
        # browser — the node must say so loudly.
        sim.sock.settimeout(0.4)
        with pytest.raises(socket.timeout):
            sim.sock.recvfrom(65536)
        _wait_for(lambda: "WILL NOT START" in caplog.text, what="miss warning")
    assert chan.latest_frame() is None


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


# ---------------------------------------------------------------------------
# Congestion-adaptive preview: PreviewBackoff + vstats wiring
# ---------------------------------------------------------------------------

def test_preview_backoff_pure():
    b = qc.PreviewBackoff()
    base = 0.1  # 10 Hz configured
    assert b.period(base) == pytest.approx(base)

    # A single TTL reset is a stray WiFi burp — the dead band holds.
    b.on_window(stale=1, base_period_s=base)
    assert b.period(base) == pytest.approx(base)
    # Two frames stale in flight in one window -> halve.
    b.on_window(stale=2, base_period_s=base)
    assert b.period(base) == pytest.approx(0.2)

    # A fully fresh window decays /1.25 back toward the configured rate.
    b.on_window(stale=0, base_period_s=base)
    assert b.period(base) == pytest.approx(0.2 / 1.25)
    # Dead band again: exactly one stale frame holds steady.
    b.on_window(stale=1, base_period_s=base)
    assert b.period(base) == pytest.approx(0.2 / 1.25)


def test_preview_backoff_clamp_and_recovery():
    b = qc.PreviewBackoff()
    base = 0.1

    # Backoff clamps exactly at the 1s period ceiling (1 Hz floor rate),
    # never overshooting past it (overshoot would only slow recovery).
    for _ in range(10):
        b.on_window(stale=50, base_period_s=base)
    assert b.period(base) == pytest.approx(1.0)
    assert b.mult == pytest.approx(10.0)  # 1.0s / 0.1s, not 2**n

    # Recovery decays back to the configured rate with no undershoot,
    # in ~10-11 clean windows from the floor (log1.25(10) ~= 10.3).
    clean = 0
    while b.mult > 1.0:
        b.on_window(stale=0, base_period_s=base)
        clean += 1
    assert 10 <= clean <= 12
    assert b.period(base) == pytest.approx(base)


def test_preview_backoff_base_slower_than_floor():
    # A configured rate at/below 1 Hz never gets slowed further.
    b = qc.PreviewBackoff()
    b.on_window(stale=100, base_period_s=1.0)
    assert b.period(1.0) == pytest.approx(1.0)


def _vstats(drop_cap: int, reset_ttl: int = 0, open_ct: int = 0) -> dict:
    return {"t": "vstats", "open": open_ct, "fin": 0, "drop_cap": drop_cap,
            "reset_ttl": reset_ttl, "dg_drop": 0}


def test_vstats_cap_drops_never_back_off(channel):
    # The regression: with the in-flight cap, admission denials
    # (drop_cap) are the pacing mechanism — the timer over-offers, the
    # governor discards the excess for free. Punishing them strangled
    # the timer below the link's completion rate and recovery never
    # fired (a paced link always shows collisions). Any amount of
    # cap-pacing must hold the rate; only TTL resets back off.
    chan, sim, _ = channel
    _make_viewer_present(chan, sim)
    base = chan._preview_period_s

    sim.ctrl(_vstats(drop_cap=0, open_ct=0))           # baseline sample
    sim.ctrl(_vstats(drop_cap=60, open_ct=30))         # heavy cap pacing
    # Give the supervisor thread a beat to process, then assert no backoff.
    time.sleep(0.2)
    assert chan._preview_backoff.mult == pytest.approx(1.0)
    assert chan._effective_preview_period_s() == pytest.approx(base)

    # Two frames going stale in flight (TTL resets) does back off.
    sim.ctrl(_vstats(drop_cap=60, open_ct=60, reset_ttl=2))
    _wait_for(lambda: chan._preview_backoff.mult == pytest.approx(2.0),
              what="backoff on stale frames")


def test_vstats_backoff_and_deadline(channel):
    chan, sim, _ = channel
    _make_viewer_present(chan, sim)
    base = chan._preview_period_s

    sim.ctrl(_vstats(0))                 # baseline sample
    sim.ctrl(_vstats(0, reset_ttl=5))    # +5 stale in one window -> back off
    _wait_for(lambda: chan._effective_preview_period_s()
              == pytest.approx(2 * base), what="backoff engaged")

    # The credit deadline advances by the EFFECTIVE period, not the base.
    now = time.monotonic()
    chan._next_preview_due = now
    chan.send_preview({"cam": b"\xff\xd8jpeg"}, ts_ns=1_000_000_000)
    assert chan._next_preview_due == pytest.approx(now + 2 * base)


def test_vstats_recovery_logs_once(channel, caplog):
    chan, sim, _ = channel
    _make_viewer_present(chan, sim)
    sim.ctrl(_vstats(0))
    sim.ctrl(_vstats(0, reset_ttl=4))
    _wait_for(lambda: chan._preview_backoff.mult > 1.0, what="backoff")
    with caplog.at_level(logging.INFO, logger="interlatent.node.teleop.quic_channel"):
        # Repeated clean windows (cumulative counters unchanged) decay back.
        for _ in range(12):
            sim.ctrl(_vstats(0, reset_ttl=4))
        _wait_for(lambda: chan._preview_backoff.mult == 1.0, what="recovery")
    recovered = [r for r in caplog.records if "recovered" in r.getMessage()]
    assert len(recovered) == 1


def test_vstats_malformed_and_stray_ignored(channel):
    chan, sim, _ = channel
    _make_viewer_present(chan, sim)
    sim.ctrl(_vstats(0))
    sim.ctrl({"t": "vstats", "drop_cap": "many"})  # malformed -> ignored
    sim.ctrl({"t": "bogus", "x": 1})  # unknown type -> ignored
    stray = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        stray.sendto(_quic_ipc.encode_ctrl(_vstats(0, reset_ttl=999)),
                     sim.parent_addr)
        time.sleep(0.2)
        assert chan._preview_backoff.mult == 1.0
    finally:
        stray.close()


def test_vstats_reconnect_resets_baseline(channel):
    chan, sim, _ = channel
    _make_viewer_present(chan, sim)
    sim.ctrl(_vstats(0))
    sim.ctrl(_vstats(0, reset_ttl=100))
    _wait_for(lambda: chan._preview_backoff.mult > 1.0, what="backoff")
    mult = chan._preview_backoff.mult

    # Reconnect: the child's governor counters restart from zero. The
    # first vstats after "connected" must be treated as a baseline (a
    # negative delta), not a spurious extra backoff.
    sim.ctrl({"t": "disconnected", "reason": "test"})
    _wait_for(lambda: not chan.connected, what="disconnected")
    sim.ctrl({"t": "connected"})
    _wait_for(lambda: chan.connected, what="reconnected")
    sim.ctrl(_vstats(0, reset_ttl=2))
    time.sleep(0.2)
    assert chan._preview_backoff.mult == pytest.approx(mult)


def test_vstats_adaptive_kill_switch(monkeypatch):
    monkeypatch.setenv("INTERLATENT_PREVIEW_ADAPTIVE", "0")
    spawned: list[FakePopen] = []
    monkeypatch.setattr(
        qc.subprocess, "Popen",
        lambda argv, **kw: spawned.append(FakePopen(argv, **kw)) or spawned[-1],
    )
    chan = QuicTeleopChannel(
        session_id="sess-noadapt", api_base="http://api.example",
        api_key="ilat_test",
    )
    chan.start()
    sim = ChildSim(spawned[0].env)
    try:
        _make_viewer_present(chan, sim)
        sim.ctrl(_vstats(0))
        sim.ctrl(_vstats(0, reset_ttl=500))
        time.sleep(0.2)
        assert chan._effective_preview_period_s() == chan._preview_period_s
        assert chan._preview_backoff.mult == 1.0
    finally:
        sim.close()
        chan.stop()


def test_vstats_payload_pure():
    assert _quic_proc._vstats_payload(None, None) is None

    class _Gov:
        opened, finished, dropped_cap, reset_ttl = 7, 6, 3, 1

    class _WT:
        def datagrams_dropped(self):
            return 9

    msg = _quic_proc._vstats_payload(_Gov(), _WT())
    assert msg == {"t": "vstats", "open": 7, "fin": 6, "drop_cap": 3,
                   "reset_ttl": 1, "dg_drop": 9}
    # No live WT session (or a broken counter) degrades to 0, never raises.
    assert _quic_proc._vstats_payload(_Gov(), None)["dg_drop"] == 0
    kind, payload = _quic_ipc.parse(_quic_ipc.encode_ctrl(msg))
    assert kind == _quic_ipc.TYPE_CTRL
    assert _quic_ipc.parse_ctrl(payload) == msg


def test_video_inflight_env_parse(monkeypatch):
    # Valid override, clamping, and garbage falling back to the default.
    monkeypatch.setenv("X_INFLIGHT", "4")
    assert _quic_proc._env_int("X_INFLIGHT", 2, 1, 16) == 4
    monkeypatch.setenv("X_INFLIGHT", "99")
    assert _quic_proc._env_int("X_INFLIGHT", 2, 1, 16) == 16
    monkeypatch.setenv("X_INFLIGHT", "0")
    assert _quic_proc._env_int("X_INFLIGHT", 2, 1, 16) == 1
    monkeypatch.setenv("X_INFLIGHT", "many")
    assert _quic_proc._env_int("X_INFLIGHT", 2, 1, 16) == 2
    monkeypatch.delenv("X_INFLIGHT")
    assert _quic_proc._env_int("X_INFLIGHT", 2, 1, 16) == 2


def test_video_governor_honors_raised_cap(monkeypatch):
    monkeypatch.setattr(_quic_proc, "_VIDEO_INFLIGHT_PER_CAM", 4)
    monkeypatch.setattr(_quic_proc, "_VIDEO_INFLIGHT_GLOBAL", 12)
    gov = _quic_proc._VideoGovernor(
        now=lambda: 0.0, is_finished=lambda sid: False, reset=lambda sid: None
    )
    for i in range(4):
        assert gov.admit("cam")
        gov.note_open(i, "cam")
    assert not gov.admit("cam")  # 5th per-cam frame hits the raised cap
    assert gov.dropped_cap == 1
