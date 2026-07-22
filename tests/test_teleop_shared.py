"""Unit tests for the shared teleop building blocks factored out of the WS and
QUIC channels: the env-knob parser + registry (`node/_env.py`), the wire
envelope (`frame.frame_with_header`), the arrival telemetry (`ArrivalTracker`),
and the safety-critical latest-frame + sticky-estop store (`LatestFrameStore`,
ADR 0016). All aioquic-free and network-free."""
from __future__ import annotations

import json
import struct

from interlatent.node import _env
from interlatent.node.teleop._frame_store import LatestFrameStore, _FRAME_STALE_MS
from interlatent.node.teleop._telemetry import ArrivalTracker
from interlatent.node.teleop.frame import TeleopFrame, frame_with_header


# --------------------------------------------------------------------------- #
# _env: clamp-parse + non-default registry                                     #
# --------------------------------------------------------------------------- #

def test_env_int_clamp_and_fallback(monkeypatch):
    monkeypatch.setenv("IL_T_INT", "5")
    assert _env.env_int("IL_T_INT", 2, 1, 16) == 5
    monkeypatch.setenv("IL_T_INT", "999")
    assert _env.env_int("IL_T_INT", 2, 1, 16) == 16   # clamp hi
    monkeypatch.setenv("IL_T_INT", "0")
    assert _env.env_int("IL_T_INT", 2, 1, 16) == 1     # clamp lo
    monkeypatch.setenv("IL_T_INT", "garbage")
    assert _env.env_int("IL_T_INT", 2, 1, 16) == 2     # fallback
    monkeypatch.delenv("IL_T_INT")
    assert _env.env_int("IL_T_INT", 2, 1, 16) == 2     # unset -> default


def test_env_float_and_bool(monkeypatch):
    monkeypatch.setenv("IL_T_HZ", "45")
    assert _env.env_float("IL_T_HZ", 10.0, 1.0, 30.0) == 30.0
    monkeypatch.setenv("IL_T_HZ", "nope")
    assert _env.env_float("IL_T_HZ", 10.0, 1.0, 30.0) == 10.0

    monkeypatch.delenv("IL_T_BOOL", raising=False)
    assert _env.env_bool("IL_T_BOOL", True) is True       # unset -> default
    monkeypatch.setenv("IL_T_BOOL", "0")
    assert _env.env_bool("IL_T_BOOL", True) is False       # "0" -> off
    monkeypatch.setenv("IL_T_BOOL", "1")
    assert _env.env_bool("IL_T_BOOL", False) is True       # non-"0" -> on


def test_env_overrides_reports_only_non_defaults(monkeypatch):
    monkeypatch.setenv("IL_T_OVR", "7")
    _env.env_int("IL_T_OVR", 2, 1, 16)          # overridden
    _env.env_int("IL_T_DEFAULTED", 2, 1, 16)     # unset -> equals default
    ov = _env.overrides()
    assert ov.get("IL_T_OVR") == 7
    assert "IL_T_DEFAULTED" not in ov


# --------------------------------------------------------------------------- #
# frame_with_header: the shared length-prefixed wire envelope                   #
# --------------------------------------------------------------------------- #

def test_frame_with_header_roundtrip():
    wire = frame_with_header({"type": "video", "cam": "wrist"}, b"\xff\xd8JPEG")
    hlen = struct.unpack(">H", wire[:2])[0]
    header = json.loads(wire[2:2 + hlen].decode("utf-8"))
    assert header == {"type": "video", "cam": "wrist"}
    assert wire[2 + hlen:] == b"\xff\xd8JPEG"


def test_frame_with_header_empty_body():
    wire = frame_with_header({"type": "spec"}, b"")
    hlen = struct.unpack(">H", wire[:2])[0]
    assert wire[2 + hlen:] == b""


# --------------------------------------------------------------------------- #
# ArrivalTracker: gap accounting + windowed summary                            #
# --------------------------------------------------------------------------- #

class _Frame:
    def __init__(self, seq: int, received_at_ns: int) -> None:
        self.seq = seq
        self.received_at_ns = received_at_ns


def test_arrival_tracker_windowed_summary():
    import time

    # Large window so the wall clock never fires it on its own; we force the
    # window boundary via _window_started to keep the gap math deterministic.
    t = ArrivalTracker(window_s=1000.0)
    assert t.note(_Frame(seq=10, received_at_ns=0)) is None            # first
    assert t.note(_Frame(seq=13, received_at_ns=20_000_000)) is None   # +20ms
    t._window_started = time.monotonic() - 2000.0  # force the window to elapse
    s = t.note(_Frame(seq=15, received_at_ns=25_000_000))              # +5ms
    assert s is not None
    assert s["n"] == 3
    assert s["gap_mean_ms"] == 12.5     # (20 + 5) / 2 gaps
    assert s["gap_max_ms"] == 20.0
    assert s["seq_span"] == 5           # 15 - 10
    # Window reset: the next note starts a fresh count.
    assert t.note(_Frame(seq=16, received_at_ns=30_000_000)) is None


def test_arrival_tracker_holds_until_window_elapses():
    t = ArrivalTracker(window_s=3600.0)  # effectively never within the test
    for i in range(5):
        assert t.note(_Frame(seq=i, received_at_ns=i * 1_000_000)) is None


# --------------------------------------------------------------------------- #
# LatestFrameStore: staleness + sticky estop (ADR 0016)                        #
# --------------------------------------------------------------------------- #

def _frame(estop: bool = False, age_ms: float = 0.0) -> TeleopFrame:
    import time
    return TeleopFrame(
        engaged=True, deadman=True, seq=1,
        received_at_ns=time.monotonic_ns() - int(age_ms * 1e6),
        estop=estop,
    )


def test_store_latest_frame_freshness():
    store = LatestFrameStore()
    assert store.latest_frame() is None
    store._store_frame(_frame(age_ms=0.0))
    assert store.latest_frame() is not None
    # A stale frame is treated as absent.
    store._store_frame(_frame(age_ms=_FRAME_STALE_MS + 50))
    assert store.latest_frame() is None


def test_sticky_estop_survives_frame_drop_and_consumes_once():
    store = LatestFrameStore()
    store._store_frame(_frame(estop=True))
    store._latch_estop()
    store._drop_frame()                 # disconnect / staleness drop
    assert store.latest_frame() is None  # frame gone...
    assert store.consume_estop() is True  # ...but the latch survived
    assert store.consume_estop() is False  # cleared exactly once
