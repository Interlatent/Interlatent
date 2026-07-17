"""Operator e-stop ingress (ADR 0016): frame field, sticky channel latch,
SafetyGate latch semantics. No sockets — the channels' receive machinery is
exercised at the decode seam their run loops share."""
from __future__ import annotations

import numpy as np
import pytest

from interlatent.node.teleop.frame import TeleopFrame
from interlatent.node.teleop.robot_profile import get_profile
from interlatent.node.teleop.safety import SafetyGate, TargetSample


# --------------------------------------------------------------------------- #
# Frame parsing (additive, back-compat)                                        #
# --------------------------------------------------------------------------- #


def test_frame_estop_absent_defaults_false():
    frame = TeleopFrame.from_json('{"engaged": true, "seq": 1}')
    assert frame is not None and frame.estop is False


def test_frame_estop_parses_true():
    frame = TeleopFrame.from_json('{"engaged": false, "estop": true, "seq": 2}')
    assert frame is not None and frame.estop is True


def test_legacy_frames_unchanged():
    # The pre-estop overlay wire shape must decode exactly as before.
    frame = TeleopFrame.from_json(
        '{"engaged": true, "deadman": true, "held_keys": ["w"], "seq": 3}'
    )
    assert frame.engaged and frame.deadman and frame.held_keys == {"w"}
    assert frame.mode == "keys" and frame.estop is False


# --------------------------------------------------------------------------- #
# Sticky channel latch (WS + QUIC share the semantics)                         #
# --------------------------------------------------------------------------- #


def _ws_channel():
    from interlatent.node.teleop.channel import TeleopChannel

    # Construct without connecting; we drive the decode seam directly.
    return TeleopChannel.__new__(TeleopChannel)


def test_ws_channel_sticky_latch_survives_frame_drop():
    import threading

    from interlatent.node.teleop.channel import TeleopChannel

    ch = TeleopChannel.__new__(TeleopChannel)
    ch._lock = threading.Lock()
    ch._latest = None
    ch._estop_seen = False

    # Simulate the receive path: estop frame decoded, then the disconnect
    # handler drops _latest (the stale-frame rule would do the same).
    frame = TeleopFrame.from_json('{"estop": true, "seq": 9}')
    with ch._lock:
        ch._latest = frame
        if frame.estop:
            ch._estop_seen = True
    with ch._lock:
        ch._latest = None  # disconnect drop — must NOT clear the latch

    assert ch.consume_estop() is True, "estop lost to the frame drop"
    assert ch.consume_estop() is False, "consume must clear exactly once"


def test_quic_channel_estop_latches_before_dedupe():
    quic = pytest.importorskip(
        "interlatent.node.teleop.quic_channel",
        reason="quic channel imports optional deps",
    )
    import threading

    ch = quic.QuicTeleopChannel.__new__(quic.QuicTeleopChannel)
    ch._lock = threading.Lock()
    ch._latest = None
    ch._estop_seen = False
    ch._dedup = quic.LatestSeqBuffer()
    ch._last_rx_at = 0.0
    ch._child_addr = ("127.0.0.1", 1)

    from interlatent.node.teleop import _quic_ipc

    ch._dedup.accept(100)  # a newer frame already won the latest-wins race
    payload = b'{"estop": true, "seq": 50}'  # late duplicate estop datagram
    ch._handle_datagram(_quic_ipc.encode_data(payload), ("127.0.0.1", 1))

    assert ch.latest_frame() is None, "deduped frame must not become latest"
    assert ch.consume_estop() is True, "late-dupe estop must still latch"


# --------------------------------------------------------------------------- #
# SafetyGate latch: step() idles until an explicit clear                       #
# --------------------------------------------------------------------------- #


def test_gate_latch_idles_until_cleared():
    profile = get_profile("nori")
    gate = SafetyGate(profile=profile, control_dt=1 / 30)
    current = np.zeros(len(profile.joint_names), dtype=np.float32)
    target = np.full_like(current, 10.0)

    gate.submit(TargetSample(
        joints=target, deadman_active=True, confidence=1.0,
        received_at=1e9, producer_timestamp_ns=1,
    ))
    gate.latch_estop("teleop_frame")
    commanded, status = gate.step(current, now=1e9)
    assert status == "estop_latched"
    np.testing.assert_allclose(commanded, current)  # hold, not move

    gate.clear_estop()
    commanded, status = gate.step(current, now=1e9)
    assert status == "ok"
    assert np.any(commanded != current), "post-clear step should move again"
