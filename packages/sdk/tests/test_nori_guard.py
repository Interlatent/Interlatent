"""Unit tests for ``NoriNativeRobot.pre_tick`` — the per-tick episode guard.

The guard collapses the three pre-arbitration rungs the old native loop
carried (session death, the daemon safety FSM, telemetry staleness) into one
``TickVerdict``. It is pure disclosure → verdict, so it is tested unbound
against a stub carrying the same properties: no daemon, no sockets.

The healthy path's equivalence with the old loop was proven by the (since
retired) frozen-copy harness; this suite covers the verdicts themselves,
including the startup-recovery window (an idle daemon is ALWAYS
watchdog-stopped when a session begins — the keep-alive pump must get a
bounded grace period to feed it back to ok).
"""
from __future__ import annotations

import time
from typing import Optional

import pytest

from interlatent.adapters.nori import robot as nori_robot
from interlatent.adapters.nori.robot import NoriNativeRobot, _STARTUP_RECOVERY_S
from interlatent.node.movement import TickVerdict


class _Stub:
    """Carries exactly the state pre_tick reads."""

    def __init__(
        self,
        *,
        session_dead: bool = False,
        dead_reason: Optional[str] = None,
        status: Optional[dict] = None,
        telemetry_fresh: bool = True,
        obs_age_ms: float = 0.0,
    ):
        self.session_dead = session_dead
        self.dead_reason = dead_reason
        self.last_status = status if status is not None else {
            "safety": "ok", "watchdog": "ok",
        }
        self.telemetry_fresh = telemetry_fresh
        self.obs_age_ms = obs_age_ms
        # Guard state, exactly as NoriNativeRobot.__init__ seeds it.
        self._guard_t0 = None
        self._guard_was_healthy = False
        self._guard_stale_warned = False


def _tick(stub) -> TickVerdict:
    return NoriNativeRobot.pre_tick(stub, obs={})


def test_healthy_daemon_proceeds_and_is_remembered():
    stub = _Stub()
    assert _tick(stub) is TickVerdict.PROCEED
    assert stub._guard_was_healthy is True, (
        "health must be latched so a later safe-stop reads as a mid-session "
        "stream break, not an idle daemon"
    )


def test_session_death_ends_the_episode():
    stub = _Stub(session_dead=True, dead_reason="fatal: bye")
    assert _tick(stub) is TickVerdict.END_EPISODE


def test_daemon_latch_is_a_hard_episode_boundary():
    """safety=latched → END_EPISODE regardless of history; clearing the latch
    is a human act (`interlatent-act --robot nori --reset-latch`)."""
    for was_healthy in (False, True):
        stub = _Stub(status={"safety": "latched", "watchdog": "ok",
                             "latch_reason": "estop_command"})
        stub._guard_was_healthy = was_healthy
        assert _tick(stub) is TickVerdict.END_EPISODE


def test_startup_stop_holds_within_the_recovery_window(monkeypatch):
    """An idle daemon rests at safe_hold/watchdog-stop; the guard holds (no
    motion, no capture) while the keep-alive pump revives it — bounded."""
    clock = {"t": 1000.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["t"])

    stub = _Stub(status={"safety": "safe_hold", "watchdog": "stop"})
    assert _tick(stub) is TickVerdict.HOLD_NO_CAPTURE
    clock["t"] += _STARTUP_RECOVERY_S - 1.0
    assert _tick(stub) is TickVerdict.HOLD_NO_CAPTURE, (
        "still inside the recovery window"
    )
    clock["t"] += 2.0
    assert _tick(stub) is TickVerdict.END_EPISODE, (
        "keep-alives are not reviving the daemon — the session is unstartable"
    )


def test_safe_stop_after_health_is_a_hard_boundary():
    """Once the session was healthy, a safe-stop means the control-frame
    stream broke mid-session: end the episode (a new session recovers it)."""
    stub = _Stub()
    assert _tick(stub) is TickVerdict.PROCEED
    stub.last_status = {"safety": "safe_hold", "watchdog": "ok"}
    assert _tick(stub) is TickVerdict.END_EPISODE


def test_watchdog_stop_alone_counts_as_stopped():
    stub = _Stub(status={"safety": "ok", "watchdog": "stop"})
    assert _tick(stub) is TickVerdict.HOLD_NO_CAPTURE, (
        "watchdog=stop without safety=safe_hold is still a stopped daemon"
    )


def test_stale_telemetry_holds_then_recovers():
    stub = _Stub()
    assert _tick(stub) is TickVerdict.PROCEED

    stub.telemetry_fresh = False
    stub.obs_age_ms = 750.0
    assert _tick(stub) is TickVerdict.HOLD_NO_CAPTURE
    assert _tick(stub) is TickVerdict.HOLD_NO_CAPTURE

    stub.telemetry_fresh = True
    assert _tick(stub) is TickVerdict.PROCEED
    assert stub._guard_stale_warned is False, "the one-shot warning re-arms"


def test_full_startup_recovery_sequence(monkeypatch):
    """stopped (idle) → ok (pump revived it) → PROCEED; a later stop is then a
    hard boundary because the session had been healthy."""
    clock = {"t": 50.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["t"])

    stub = _Stub(status={"safety": "safe_hold", "watchdog": "stop"})
    assert _tick(stub) is TickVerdict.HOLD_NO_CAPTURE

    clock["t"] += 2.0
    stub.last_status = {"safety": "ok", "watchdog": "warn"}
    assert _tick(stub) is TickVerdict.PROCEED
    assert stub._guard_was_healthy is True

    stub.last_status = {"safety": "safe_hold", "watchdog": "stop"}
    assert _tick(stub) is TickVerdict.END_EPISODE


def test_missing_status_block_proceeds_when_fresh():
    """No telemetry.status yet (very first ticks): neither latched nor
    stopped — drive, gated only on telemetry freshness."""
    stub = _Stub(status=None)
    # last_status None means (st or {}).get(...) sees nothing.
    stub.last_status = None
    assert _tick(stub) is TickVerdict.PROCEED
    assert stub._guard_was_healthy is False, (
        "no status is not evidence of health"
    )


def test_guard_module_constant_matches_hardware_verified_window():
    # The 10 s figure was verified on hardware 2026-07-10 (see robot.py); a
    # drive-by "tidy" of the constant should have to face this assertion.
    assert nori_robot._STARTUP_RECOVERY_S == pytest.approx(10.0)
