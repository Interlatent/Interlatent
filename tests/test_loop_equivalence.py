"""Tick-for-tick equivalence: the migrated lerobot loop vs its frozen ancestor.

``tests/test_loop_contract.py`` asserts *intent* — invariants every loop owes
regardless of history. This suite asserts *sameness*: the loop that runs on
``looprunner.run_control_loop`` + ``CommandBus.drive()`` must do exactly what
the pre-migration inline loop (frozen at ``tests/_frozen/``) did, in the same
order, with the same numbers. That catches the one failure class the contract
suite cannot: side effects that all still happen but got silently *reordered*
when the branch bodies were folded into ``drive()`` — precisely the risk the
old ``movement.py`` docstring warned about when it left them in the loop.

Both sides run against the same doubles and the same monkeypatched trace
points, so the comparison is an ordered event stream plus the actual numeric
actions sent and captured. Determinism holds because both loops pass their own
``loop_start`` as the gate's ``now`` *and* the submitted sample's
``received_at``, so the gate's frame-age is identically zero on either side;
everything else on the motion path (Butterworth filter, delta clamp, calib
coercion) is pure arithmetic.

The robot double here deliberately has no ``estop()`` and no Nori surface:
real lerobot robots expose neither, and giving the fake an ``estop()`` would
make the *new* side forward a hardware latch the old loop never did — a
difference that exists only for robots that cannot occur on this path.
"""
from __future__ import annotations


import numpy as np
import pytest

from _frozen.lerobot_loop_pre_bus import lerobot_control_loop as frozen_loop
from test_loop_contract import FakeChannel, FakeClient, Frame, Trace

from interlatent.node import control as _ctrl
from interlatent.node.teleop.robot_profile import get_profile

_ROBOT_KIND = "so101"
_FPS = 100

_PROFILE = get_profile(_ROBOT_KIND)
assert _PROFILE is not None, "equivalence scenarios need the so101 teleop profile"
_ACTION_KEYS = [f"{n}.pos" for n in _PROFILE.joint_names]
_N = len(_ACTION_KEYS)


class EquivRobot:
    """Lerobot-shaped robot double: marks tick boundaries, records sends.

    Unlike the contract suite's ``FakeRobot`` it exposes neither ``estop()``
    nor the Nori daemon surface — see the module docstring.
    """

    def __init__(self, trace: Trace):
        self._trace = trace

    @property
    def action_features(self) -> list[str]:
        return list(_ACTION_KEYS)

    @property
    def joint_specs(self):
        return []

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        self._trace.events.append("disconnect")

    def get_observation(self) -> dict:
        # One get_observation per tick in both generations, so this event is
        # the tick delimiter the comparison segments on.
        self._trace.events.append("tick")
        return {k: 0.0 for k in _ACTION_KEYS}

    def send_action(self, action) -> None:
        self._trace.events.append("send")
        self._trace.sends.append(dict(action) if isinstance(action, dict) else action)


# Routes the shared monkeypatched trace points (which are installed once per
# test, but must record into whichever side is currently running) to the live
# side's doubles.
_CURRENT: dict = {"trace": None, "robot": None}


def _install_patches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Trace capture/clamp/gate for whichever side is running, via _CURRENT."""
    monkeypatch.setattr(
        _ctrl, "_make_lerobot_robot", lambda *a, **k: _CURRENT["robot"]
    )

    def _capture(_client, _obs, action, _step, *, control_source=None):
        t: Trace = _CURRENT["trace"]
        t.events.append("capture:%s" % (control_source or "policy"))
        t.captures.append((control_source or "policy", np.asarray(action).copy()))
        return list(_ACTION_KEYS)

    monkeypatch.setattr(_ctrl, "_capture_tick", _capture)
    monkeypatch.setattr(_ctrl, "_report_robot_features", lambda *a, **k: True)

    real_clamp = _ctrl._clamp_action_delta

    def _clamp(action, actual, max_step, keys, step, *, source):
        _CURRENT["trace"].events.append("clamp:%s" % source)
        return real_clamp(action, actual, max_step, keys, step, source=source)

    monkeypatch.setattr(_ctrl, "_clamp_action_delta", _clamp)

    from interlatent.node.teleop import safety as _safety

    real_step = _safety.SafetyGate.step
    real_latch = _safety.SafetyGate.latch_estop

    def _gate_step(self, current_joints, now=None):
        _CURRENT["trace"].events.append("gate.step")
        return real_step(self, current_joints, now=now)

    def _gate_latch(self, reason):
        _CURRENT["trace"].events.append("gate.latch")
        return real_latch(self, reason)

    monkeypatch.setattr(_safety.SafetyGate, "step", _gate_step)
    monkeypatch.setattr(_safety.SafetyGate, "latch_estop", _gate_latch)

    class _NullProfiler:
        def __init__(self, **_):
            pass

        def record_tick(self, **_):
            pass

        def close(self):
            pass

    # control.py binds the class at module import (`from .teleop_profiler
    # import NodeTeleopProfiler`), so patch the *control-module* binding — both
    # generations resolve it there at call time.
    monkeypatch.setattr(_ctrl, "NodeTeleopProfiler", _NullProfiler)


def _run_side(loop_fn, *, frames, policy_enabled: bool, ticks: int,
              client_action: bool) -> Trace:
    trace = Trace()
    robot = EquivRobot(trace)
    client = FakeClient(trace, _N, action=client_action)
    _CURRENT["trace"] = trace
    _CURRENT["robot"] = robot

    remaining = {"n": ticks}

    def should_stop() -> bool:
        if remaining["n"] <= 0:
            return True
        remaining["n"] -= 1
        return False

    loop_fn(
        client=client,
        session={"id": "equiv-session", "fps": _FPS},
        should_stop=should_stop,
        robot_kind=_ROBOT_KIND,
        robot_port=None,
        robot_extra={"max_step": "5.0"},
        robot_cameras={},
        api_key="k",
        api_base="http://localhost",
        teleop_channel=FakeChannel(list(frames)),
        node_id="node-1",
        image_resize=None,
        bypass_key=None,
        policy_enabled=policy_enabled,
    )
    return trace


def _send_vector(sent) -> np.ndarray:
    """A send as an ordered float vector, whether coerce produced dict or array."""
    if isinstance(sent, dict):
        return np.asarray([sent[k] for k in _ACTION_KEYS], dtype=np.float64)
    return np.asarray(sent, dtype=np.float64).reshape(-1)


def _assert_equivalent(old: Trace, new: Trace, scenario: str) -> None:
    assert old.events == new.events, (
        "[%s] ordered event streams diverge.\n  frozen : %r\n  runner : %r"
        % (scenario, old.events, new.events)
    )
    assert old.step_calls == new.step_calls, scenario
    assert len(old.sends) == len(new.sends), scenario
    for i, (a, b) in enumerate(zip(old.sends, new.sends)):
        np.testing.assert_allclose(
            _send_vector(a), _send_vector(b), atol=1e-6,
            err_msg="[%s] send #%d differs" % (scenario, i),
        )
    assert old.captured_sources() == new.captured_sources(), scenario
    for i, ((_, a), (_, b)) in enumerate(zip(old.captures, new.captures)):
        np.testing.assert_allclose(
            np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64),
            atol=1e-6, err_msg="[%s] capture #%d differs" % (scenario, i),
        )


def _engaged(**kw) -> Frame:
    kw.setdefault("joint_targets", [0.1] * _N)
    return Frame(**kw)


# Each scenario is (frames, policy_enabled, client_action). Frames are rebuilt
# per side via the factory so the two runs share no mutable state.
_SCENARIOS = {
    # The highest-traffic path: pure policy inference, no producer connected.
    "pure_policy": (lambda: [None] * 6, True, True),
    # DRTC cold start / RTC cooldown: client.step() yields no action yet.
    "policy_no_action": (lambda: [None] * 4, True, False),
    # TeleopRecording assignment, operator never engages: all-hold episode.
    "teleop_recording_hold": (lambda: [None] * 4, False, True),
    # Operator drives continuously.
    "teleop_engaged": (lambda: [_engaged()] * 6, True, True),
    # Engage → release → policy resumes (the smoother-reset / flush handback).
    "teleop_handback": (
        lambda: [_engaged(), _engaged(), None, None, None], True, True,
    ),
    # E-stop mid-drive: must latch on tick 1 and stay latched (level, not edge).
    "estop_mid_run": (
        lambda: [_engaged(), _engaged(estop=True), _engaged(), _engaged()],
        True, True,
    ),
    # TeleopRecording with engage gaps: hold ↔ teleop transitions, gate resets.
    "teleop_recording_gaps": (
        lambda: [None, _engaged(), _engaged(), None], False, True,
    ),
    # Malformed frame (wrong target arity): teleop still owns the tick but the
    # gate idles toward the measured pose.
    "malformed_target_len": (
        lambda: [_engaged(joint_targets=[0.1] * (_N + 1))] * 3, True, True,
    ),
    # A pose-mode frame that escaped pod-side retargeting: hold pose + one-shot
    # warning (ADR 0009, second amendment).
    "pose_mode_frame": (
        lambda: [_engaged(mode="pose", joint_targets=None)] * 3, True, True,
    ),
}


@pytest.mark.parametrize("scenario", sorted(_SCENARIOS), ids=str)
def test_runner_loop_matches_frozen_loop(scenario, monkeypatch):
    frames_fn, policy_enabled, client_action = _SCENARIOS[scenario]
    _install_patches(monkeypatch)

    ticks = len(frames_fn())
    old = _run_side(
        frozen_loop, frames=frames_fn(), policy_enabled=policy_enabled,
        ticks=ticks, client_action=client_action,
    )
    new = _run_side(
        _ctrl.lerobot_control_loop, frames=frames_fn(),
        policy_enabled=policy_enabled, ticks=ticks, client_action=client_action,
    )
    _assert_equivalent(old, new, scenario)


def test_frozen_loop_is_actually_exercising_motion(monkeypatch):
    """Guard against vacuous equivalence: the frozen side must send, capture,
    flush, and step the gate across the scenario matrix, or a regression that
    silences both sides equally would pass every comparison above."""
    _install_patches(monkeypatch)
    seen: set = set()
    for scenario, (frames_fn, policy_enabled, client_action) in _SCENARIOS.items():
        trace = _run_side(
            frozen_loop, frames=frames_fn(), policy_enabled=policy_enabled,
            ticks=len(frames_fn()), client_action=client_action,
        )
        seen.update(trace.events)
    for required in ("send", "gate.step", "gate.latch", "flush",
                     "capture:policy", "capture:teleop", "capture:hold"):
        assert required in seen, (
            "no scenario produced %r on the frozen side — the matrix has a "
            "hole and the equivalence assertions are weaker than they look"
            % required
        )
