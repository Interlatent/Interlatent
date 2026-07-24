"""Tick-for-tick equivalence: each migrated loop vs its frozen ancestor.

``tests/test_loop_contract.py`` asserts *intent* — invariants every loop owes
regardless of history. This suite asserts *sameness*: a loop that now runs on
``looprunner.run_control_loop`` + ``CommandBus.drive()`` must do exactly what
its pre-migration inline ancestor (frozen at ``tests/_frozen/``) did, in the
same order, with the same numbers. That catches the one failure class the
contract suite cannot: side effects that all still happen but got silently
*reordered* when the branch bodies were folded into ``drive()`` — precisely
the risk the old ``movement.py`` docstring warned about when it left them in
the loop.

Both sides run against the same doubles and the same monkeypatched trace
points, so the comparison is an ordered event stream plus the actual numeric
actions sent and captured. Determinism holds because both generations pass
their own ``loop_start`` as the gate's ``now`` *and* the submitted sample's
``received_at``, so the gate's frame-age is identically zero on either side;
everything else on the motion path (Butterworth filter, delta clamp, coerce)
is pure arithmetic.

The robot double deliberately has no ``estop()`` and no Nori daemon surface:
neither lerobot robots nor YAM expose them, and giving the fake an ``estop()``
would make the *new* side forward a hardware latch the old loops never did — a
difference that exists only for robots that cannot occur on these paths.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pytest

from _frozen.lerobot_loop_pre_bus import lerobot_control_loop as frozen_lerobot
from _frozen.nori_loop_pre_bus import control_loop as frozen_nori
from _frozen.yam_loop_pre_bus import control_loop as frozen_yam
from test_loop_contract import FakeChannel, FakeClient, Frame, Trace

from interlatent.node import control as _ctrl
from interlatent.node.teleop.robot_profile import get_profile

_FPS = 100


class EquivRobot:
    """Robot double shared by every pair: marks tick boundaries, records sends.

    Unlike the contract suite's ``FakeRobot`` it exposes neither ``estop()``
    nor the Nori daemon surface — see the module docstring.
    """

    def __init__(self, trace: Trace, action_keys: list):
        self._trace = trace
        self._action_keys = list(action_keys)

    @property
    def action_features(self) -> list[str]:
        return list(self._action_keys)

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
        return {k: 0.0 for k in self._action_keys}

    def send_action(self, action) -> None:
        self._trace.events.append("send")
        self._trace.sends.append(dict(action) if isinstance(action, dict) else action)


class NoriEquivRobot(EquivRobot):
    """Nori-surfaced double: healthy daemon disclosure plus a traced hardware
    e-stop forward. Deliberately no ``pre_tick`` — the frozen loop's guard
    rungs all pass on a healthy daemon, so the migrated side must behave
    identically *without* a guard; the real ``NoriNativeRobot.pre_tick`` logic
    has its own unit suite (``packages/sdk/tests/test_nori_guard.py``)."""

    @property
    def session_dead(self) -> bool:
        return False

    @property
    def dead_reason(self):
        return None

    @property
    def last_status(self) -> dict:
        return {"safety": "ok", "watchdog": "ok"}

    @property
    def telemetry_fresh(self) -> bool:
        return True

    @property
    def obs_age_ms(self) -> float:
        return 0.0

    def estop(self) -> None:
        self._trace.events.append("robot.estop")


@dataclass(frozen=True)
class LoopPair:
    """One frozen-ancestor / migrated-loop pair under comparison."""

    name: str
    robot_kind: str
    frozen_fn: Callable
    live_path: str            # "module:function", resolved at test time
    robot_module: str         # module whose *NativeRobot is swapped, "" = lerobot
    robot_cls: type = EquivRobot

    @property
    def live_fn(self) -> Callable:
        mod_name, fn_name = self.live_path.split(":")
        return getattr(importlib.import_module(mod_name), fn_name)

    @property
    def action_keys(self) -> list:
        profile = get_profile(self.robot_kind)
        assert profile is not None, (
            "equivalence scenarios need a teleop profile for %r" % self.robot_kind
        )
        return [f"{n}.pos" for n in profile.joint_names]


PAIRS = [
    LoopPair("lerobot", "so101", frozen_lerobot,
             "interlatent.node.control:lerobot_control_loop", ""),
    LoopPair("yam", "yam", frozen_yam,
             "interlatent.adapters.yam.loop:control_loop",
             "interlatent.adapters.yam.robot"),
    LoopPair("nori", "nori", frozen_nori,
             "interlatent.adapters.nori.loop:control_loop",
             "interlatent.adapters.nori.robot", robot_cls=NoriEquivRobot),
]


# Routes the shared monkeypatched trace points (installed once per test, but
# recording into whichever side is currently running) to the live side's
# doubles.
_CURRENT: dict = {"trace": None, "robot": None, "keys": None}


def _install_patches(monkeypatch: pytest.MonkeyPatch, pair: LoopPair) -> None:
    """Trace capture/clamp/gate for whichever side is running, via _CURRENT."""
    if pair.robot_module:
        rmod = importlib.import_module(pair.robot_module)
        cls_name = next(
            n for n in dir(rmod) if n.endswith("NativeRobot") and not n.startswith("_")
        )
        monkeypatch.setattr(rmod, cls_name, lambda _cfg: _CURRENT["robot"])
        cmod = importlib.import_module(pair.robot_module.rsplit(".", 1)[0] + ".config")
        monkeypatch.setattr(cmod, "build_adapter_config", lambda *a, **k: object())
    else:
        monkeypatch.setattr(
            _ctrl, "_make_lerobot_robot", lambda *a, **k: _CURRENT["robot"]
        )

    def _capture(_client, _obs, action, _step, *, control_source=None):
        t: Trace = _CURRENT["trace"]
        t.events.append("capture:%s" % (control_source or "policy"))
        t.captures.append((control_source or "policy", np.asarray(action).copy()))
        return list(_CURRENT["keys"])

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

    # The lerobot path binds the class at control-module import; the native
    # loops from-import it inside the function body at call time. Patch both
    # homes so every generation of every pair resolves the null profiler.
    from interlatent.node import teleop_profiler as _tp

    monkeypatch.setattr(_ctrl, "NodeTeleopProfiler", _NullProfiler)
    monkeypatch.setattr(_tp, "NodeTeleopProfiler", _NullProfiler)


def _run_side(loop_fn, pair: LoopPair, *, frames, policy_enabled: bool,
              ticks: int, client_action: bool) -> Trace:
    keys = pair.action_keys
    trace = Trace()
    robot = pair.robot_cls(trace, keys)
    client = FakeClient(trace, len(keys), action=client_action)
    _CURRENT["trace"] = trace
    _CURRENT["robot"] = robot
    _CURRENT["keys"] = keys

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
        robot_kind=pair.robot_kind,
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


def _send_vector(sent, keys: list) -> np.ndarray:
    """A send as an ordered float vector, whether coerce produced dict or array."""
    if isinstance(sent, dict):
        return np.asarray([sent[k] for k in keys], dtype=np.float64)
    return np.asarray(sent, dtype=np.float64).reshape(-1)


def _assert_equivalent(old: Trace, new: Trace, keys: list, label: str) -> None:
    assert old.events == new.events, (
        "[%s] ordered event streams diverge.\n  frozen : %r\n  runner : %r"
        % (label, old.events, new.events)
    )
    assert old.step_calls == new.step_calls, label
    assert len(old.sends) == len(new.sends), label
    for i, (a, b) in enumerate(zip(old.sends, new.sends)):
        np.testing.assert_allclose(
            _send_vector(a, keys), _send_vector(b, keys), atol=1e-6,
            err_msg="[%s] send #%d differs" % (label, i),
        )
    assert old.captured_sources() == new.captured_sources(), label
    for i, ((_, a), (_, b)) in enumerate(zip(old.captures, new.captures)):
        np.testing.assert_allclose(
            np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64),
            atol=1e-6, err_msg="[%s] capture #%d differs" % (label, i),
        )


def _engaged(n: int, **kw) -> Frame:
    kw.setdefault("joint_targets", [0.1] * n)
    return Frame(**kw)


# Each scenario maps n (joint arity) → (frames, policy_enabled, client_action).
# Frames are rebuilt per side so the two runs share no mutable state.
_SCENARIOS = {
    # The highest-traffic path: pure policy inference, no producer connected.
    "pure_policy": lambda n: ([None] * 6, True, True),
    # DRTC cold start / RTC cooldown: client.step() yields no action yet.
    "policy_no_action": lambda n: ([None] * 4, True, False),
    # TeleopRecording assignment, operator never engages: all-hold episode.
    "teleop_recording_hold": lambda n: ([None] * 4, False, True),
    # Operator drives continuously.
    "teleop_engaged": lambda n: ([_engaged(n)] * 6, True, True),
    # Engage → release → policy resumes (the smoother-reset / flush handback).
    "teleop_handback": lambda n: (
        [_engaged(n), _engaged(n), None, None, None], True, True,
    ),
    # E-stop mid-drive: must latch on tick 1 and stay latched (level, not edge).
    "estop_mid_run": lambda n: (
        [_engaged(n), _engaged(n, estop=True), _engaged(n), _engaged(n)],
        True, True,
    ),
    # TeleopRecording with engage gaps: hold ↔ teleop transitions, gate resets.
    "teleop_recording_gaps": lambda n: (
        [None, _engaged(n), _engaged(n), None], False, True,
    ),
    # Malformed frame (wrong target arity): teleop still owns the tick but the
    # gate idles toward the measured pose.
    "malformed_target_len": lambda n: (
        [_engaged(n, joint_targets=[0.1] * (n + 1))] * 3, True, True,
    ),
    # A pose-mode frame that escaped pod-side retargeting: hold pose + one-shot
    # warning (ADR 0009, second amendment).
    "pose_mode_frame": lambda n: (
        [_engaged(n, mode="pose", joint_targets=None)] * 3, True, True,
    ),
}


@pytest.mark.parametrize("pair", PAIRS, ids=lambda p: p.name)
@pytest.mark.parametrize("scenario", sorted(_SCENARIOS), ids=str)
def test_runner_loop_matches_frozen_loop(pair, scenario, monkeypatch):
    n = len(pair.action_keys)
    frames, policy_enabled, client_action = _SCENARIOS[scenario](n)
    ticks = len(frames)
    _install_patches(monkeypatch, pair)

    old = _run_side(
        pair.frozen_fn, pair, frames=_SCENARIOS[scenario](n)[0],
        policy_enabled=policy_enabled, ticks=ticks, client_action=client_action,
    )
    new = _run_side(
        pair.live_fn, pair, frames=_SCENARIOS[scenario](n)[0],
        policy_enabled=policy_enabled, ticks=ticks, client_action=client_action,
    )
    _assert_equivalent(old, new, pair.action_keys, "%s/%s" % (pair.name, scenario))


@pytest.mark.parametrize("pair", PAIRS, ids=lambda p: p.name)
def test_frozen_loop_is_actually_exercising_motion(pair, monkeypatch):
    """Guard against vacuous equivalence: the frozen side must send, capture,
    flush, and step the gate across the scenario matrix, or a regression that
    silences both sides equally would pass every comparison above."""
    _install_patches(monkeypatch, pair)
    n = len(pair.action_keys)
    seen: set = set()
    for scenario in _SCENARIOS:
        frames, policy_enabled, client_action = _SCENARIOS[scenario](n)
        trace = _run_side(
            pair.frozen_fn, pair, frames=frames, policy_enabled=policy_enabled,
            ticks=len(frames), client_action=client_action,
        )
        seen.update(trace.events)
    for required in ("send", "gate.step", "gate.latch", "flush",
                     "capture:policy", "capture:teleop", "capture:hold"):
        assert required in seen, (
            "no scenario produced %r on the frozen side for %s — the matrix "
            "has a hole and the equivalence assertions are weaker than they "
            "look" % (required, pair.name)
        )
