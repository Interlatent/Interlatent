"""Contract every node control loop must satisfy, asserted against all four.

There are four per-tick orchestrations in this tree — ``node/control.py`` plus a
``loop.py`` per native adapter (``adapters/{yam,nori,axol}``) — and nothing has
ever asserted they agree. ``CLAUDE.md`` calls this out as *"Node control-loop
drift … extract the robot-agnostic scaffolding so a new adapter can't silently
miss teleop/recording behavior."* It already has: this suite is expected to FAIL
for ``yam`` (no e-stop rung at all) and ``axol`` (no ``policy_enabled``) until
the hotfix lands.

This asserts the *contract*, not current behavior — deliberately unlike
``packages/sdk/tests/test_movement.py``, which pins the arbiter to the cascade it
replaced. A test that encodes what the loops do today would have ratified the
missing e-stop; this one fails on it.

The four invariants, each traceable to a documented promise:

1. **E-stop is level-triggered.** ADR 0016 / ``CONTEXT.md``: an e-stop latches
   the ``SafetyGate`` and *stays* latched until a human clears it. So the tick
   after the one carrying the flag must still refuse motion — the latch is
   re-read from gate state every tick, never consumed as an edge.
2. **``policy_enabled=False`` never infers.** ``CONTEXT.md:137``: a
   TeleopRecording *"runs its loop with policy_enabled=False — never
   client.step(); engaged ticks record control_source="teleop", disengaged ticks
   hold pose and record control_source="hold"."*
3. **Every send is clamped.** The delta clamp is the last-line execution-safety
   guard against a single-tick joint slam, and the SDK glossary calls it
   *"source-agnostic … for every action — policy and teleop alike."*
4. **Teleop motion converges on the gate.** ``CONTEXT.md:97``: *"All motion
   converges on one node-side path: absolute target → SafetyGate →
   send_action."*

The harness records an ordered event stream rather than a set of booleans, so a
loop that does the right things in the wrong order (sends before clamping, say)
still fails.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pytest

from interlatent.node.teleop.robot_profile import get_profile

# Ticks each scenario drives. Small: every assertion here is about *which* tick
# an effect lands on, never about steady-state behavior.
_TICKS = 4
# High enough that the per-tick sleep is noise, low enough that the SafetyGate's
# velocity clamp (control_dt = 1/fps) still permits visible motion.
_FPS = 100


# --------------------------------------------------------------------------
# Recording doubles
# --------------------------------------------------------------------------


@dataclass
class Trace:
    """Ordered record of everything a loop did to the robot and the client."""

    events: list[str] = field(default_factory=list)
    sends: list[dict] = field(default_factory=list)
    captures: list[tuple[str, np.ndarray]] = field(default_factory=list)
    step_calls: int = 0

    def captured_sources(self) -> list[str]:
        return [src for src, _ in self.captures]

    def tick_index_of(self, event: str) -> list[int]:
        """Indices into ``events`` where this event occurs."""
        return [i for i, e in enumerate(self.events) if e == event]


@dataclass
class Frame:
    """A teleop wire frame, in the shape the loops destructure."""

    engaged: bool = True
    deadman: bool = True
    estop: bool = False
    mode: str = "targets"
    joint_targets: Optional[list] = None
    confidence: float = 1.0
    seq: int = 0
    received_at_ns: int = 0


class FakeChannel:
    """Teleop channel serving a scripted frame per tick.

    Deliberately defines only ``latest_frame`` and ``consume_estop``: the loops
    reach for ``send_state`` / ``preview_due`` / ``note_applied`` through
    ``getattr(..., None)``, so omitting them exercises the guarded path and
    keeps the trace free of tee noise.
    """

    def __init__(self, frames: list[Optional[Frame]]):
        self._frames = list(frames)
        self._i = 0
        self._estop_pending = False

    def latest_frame(self) -> Optional[Frame]:
        f = self._frames[self._i] if self._i < len(self._frames) else None
        self._i += 1
        if f is not None and f.estop:
            # The real channel latches stickily and hands the latch over once;
            # see node/teleop/_frame_store.py.
            self._estop_pending = True
        return f

    def consume_estop(self) -> bool:
        """One-shot, exactly like the real channel — which is precisely why a
        loop must not treat it as its only e-stop signal."""
        hit, self._estop_pending = self._estop_pending, False
        return hit


class FakeSchedule:
    def __init__(self, trace: Trace):
        self._trace = trace

    def flush(self) -> None:
        self._trace.events.append("flush")


class FakeClient:
    def __init__(self, trace: Trace, n_joints: int, action: bool = True):
        self._trace = trace
        self._n = n_joints
        self._action = action
        self.schedule = FakeSchedule(trace)

    def step(self, encode_fn: Callable[[], Any], codec: str = "npz"):
        # Never invoke encode_fn: it would drag in the JPEG backends, and no
        # assertion here depends on the wire payload.
        self._trace.step_calls += 1
        self._trace.events.append("client.step")
        return np.zeros(self._n, dtype=np.float32) if self._action else None


class FakeRobot:
    """Minimal RobotAdapter (adapters/base.py:77) that records its sends."""

    def __init__(self, trace: Trace, action_keys: list[str]):
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
        pass

    def get_observation(self) -> dict:
        # Flat lerobot-shaped obs, no camera arrays (nothing here asserts on
        # video, and omitting them keeps the encoders out of the test).
        return {k: 0.0 for k in self._action_keys}

    def send_action(self, action) -> None:
        self._trace.events.append("send")
        self._trace.sends.append(dict(action) if isinstance(action, dict) else action)

    # --- Nori-only surface, benign for the others -------------------------
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

    def estop(self) -> None:
        self._trace.events.append("robot.estop")


# --------------------------------------------------------------------------
# Adapter matrix
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LoopUnderTest:
    name: str
    robot_kind: str
    loop_path: str      # "module:function"
    robot_module: str   # module whose *NativeRobot attr is swapped, "" = lerobot

    @property
    def teleop_capable(self) -> bool:
        """A loop can only be held to the teleop contract if a RobotProfile
        exists for its kind — without one, SafetyGate refuses the path by
        design (ADR 0011:111)."""
        return get_profile(self.robot_kind) is not None


LOOPS = [
    LoopUnderTest("lerobot", "so101", "interlatent.node.control:lerobot_control_loop", ""),
    LoopUnderTest("yam", "yam", "interlatent.adapters.yam.loop:control_loop",
                  "interlatent.adapters.yam.robot"),
    LoopUnderTest("nori", "nori", "interlatent.adapters.nori.loop:control_loop",
                  "interlatent.adapters.nori.robot"),
    LoopUnderTest("axol", "axol", "interlatent.adapters.axol.loop:control_loop",
                  "interlatent.adapters.axol.robot"),
]


def _action_keys_for(spec: LoopUnderTest) -> list[str]:
    """Action keys whose arity matches the profile, so the teleop schema check
    (``len(action_keys) == len(profile.joint_names)``) passes."""
    profile = get_profile(spec.robot_kind)
    if profile is not None:
        return [f"{n}.pos" for n in profile.joint_names]
    return [f"joint_{i}.pos" for i in range(6)]


def _drive(
    spec: LoopUnderTest,
    monkeypatch: pytest.MonkeyPatch,
    *,
    frames: Optional[list[Optional[Frame]]] = None,
    policy_enabled: bool = True,
    ticks: int = _TICKS,
) -> Trace:
    """Run one loop for ``ticks`` iterations against the doubles, return its trace."""
    import importlib

    from interlatent.node import control as _ctrl

    trace = Trace()
    action_keys = _action_keys_for(spec)
    robot = FakeRobot(trace, action_keys)
    client = FakeClient(trace, len(action_keys))

    # --- Robot construction ------------------------------------------------
    if spec.robot_module:
        rmod = importlib.import_module(spec.robot_module)
        cls_name = next(
            n for n in dir(rmod) if n.endswith("NativeRobot") and not n.startswith("_")
        )
        monkeypatch.setattr(rmod, cls_name, lambda _cfg: robot)
        cmod = importlib.import_module(spec.robot_module.rsplit(".", 1)[0] + ".config")
        monkeypatch.setattr(cmod, "build_adapter_config", lambda *a, **k: object())
    else:
        monkeypatch.setattr(_ctrl, "_make_lerobot_robot", lambda *a, **k: robot)

    # --- Shared wire helpers: record instead of doing I/O ------------------
    def _capture(_client, _obs, action, _step, *, control_source=None):
        trace.events.append("capture:%s" % (control_source or "policy"))
        trace.captures.append((control_source or "policy", np.asarray(action).copy()))
        return list(action_keys)

    monkeypatch.setattr(_ctrl, "_capture_tick", _capture)
    monkeypatch.setattr(_ctrl, "_report_robot_features", lambda *a, **k: True)

    real_clamp = _ctrl._clamp_action_delta

    def _clamp(action, actual, max_step, keys, step, *, source):
        trace.events.append("clamp:%s" % source)
        return real_clamp(action, actual, max_step, keys, step, source=source)

    monkeypatch.setattr(_ctrl, "_clamp_action_delta", _clamp)

    # The gate is real — it is the thing under test — but we need to see when
    # it is stepped, and in what order relative to the send.
    from interlatent.node.teleop import safety as _safety

    real_step = _safety.SafetyGate.step
    real_latch = _safety.SafetyGate.latch_estop

    def _gate_step(self, current_joints, now=None):
        trace.events.append("gate.step")
        return real_step(self, current_joints, now=now)

    def _gate_latch(self, reason):
        trace.events.append("gate.latch")
        return real_latch(self, reason)

    monkeypatch.setattr(_safety.SafetyGate, "step", _gate_step)
    monkeypatch.setattr(_safety.SafetyGate, "latch_estop", _gate_latch)

    # Profiler writes a CSV; silence it without touching the loop's call sites.
    from interlatent.node import teleop_profiler as _tp

    class _NullProfiler:
        def __init__(self, **_):
            pass

        def record_tick(self, **_):
            pass

        def close(self):
            pass

    monkeypatch.setattr(_tp, "NodeTeleopProfiler", _NullProfiler)

    # --- Drive -------------------------------------------------------------
    remaining = {"n": ticks}

    def should_stop() -> bool:
        if remaining["n"] <= 0:
            return True
        remaining["n"] -= 1
        return False

    mod_name, fn_name = spec.loop_path.split(":")
    loop_fn = getattr(importlib.import_module(mod_name), fn_name)

    loop_fn(
        client=client,
        session={"id": "test-session", "fps": _FPS},
        should_stop=should_stop,
        robot_kind=spec.robot_kind,
        robot_port=None,
        robot_extra={"max_step": "5.0"},
        robot_cameras={},
        api_key="k",
        api_base="http://localhost",
        teleop_channel=FakeChannel(frames or [None] * ticks),
        node_id="node-1",
        image_resize=None,
        bypass_key=None,
        policy_enabled=policy_enabled,
    )
    return trace


def _ids(spec: "LoopUnderTest") -> str:
    return spec.name


# --------------------------------------------------------------------------
# 1. E-stop is level-triggered (ADR 0016)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("spec", [s for s in LOOPS if s.teleop_capable], ids=_ids)
def test_estop_latches_and_stays_latched(spec, monkeypatch):
    """An e-stop on tick 1 must suppress motion on tick 1 AND every tick after.

    This is the level-vs-edge distinction. ``consume_estop()`` is one-shot and
    ``frame.estop`` only holds while the operator's frames say so, so a loop
    that branches on the *event* resumes driving on the next tick. Only a loop
    that re-reads ``gate.config.estop_latched`` each tick stays safe.
    """
    n = len(_action_keys_for(spec))
    engaged = Frame(joint_targets=[0.1] * n)
    frames = [engaged, Frame(estop=True, joint_targets=[0.1] * n), engaged, engaged]

    trace = _drive(spec, monkeypatch, frames=frames, ticks=4)

    # Tick 0 established that this loop *can* drive, or the rest proves nothing.
    assert "send" in trace.events, (
        "%s never sent an action even before the e-stop — the scenario is not "
        "exercising the teleop path, so the assertions below would pass "
        "vacuously" % spec.name
    )
    assert "gate.latch" in trace.events, (
        "%s never latched the SafetyGate on an e-stop frame. ADR 0016 makes the "
        "latch the robot-agnostic half of e-stop: every loop owes it, whether or "
        "not the robot also has a hardware latch to forward." % spec.name
    )

    # The latch is the boundary. Anchoring on it rather than on the flush
    # matters: the teleop path flushes on every engaged tick (control.py:446),
    # so a flush is not evidence of an e-stop.
    boundary = trace.events.index("gate.latch")
    after = [
        e for e in trace.events[boundary:]
        if e == "send" or e.startswith("capture:")
    ]
    assert not after, (
        "%s executed %r after the e-stop latched. The latch must be re-read from "
        "SafetyGate state every tick (level), not consumed as an edge — "
        "consume_estop() is one-shot, so an edge-triggered loop resumes driving "
        "on the very next tick." % (spec.name, after)
    )
    assert "flush" in trace.events[boundary:], (
        "%s latched but never flushed the DRTC schedule — queued policy chunks "
        "would fire the moment the latch clears" % spec.name
    )


# --------------------------------------------------------------------------
# 2. policy_enabled=False never infers (CONTEXT.md:137)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("spec", LOOPS, ids=_ids)
def test_policy_disabled_never_steps_and_holds(spec, monkeypatch):
    """A TeleopRecording assignment must never reach the policy, and must keep
    recording so the episode stays continuous across disengage gaps."""
    trace = _drive(spec, monkeypatch, policy_enabled=False, ticks=_TICKS)

    assert trace.step_calls == 0, (
        "%s called client.step() %d time(s) with policy_enabled=False. A "
        "teleop-recording session has no policy loaded (CONTEXT.md:137)."
        % (spec.name, trace.step_calls)
    )
    assert trace.captured_sources() == ["hold"] * _TICKS, (
        "%s recorded %r; a disengaged teleop recording must record every tick "
        'as "hold" so the episode is continuous.'
        % (spec.name, trace.captured_sources())
    )
    assert not trace.sends, (
        "%s commanded the robot on a disengaged hold tick; the motors must "
        "simply hold." % spec.name
    )


# --------------------------------------------------------------------------
# 3. Every send is clamped (layered client-side safety model)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("spec", LOOPS, ids=_ids)
def test_policy_path_sends_are_clamped(spec, monkeypatch):
    """The delta clamp is source-agnostic: it guards the policy path too.

    A first policy chunk whose absolute target is far from the current pose is
    exactly the single-tick slam the clamp exists to bound.
    """
    trace = _drive(spec, monkeypatch, policy_enabled=True, ticks=_TICKS)

    assert "send" in trace.events, "%s never sent on the policy path" % spec.name
    _assert_each_send_preceded_by(trace, "clamp:policy", spec.name)


@pytest.mark.parametrize("spec", [s for s in LOOPS if s.teleop_capable], ids=_ids)
def test_teleop_path_sends_are_clamped(spec, monkeypatch):
    n = len(_action_keys_for(spec))
    frames = [Frame(joint_targets=[0.1] * n)] * _TICKS
    trace = _drive(spec, monkeypatch, frames=frames, ticks=_TICKS)

    assert "send" in trace.events, "%s never sent on the teleop path" % spec.name
    _assert_each_send_preceded_by(trace, "clamp:teleop", spec.name)


def _assert_each_send_preceded_by(trace: Trace, guard: str, name: str) -> None:
    """Every ``send`` must have ``guard`` between it and the previous ``send``."""
    last = -1
    for i, event in enumerate(trace.events):
        if event != "send":
            continue
        window = trace.events[last + 1:i]
        assert guard in window, (
            "%s sent an action with no %r guarding it (events since the previous "
            "send: %r). The clamp is the last-line guard against a single-tick "
            "joint slam and must run on every source." % (name, guard, window)
        )
        last = i


# --------------------------------------------------------------------------
# 4. Teleop motion converges on the SafetyGate (CONTEXT.md:97)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("spec", [s for s in LOOPS if s.teleop_capable], ids=_ids)
def test_teleop_motion_passes_the_gate(spec, monkeypatch):
    """Human-driven motion reaches the motors only through the SafetyGate —
    the single safety authority (CONTEXT.md:147)."""
    n = len(_action_keys_for(spec))
    frames = [Frame(joint_targets=[0.1] * n)] * _TICKS
    trace = _drive(spec, monkeypatch, frames=frames, ticks=_TICKS)

    assert "gate.step" in trace.events, (
        "%s drove the robot from a teleop frame without stepping the SafetyGate"
        % spec.name
    )
    _assert_each_send_preceded_by(trace, "gate.step", spec.name)
