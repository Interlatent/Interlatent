"""Tests for the node movement arbiter (node/movement.py).

The arbiter must reproduce the *exact* control-loop decision it replaced, with
one addition: an ``ESTOP`` rung above every other source. This enumerates the
inputs exhaustively and checks the arbitrated source against the original
boolean cascade, extended by the latch:

    estop_latched                                     -> ESTOP
    engaged = frame and frame.engaged and frame.deadman
    teleop_ok = engaged and gate is not None and action_keys \\
                and len(action_keys) == len(profile.joint_names)
    -> TELEOP if teleop_ok
    -> HOLD   if (not teleop_ok) and (not policy_enabled)
    -> POLICY otherwise

Note on imports: an earlier revision loaded ``movement.py`` in isolation via
``importlib`` to avoid importing ``interlatent.node``, because the module was
stdlib-only. It no longer is — it imports numpy and the SafetyGate's
``TargetSample`` — and the invariant that actually matters (enforced by
``tests/test_lazy_imports.py``) is that no module imports an *optional extra*.
numpy is a base dependency, so a plain import is correct here.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from interlatent.node.movement import (
    Arbiter,
    CommandBus,
    MovementSource,
    TeleopReadiness,
    TickVerdict,
    WireHelpers,
)


@dataclass
class _FakeFrame:
    engaged: bool
    deadman: bool
    estop: bool = False


@dataclass
class _FakeProfile:
    joint_names: tuple


@dataclass
class _FakeConfig:
    estop_latched: bool = False


@dataclass
class _FakeGate:
    """Only the surface the arbiter reads: ``gate.config.estop_latched``."""

    config: _FakeConfig = field(default_factory=_FakeConfig)
    latched_reasons: list = field(default_factory=list)

    def latch_estop(self, reason: str) -> None:
        self.config.estop_latched = True
        self.latched_reasons.append(reason)


class _FakeChannel:
    def __init__(self, frame, estop_pending: bool = False):
        self._frame = frame
        self._estop_pending = estop_pending

    def latest_frame(self):
        return self._frame

    def consume_estop(self) -> bool:
        hit, self._estop_pending = self._estop_pending, False
        return hit


def _reference_decision(*, frame, gate, profile, action_keys, policy_enabled):
    """The original control-loop cascade, verbatim (pre-ESTOP-rung)."""
    engaged = bool(frame and frame.engaged and frame.deadman)
    teleop_ok = (
        engaged
        and gate is not None
        and action_keys
        and len(action_keys) == len(profile.joint_names)
    )
    if teleop_ok:
        return MovementSource.TELEOP
    if not policy_enabled:
        return MovementSource.HOLD
    return MovementSource.POLICY


def _bus(frame, gate, profile, policy_enabled, **kw):
    return CommandBus(
        teleop_channel=_FakeChannel(frame, **kw),
        teleop_gate=gate,
        teleop_profile=profile,
        policy_enabled=policy_enabled,
    )


def test_arbiter_matches_original_cascade_exhaustively():
    """With no latch held, every input combination must decide exactly as the
    inline cascade this replaced."""
    profile = _FakeProfile(joint_names=("a", "b", "c"))  # arity 3

    frame_opts = [
        None,
        _FakeFrame(engaged=True, deadman=True),
        _FakeFrame(engaged=True, deadman=False),
        _FakeFrame(engaged=False, deadman=True),
        _FakeFrame(engaged=False, deadman=False),
    ]
    profile_opts = [profile, None]
    action_key_opts = [["a", "b", "c"], ["a", "b"], []]  # match / mismatch / empty
    policy_opts = [True, False]

    checked = 0
    for frame, prof, akeys, policy_enabled in itertools.product(
        frame_opts, profile_opts, action_key_opts, policy_opts
    ):
        # The gate only exists when a profile exists (loop invariant), so skip
        # the impossible gate-without-profile combos.
        for gate in ([_FakeGate(), None] if prof is not None else [None]):
            bus = _bus(frame, gate, prof, policy_enabled)
            got = bus.arbitrate(bus.sample_teleop(), akeys)
            want = _reference_decision(
                frame=frame, gate=gate, profile=prof,
                action_keys=akeys, policy_enabled=policy_enabled,
            )
            assert got is want, (
                f"mismatch: frame={frame} gate={'set' if gate else None} "
                f"profile={'set' if prof else None} action_keys={akeys} "
                f"policy_enabled={policy_enabled}: got {got} want {want}"
            )
            checked += 1
    assert checked > 0


def test_estop_outranks_every_other_source():
    """A held latch wins regardless of what else the tick looks like."""
    profile = _FakeProfile(joint_names=("a", "b", "c"))
    akeys = ["a", "b", "c"]

    frame_opts = [
        None,
        _FakeFrame(engaged=True, deadman=True),    # would be TELEOP
        _FakeFrame(engaged=False, deadman=False),  # would be HOLD/POLICY
    ]
    for frame, policy_enabled in itertools.product(frame_opts, [True, False]):
        gate = _FakeGate(config=_FakeConfig(estop_latched=True))
        bus = _bus(frame, gate, profile, policy_enabled)
        got = bus.arbitrate(bus.sample_teleop(), akeys)
        assert got is MovementSource.ESTOP, (
            f"latch held but arbitrated {got} for frame={frame} "
            f"policy_enabled={policy_enabled}"
        )


def test_latch_is_level_not_edge():
    """The decision reads gate state, so it survives the one-shot channel latch.

    ``consume_estop()`` returns True exactly once. A loop that keyed off that
    event would resume driving on the next tick; keying off the latch does not.
    """
    profile = _FakeProfile(joint_names=("a",))
    gate = _FakeGate()
    frame = _FakeFrame(engaged=True, deadman=True)
    bus = _bus(frame, gate, profile, True, estop_pending=True)

    bus.observe_estop(bus.sample_teleop())          # tick 1: the latch arrives
    assert gate.config.estop_latched is True
    assert bus.arbitrate(bus.sample_teleop(), ["a"]) is MovementSource.ESTOP

    # Tick 2: the sticky latch is spent and no frame carries the flag.
    assert bus._teleop_channel.consume_estop() is False
    assert bus.arbitrate(bus.sample_teleop(), ["a"]) is MovementSource.ESTOP


def test_estop_forwards_to_hardware_once_and_retries_on_failure():
    """Robots with their own hard latch expose ``estop()``; it is forwarded once
    per latch, and a failure must not consume the one-shot."""
    profile = _FakeProfile(joint_names=("a",))

    class _Robot:
        def __init__(self, fail_first: bool):
            self.calls = 0
            self._fail_first = fail_first

        def estop(self):
            self.calls += 1
            if self._fail_first and self.calls == 1:
                raise RuntimeError("daemon unreachable")

    robot = _Robot(fail_first=False)
    gate = _FakeGate()
    bus = CommandBus(
        teleop_channel=_FakeChannel(_FakeFrame(True, True, estop=True)),
        teleop_gate=gate, teleop_profile=profile, policy_enabled=True, robot=robot,
    )
    bus.observe_estop(bus.sample_teleop())
    bus.observe_estop(bus.sample_teleop())
    assert robot.calls == 1, "hardware e-stop forwarded more than once per latch"

    failing = _Robot(fail_first=True)
    bus2 = CommandBus(
        teleop_channel=_FakeChannel(_FakeFrame(True, True, estop=True)),
        teleop_gate=_FakeGate(), teleop_profile=profile, policy_enabled=True,
        robot=failing,
    )
    try:
        bus2.observe_estop(bus2.sample_teleop())
    except RuntimeError:
        pass
    bus2.observe_estop(bus2.sample_teleop())
    assert failing.calls == 2, "a failed forward must be retried, not swallowed"


def test_no_teleop_channel_is_policy_or_hold():
    bus = CommandBus(
        teleop_channel=None, teleop_gate=None,
        teleop_profile=None, policy_enabled=True,
    )
    assert bus.sample_teleop() is None
    assert bus.arbitrate(None, ["a"]) is MovementSource.POLICY

    bus_hold = CommandBus(
        teleop_channel=None, teleop_gate=None,
        teleop_profile=None, policy_enabled=False,
    )
    assert bus_hold.arbitrate(None, ["a"]) is MovementSource.HOLD


def test_source_values_match_legacy_labels():
    """The recorded control_source strings must not change — these enum values
    ARE the dataset's wire format."""
    assert MovementSource.TELEOP.value == "teleop"
    assert MovementSource.HOLD.value == "hold"
    assert MovementSource.POLICY.value == "policy"


def test_estop_is_not_a_dataset_label():
    """CONTEXT.md pins control_source to exactly three values. ESTOP is a
    movement source but never a recorded label — e-stop ticks are not captured."""
    recorded = {
        MovementSource.TELEOP.value,
        MovementSource.HOLD.value,
        MovementSource.POLICY.value,
    }
    assert recorded == {"teleop", "hold", "policy"}
    assert MovementSource.ESTOP.value not in recorded


def test_arbiter_is_usable_without_a_bus():
    """The decision is a pure function of booleans, so it stays testable and
    reusable on its own."""
    ready = TeleopReadiness(engaged=True, gated=True, schema_ok=True)
    a = Arbiter()
    assert a.decide(teleop_ready=ready, policy_enabled=True) is MovementSource.TELEOP
    assert a.decide(
        teleop_ready=ready, policy_enabled=True, estop_latched=True
    ) is MovementSource.ESTOP


def test_tick_verdict_vocabulary():
    """The three outcomes an adapter guard can return."""
    assert {v.value for v in TickVerdict} == {
        "proceed", "hold_no_capture", "end_episode",
    }


# ---------------------------------------------------------------------------
# CommandBus.drive() — the motion path
#
# The bus now owns ordering that used to be spelled out in four forked loops,
# and ordering is the thing a refactor breaks silently. These record an ordered
# event trace rather than a set of booleans, so "did the right things in the
# wrong order" fails.
# ---------------------------------------------------------------------------

_KEYS = ["j0.pos", "j1.pos", "j2.pos"]


class _TraceGate:
    def __init__(self, latched: bool = False, trace=None):
        self.config = _FakeConfig(estop_latched=latched)
        self.trace = trace if trace is not None else []
        self.submitted = []

    def submit(self, sample):
        self.trace.append("gate.submit")
        self.submitted.append(sample)

    def step(self, current, now=None):
        self.trace.append("gate.step")
        s = self.submitted[-1] if self.submitted else None
        return (s.joints if s is not None else current), "ok"

    def reset(self):
        self.trace.append("gate.reset")

    def latch_estop(self, reason):
        self.config.estop_latched = True


class _TraceSchedule:
    def __init__(self, trace):
        self._t = trace

    def flush(self):
        self._t.append("flush")


class _TraceClient:
    def __init__(self, trace, action=None):
        self.schedule = _TraceSchedule(trace)
        self._t = trace
        self._action = action
        self.steps = 0

    def step(self, encode_fn, codec="npz"):
        self._t.append("client.step")
        self.steps += 1
        return self._action


class _TraceRobot:
    def __init__(self, trace):
        self._t = trace
        self.sent = []

    def send_action(self, action):
        self._t.append("send")
        self.sent.append(action)


class _TraceFilter:
    def __init__(self, trace):
        self._t = trace

    def filter(self, arr):
        self._t.append("filter")
        return arr

    def reset(self):
        self._t.append("filter.reset")


def _helpers(trace):
    def extract(obs, keys):
        return np.array([float(obs[k]) for k in keys], dtype=np.float32)

    def clamp(action, actual, max_step, keys, step, *, source):
        trace.append(f"clamp:{source}")
        return action

    def coerce(action, keys):
        return {k: float(action[i]) for i, k in enumerate(keys)}

    def encode(obs):
        return b""

    return WireHelpers(extract=extract, clamp=clamp, coerce=coerce, encode=encode)


def _drive_bus(trace, *, frame, policy_enabled=True, latched=False, action=None):
    gate = _TraceGate(latched=latched, trace=trace)
    return CommandBus(
        teleop_channel=_FakeChannel(frame),
        teleop_gate=gate,
        teleop_profile=_FakeProfile(joint_names=tuple(_KEYS)),
        policy_enabled=policy_enabled,
        robot=_TraceRobot(trace),
        client=_TraceClient(trace, action=action),
        action_keys=list(_KEYS),
        helpers=_helpers(trace),
        max_step=5.0,
        action_filter=_TraceFilter(trace),
    ), gate


_OBS = {k: 0.0 for k in _KEYS}


def _teleop_frame():
    return type(
        "F", (), dict(
            engaged=True, deadman=True, estop=False, mode="targets",
            joint_targets=[0.1, 0.2, 0.3], confidence=1.0, seq=7,
            received_at_ns=None,
        ),
    )()


def test_drive_estop_suppresses_motion_and_capture():
    trace = []
    bus, _ = _drive_bus(trace, frame=_teleop_frame(), latched=True)
    out = bus.drive(_OBS, step=0, now=1.0)

    assert out.source is MovementSource.ESTOP
    assert out.sent is False and out.should_record is False
    assert out.control_source is None, "an e-stop tick must carry no dataset label"
    assert "send" not in trace
    assert trace == ["flush", "filter.reset"], (
        f"e-stop must flush queued chunks and drop smoother state, got {trace}"
    )


def test_drive_hold_records_without_commanding():
    trace = []
    bus, _ = _drive_bus(trace, frame=None, policy_enabled=False)
    out = bus.drive(_OBS, step=0, now=1.0)

    assert out.source is MovementSource.HOLD
    assert out.control_source == "hold" and out.should_record is True
    assert out.sent is False and "send" not in trace
    np.testing.assert_allclose(out.action, np.zeros(3, dtype=np.float32))


def test_drive_teleop_orders_gate_then_clamp_then_send():
    trace = []
    bus, _ = _drive_bus(trace, frame=_teleop_frame())
    out = bus.drive(_OBS, step=0, now=1.0)

    assert out.source is MovementSource.TELEOP
    assert out.control_source == "teleop" and out.should_record is True
    assert out.sent is True and out.cmd_at is not None
    assert trace == [
        "gate.submit", "gate.step", "clamp:teleop", "send", "flush", "filter.reset",
    ], f"teleop ordering changed: {trace}"
    # The commanded (post-gate) action is what gets reported for recording.
    np.testing.assert_allclose(out.action, np.array([0.1, 0.2, 0.3], dtype=np.float32))


def test_drive_policy_smooths_then_clamps_then_sends():
    trace = []
    bus, _ = _drive_bus(
        trace, frame=None, action=np.array([1.0, 2.0, 3.0], dtype=np.float32)
    )
    out = bus.drive(_OBS, step=0, now=1.0)

    assert out.source is MovementSource.POLICY
    assert out.control_source == "policy" and out.should_record is True
    assert trace == [
        "gate.reset", "client.step", "filter", "clamp:policy", "send",
    ], f"policy ordering changed: {trace}"


def test_drive_policy_without_an_action_yet_does_nothing():
    """client.step() returns None while the first chunk is in flight."""
    trace = []
    bus, _ = _drive_bus(trace, frame=None, action=None)
    out = bus.drive(_OBS, step=0, now=1.0)

    assert out.source is MovementSource.POLICY
    assert out.sent is False and out.should_record is False
    assert "send" not in trace


def test_drive_requires_motion_collaborators():
    """A bus built for arbitration only must refuse to drive rather than
    silently no-op."""
    bus = CommandBus(
        teleop_channel=None, teleop_gate=None,
        teleop_profile=None, policy_enabled=True,
    )
    try:
        bus.drive(_OBS, step=0, now=1.0)
    except RuntimeError as exc:
        assert "arbitration only" in str(exc)
    else:
        raise AssertionError("drive() must raise without robot + helpers")


def test_drive_latches_before_arbitrating():
    """An e-stop arriving on this tick suppresses this tick, not the next one."""
    trace = []
    frame = _teleop_frame()
    frame.estop = True
    bus, gate = _drive_bus(trace, frame=frame)
    out = bus.drive(_OBS, step=0, now=1.0)

    assert gate.config.estop_latched is True
    assert out.source is MovementSource.ESTOP
    assert "send" not in trace
