"""Tests for the shared tick skeleton (node/looprunner.py).

The runner owns everything that is *not* motion, plus the one new seam an
adapter can implement: ``pre_tick``. Motion itself is covered by the
``CommandBus.drive()`` tests in ``test_movement.py``; here the bus is a stub, so
these assert the skeleton's own obligations — that a guard can end an episode or
suppress a capture, that recording follows the outcome rather than second-
guessing it, and that the robot is always disconnected.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from interlatent.node.looprunner import run_control_loop
from interlatent.node.movement import MovementSource, TickOutcome, TickVerdict

_KEYS = ["j0.pos"]


class _StubBus:
    """Stands in for CommandBus: returns a scripted outcome per tick."""

    def __init__(self, outcome: Optional[TickOutcome] = None):
        self._outcome = outcome or TickOutcome(
            source=MovementSource.POLICY,
            action=[0.0],
            control_source="policy",
            should_record=True,
            sent=True,
        )
        self.calls = 0
        self.interrupts: list = []

    def drive(self, obs, *, step, now):
        self.calls += 1
        return self._outcome

    def guard_interrupt(self, verdict):
        self.interrupts.append(verdict)


@dataclass
class _Robot:
    verdicts: Optional[list] = None
    observations: int = 0
    disconnected: bool = False
    _i: int = 0

    @property
    def action_features(self):
        return list(_KEYS)

    def get_observation(self):
        self.observations += 1
        return {k: 0.0 for k in _KEYS}

    def disconnect(self):
        self.disconnected = True

    def __post_init__(self):
        if self.verdicts is not None:
            # Only expose pre_tick when this robot actually has a guard, so the
            # getattr discovery path is exercised in both directions.
            self.pre_tick = self._pre_tick

    def _pre_tick(self, obs):
        v = self.verdicts[min(self._i, len(self.verdicts) - 1)]
        self._i += 1
        return v


def _stopper(n: int):
    remaining = {"n": n}

    def should_stop() -> bool:
        if remaining["n"] <= 0:
            return True
        remaining["n"] -= 1
        return False

    return should_stop


def _run(robot, bus, ticks: int, captures: list):
    def capture_fn(obs, action, step, *, control_source=None):
        captures.append((control_source, step))
        return list(_KEYS)

    run_control_loop(
        robot=robot, bus=bus, should_stop=_stopper(ticks),
        fps=1000, action_keys=list(_KEYS), capture_fn=capture_fn,
    )


def test_runs_and_records_each_tick():
    robot, bus, captures = _Robot(), _StubBus(), []
    _run(robot, bus, 3, captures)

    assert bus.calls == 3
    assert captures == [("policy", 0), ("policy", 1), ("policy", 2)], (
        "the step counter must advance only on recorded ticks"
    )
    assert robot.disconnected is True


def test_no_guard_is_fine():
    """Most adapters implement no pre_tick at all."""
    robot = _Robot()
    assert not hasattr(robot, "pre_tick")
    _run(robot, _StubBus(), 2, [])


def test_guard_can_end_the_episode():
    """END_EPISODE returns immediately — the bus is never consulted, and the
    robot is still disconnected on the way out."""
    robot = _Robot(verdicts=[TickVerdict.END_EPISODE])
    bus, captures = _StubBus(), []
    _run(robot, bus, 5, captures)

    assert bus.calls == 0, "a dead session must not be driven"
    assert captures == []
    assert robot.observations == 1
    assert robot.disconnected is True
    assert bus.interrupts == [TickVerdict.END_EPISODE], (
        "the bus must get the interrupt so it can flush queued chunks"
    )


def test_guard_can_hold_without_capturing():
    """HOLD_NO_CAPTURE suppresses motion and capture but continues the episode.

    This is the stale-telemetry case: recording a stale pose as live state would
    poison the dataset, which is worse than a gap.
    """
    robot = _Robot(verdicts=[
        TickVerdict.HOLD_NO_CAPTURE, TickVerdict.HOLD_NO_CAPTURE, TickVerdict.PROCEED,
    ])
    bus, captures = _StubBus(), []
    _run(robot, bus, 3, captures)

    assert bus.calls == 1, "held ticks must not reach the motion path"
    assert captures == [("policy", 0)], "held ticks must not be recorded"
    assert robot.observations == 3, (
        "the observation read must happen every tick — for a daemon-driven robot "
        "it doubles as the keep-alive liveness proof"
    )
    assert bus.interrupts == [TickVerdict.HOLD_NO_CAPTURE] * 2, (
        "each held tick must hand the interrupt to the bus for gate/smoother "
        "hygiene"
    )


def test_outcome_drives_recording_not_the_runner():
    """The runner records what drive() reports, and nothing else."""
    silent = TickOutcome(source=MovementSource.ESTOP)  # should_record defaults False
    robot, captures = _Robot(), []
    _run(robot, _StubBus(silent), 3, captures)
    assert captures == []


def test_disconnect_happens_even_when_a_tick_raises():
    class _Boom(_StubBus):
        def drive(self, obs, *, step, now):
            raise RuntimeError("bang")

    robot = _Robot()
    try:
        _run(robot, _Boom(), 2, [])
    except RuntimeError:
        pass
    assert robot.disconnected is True
