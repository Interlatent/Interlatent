"""LWW action schedule (the heart of DRTC chunk merging)."""
import numpy as np

from interlatent.inference.client.merge import ActionSchedule, ScheduledAction


def sa(step: int, ts: int, value: float = 0.0) -> ScheduledAction:
    return ScheduledAction(
        action_step=step,
        control_timestamp=ts,
        vector=np.full(3, value, dtype=np.float32),
    )


def test_merge_and_pop_in_order():
    sched = ActionSchedule()
    assert sched.merge([sa(0, 1), sa(1, 1), sa(2, 1)]) == 3
    assert sched.pop_next().action_step == 0
    assert sched.pop_next().action_step == 1
    assert sched.pop_next().action_step == 2
    assert sched.pop_next() is None


def test_lww_newer_timestamp_wins():
    sched = ActionSchedule()
    sched.merge([sa(0, ts=1, value=1.0)])
    sched.merge([sa(0, ts=2, value=2.0)])  # newer overwrites
    sched.merge([sa(0, ts=1, value=9.0)])  # older is ignored
    a = sched.pop_next()
    assert float(a.vector[0]) == 2.0


def test_merge_is_idempotent():
    sched = ActionSchedule()
    assert sched.merge([sa(5, 10)]) == 1
    assert sched.merge([sa(5, 10)]) == 0  # same timestamp: not "strictly newer"


def test_already_executed_steps_are_ignored():
    sched = ActionSchedule()
    sched.merge([sa(0, 1), sa(1, 1)])
    sched.pop_next()  # cursor -> 1
    assert sched.merge([sa(0, 99)]) == 0  # behind cursor, dropped
    assert sched.queue_depth() == 1


def test_pop_next_does_not_skip_gaps():
    sched = ActionSchedule()
    sched.merge([sa(1, 1)])  # step 0 missing
    assert sched.pop_next() is None  # waits at cursor 0
    assert sched.cursor() == 0
    sched.merge([sa(0, 1)])
    assert sched.pop_next().action_step == 0
    assert sched.pop_next().action_step == 1


def test_scheduled_spans_run_length():
    sched = ActionSchedule()
    sched.merge([sa(s, 1) for s in (0, 1, 2, 5, 6, 9)])
    assert sched.scheduled_spans() == [(0, 2), (5, 6), (9, 9)]


def test_flush_keeps_cursor():
    sched = ActionSchedule()
    sched.merge([sa(0, 1), sa(1, 1)])
    sched.pop_next()
    assert sched.flush() == 1
    assert sched.queue_depth() == 0
    assert sched.cursor() == 1  # next chunk anchors at the same step


def test_next_action_step_is_cursor():
    sched = ActionSchedule()
    sched.merge([sa(0, 1), sa(1, 1), sa(2, 1)])
    sched.pop_next()
    assert sched.next_action_step() == 1


def test_flush_barrier_rejects_in_flight_pre_barrier_chunk():
    # An Infer dispatched before a teleop takeover must not install its
    # actions after the takeover's barrier flush.
    sched = ActionSchedule()
    sched.merge([sa(0, ts=5), sa(1, ts=5)])
    assert sched.flush(barrier_ts=10) == 2
    # In-flight chunk stamped before the barrier: rejected entirely.
    assert sched.merge([sa(1, ts=8), sa(2, ts=8)]) == 0
    assert sched.queue_depth() == 0
    # Cursor untouched by the barrier flush.
    assert sched.cursor() == 0


def test_flush_barrier_accepts_newer_chunk():
    sched = ActionSchedule()
    sched.flush(barrier_ts=10)
    assert sched.merge([sa(0, ts=11), sa(1, ts=12)]) == 2
    assert sched.pop_next().action_step == 0


def test_flush_barrier_is_monotonic():
    # A later flush with a smaller (or absent) barrier never lowers the floor.
    sched = ActionSchedule()
    sched.flush(barrier_ts=10)
    sched.flush(barrier_ts=3)
    sched.flush()
    assert sched.merge([sa(0, ts=9)]) == 0
    assert sched.merge([sa(0, ts=11)]) == 1
