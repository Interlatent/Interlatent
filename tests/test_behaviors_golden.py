"""Golden test: the emitted target stream for the built-in SO-101 ``hello`` wave.

Runs ``hello`` against the fake adapter at the default 30 Hz and pins the exact
sampled command sequence — both a full-stream digest (any change to the interpolation
math, keyframes, or sampling trips it) and human-readable keyframe checkpoints.
"""
from __future__ import annotations

import hashlib

import numpy as np

from behavior_fakes import FakeAdapter

from interlatent.behaviors.executor import TrajectoryExecutor
from interlatent.behaviors.registry import BehaviorRegistry
from interlatent.node.teleop.robot_profile import get_profile

# sha256 of np.round(commands, 3).tobytes() for `hello` at 30 Hz from rest.
# Regenerate deliberately (and review the diff) if the wave is intentionally changed.
_HELLO_SHA256 = "b60d10529250824db8942994e0a5087dc64f68412f4689b35e68dc086dd7e554"

# (tick index, {joint index: expected value}) at each keyframe boundary.
# Joint order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper.
_CHECKPOINTS = [
    (45, {1: -30.0, 2: -40.0, 4: 0.0}),
    (63, {1: -30.0, 2: -40.0, 4: 35.0}),
    (81, {4: -35.0}),
    (99, {4: 35.0}),
    (117, {4: -35.0}),
    (135, {4: 0.0}),
    (180, {1: 0.0, 2: 0.0, 4: 0.0}),
]


def _run_hello() -> np.ndarray:
    a = FakeAdapter()
    a.connect()
    ex = TrajectoryExecutor(a, get_profile("so101"), realtime=False)
    ex.act(BehaviorRegistry.for_robot("so101").resolve("hello"), wait=True)
    return a.command_array


def test_hello_stream_matches_golden_digest():
    cmds = np.round(_run_hello(), 3)
    assert len(cmds) == 181
    digest = hashlib.sha256(np.ascontiguousarray(cmds).tobytes()).hexdigest()
    assert digest == _HELLO_SHA256


def test_hello_keyframe_checkpoints():
    cmds = _run_hello()
    for tick, expected in _CHECKPOINTS:
        for j, value in expected.items():
            assert cmds[tick, j] == np.float32(value), (tick, j, cmds[tick, j])


def test_hello_stream_is_deterministic():
    assert np.array_equal(_run_hello(), _run_hello())
