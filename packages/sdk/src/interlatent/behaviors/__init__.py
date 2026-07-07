"""Named, deterministic robot behaviors — no cloud, no GPU, no policy.

A small, fully-offline layer for driving a robot through named moves and trajectories:

    import interlatent as il

    with il.Robot("so101", port="/dev/ttyACM0") as robot:
        robot.act("home")            # move to the profile rest pose
        robot.act("hello")           # play the built-in wave
        robot.move(wrist_roll=30, duration=0.5)

Behaviors are **data** (TOML: poses + keyframe trajectories) validated against the
robot's :class:`~interlatent.node.teleop.robot_profile.RobotProfile`, executed by a
local min-jerk control loop that drives the *existing* adapter action path (so the
adapter's delta clamp still applies). See :mod:`interlatent.behaviors.registry` for
the load order and the :func:`behavior` decorator, and :mod:`interlatent.behaviors.executor`
for the control loop.

The user-facing entry points (``Robot``, ``behavior``) are re-exported at the top
level of the package: ``interlatent.Robot`` and ``interlatent.behavior``.
"""
from __future__ import annotations

from .arbitration import RobotBusyError
from .executor import ActHandle, ActResult, TrajectoryExecutor
from .registry import USER_BEHAVIORS_PATH, BehaviorRegistry, behavior
from .schema import (
    BehaviorError,
    BehaviorExecutionError,
    BehaviorValidationError,
    Keyframe,
    PoseBehavior,
    ProceduralBehavior,
    TrajectoryBehavior,
)

__all__ = [
    "behavior",
    "BehaviorRegistry",
    "USER_BEHAVIORS_PATH",
    "TrajectoryExecutor",
    "ActResult",
    "ActHandle",
    "RobotBusyError",
    "BehaviorError",
    "BehaviorValidationError",
    "BehaviorExecutionError",
    "Keyframe",
    "PoseBehavior",
    "TrajectoryBehavior",
    "ProceduralBehavior",
]
