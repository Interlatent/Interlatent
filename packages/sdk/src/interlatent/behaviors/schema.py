"""Behavior data model + error types.

Behaviors are **data**, not code (with one escape hatch — see ``ProceduralBehavior``).
Two declarative shapes load from TOML:

- :class:`PoseBehavior` — a single joint target reached over a ``duration``.
- :class:`TrajectoryBehavior` — a list of timed :class:`Keyframe` s with an
  interpolation profile between them.

Joint targets are always in the **robot's own units** — the same units the adapter
and :class:`~interlatent.node.teleop.robot_profile.RobotProfile` use (degrees for
SO-101/Koch, radians for YAM revolute joints; grippers in their own 0..100 / 0..1
range). Nothing here is Cartesian: behaviors are joint-space only (ADR 0013).

A third shape, :class:`ProceduralBehavior`, wraps a Python callable registered with
the :func:`~interlatent.behaviors.registry.behavior` decorator — an imperative escape
hatch for behaviors that are awkward to express as static keyframes (loops, counts).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Union

from .._exceptions import InterlatentError

# The interpolation profiles a trajectory (or pose) may request between keyframes.
INTERPOLATIONS: tuple[str, ...] = ("min_jerk", "linear", "trapezoidal")

# Keys in a TOML behavior table that are *not* joint targets.
_RESERVED_KEYS: frozenset[str] = frozenset(
    {"type", "duration", "interpolation", "keyframes", "description"}
)


class BehaviorError(InterlatentError):
    """Base class for every error raised by the behaviors module."""


class BehaviorValidationError(BehaviorError, ValueError):
    """A behavior is invalid against the robot profile.

    Raised at load time (unknown joint, out-of-limit target) and at plan time
    (velocity-cap violation given the requested timing / ``speed``). Subclasses
    :class:`ValueError` so the CLI's existing contract-violation handling catches it.
    """


class BehaviorExecutionError(BehaviorError, RuntimeError):
    """A behavior aborted mid-motion (adapter error or safety-clamp saturation).

    A *cancellation* is not an error and does not raise this — it returns an
    :class:`~interlatent.behaviors.executor.ActResult` with ``aborted=True``.
    """


@dataclass(frozen=True)
class Keyframe:
    """One timed waypoint of a trajectory.

    ``t`` is seconds from the start of the behavior. ``targets`` maps bare joint
    names to absolute targets; joints omitted from a keyframe **hold their previous
    value** (see the executor). The first keyframe (``t == 0``) must name every joint
    the trajectory ever moves, so the whole trajectory can be velocity-validated
    without knowing the live pose.
    """

    t: float
    targets: dict[str, float]


@dataclass(frozen=True)
class PoseBehavior:
    """A single joint target reached from the live pose over ``duration`` seconds.

    ``duration is None`` means *auto*: the executor sizes the move to the profile's
    velocity caps (with comfort headroom), so a bare ``home`` is always feasible from
    any start pose. An explicit ``duration`` is honored as-is and **raises** if it
    would violate a velocity cap (the caller chose the timing).
    """

    name: str
    targets: dict[str, float]
    duration: "float | None"
    interpolation: str = "min_jerk"
    kind: str = "pose"


@dataclass(frozen=True)
class TrajectoryBehavior:
    """An ordered sequence of :class:`Keyframe` s with an interpolation profile."""

    name: str
    keyframes: tuple[Keyframe, ...]
    interpolation: str = "min_jerk"
    kind: str = "trajectory"

    @property
    def duration(self) -> float:
        """Total wall-clock length: the time of the last keyframe."""
        return self.keyframes[-1].t if self.keyframes else 0.0


@dataclass(frozen=True)
class ProceduralBehavior:
    """A behavior implemented as a Python callable ``fn(robot)``.

    Registered via the :func:`~interlatent.behaviors.registry.behavior` decorator.
    ``robot`` restricts it to one robot kind (``None`` = available on any robot).
    Not validated at load — it drives the robot through the same public ``move()`` /
    ``act()`` calls, each of which validates on its own.
    """

    name: str
    fn: Callable[..., None]
    robot: "str | None" = None
    kind: str = "procedural"
    description: str = ""


# A behavior that carries its motion as data (the executor can plan + run it).
DataBehavior = Union[PoseBehavior, TrajectoryBehavior]
# Anything the registry can resolve by name.
AnyBehavior = Union[PoseBehavior, TrajectoryBehavior, ProceduralBehavior]


def behavior_summary(b: AnyBehavior) -> tuple[str, str, str]:
    """``(name, type, duration)`` row for ``behavior ls`` — duration as display text."""
    if isinstance(b, PoseBehavior):
        dur = "auto" if b.duration is None else f"{b.duration:.2f}s"
        return (b.name, "pose", dur)
    if isinstance(b, TrajectoryBehavior):
        return (b.name, "trajectory", f"{b.duration:.2f}s")
    return (b.name, "procedural", "—")


__all__ = [
    "INTERPOLATIONS",
    "BehaviorError",
    "BehaviorValidationError",
    "BehaviorExecutionError",
    "Keyframe",
    "PoseBehavior",
    "TrajectoryBehavior",
    "ProceduralBehavior",
    "DataBehavior",
    "AnyBehavior",
    "behavior_summary",
    "_RESERVED_KEYS",
]
