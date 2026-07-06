"""Loading, validating, and resolving named behaviors.

A :class:`BehaviorRegistry` binds a set of behaviors to one robot kind and its
:class:`~interlatent.node.teleop.robot_profile.RobotProfile`. Behaviors come from four
layers, each overriding the previous **by name**:

1. **Built-in defaults** — ``home`` (derived from the profile's ``rest_pose``, so it
   always matches the robot) plus any packaged ``data/<robot>.toml`` (SO-101 ships a
   ``hello`` wave).
2. **User file** — ``~/.interlatent/behaviors.toml`` if present.
3. **Explicit file** — the path passed to ``Robot(behaviors=...)`` / ``--behaviors``.
4. **Procedural** — functions registered with the :func:`behavior` decorator (global
   or scoped to this robot kind).

Every declarative behavior is validated against the profile as it is loaded: unknown
joint names, out-of-limit targets, and (for trajectories) velocity-cap violations all
raise :class:`BehaviorValidationError` naming the behavior, joint, value, and limit.
Pose behaviors defer their velocity check to plan time, where the live pose is known.

The registry needs no hardware — ``behavior ls`` / ``behavior validate`` build one
from the profile alone.
"""
from __future__ import annotations

import logging
import tomllib
from importlib.resources import files
from pathlib import Path
from typing import Callable, Optional

from ..node.teleop.robot_profile import RobotProfile, get_profile
from .interpolation import peak_velocity_factor
from .schema import (
    _RESERVED_KEYS,
    AnyBehavior,
    BehaviorValidationError,
    DataBehavior,
    Keyframe,
    PoseBehavior,
    ProceduralBehavior,
    TrajectoryBehavior,
)

_LOG = logging.getLogger("interlatent.behaviors.registry")

# Default location for the user's personal behavior library.
USER_BEHAVIORS_PATH = Path.home() / ".interlatent" / "behaviors.toml"

# Small tolerance so float round-off in a hand-authored TOML doesn't trip validation.
_VEL_EPS = 1e-6

# Map a robot kind onto the packaged built-in data file (stem). Kinds that share a
# topology share a file; kinds with no file still get a generated ``home``.
_BUILTIN_DATA: dict[str, str] = {
    "so101": "so101",
    "so101_follower": "so101",
}

# ---------------------------------------------------------------------------
# Procedural (decorator) registration
# ---------------------------------------------------------------------------

# Keyed by (robot_kind_or_None, name). A None robot key applies to every robot.
_PROCEDURAL: dict[tuple[Optional[str], str], ProceduralBehavior] = {}


def behavior(
    name: str, *, robot: Optional[str] = None, description: str = ""
) -> Callable[[Callable[..., None]], Callable[..., None]]:
    """Register ``name`` as a procedural behavior implemented by the decorated function.

    The function receives the :class:`~interlatent.robot.Robot` facade and drives it
    through the public API::

        @il.behavior("nod")
        def nod(robot):
            for _ in range(2):
                robot.move(wrist_flex=-20, duration=0.3)
                robot.move(wrist_flex=0, duration=0.3)

    ``robot`` scopes the behavior to one robot kind (default: available on any robot).
    Returns the function unchanged, so it stays directly callable/testable.
    """

    def _decorator(fn: Callable[..., None]) -> Callable[..., None]:
        key = (robot.lower().strip() if robot else None, name)
        _PROCEDURAL[key] = ProceduralBehavior(
            name=name, fn=fn, robot=(robot.lower().strip() if robot else None),
            description=description or (fn.__doc__ or "").strip().split("\n", 1)[0],
        )
        return fn

    return _decorator


def _procedural_for(robot_kind: str) -> dict[str, ProceduralBehavior]:
    """Procedural behaviors visible to ``robot_kind`` (global first, kind-specific wins)."""
    kind = robot_kind.lower().strip()
    out: dict[str, ProceduralBehavior] = {}
    for (scope, name), proc in _PROCEDURAL.items():
        if scope is None or scope == kind:
            out[name] = proc
    return out


class BehaviorRegistry:
    """A robot kind's resolved, validated behavior set."""

    def __init__(self, robot_kind: str, profile: RobotProfile) -> None:
        self.robot_kind = robot_kind
        self.profile = profile
        self.joint_names: tuple[str, ...] = tuple(profile.joint_names)
        self._index: dict[str, int] = {n: i for i, n in enumerate(self.joint_names)}
        self._data: dict[str, DataBehavior] = {}
        self._procedural: dict[str, ProceduralBehavior] = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def for_robot(
        cls,
        robot_kind: str,
        *,
        explicit: "str | Path | None" = None,
        user_file: "str | Path | None" = USER_BEHAVIORS_PATH,
        include_procedural: bool = True,
    ) -> "BehaviorRegistry":
        """Build the layered registry for ``robot_kind`` (see module docstring)."""
        profile = get_profile(robot_kind)
        if profile is None:
            raise BehaviorValidationError(
                f"no RobotProfile for robot kind {robot_kind!r}: behaviors need a "
                "profile (joint limits + velocity caps). Add one in "
                "interlatent.node.teleop.robot_profile."
            )
        reg = cls(robot_kind, profile)
        reg._load_builtin()
        if user_file is not None and Path(user_file).expanduser().is_file():
            reg._load_file(Path(user_file).expanduser(), source="user file")
        if explicit is not None:
            reg._load_file(Path(explicit).expanduser(), source="explicit path")
        if include_procedural:
            reg._procedural = _procedural_for(robot_kind)
        return reg

    def _load_builtin(self) -> None:
        # `home` is derived from the profile's rest pose so it always matches the
        # robot — never a stale literal. Auto duration so it is feasible from anywhere.
        self._data["home"] = PoseBehavior(
            name="home",
            targets={n: float(self.profile.rest_pose[i]) for i, n in enumerate(self.joint_names)},
            duration=None,
        )
        stem = _BUILTIN_DATA.get(self.robot_kind.lower().strip())
        if stem is None:
            return
        try:
            resource = files("interlatent.behaviors").joinpath("data", f"{stem}.toml")
            raw = resource.read_text(encoding="utf-8")
        except (FileNotFoundError, ModuleNotFoundError, OSError):
            _LOG.debug("no packaged behavior data for %r", self.robot_kind)
            return
        self._load_toml_text(raw, source=f"built-in ({stem}.toml)")

    def _load_file(self, path: Path, *, source: str) -> None:
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise BehaviorValidationError(f"cannot read behaviors {source} {str(path)!r}: {exc}")
        self._load_toml_text(raw, source=f"{source} {path}")

    def _load_toml_text(self, raw: str, *, source: str) -> None:
        try:
            data = tomllib.loads(raw)
        except tomllib.TOMLDecodeError as exc:
            raise BehaviorValidationError(f"invalid TOML in {source}: {exc}")
        for name, spec in data.items():
            if not isinstance(spec, dict):
                raise BehaviorValidationError(
                    f"{source}: behavior {name!r} must be a table, got {type(spec).__name__}"
                )
            self._data[name] = self._parse_behavior(name, spec, source)

    # ------------------------------------------------------------------
    # Parsing + validation
    # ------------------------------------------------------------------

    def _parse_behavior(self, name: str, spec: dict, source: str) -> DataBehavior:
        btype = str(spec.get("type", "")).strip().lower()
        if btype == "pose":
            beh = self._parse_pose(name, spec, source)
        elif btype == "trajectory":
            beh = self._parse_trajectory(name, spec, source)
        else:
            raise BehaviorValidationError(
                f"{source}: behavior {name!r} has type={spec.get('type')!r}; expected "
                "'pose' or 'trajectory'."
            )
        return beh

    def _joint_targets(self, name: str, table: dict, source: str) -> dict[str, float]:
        """Extract + validate ``{joint: value}`` from a table (non-reserved keys)."""
        targets: dict[str, float] = {}
        for key, value in table.items():
            if key in _RESERVED_KEYS:
                continue
            if key not in self._index:
                raise BehaviorValidationError(
                    f"{source}: behavior {name!r} names unknown joint {key!r}; known "
                    f"joints for {self.robot_kind!r}: {list(self.joint_names)}"
                )
            try:
                targets[key] = float(value)
            except (TypeError, ValueError):
                raise BehaviorValidationError(
                    f"{source}: behavior {name!r} joint {key!r} target must be a number, "
                    f"got {value!r}"
                )
        return targets

    def _check_limits(self, name: str, targets: dict[str, float]) -> None:
        for joint, value in targets.items():
            lo, hi = self.profile.joint_limits[self._index[joint]]
            if not (lo <= value <= hi):
                raise BehaviorValidationError(
                    f"behavior {name!r}: joint {joint!r} target {value:g} is outside its "
                    f"limit [{lo:g}, {hi:g}]"
                )

    def _parse_pose(self, name: str, spec: dict, source: str) -> PoseBehavior:
        targets = self._joint_targets(name, spec, source)
        if not targets:
            raise BehaviorValidationError(
                f"{source}: pose behavior {name!r} names no joint targets."
            )
        duration = spec.get("duration")
        if duration is not None:
            try:
                duration = float(duration)
            except (TypeError, ValueError):
                raise BehaviorValidationError(
                    f"{source}: behavior {name!r} duration must be a number, got {duration!r}"
                )
            if duration <= 0:
                raise BehaviorValidationError(
                    f"{source}: behavior {name!r} duration must be > 0, got {duration:g}"
                )
        interp = self._check_interpolation(name, spec, source)
        self._check_limits(name, targets)
        return PoseBehavior(name=name, targets=targets, duration=duration, interpolation=interp)

    def _parse_trajectory(self, name: str, spec: dict, source: str) -> TrajectoryBehavior:
        raw_kfs = spec.get("keyframes")
        if not isinstance(raw_kfs, list) or not raw_kfs:
            raise BehaviorValidationError(
                f"{source}: trajectory {name!r} needs a non-empty 'keyframes' array."
            )
        interp = self._check_interpolation(name, spec, source)
        keyframes: list[Keyframe] = []
        for i, kf in enumerate(raw_kfs):
            if not isinstance(kf, dict) or "t" not in kf:
                raise BehaviorValidationError(
                    f"{source}: trajectory {name!r} keyframe #{i} must be a table with a 't'."
                )
            try:
                t = float(kf["t"])
            except (TypeError, ValueError):
                raise BehaviorValidationError(
                    f"{source}: trajectory {name!r} keyframe #{i} 't' must be a number, "
                    f"got {kf['t']!r}"
                )
            targets = self._joint_targets(name, {k: v for k, v in kf.items() if k != "t"}, source)
            keyframes.append(Keyframe(t=t, targets=targets))
        self._validate_trajectory_shape(name, keyframes)
        traj = TrajectoryBehavior(name=name, keyframes=tuple(keyframes), interpolation=interp)
        self._validate_trajectory_velocity(traj, speed=1.0)
        return traj

    def _check_interpolation(self, name: str, spec: dict, source: str) -> str:
        interp = str(spec.get("interpolation", "min_jerk")).strip().lower()
        from .schema import INTERPOLATIONS

        if interp not in INTERPOLATIONS:
            raise BehaviorValidationError(
                f"{source}: behavior {name!r} interpolation={interp!r}; expected one of "
                f"{list(INTERPOLATIONS)}"
            )
        return interp

    def _validate_trajectory_shape(self, name: str, keyframes: list[Keyframe]) -> None:
        times = [kf.t for kf in keyframes]
        if times[0] != 0.0:
            raise BehaviorValidationError(
                f"behavior {name!r}: first keyframe must be at t=0 (got t={times[0]:g})."
            )
        for a, b in zip(times, times[1:]):
            if b <= a:
                raise BehaviorValidationError(
                    f"behavior {name!r}: keyframe times must strictly increase "
                    f"(got {a:g} then {b:g})."
                )
        # Every joint the trajectory ever moves must be initialized at t=0, so the
        # whole path is velocity-checkable without the live pose.
        moved = set().union(*(kf.targets.keys() for kf in keyframes))
        missing = moved - set(keyframes[0].targets)
        if missing:
            raise BehaviorValidationError(
                f"behavior {name!r}: joint(s) {sorted(missing)} appear in a later "
                "keyframe but not in the t=0 keyframe; initialize every moved joint at "
                "t=0 so the trajectory can be velocity-validated."
            )
        for kf in keyframes:
            self._check_limits(name, kf.targets)

    def _validate_trajectory_velocity(self, traj: TrajectoryBehavior, *, speed: float) -> None:
        """Raise if any segment's peak velocity would exceed a joint's cap at ``speed``."""
        factor = peak_velocity_factor(traj.interpolation)
        # Resolve keyframes forward; joints not yet seen are skipped (they hold the
        # live pose, contributing zero velocity until first specified — and the shape
        # check guarantees moved joints are set at t=0).
        resolved: list[dict[str, float]] = []
        for kf in traj.keyframes:
            cur = dict(resolved[-1]) if resolved else {}
            cur.update(kf.targets)
            resolved.append(cur)
        for k in range(len(traj.keyframes) - 1):
            seg_t = (traj.keyframes[k + 1].t - traj.keyframes[k].t) / speed
            for joint in resolved[k + 1]:
                if joint not in resolved[k]:
                    continue
                j = self._index[joint]
                delta = abs(resolved[k + 1][joint] - resolved[k][joint])
                if delta == 0.0:
                    continue
                peak = factor * delta / seg_t
                cap = self.profile.max_velocity[j]
                if peak > cap * (1.0 + _VEL_EPS):
                    raise BehaviorValidationError(
                        f"behavior {traj.name!r}: joint {joint!r} peak velocity "
                        f"{peak:.2f} exceeds its cap {cap:.2f} (units/s) over the "
                        f"{seg_t:.2f}s segment"
                        + (f" at speed={speed:g}" if speed != 1.0 else "")
                        + "; slow the segment or lower speed."
                    )

    def revalidate_velocity(self, behavior: DataBehavior, *, speed: float) -> None:
        """Re-check velocity caps for a time-scaled behavior (trajectory path).

        Pose behaviors are checked by the executor at plan time (they need the live
        pose); this covers trajectories, whose keyframes are absolute.
        """
        if isinstance(behavior, TrajectoryBehavior):
            self._validate_trajectory_velocity(behavior, speed=speed)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def validate_move_targets(self, targets: dict[str, float]) -> None:
        """Validate ad-hoc ``move()`` joint names + limits (velocity is checked later)."""
        for joint in targets:
            if joint not in self._index:
                raise BehaviorValidationError(
                    f"move(): unknown joint {joint!r} for {self.robot_kind!r}; known "
                    f"joints: {list(self.joint_names)}"
                )
        self._check_limits("move", targets)

    def resolve(self, name: str) -> AnyBehavior:
        """Return the behavior named ``name`` (data behaviors win over procedural)."""
        if name in self._data:
            return self._data[name]
        if name in self._procedural:
            return self._procedural[name]
        raise BehaviorValidationError(
            f"unknown behavior {name!r} for {self.robot_kind!r}; available: {self.names()}"
        )

    def names(self) -> list[str]:
        """Sorted list of every resolvable behavior name."""
        return sorted(set(self._data) | set(self._procedural))

    def summaries(self) -> list[tuple[str, str, str]]:
        """``(name, type, duration)`` rows for every behavior, sorted by name."""
        from .schema import behavior_summary

        rows: dict[str, tuple[str, str, str]] = {}
        for n, b in self._data.items():
            rows[n] = behavior_summary(b)
        for n, p in self._procedural.items():
            rows.setdefault(n, behavior_summary(p))
        return [rows[n] for n in sorted(rows)]


__all__ = ["BehaviorRegistry", "behavior", "USER_BEHAVIORS_PATH"]
