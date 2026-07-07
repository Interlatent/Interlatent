"""The local control loop that turns a behavior into motor commands.

The executor **plans** a behavior into a dense sequence of joint-vector samples at a
fixed control rate, validates it against the profile's velocity caps *before* moving,
then **runs** it by calling the adapter's ordinary ``send_action`` once per tick — the
same fire-and-forget seam the manual ``action()`` path uses. It never touches the
motor bus directly, so the adapter's own delta clamp stays in force as a final
backstop. Because the plan is already velocity-safe by construction (min-jerk / linear
/ trapezoidal within the caps), that clamp should never trigger; if it (or the arm)
saturates for too many consecutive ticks the run aborts cleanly.

Design seam for record-by-demonstration (out of scope now): a plan *is* a joint
target stream, and :meth:`ActHandle.wait` already returns the realized errors — a
recorded joint stream reverses straight back into trajectory keyframes.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from ..node.control import _joint_name
from ..node.teleop.robot_profile import RobotProfile
from .interpolation import build_samples, peak_velocity_factor, smoothstep
from .schema import (
    BehaviorExecutionError,
    BehaviorValidationError,
    DataBehavior,
    PoseBehavior,
    TrajectoryBehavior,
)

_LOG = logging.getLogger("interlatent.behaviors.executor")

_VEL_EPS = 1e-6


@dataclass
class ActResult:
    """Outcome of a behavior run.

    ``reached`` is True when the plan ran to completion; ``aborted`` is True when it
    stopped early (cancellation, adapter error, or saturation). ``joint_error`` is the
    signed ``measured − target`` per joint at the end, in the robot's units.
    """

    behavior: str
    reached: bool
    aborted: bool
    elapsed: float
    joint_error: dict[str, float] = field(default_factory=dict)
    reason: str = ""

    def __bool__(self) -> bool:  # `if robot.act(...):` reads as "did it reach?"
        return self.reached and not self.aborted


@dataclass
class _Plan:
    """A validated, ready-to-run behavior: dense samples in adapter joint order."""

    behavior: str
    samples: np.ndarray  # (N, n_joints), float64
    target: np.ndarray   # (n_joints,) final target
    dt: float


class ActHandle:
    """A running (or finished) non-blocking behavior.

    ``act(wait=False)`` returns this. Call :meth:`wait` to block for the result (which
    re-raises any execution error) or :meth:`cancel` to decelerate smoothly to a stop.
    """

    def __init__(self, behavior: str) -> None:
        self.behavior = behavior
        self._cancel = threading.Event()
        self._done = threading.Event()
        self._result: Optional[ActResult] = None
        self._error: Optional[BaseException] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def cancel_requested(self) -> bool:
        return self._cancel.is_set()

    @property
    def done(self) -> bool:
        return self._done.is_set()

    @property
    def result(self) -> Optional[ActResult]:
        return self._result

    def cancel(self) -> None:
        """Request a smooth deceleration to a stop (not an instant freeze)."""
        self._cancel.set()

    def wait(self, timeout: Optional[float] = None) -> ActResult:
        """Block until the behavior finishes; re-raise any execution error."""
        if self._thread is not None:
            self._done.wait(timeout)
            if not self._done.is_set():
                raise TimeoutError(f"behavior {self.behavior!r} still running after {timeout}s")
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result


class TrajectoryExecutor:
    """Plans and runs behaviors against one connected adapter."""

    def __init__(
        self,
        adapter: Any,
        profile: RobotProfile,
        *,
        control_hz: float = 30.0,
        realtime: bool = True,
        comfort: float = 0.7,
        min_auto_duration: float = 0.6,
        saturation_ticks: int = 12,
        saturation_factor: float = 6.0,
    ) -> None:
        self.adapter = adapter
        self.profile = profile
        self.control_hz = float(control_hz) if control_hz > 0 else 30.0
        self.dt = 1.0 / self.control_hz
        self.realtime = realtime
        self.comfort = comfort
        self.min_auto_duration = min_auto_duration
        self.saturation_ticks = saturation_ticks
        self.saturation_factor = saturation_factor

        # Joint order comes from the adapter's action_features (bare names), which the
        # base adapter guarantees equals the profile's joint order.
        self.features: list[str] = list(getattr(adapter, "action_features", None) or [])
        self.joint_names: list[str] = [_joint_name(f) for f in self.features]
        self.n = len(self.joint_names)
        self._index = {n: i for i, n in enumerate(self.joint_names)}
        self._lo = np.array([lim[0] for lim in profile.joint_limits], dtype=np.float64)
        self._hi = np.array([lim[1] for lim in profile.joint_limits], dtype=np.float64)
        self._cap = np.array(profile.max_velocity, dtype=np.float64)
        # Position-mode joints participate in the saturation (stall) guard; grippers
        # and other non-position DOFs do not — they may never reach a measured target.
        self._is_position = self._position_mask()

    def _position_mask(self) -> np.ndarray:
        specs = list(getattr(self.adapter, "joint_specs", None) or [])
        by_name = {s.name: s for s in specs}
        mask = np.ones(self.n, dtype=bool)
        for i, name in enumerate(self.joint_names):
            spec = by_name.get(name)
            if spec is not None and spec.control_mode != "position":
                mask[i] = False
        return mask

    # ------------------------------------------------------------------
    # Planning (validates + may raise BEFORE any motion)
    # ------------------------------------------------------------------

    def _read_pose(self) -> np.ndarray:
        obs = self.adapter.get_observation()
        out = np.zeros(self.n, dtype=np.float64)
        for i, feature in enumerate(self.features):
            v = obs.get(feature)
            if v is None:
                continue
            try:
                out[i] = float(np.asarray(v).reshape(-1)[0])
            except (TypeError, ValueError, IndexError):
                continue
        return out

    def _auto_duration(self, p0: np.ndarray, p1: np.ndarray, factor: float) -> float:
        """Shortest duration that keeps every joint within ``comfort × cap``."""
        delta = np.abs(p1 - p0)
        needs = factor * delta / (self._cap * self.comfort)
        return float(max(self.min_auto_duration, needs.max() if needs.size else 0.0))

    def _check_segment_velocity(
        self, behavior: str, p0: np.ndarray, p1: np.ndarray, seg_t: float, factor: float, speed: float
    ) -> None:
        delta = np.abs(p1 - p0)
        peak = factor * delta / seg_t
        over = peak > self._cap * (1.0 + _VEL_EPS)
        if np.any(over):
            j = int(np.argmax(peak - self._cap))
            raise BehaviorValidationError(
                f"behavior {behavior!r}: joint {self.joint_names[j]!r} peak velocity "
                f"{peak[j]:.2f} exceeds its cap {self._cap[j]:.2f} (units/s) over the "
                f"{seg_t:.2f}s segment"
                + (f" at speed={speed:g}" if speed != 1.0 else "")
                + "; slow it down or lower speed."
            )

    def plan(self, behavior: DataBehavior, *, speed: float = 1.0) -> _Plan:
        """Resolve, velocity-validate, and sample a behavior. Reads the live pose."""
        if speed <= 0:
            raise BehaviorValidationError(f"speed must be > 0, got {speed:g}")
        if isinstance(behavior, PoseBehavior):
            return self._plan_pose(behavior, speed)
        if isinstance(behavior, TrajectoryBehavior):
            return self._plan_trajectory(behavior, speed)
        raise BehaviorValidationError(f"cannot plan behavior of type {type(behavior).__name__}")

    def _plan_pose(self, behavior: PoseBehavior, speed: float) -> _Plan:
        live = self._read_pose()
        target = live.copy()
        for name, value in behavior.targets.items():
            target[self._index[name]] = value
        target = np.clip(target, self._lo, self._hi)
        factor = peak_velocity_factor(behavior.interpolation)
        if behavior.duration is None:
            # Auto: size to the caps, then a faster `speed` shrinks it (and may raise).
            dur = self._auto_duration(live, target, factor) / speed
        else:
            dur = behavior.duration / speed
        self._check_segment_velocity(behavior.name, live, target, dur, factor, speed)
        samples = build_samples([live, target], [dur], behavior.interpolation, self.dt)
        return _Plan(behavior=behavior.name, samples=samples, target=target, dt=self.dt)

    def _plan_trajectory(self, behavior: TrajectoryBehavior, speed: float) -> _Plan:
        live = self._read_pose()
        # Resolve keyframes to full joint vectors; unspecified joints hold the previous
        # value (the first keyframe's unspecified joints hold the live pose).
        waypoints: list[np.ndarray] = []
        prev = live.copy()
        for kf in behavior.keyframes:
            cur = prev.copy()
            for name, value in kf.targets.items():
                cur[self._index[name]] = value
            cur = np.clip(cur, self._lo, self._hi)
            waypoints.append(cur)
            prev = cur
        times = [kf.t for kf in behavior.keyframes]
        seg_durs = [(times[k + 1] - times[k]) / speed for k in range(len(times) - 1)]
        factor = peak_velocity_factor(behavior.interpolation)
        for k, seg_t in enumerate(seg_durs):
            self._check_segment_velocity(
                behavior.name, waypoints[k], waypoints[k + 1], seg_t, factor, speed
            )

        # Approach: if the live pose differs from the trajectory's first waypoint,
        # prepend an auto-timed, cap-respecting min-jerk move so the start is smooth.
        pieces: list[np.ndarray] = []
        wp0 = waypoints[0]
        if np.max(np.abs(live - wp0)) > 1e-6:
            a_dur = self._auto_duration(live, wp0, peak_velocity_factor("min_jerk"))
            approach = build_samples([live, wp0], [a_dur], "min_jerk", self.dt)
            pieces.append(approach[:-1])  # drop wp0; the trajectory begins with it
        traj = build_samples(waypoints, seg_durs, behavior.interpolation, self.dt)
        pieces.append(traj)
        samples = np.concatenate(pieces, axis=0) if len(pieces) > 1 else pieces[0]
        return _Plan(behavior=behavior.name, samples=samples, target=waypoints[-1], dt=self.dt)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def act(self, behavior: DataBehavior, *, speed: float = 1.0, wait: bool = True):
        """Plan then run ``behavior``. Blocking returns :class:`ActResult`; ``wait=False``
        returns an :class:`ActHandle`."""
        plan = self.plan(behavior, speed=speed)  # may raise before any motion
        handle = ActHandle(behavior.name)
        if wait:
            self._run(plan, handle)
            if handle._error is not None:
                raise handle._error
            return handle._result
        thread = threading.Thread(
            target=self._run, args=(plan, handle), name=f"behavior:{behavior.name}", daemon=True
        )
        handle._thread = thread
        thread.start()
        return handle

    def _to_action(self, vec: np.ndarray) -> dict[str, float]:
        return {f: float(vec[i]) for i, f in enumerate(self.features)}

    def _run(self, plan: _Plan, handle: ActHandle) -> None:
        start = time.monotonic()
        samples = plan.samples
        sat_margin = self.saturation_factor * self._cap * plan.dt
        sat_count = np.zeros(self.n, dtype=int)
        last_cmd: Optional[np.ndarray] = None
        prev_cmd: Optional[np.ndarray] = None
        aborted = False
        reason = ""
        error: Optional[BaseException] = None
        try:
            for i in range(len(samples)):
                if handle.cancel_requested:
                    self._decelerate(last_cmd, prev_cmd, plan, start, i)
                    aborted, reason = True, "cancelled"
                    break
                cmd = np.clip(samples[i], self._lo, self._hi)
                self.adapter.send_action(self._to_action(cmd))
                prev_cmd, last_cmd = last_cmd, cmd

                measured = self._read_pose()
                lag = np.abs(cmd - measured)
                over = (lag > sat_margin) & self._is_position
                sat_count = np.where(over, sat_count + 1, 0)
                if np.any(sat_count >= self.saturation_ticks):
                    j = int(np.argmax(sat_count))
                    self._decelerate(last_cmd, prev_cmd, plan, start, i + 1)
                    raise BehaviorExecutionError(
                        f"behavior {plan.behavior!r} aborted: joint "
                        f"{self.joint_names[j]!r} lagged the commanded target by "
                        f"{lag[j]:.3f} for {self.saturation_ticks} consecutive ticks "
                        "(safety-clamp saturation / stalled joint)."
                    )
                self._pace(start, i + 1, plan.dt)
        except BehaviorExecutionError as exc:
            aborted, reason, error = True, str(exc), exc
        except Exception as exc:  # noqa: BLE001 — any adapter error aborts cleanly
            aborted, reason = True, f"adapter error: {exc}"
            error = BehaviorExecutionError(
                f"behavior {plan.behavior!r} aborted on adapter error: {exc}"
            )

        elapsed = time.monotonic() - start
        try:
            measured = self._read_pose()
            joint_error = {n: float(measured[i] - plan.target[i]) for i, n in enumerate(self.joint_names)}
        except Exception:  # noqa: BLE001
            joint_error = {}
        handle._result = ActResult(
            behavior=plan.behavior,
            reached=not aborted,
            aborted=aborted,
            elapsed=elapsed,
            joint_error=joint_error,
            reason=reason,
        )
        handle._error = error
        handle._done.set()

    def _decelerate(
        self,
        last_cmd: Optional[np.ndarray],
        prev_cmd: Optional[np.ndarray],
        plan: _Plan,
        start: float,
        tick: int,
    ) -> None:
        """Ramp the commanded velocity to zero over a short window (smooth stop)."""
        if last_cmd is None:
            return
        vel = (last_cmd - prev_cmd) if prev_cmd is not None else np.zeros(self.n)
        stop_ticks = max(1, int(round(0.2 / plan.dt)))  # ~200 ms
        pos = last_cmd.astype(np.float64).copy()
        for k in range(1, stop_ticks + 1):
            factor = 1.0 - smoothstep(k / stop_ticks)  # 1 → 0
            pos = np.clip(pos + vel * factor, self._lo, self._hi)
            try:
                self.adapter.send_action(self._to_action(pos))
            except Exception:  # noqa: BLE001 — best-effort stop
                _LOG.warning("adapter error during deceleration", exc_info=True)
                return
            self._pace(start, tick + k, plan.dt)

    def _pace(self, start: float, sent: int, dt: float) -> None:
        if not self.realtime:
            return
        target_time = start + sent * dt
        remaining = target_time - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)


__all__ = ["TrajectoryExecutor", "ActResult", "ActHandle"]
