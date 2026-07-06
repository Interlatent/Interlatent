"""``interlatent.Robot`` — a dead-simple facade for named, offline behaviors.

    import interlatent as il

    robot = il.Robot("so101", port="/dev/ttyACM0")
    robot.act("home")                       # move to the profile rest pose, block
    robot.act("hello")                      # play the built-in wave
    robot.act("home", speed=0.5)            # time-scale a behavior (gentler)
    print(robot.pose())                     # {joint: position}
    robot.move(wrist_roll=30, duration=0.5) # ad-hoc single joint move
    robot.close()

Or as a context manager (``with il.Robot(...) as robot:``).

This is the *manual*, no-cloud path. It resolves the robot kind to an adapter exactly
as ``interlatent-act`` does, opens it, loads the layered behavior registry for that
robot, and drives motion through the :class:`~interlatent.behaviors.executor.TrajectoryExecutor`
— which samples validated min-jerk trajectories through the adapter's ordinary action
path (delta clamp intact). It never runs a policy and needs no API key.

**Arbitration.** If the node daemon (or another ``Robot``) is already driving this
robot, the constructor raises :class:`~interlatent.behaviors.arbitration.RobotBusyError`
rather than fighting it for the bus; pass ``force=True`` to override (dangerous — it
can corrupt a live inference session). Detection is best-effort (client-side lockfile +
OS serial lock); see :mod:`interlatent.behaviors.arbitration`.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

from .behaviors.arbitration import acquire_bus_lock
from .behaviors.executor import ActHandle, ActResult, TrajectoryExecutor
from .behaviors.registry import BehaviorRegistry
from .behaviors.schema import PoseBehavior, ProceduralBehavior
from .node.control import _joint_name
from .node.teleop.robot_profile import get_profile

_LOG = logging.getLogger("interlatent.robot")

# What act()/move() return: a result when blocking, a handle when wait=False.
ActReturn = Union[ActResult, ActHandle]


class Robot:
    """A connected robot you can drive with named behaviors and ad-hoc moves."""

    def __init__(
        self,
        robot_type: str,
        *,
        port: Optional[str] = None,
        behaviors: "str | Path | None" = None,
        robot_arg: Optional[dict[str, str]] = None,
        cameras: Optional[dict[str, str]] = None,
        control_hz: float = 30.0,
        realtime: bool = True,
        force: bool = False,
        connect: bool = True,
    ) -> None:
        from .adapters import resolve_adapter  # lazy: keeps `import interlatent` light

        self.robot_type = robot_type
        self._adapter = resolve_adapter(
            robot_type, port=port, extra=robot_arg, cameras=cameras
        )
        # The adapter picks its concrete kind (e.g. yam vs yam_left); use it for the
        # profile + registry so bimanual/unimanual resolve correctly.
        self.robot_kind = getattr(self._adapter, "robot_kind", robot_type)

        self._lock = acquire_bus_lock(self.robot_kind, port, robot_arg, force=force)
        self._closed = False
        self._executor: Optional[TrajectoryExecutor] = None
        try:
            profile = get_profile(self.robot_kind)
            if profile is None:
                raise ValueError(
                    f"no RobotProfile for robot kind {self.robot_kind!r}: behaviors "
                    "need a profile. Add one in interlatent.node.teleop.robot_profile."
                )
            self._profile = profile
            self._registry = BehaviorRegistry.for_robot(self.robot_kind, explicit=behaviors)
            if connect:
                self._adapter.connect()
                self._executor = TrajectoryExecutor(
                    self._adapter, profile, control_hz=control_hz, realtime=realtime
                )
        except BaseException:
            self._lock.release()
            raise
        self._control_hz = control_hz
        self._realtime = realtime

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def behaviors(self) -> list[str]:
        """Names of every behavior available on this robot."""
        return self._registry.names()

    def pose(self) -> dict[str, float]:
        """The live joint positions as ``{joint_name: value}`` (robot units)."""
        obs = self._adapter.get_observation()
        out: dict[str, float] = {}
        for feature in self._adapter.action_features:
            if feature in obs:
                try:
                    out[_joint_name(feature)] = float(obs[feature])
                except (TypeError, ValueError):
                    continue
        return out

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------

    def act(self, name: str, *, speed: float = 1.0, wait: bool = True) -> ActReturn:
        """Run the named behavior. Blocks by default, returning an :class:`ActResult`;
        ``wait=False`` returns an :class:`ActHandle` with ``.wait()`` / ``.cancel()``.

        ``speed`` time-scales the behavior (``0.5`` = half speed / gentler; ``2.0`` =
        twice as fast — which **raises** before moving if it would break a velocity cap).
        """
        self._ensure_open()
        behavior = self._registry.resolve(name)
        if isinstance(behavior, ProceduralBehavior):
            return self._run_procedural(behavior, wait=wait)
        return self._executor.act(behavior, speed=speed, wait=wait)

    def move(
        self, *, duration: float = 0.5, speed: float = 1.0, wait: bool = True, **joints: float
    ) -> ActReturn:
        """Move the named joints to absolute targets over ``duration`` seconds.

        Unnamed joints hold their current position. Targets are joint values in the
        robot's own units (degrees for SO-101, radians for YAM revolute joints). Raises
        :class:`~interlatent.behaviors.schema.BehaviorValidationError` on an unknown
        joint, an out-of-limit target, or a too-fast move — all *before* any motion.
        """
        self._ensure_open()
        if not joints:
            raise ValueError("move() needs at least one joint=value target.")
        targets = {k: float(v) for k, v in joints.items()}
        self._registry.validate_move_targets(targets)
        behavior = PoseBehavior(name="move", targets=targets, duration=float(duration))
        return self._executor.act(behavior, speed=speed, wait=wait)

    def _run_procedural(self, behavior: ProceduralBehavior, *, wait: bool) -> ActReturn:
        import threading
        import time

        handle = ActHandle(behavior.name)

        def _invoke() -> None:
            start = time.monotonic()
            error: Optional[BaseException] = None
            try:
                behavior.fn(self)
            except Exception as exc:  # noqa: BLE001 — surface via the handle
                from .behaviors.schema import BehaviorExecutionError

                error = BehaviorExecutionError(
                    f"procedural behavior {behavior.name!r} raised: {exc}"
                )
            handle._result = ActResult(
                behavior=behavior.name,
                reached=error is None,
                aborted=error is not None,
                elapsed=time.monotonic() - start,
                reason="" if error is None else str(error),
            )
            handle._error = error
            handle._done.set()

        if wait:
            _invoke()
            if handle._error is not None:
                raise handle._error
            return handle._result
        thread = threading.Thread(target=_invoke, name=f"behavior:{behavior.name}", daemon=True)
        handle._thread = thread
        thread.start()
        return handle

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Robot is closed; open a new Robot to drive it again.")
        if self._executor is None:
            raise RuntimeError("Robot adapter is not connected.")

    def close(self) -> None:
        """Disconnect the adapter and release the bus lock (idempotent)."""
        if self._closed:
            return
        self._closed = True
        try:
            self._adapter.disconnect()
        except Exception:  # noqa: BLE001
            _LOG.warning("adapter disconnect failed", exc_info=True)
        finally:
            self._lock.release()

    def __enter__(self) -> "Robot":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:  # best-effort safety net
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass


__all__ = ["Robot"]
