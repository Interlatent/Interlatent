"""The formal robot-adapter interface and the shared manual ``action()`` seam.

Every robot adapter (`interlatent.adapters.<kind>`) exposes the same two-level
action interface, sitting **below** the DRTC ``ActionSchedule`` — a final actuator,
not a source that merges into the schedule (see ADR 0013):

- ``send_action(action)`` — non-blocking, fire-and-forget, latest-wins. The engine
  control loop calls it once per tick; each action is a *waypoint*, not a destination.
  Adapters implement this themselves (it talks to their motor stack).
- ``action(**named, hold_missing=…, timeout=…)`` — the manual/programmatic call:
  **named joints**, **block-then-settle**. It is defined *once* here in
  :class:`ManualActionInterface` and inherited by every adapter, composed entirely
  from the adapter's own ``send_action`` + ``get_observation``.

Manual ``action()`` is human-driven motion, so it reuses the existing client-side
safety model rather than a bespoke guard: every commanded step passes through the
:class:`~interlatent.node.teleop.safety.SafetyGate` (workspace / velocity / deadman)
— whose velocity-limited ``step()`` *is* the block-then-settle stepping mechanism —
and then the adapter's own delta clamp inside ``send_action``. Joint ranges come from
the existing :class:`~interlatent.node.teleop.robot_profile.RobotProfile`; a robot kind
with no profile **refuses** manual motion rather than running unguarded.

All actions are joint-space: a vector of joint targets, one per ``action_feature``.
There is no IK / Cartesian frame anywhere in the robot-side stack.

Import-weight note: this module must stay importable on a barebones Pi — it imports
only numpy and the (lerobot-free) teleop safety helpers, never lerobot/almond_axol.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Protocol, Sequence, runtime_checkable

import numpy as np

# node.control imports numpy only at module load (lerobot is imported lazily inside
# its functions), so reusing these two pure helpers keeps base.py Pi-importable.
from ..node.control import _extract_joint_state, _joint_name
from ..node.teleop.robot_profile import RobotProfile, get_profile
from ..node.teleop.safety import SafetyGate, TargetSample

_LOG = logging.getLogger("interlatent.adapters.base")

# Control modes that wait for the joint to physically reach its target before the
# move is considered settled. Anything else (a gripper that closes on an object and
# never reaches its position target; a velocity/effort DOF) settles on "command
# issued" — see ``ManualActionInterface._settled``.
_POSITION_MODE = "position"


@dataclass(frozen=True)
class JointSpec:
    """Per-joint metadata an adapter declares for the manual ``action()`` path.

    Ranges are **not** declared here — they come from the robot's
    :class:`RobotProfile` (the established source of truth, also used by teleop), so
    limits live in exactly one place. The adapter declares only what the profile does
    not carry: how the joint settles.

    - ``name``: bare joint name (e.g. ``"shoulder_pan"``), matching the profile's
      ``joint_names`` and the bare form of the adapter's ``action_features``.
    - ``control_mode``: ``"position"`` waits for ``|measured - target| <=
      settle_tolerance``; any other value (e.g. ``"gripper"``, ``"effort"``,
      ``"velocity"``) settles immediately once commanded.
    - ``settle_tolerance``: convergence band for a position joint, in the joint's
      own units (degrees for lerobot SO-101). Should be >= the joint's
      resolution/deadband or it will never settle.
    """

    name: str
    control_mode: str = _POSITION_MODE
    settle_tolerance: float = 2.0


@runtime_checkable
class RobotAdapter(Protocol):
    """The duck type every adapter satisfies (``AxolNativeRobot`` already does).

    Lifecycle + observe/act, plus the metadata the manual seam needs. The concrete
    ``action()`` is supplied by :class:`ManualActionInterface`.
    """

    robot_kind: str

    @property
    def action_features(self) -> list[str]: ...

    @property
    def joint_specs(self) -> Sequence[JointSpec]: ...

    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def get_observation(self) -> Dict[str, Any]: ...

    def send_action(self, action: Dict[str, Any]) -> Any: ...


class ManualActionInterface:
    """Mixin providing the manual, block-then-settle ``action()`` call.

    Mixed into every adapter. Relies on the host exposing ``robot_kind``,
    ``action_features`` (ordered ``"<name>.pos"`` keys), ``joint_specs`` (ordered,
    aligned with ``action_features``), ``get_observation`` and ``send_action``.
    Never used on the engine path — that path calls ``send_action`` directly.
    """

    # Provided by the concrete adapter.
    robot_kind: str
    action_features: list[str]
    joint_specs: Sequence[JointSpec]

    def get_observation(self) -> Dict[str, Any]:  # pragma: no cover - provided by host
        raise NotImplementedError

    def send_action(self, action: Dict[str, Any]) -> Any:  # pragma: no cover
        raise NotImplementedError

    def action(
        self,
        *,
        hold_missing: bool = False,
        timeout: float = 10.0,
        rate_hz: float = 30.0,
        **named: float,
    ) -> None:
        """Drive the named joints to absolute targets, blocking until settled.

        Named joints are the contract: ``adapter.action(shoulder_pan=0.0,
        gripper=80.0)``. The call returns once every position joint is within its
        ``settle_tolerance`` of its target, and **raises** ``TimeoutError`` if that
        does not happen within ``timeout`` seconds.

        - **Unknown joint name → ``ValueError``** (cross-embodiment / typo guard;
          ``hold_missing`` does not suppress it).
        - **Omitted known joint → ``ValueError``, unless ``hold_missing=True``** →
          held at its measured present position (read once, up front), and the held
          joints are logged so a silent embodiment mismatch is visible.
        - Targets are pre-validated against the robot profile's joint limits and a
          violation **raises before any motion**.

        Motion is gated: each step is velocity/workspace/deadman-clamped by the
        :class:`SafetyGate` and then delta-clamped inside ``send_action``.
        """
        names = [_joint_name(f) for f in self.action_features]
        name_to_idx = {n: i for i, n in enumerate(names)}

        # Unknown-name guard (always raises).
        unknown = [k for k in named if k not in name_to_idx]
        if unknown:
            raise ValueError(
                f"action() got unknown joint(s) {unknown} for robot "
                f"{self.robot_kind!r}; known joints: {names}"
            )

        profile = get_profile(self.robot_kind)
        if profile is None:
            raise RuntimeError(
                f"no RobotProfile for robot kind {self.robot_kind!r}: manual "
                "action() refuses to run without a safety envelope. Add a profile "
                "in interlatent.node.teleop.robot_profile."
            )
        if list(profile.joint_names) != names:
            raise RuntimeError(
                f"profile/adapter joint mismatch for {self.robot_kind!r}: profile "
                f"{list(profile.joint_names)} != adapter {names}. They must share "
                "order so the SafetyGate operates in the adapter's joint frame."
            )

        # One measured snapshot, used both to fill held joints and as the gate's
        # starting pose. No extra bus read per partial call.
        snapshot = self._joint_vector(self.get_observation())

        held: list[str] = []
        target = snapshot.copy()
        for i, name in enumerate(names):
            if name in named:
                target[i] = float(named[name])
            elif hold_missing:
                held.append(name)  # keep snapshot value
            else:
                raise ValueError(
                    f"action() missing joint {name!r}; pass it, or hold_missing=True "
                    "to keep it at its present position."
                )
        if held:
            _LOG.info(
                "action(): holding %d unspecified joint(s) at measured position: %s",
                len(held), held,
            )

        # Range pre-validation (raise before moving). Held joints are at the live
        # pose, so only validate the ones the caller explicitly commanded.
        lo = np.array([lim[0] for lim in profile.joint_limits], dtype=np.float32)
        hi = np.array([lim[1] for lim in profile.joint_limits], dtype=np.float32)
        for i, name in enumerate(names):
            if name in named and not (lo[i] <= target[i] <= hi[i]):
                raise ValueError(
                    f"target {target[i]:.3f} for joint {name!r} is outside its "
                    f"limit [{lo[i]:.3f}, {hi[i]:.3f}]"
                )

        self._run_settle(profile, names, target, timeout=timeout, rate_hz=rate_hz)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _joint_vector(self, obs: Dict[str, Any]) -> np.ndarray:
        """Measured joint positions in ``action_features`` order."""
        return _extract_joint_state(obs, list(self.action_features))

    def _settled(
        self,
        current: np.ndarray,
        target: np.ndarray,
        commanded: "np.ndarray | None" = None,
    ) -> bool:
        """True once the move is complete.

        - **Position joints** settle when the *measured* position is within
          ``settle_tolerance`` of the target.
        - **Non-position joints** (gripper / effort / velocity) don't track to a
          measured position — a gripper closing on an object never reaches its
          position target — so they settle once the *commanded* trajectory has
          reached the target (the SafetyGate has finished issuing the move). This
          still **waits for the command to be fully issued**, so the joint isn't
          cut off mid-approach, but does not block on a measured value it may never
          reach.

        Until the first command has been issued (``commanded is None``) the move is
        never considered settled, so at least one action is always sent — a target
        that happens to start within tolerance still gets commanded.
        """
        if commanded is None:
            return False
        for i, (spec, cur, tgt) in enumerate(zip(self.joint_specs, current, target)):
            if spec.control_mode == _POSITION_MODE:
                if abs(float(cur) - float(tgt)) > spec.settle_tolerance:
                    return False
            else:
                # Non-position: wait until the commanded trajectory reaches target.
                if abs(float(commanded[i]) - float(tgt)) > 1e-3:
                    return False
        return True

    def _run_settle(
        self,
        profile: RobotProfile,
        names: Sequence[str],
        target: np.ndarray,
        *,
        timeout: float,
        rate_hz: float,
    ) -> None:
        control_dt = 1.0 / rate_hz if rate_hz > 0 else 1.0 / 30.0
        gate = SafetyGate(profile=profile, control_dt=control_dt)
        deadline = time.monotonic() + timeout

        # The last vector we actually commanded. ``None`` until the first command is
        # issued — _settled() treats that as "not settled yet", so we always send at
        # least one action even if the arm starts within tolerance of the target.
        last_commanded: "np.ndarray | None" = None

        while True:
            now = time.monotonic()
            current = self._joint_vector(self.get_observation())

            if self._settled(current, target, last_commanded):
                return
            if now >= deadline:
                err = np.abs(current - target)
                worst = int(np.argmax(err))
                raise TimeoutError(
                    f"action() did not settle within {timeout:.1f}s; worst joint "
                    f"{names[worst]!r} still {err[worst]:.3f} from target."
                )

            gate.submit(
                TargetSample(
                    joints=target.astype(np.float32),
                    deadman_active=True,   # a manual action is intentional motion
                    confidence=1.0,
                    received_at=now,
                    producer_timestamp_ns=time.monotonic_ns(),
                )
            )
            commanded, status = gate.step(current, now=now)
            if status != "ok":
                # The gate is idling (e.g. it just anchored on the first tick); retry
                # next tick. It should reach "ok" once a fresh engaged sample is seen.
                _LOG.debug("action(): gate status=%s (not commanding this tick)", status)
            else:
                last_commanded = np.asarray(commanded, dtype=np.float32)
                self.send_action(
                    {f: float(commanded[i]) for i, f in enumerate(self.action_features)}
                )

            elapsed = time.monotonic() - now
            if elapsed < control_dt:
                time.sleep(control_dt - elapsed)


__all__ = ["JointSpec", "RobotAdapter", "ManualActionInterface"]
