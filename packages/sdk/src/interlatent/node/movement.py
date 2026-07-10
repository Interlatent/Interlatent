"""Unified movement-command ingress for the robot node.

Every physical movement the node executes — teleoperation, human
intervention, and policy inference — is decided here, at one point of
access, before it reaches the SafetyGate and the single
``robot.send_action`` sink. Today those sources arrive over two transports
(teleop over QUIC/WS datagrams, inference over the DRTC client); the goal of
the unify-on-QUIC work is to collapse the *decision* — and later the
transport — behind this module so the control loop reasons about one thing:
"who is driving the robot this tick, and hand me their command."

This module owns three pieces:

* :class:`MovementSource` — the vocabulary naming who is driving the robot.
* :class:`Arbiter` — the single authority that decides, per control tick,
  which source wins. Its priority order is the one place to reason about
  (and, later, to slot an e-stop above everything).
* :class:`CommandBus` — the node-side aggregator the control loop consults.
  It holds the realtime teleop ingress and whether a policy is loaded, and
  returns the arbitrated source each tick.

**Scope (Phase 1).** This unifies the *decision*. The branch bodies that
actually produce the action for the winning source still live in the control
loop, because each has heterogeneous side effects (record capture, policy
chunk flush, smoother reset, latency accounting) that must not be reordered.
Phase 2 brings policy inference onto the same QUIC transport and folds action
production behind this bus. Kept stdlib-only so the node stays importable on
a barebones Pi.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class MovementSource(str, Enum):
    """Who is commanding the robot on a given control tick.

    ``str``-valued so the member's value doubles as the recorded
    ``control_source`` label in the dataset — the existing wire labels stay
    byte-for-byte identical.
    """

    TELEOP = "teleop"   # a human is driving (also the label for intervention today)
    HOLD = "hold"       # no policy loaded and human disengaged: servos hold in place
    POLICY = "policy"   # autonomous inference chunk

    # Reserved for Phase 2 — declared here so the priority ladder has one home:
    #   ESTOP        — highest priority; overrides every source (safety latch)
    #   INTERVENTION — human override *of a running policy*; identical on the
    #                  wire to TELEOP today, split out later so DAgger gets clean
    #                  correction labels.


@dataclass(frozen=True)
class TeleopReadiness:
    """Precomputed teleop-side inputs the arbiter needs, so the decision itself
    stays a pure function of booleans (easy to reason about and test)."""

    engaged: bool     # a frame is present, engaged, and the deadman is held
    gated: bool       # a SafetyGate exists for this robot kind
    schema_ok: bool   # action_keys present and match the profile's joint arity

    @property
    def teleop_available(self) -> bool:
        return self.engaged and self.gated and self.schema_ok


class Arbiter:
    """Single authority deciding which source controls the robot this tick.

    Priority, highest first:

        1. ``TELEOP`` — a human is engaged (deadman held) *and* the gated,
           schema-matched teleop path is available for this robot.
        2. ``HOLD``   — no policy is loaded (a teleop-recording assignment) and
           the human is not driving: send nothing so the servos hold, while the
           loop keeps recording a continuous episode.
        3. ``POLICY`` — autonomous inference.

    A future ``ESTOP`` source will sit above ``TELEOP``, and ``INTERVENTION``
    will split off from it. Keeping the decision here means those changes land
    in exactly one place instead of an inline ``if/elif`` in the control loop.

    This preserves the original control-loop semantics exactly: ``TELEOP`` iff
    the old ``teleop_ok`` was true; otherwise ``HOLD`` iff ``not
    policy_enabled``; otherwise ``POLICY``.
    """

    def decide(
        self, *, teleop_ready: TeleopReadiness, policy_enabled: bool
    ) -> MovementSource:
        if teleop_ready.teleop_available:
            return MovementSource.TELEOP
        if not policy_enabled:
            return MovementSource.HOLD
        return MovementSource.POLICY


class CommandBus:
    """One point of access to the node's movement ingress.

    Each tick the control loop asks the bus for the latest realtime (teleop)
    frame and for the arbitrated source, instead of poking the teleop channel
    and inspecting the policy flag inline. The bus does not own the action
    production or the ``send_action`` sink (Phase 1) — it owns the *decision*.
    """

    def __init__(
        self,
        *,
        teleop_channel: Optional[Any],
        teleop_gate: Optional[Any],
        teleop_profile: Optional[Any],
        policy_enabled: bool,
        arbiter: Optional[Arbiter] = None,
    ) -> None:
        self._teleop_channel = teleop_channel
        self._teleop_gate = teleop_gate
        self._teleop_profile = teleop_profile
        self._policy_enabled = policy_enabled
        self._arbiter = arbiter or Arbiter()

    def sample_teleop(self) -> Optional[Any]:
        """The latest teleop frame, or ``None`` when no producer is connected or
        the last frame is stale (the channel drops frames older than ~250 ms)."""
        if self._teleop_channel is None:
            return None
        return self._teleop_channel.latest_frame()

    def readiness(self, frame: Optional[Any], action_keys: list) -> TeleopReadiness:
        engaged = bool(frame and frame.engaged and frame.deadman)
        gated = self._teleop_gate is not None
        # ``teleop_profile is not None`` is implied by ``gated`` (the gate is
        # only built when a profile exists), but we guard explicitly so
        # readiness never dereferences a missing profile.
        schema_ok = bool(
            action_keys
            and self._teleop_profile is not None
            and len(action_keys) == len(self._teleop_profile.joint_names)
        )
        return TeleopReadiness(engaged=engaged, gated=gated, schema_ok=schema_ok)

    def arbitrate(self, frame: Optional[Any], action_keys: list) -> MovementSource:
        """Decide which source controls the robot this tick."""
        return self._arbiter.decide(
            teleop_ready=self.readiness(frame, action_keys),
            policy_enabled=self._policy_enabled,
        )
