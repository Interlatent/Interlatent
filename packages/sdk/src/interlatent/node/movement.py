"""Unified movement ingress *and* motion path for the robot node.

Every physical movement the node executes — teleoperation, human intervention,
and policy inference — is decided **and produced** here, at one point of access,
before it reaches the single ``robot.send_action`` sink. The control loop asks
one question per tick: *"drive the robot, and tell me what happened."*

This module owns:

* :class:`MovementSource` — the vocabulary naming who is driving the robot.
* :class:`Arbiter` — the single authority that decides, per control tick,
  which source wins. Its priority ladder is the one place to reason about.
* :class:`CommandBus` — the aggregator the control loop consults. It holds the
  realtime teleop ingress, the safety gate, the smoother, and the robot, and
  its :meth:`CommandBus.drive` runs the whole motion path.
* :class:`TickOutcome` — what ``drive()`` reports back, so the loop can record
  and instrument without re-deriving anything.
* :class:`TickVerdict` — the vocabulary an adapter's optional ``pre_tick``
  guard speaks, for per-robot conditions that must be checked *before* any
  movement is arbitrated (a dead session, a stale telemetry read).

**What the bus owns, and what it does not.** The bus owns *motion*: arbitration,
action production, the :class:`SafetyGate`, the delta clamp, ``send_action``, and
the discontinuity bookkeeping that goes with them (schedule flush, smoother
reset). It does **not** own the *dataset* — recording, preview video, the
feature report, latency logging, and pacing stay in the loop
(:mod:`interlatent.node.looprunner`). ``drive()`` reports what should be
recorded; it never records.

Two consequences of that split are load-bearing:

* ``CONTEXT.md``'s *"all motion converges on one node-side path: absolute target
  → SafetyGate → send_action"* becomes structurally true rather than a
  convention four forked loops were each trusted to honour.
* JPEG encoding and dataset concerns stay out of a module named ``movement``.

**Wire helpers are injected, never imported.** :mod:`interlatent.node.control`
imports this module, so importing it back would cycle. The four helpers the
motion path needs travel in as :class:`WireHelpers`. That is not merely
cycle-avoidance: ``coerce`` is where the OLD→NEW calibration affine does or does
not get applied, and that differs by caller (a policy commands in *model* frame,
a human commands the arm directly). Injecting it keeps the frame policy with the
caller instead of hiding it in the bus.

**On numpy.** An earlier revision of this module was stdlib-only "so the node
stays importable on a barebones Pi". The invariant that is actually enforced
(``tests/test_lazy_imports.py``) is *no optional extra at import time*; numpy is
a base dependency, and ``node/teleop/safety.py`` and ``node/smoothing.py``
already import it at module scope. Nothing here may import an extra.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np

from .teleop.safety import TargetSample

_LOG = logging.getLogger(__name__)


class MovementSource(str, Enum):
    """Who is commanding the robot on a given control tick.

    ``str``-valued so a member's value doubles as the recorded
    ``control_source`` label in the dataset — the existing wire labels stay
    byte-for-byte identical.

    ``ESTOP`` is the one member whose value is **not** a ``control_source``:
    e-stop ticks are never captured, so the dataset's three-value contract
    (``{"policy","teleop","hold"}``, ``CONTEXT.md``) is preserved.
    :attr:`TickOutcome.control_source` is ``None`` on those ticks.
    """

    ESTOP = "estop"     # safety latch held — overrides every other source
    TELEOP = "teleop"   # a human is driving (also the label for intervention today)
    HOLD = "hold"       # no policy loaded and human disengaged: servos hold in place
    POLICY = "policy"   # autonomous inference chunk

    # Reserved:
    #   INTERVENTION — human override *of a running policy*; identical on the
    #                  wire to TELEOP today, split out later so interventions get
    #                  clean correction labels.


class TickVerdict(str, Enum):
    """An adapter's per-tick pre-flight result, checked before any arbitration.

    This is the seam for robot conditions the generic path cannot know about —
    a supervising daemon that died, a safety FSM that latched, a telemetry read
    that went stale mid-reconnect. Adapters that have no such conditions (most)
    implement no guard at all.
    """

    PROCEED = "proceed"
    #: No motion and **no capture** this tick, but the episode continues. For a
    #: stale telemetry read: recording a stale pose as live state would poison
    #: the dataset, which is worse than a gap.
    HOLD_NO_CAPTURE = "hold_no_capture"
    #: End the episode now. One DRTC session is one episode, so returning also
    #: releases whatever single-client resource the robot holds.
    END_EPISODE = "end_episode"


@dataclass(frozen=True)
class WireHelpers:
    """The four :mod:`interlatent.node.control` helpers the motion path needs.

    Bundled so the bus takes one collaborator instead of four loose callables,
    and injected rather than imported (see the module docstring).
    """

    #: ``(obs, action_keys) -> ndarray`` — joint scalars in action order.
    extract: Callable[[dict, list], np.ndarray]
    #: ``(action, actual, max_step, action_keys, step, *, source) -> ndarray``
    #: — the measured-pose-anchored execution-safety clamp. Distinct from the
    #: adapter's own clamp inside ``send_action``, which is anchored to the
    #: last *accepted command* and exempts grippers. Both must survive.
    clamp: Callable[..., np.ndarray]
    #: ``(action, action_keys) -> Any`` — flat vector to whatever
    #: ``send_action`` accepts. **This is where the calibration frame is
    #: decided**; see the module docstring.
    coerce: Callable[[np.ndarray, list], Any]
    #: ``(obs) -> bytes`` — the DRTC inference payload for this observation.
    encode: Callable[[dict], bytes]


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


@dataclass(frozen=True)
class TickOutcome:
    """What :meth:`CommandBus.drive` did, and what the loop still owes.

    Everything the loop needs for recording and instrumentation is here, so no
    caller re-derives arbitration state or re-reads the gate.
    """

    source: MovementSource
    #: The action actually commanded (post-gate, post-clamp), or the measured
    #: pose on a HOLD tick. ``None`` when nothing was produced.
    action: Optional[np.ndarray] = None
    #: The dataset label for this tick, or ``None`` when it must not be
    #: recorded. Always one of ``{"policy","teleop","hold"}`` when set.
    control_source: Optional[str] = None
    #: Whether the loop should call its capture helper this tick.
    should_record: bool = False
    #: Whether ``send_action`` was actually called.
    sent: bool = False
    #: ``perf_counter()`` at the moment of the send, for the profiler.
    cmd_at: Optional[float] = None
    #: Age of the executed teleop frame, teleop ticks only.
    frame_age_ms: Optional[float] = None
    # --- arbitration state, surfaced for logging/profiling ---
    engaged: bool = False
    teleop_ok: bool = False
    estop_latched: bool = False


class Arbiter:
    """Single authority deciding which source controls the robot this tick.

    Priority, highest first:

        1. ``ESTOP``  — a safety latch is held on the SafetyGate. Overrides
           everything: no motion, no capture, until a human clears it.
        2. ``TELEOP`` — a human is engaged (deadman held) *and* the gated,
           schema-matched teleop path is available for this robot.
        3. ``HOLD``   — no policy is loaded (a teleop-recording assignment) and
           the human is not driving: send nothing so the servos hold, while the
           loop keeps recording a continuous episode.
        4. ``POLICY`` — autonomous inference.

    The e-stop rung lives here rather than as an ``if`` above the branch,
    because "what may drive the robot" is exactly one question and it deserves
    exactly one answer. ``INTERVENTION`` will later split off from ``TELEOP``.
    """

    def decide(
        self,
        *,
        teleop_ready: TeleopReadiness,
        policy_enabled: bool,
        estop_latched: bool = False,
    ) -> MovementSource:
        if estop_latched:
            return MovementSource.ESTOP
        if teleop_ready.teleop_available:
            return MovementSource.TELEOP
        if not policy_enabled:
            return MovementSource.HOLD
        return MovementSource.POLICY


class CommandBus:
    """One point of access to the node's movement ingress *and* motion path.

    Construct with the teleop ingress alone to use it as a decision oracle
    (:meth:`sample_teleop` / :meth:`readiness` / :meth:`arbitrate`), or with the
    full motion collaborators to use :meth:`drive`, which runs the tick.
    """

    def __init__(
        self,
        *,
        teleop_channel: Optional[Any],
        teleop_gate: Optional[Any],
        teleop_profile: Optional[Any],
        policy_enabled: bool,
        arbiter: Optional[Arbiter] = None,
        # --- motion collaborators; required by drive(), unused by arbitrate() ---
        robot: Optional[Any] = None,
        client: Optional[Any] = None,
        action_keys: Optional[list] = None,
        helpers: Optional[WireHelpers] = None,
        max_step: Optional[float] = None,
        action_filter: Optional[Any] = None,
    ) -> None:
        self._teleop_channel = teleop_channel
        self._teleop_gate = teleop_gate
        self._teleop_profile = teleop_profile
        self._policy_enabled = policy_enabled
        self._arbiter = arbiter or Arbiter()

        self._robot = robot
        self._client = client
        self._action_keys = list(action_keys or [])
        self._helpers = helpers
        self._max_step = max_step
        self._action_filter = action_filter

        # One-shot warnings / latches, mirroring the loops they replace.
        self._teleop_warned = False
        self._estop_forwarded = False

    # ------------------------------------------------------------------
    # Decision surface (Phase 1; still used directly by tests)
    # ------------------------------------------------------------------

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

    @property
    def estop_latched(self) -> bool:
        """Whether the safety latch is currently held.

        Read from gate state every tick — never from the arriving event. The
        channel's sticky latch is one-shot and ``frame.estop`` only holds while
        the operator's frames say so, so an edge-triggered check resumes driving
        on the very next tick.
        """
        return (
            self._teleop_gate is not None
            and self._teleop_gate.config.estop_latched
        )

    def arbitrate(self, frame: Optional[Any], action_keys: list) -> MovementSource:
        """Decide which source controls the robot this tick."""
        return self._arbiter.decide(
            teleop_ready=self.readiness(frame, action_keys),
            policy_enabled=self._policy_enabled,
            estop_latched=self.estop_latched,
        )

    # ------------------------------------------------------------------
    # E-stop ingress
    # ------------------------------------------------------------------

    def observe_estop(self, frame: Optional[Any]) -> None:
        """Latch the gate if this tick carries an e-stop, and forward it once.

        Latching is robot-agnostic. Forwarding is not: a robot driven through a
        supervising daemon exposes ``estop()`` to trip its own hard latch, and
        we call it once, retrying on failure. Clearing is a human act and never
        happens here (ADR 0016).
        """
        consume = getattr(self._teleop_channel, "consume_estop", None)
        hit = bool(frame is not None and getattr(frame, "estop", False)) or bool(
            consume is not None and consume()
        )
        if not hit:
            return

        if self._teleop_gate is not None and not self._teleop_gate.config.estop_latched:
            self._teleop_gate.latch_estop("teleop_frame")

        forward = getattr(self._robot, "estop", None)
        if forward is not None and not self._estop_forwarded:
            self._estop_forwarded = True
            try:
                forward()
            except Exception:
                self._estop_forwarded = False  # retry next tick
                raise

    # ------------------------------------------------------------------
    # The motion path
    # ------------------------------------------------------------------

    def drive(self, obs: dict, *, step: int, now: float) -> TickOutcome:
        """Run this tick's motion, end to end, and report what happened.

        Order is fixed and load-bearing: arbitrate → produce → gate → clamp →
        send → discontinuity bookkeeping. ``now`` is the caller's
        ``perf_counter()`` at the top of the tick, so gate timing and the
        profiler agree on one clock.
        """
        if self._helpers is None or self._robot is None:
            raise RuntimeError(
                "CommandBus.drive() needs robot + helpers; this bus was built "
                "for arbitration only"
            )

        frame = self.sample_teleop()
        self.observe_estop(frame)

        ready = self.readiness(frame, self._action_keys)
        latched = self.estop_latched
        source = self._arbiter.decide(
            teleop_ready=ready,
            policy_enabled=self._policy_enabled,
            estop_latched=latched,
        )
        base = dict(
            engaged=ready.engaged,
            teleop_ok=ready.teleop_available,
            estop_latched=latched,
        )

        if source is MovementSource.ESTOP:
            # No motion, no capture. Queued policy chunks are dropped so nothing
            # stale fires on reset; the smoother is dropped so a post-reset
            # resume warm-starts from the live pose.
            self._flush_schedule()
            self._reset_filter()
            return TickOutcome(source=source, **base)

        if source is MovementSource.TELEOP:
            return self._drive_teleop(obs, frame, step=step, now=now, base=base)

        if source is MovementSource.HOLD:
            # No policy to fall back to: send nothing (the servos hold), but
            # report a capture so the episode stays continuous across the
            # human's engage/disengage gaps.
            self._reset_gate()
            actual = self._helpers.extract(obs, self._action_keys)
            return TickOutcome(
                source=source, action=actual, control_source=MovementSource.HOLD.value,
                should_record=True, **base
            )

        return self._drive_policy(obs, step=step, base=base)

    # --- per-source production ---------------------------------------

    def _drive_teleop(self, obs, frame, *, step: int, now: float, base: dict) -> TickOutcome:
        """The hosted teleop engine already resolved an absolute joint target;
        route it through the SafetyGate (the single safety authority for
        human-driven motion) and the delta clamp, then report the *commanded*
        (post-gate) action so the dataset reflects what the robot was actually
        told to do."""
        helpers = self._helpers
        actual = helpers.extract(obs, self._action_keys)

        if (
            frame.mode == "targets"
            and frame.joint_targets is not None
            and len(frame.joint_targets) == len(self._action_keys)
        ):
            target = np.asarray(frame.joint_targets, dtype=np.float32)
        else:
            # Malformed/length-mismatched, or a keys/pose frame the node can't
            # compute locally: hold pose (the gate idles toward it). Pose frames
            # should have been converted to targets on the compute pod.
            if frame.mode == "pose" and not self._teleop_warned:
                self._teleop_warned = True
                _LOG.warning(
                    "Teleop frame mode='pose' reached the node — the pod-side "
                    "retarget stage should have converted it to 'targets' (is "
                    "the relay running without a teleop_view hook?); holding "
                    "pose. See ADR 0009, second amendment."
                )
            target = actual.copy()

        self._teleop_gate.submit(TargetSample(
            joints=target.reshape(-1),
            deadman_active=frame.deadman,
            confidence=frame.confidence,
            received_at=now,
            producer_timestamp_ns=time.monotonic_ns(),
        ))
        commanded, _status = self._teleop_gate.step(actual, now=now)
        action = np.asarray(commanded, dtype=np.float32).reshape(-1)
        # Uniform final guard. The gate already velocity-clamped, so this is
        # typically a no-op, but it keeps one execution-safety invariant across
        # every source.
        action = helpers.clamp(
            action, actual, self._max_step, self._action_keys, step, source="teleop",
        )
        cmd_at = self._send(action)

        # Echo the executed target's seq back so the producer can compute
        # command round-trip latency against its own clock.
        note = getattr(self._teleop_channel, "note_applied", None)
        if note is not None:
            try:
                note(int(frame.seq))
            except Exception:
                pass

        age_ms = None
        received = getattr(frame, "received_at_ns", None)
        if received is not None:
            age_ms = (time.monotonic_ns() - received) / 1e6

        # The policy stream is interrupted: drop queued chunks so they don't
        # apply when the human releases, and drop the smoother's state so the
        # first action after release warm-starts from the live pose.
        self._flush_schedule()
        self._reset_filter()

        return TickOutcome(
            source=MovementSource.TELEOP, action=action,
            control_source=MovementSource.TELEOP.value, should_record=True,
            sent=True, cmd_at=cmd_at, frame_age_ms=age_ms, **base
        )

    def _drive_policy(self, obs, *, step: int, base: dict) -> TickOutcome:
        helpers = self._helpers
        # Reset the gate so the next engage starts from the live pose (the gate
        # is only stepped while engaged).
        self._reset_gate()

        # Encode lazily: client.step() only builds the payload on ticks where
        # DRTC actually sends an observation, so we skip the encode on most.
        action = self._client.step(lambda o=obs: helpers.encode(o), codec="npz")
        if action is None:
            return TickOutcome(source=MovementSource.POLICY, **base)

        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        # Low-pass the policy stream to damp per-tick volatility (chunk-boundary
        # / model jitter) before any safety guard. Warm-started, so it does not
        # ramp from zero.
        if self._action_filter is not None:
            arr = self._action_filter.filter(arr)
        if self._action_keys:
            actual = helpers.extract(obs, self._action_keys)
            arr = helpers.clamp(
                arr, actual, self._max_step, self._action_keys, step, source="policy",
            )
        cmd_at = self._send(arr)
        return TickOutcome(
            source=MovementSource.POLICY, action=arr,
            control_source=MovementSource.POLICY.value, should_record=True,
            sent=True, cmd_at=cmd_at, **base
        )

    # --- collaborators ------------------------------------------------

    def _send(self, action: np.ndarray) -> float:
        """The single ``send_action`` sink. Every source funnels through here."""
        self._robot.send_action(self._helpers.coerce(action, self._action_keys))
        return time.perf_counter()

    def _flush_schedule(self) -> None:
        try:
            self._client.schedule.flush()
        except Exception:
            pass

    def _reset_filter(self) -> None:
        if self._action_filter is not None:
            self._action_filter.reset()

    def _reset_gate(self) -> None:
        if self._teleop_gate is not None:
            self._teleop_gate.reset()
