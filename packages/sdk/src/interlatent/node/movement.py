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


def _should_log(n: int) -> bool:
    """Loud for the first few, then every hundredth.

    A rejection at 30 Hz would otherwise emit 1800 lines a minute; silencing
    it after the first would hide a policy that recovers and re-breaks.
    """
    return n <= 3 or n % 100 == 0


class MovementSource(str, Enum):
    """Who is commanding the robot on a given control tick.

    ``str``-valued so a member's value doubles as the recorded
    ``control_source`` label in the dataset — the existing wire labels stay
    byte-for-byte identical.

    ``ESTOP`` is the one member whose value is **not** a ``control_source``:
    e-stop ticks are never captured, so the dataset's four-value contract
    (``{"policy","teleop","hold","intervention"}``, ``CONTEXT.md``) is
    preserved. :attr:`TickOutcome.control_source` is ``None`` on those ticks.
    """

    ESTOP = "estop"     # safety latch held — overrides every other source
    TELEOP = "teleop"   # a human is driving a policy-less recording session
    #: A human override *of a running policy* (deadman held while
    #: ``policy_enabled``). Split from TELEOP so interventions carry clean
    #: correction labels for DAgger-style training (ADR 0034, platform repo).
    INTERVENTION = "intervention"
    HOLD = "hold"       # human disengaged and no policy action to execute
    POLICY = "policy"   # autonomous inference chunk


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


def dict_coerce(action: np.ndarray, action_keys: list) -> dict:
    """The identity :attr:`WireHelpers.coerce` for robots with no calibration
    frame: a plain name→value zip in the robot's native units. Callers whose
    policy commands in a different frame (the engine LeRobot path and its
    OLD→NEW affine) inject their own coerce instead."""
    return {k: float(action[i]) for i, k in enumerate(action_keys)}


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
    #: recorded. Always one of ``{"policy","teleop","hold","intervention"}``
    #: when set.
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

        1. ``ESTOP``        — a safety latch is held on the SafetyGate.
           Overrides everything: no motion, no capture, until a human clears
           it.
        2. ``INTERVENTION`` / ``TELEOP`` — a human is engaged (deadman held)
           *and* the gated, schema-matched teleop path is available for this
           robot. The label depends on what the human is overriding:
           ``INTERVENTION`` when a policy is running (a DAgger correction),
           ``TELEOP`` on a policy-less teleop-recording assignment.
        3. ``HOLD``         — no policy is loaded (a teleop-recording
           assignment) and the human is not driving: send nothing so the
           servos hold, while the loop keeps recording a continuous episode.
        4. ``POLICY``       — autonomous inference.

    The e-stop rung lives here rather than as an ``if`` above the branch,
    because "what may drive the robot" is exactly one question and it deserves
    exactly one answer.
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
            return (
                MovementSource.INTERVENTION
                if policy_enabled
                else MovementSource.TELEOP
            )
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
        handback_grace_ticks: int = 8,
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

        # Disengage grace (ADR 0034). Under shadow inference the schedule is
        # FULL during an intervention, so a single dropped/stale teleop frame
        # would otherwise fall straight through to POLICY and execute a policy
        # action mid-intervention (a jerk, and a mislabeled frame). A stale
        # frame therefore holds for up to this many ticks before handing back;
        # an *explicit* disengage (a fresh frame with the deadman released)
        # hands back instantly. ~8 ticks ≈ 250 ms at 30 Hz, matching the
        # channel's frame-staleness TTL.
        self._handback_grace_ticks = max(0, int(handback_grace_ticks))
        self._grace_left = 0
        self._prev_source: Optional[MovementSource] = None

        # One-shot warnings / latches, mirroring the loops they replace.
        self._teleop_warned = False
        self._estop_forwarded = False
        # Malformed-chunk rejection counters (see _drive_policy). Logged on the
        # first occurrence and then geometrically, so a persistently broken
        # policy neither hides nor floods the journal.
        self._rejected_nonfinite = 0
        self._rejected_arity = 0

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
            _LOG.warning(
                "Operator e-stop — SafetyGate latched; forwarding the hardware "
                "latch via robot.estop()."
            )
            try:
                forward()
            except Exception:
                # Never propagate: the gate latch above already suppresses all
                # motion, and raising would end the loop — killing the retry
                # this flag exists for.
                self._estop_forwarded = False  # retry next tick
                _LOG.error("Hardware e-stop forward failed", exc_info=True)

    def guard_interrupt(self, verdict: TickVerdict) -> None:
        """Discontinuity bookkeeping when an adapter's ``pre_tick`` guard stops
        the tick before any motion is arbitrated.

        A hold means the input stream broke (stale telemetry, a supervisor
        walking its watchdog back): drop the gate and smoother state so the
        eventual resume warm-starts from the live pose. An episode end drops
        queued policy chunks so nothing stale fires during teardown. Guards
        themselves stay pure verdicts — the bus owns the collaborators, so an
        adapter cannot forget this hygiene.
        """
        if verdict is TickVerdict.END_EPISODE:
            self._flush_schedule()
        elif verdict is TickVerdict.HOLD_NO_CAPTURE:
            self._reset_gate()
            self._reset_filter()

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
        if self._helpers is None or self._robot is None or self._client is None:
            raise RuntimeError(
                "CommandBus.drive() needs robot + client + helpers; this bus "
                "was built for arbitration only"
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
            # No motion, no capture. Queued policy chunks are dropped — with a
            # merge barrier, so an in-flight pre-estop chunk can't land after
            # the flush — and the smoother is dropped so a post-reset resume
            # warm-starts from the live pose.
            self._flush_schedule(barrier=True)
            self._reset_filter()
            self._grace_left = 0
            return self._finish(TickOutcome(source=source, **base))

        if source is MovementSource.INTERVENTION:
            return self._finish(
                self._drive_intervention(obs, frame, step=step, now=now, base=base)
            )

        if source is MovementSource.TELEOP:
            self._grace_left = 0
            return self._finish(
                self._drive_teleop(obs, frame, step=step, now=now, base=base)
            )

        if source is MovementSource.HOLD:
            # No policy to fall back to: send nothing (the servos hold), but
            # report a capture so the episode stays continuous across the
            # human's engage/disengage gaps.
            self._grace_left = 0
            self._reset_gate()
            actual = self._helpers.extract(obs, self._action_keys)
            return self._finish(TickOutcome(
                source=source, action=actual, control_source=MovementSource.HOLD.value,
                should_record=True, **base
            ))

        # POLICY. A *stale* teleop stream right after an intervention gets a
        # grace hold instead of falling through: under shadow inference the
        # schedule is full, so without it a 1-tick frame drop mid-intervention
        # would execute a policy action (a jerk, and a mislabeled frame). A
        # fresh frame with the deadman released is an explicit disengage and
        # hands back instantly.
        if self._grace_left > 0:
            if frame is None:
                self._grace_left -= 1
                return self._finish(self._drive_grace_hold(obs, base=base))
            self._grace_left = 0
        return self._finish(self._drive_policy(obs, step=step, base=base))

    def _finish(self, outcome: TickOutcome) -> TickOutcome:
        self._prev_source = outcome.source
        return outcome

    # --- per-source production ---------------------------------------

    def _drive_teleop(self, obs, frame, *, step: int, now: float, base: dict) -> TickOutcome:
        """Human motion on a policy-less teleop-recording assignment.

        Nothing competes with the human here, so the discontinuity bookkeeping
        is unconditional: any queued (placeholder-backend) chunks are dropped
        and the smoother state cleared every tick, exactly as before the
        INTERVENTION split.
        """
        outcome = self._produce_human_action(
            obs, frame, step=step, now=now, base=base,
            source=MovementSource.TELEOP,
        )
        # The policy stream is interrupted: drop queued chunks so they don't
        # apply when the human releases, and drop the smoother's state so the
        # first action after release warm-starts from the live pose.
        self._flush_schedule()
        self._reset_filter()
        return outcome

    def _drive_intervention(self, obs, frame, *, step: int, now: float, base: dict) -> TickOutcome:
        """Human override of a running policy (DAgger correction, ADR 0034).

        On the engage edge the queued policy chunks are flushed *with a merge
        barrier*, so even an Infer already in flight — computed from a
        pre-takeover observation — can never install actions once the human
        has control. After that the DRTC loop is kept warm every tick
        (*shadow inference*): observations keep flowing, chunks keep merging,
        and the schedule cursor keeps advancing while the popped policy
        actions are discarded in favor of the human's targets. At disengage
        the very next tick pops a chunk computed from a live observation —
        handback latency ≈ one control tick, instead of a frozen-cooldown
        stall plus a full inference round trip.
        """
        was_in_grace = self._grace_left > 0
        if self._prev_source is not MovementSource.INTERVENTION and not was_in_grace:
            # Engage edge only — flushing per tick would discard the fresh
            # shadow chunks this branch exists to keep.
            self._flush_schedule(barrier=True)
            self._reset_filter()
        self._grace_left = self._handback_grace_ticks
        self._shadow_step(obs)
        return self._produce_human_action(
            obs, frame, step=step, now=now, base=base,
            source=MovementSource.INTERVENTION,
        )

    def _drive_grace_hold(self, obs, *, base: dict) -> TickOutcome:
        """The teleop stream went stale mid-intervention (dropped datagrams,
        not an explicit disengage): hold pose instead of executing a policy
        action, keep shadow inference warm, and record the measured pose as a
        ``hold`` frame so the episode stays continuous."""
        self._shadow_step(obs)
        self._reset_gate()
        actual = self._helpers.extract(obs, self._action_keys)
        return TickOutcome(
            source=MovementSource.HOLD, action=actual,
            control_source=MovementSource.HOLD.value, should_record=True, **base
        )

    def _shadow_step(self, obs) -> None:
        """Step the DRTC client without executing its action.

        The side effects are the point: the cooldown keeps ticking,
        observations keep going up, fresh chunks keep LWW-merging, the
        schedule cursor keeps advancing in step with wall-clock ticks, and
        the server's chunk-continuity memory keeps tracking the motion the
        human is actually producing.

        **Not in synchronous mode** (ADR 0037). Every side effect above
        assumes overlapping chunking. Sync mode fires only on a drained
        schedule, so there is no cooldown to keep warm, no unexecuted tail to
        merge against, and nothing to prefetch into — handback costs a full
        inference either way, and shadowing cannot shorten it. What it *would*
        do is spend a multi-second forward on a multi-GPU box, once per cycle,
        for the whole intervention, and throw every result away.

        The server's context stays fresh regardless: the world-model context
        ring is fed node-side on every tick from the control loop, so it keeps
        recording the human's motion whether or not an observation is sent.
        That is what makes skipping this safe rather than merely cheaper.
        """
        if self._client is None or self._is_synchronous():
            return
        helpers = self._helpers
        try:
            self._client.step(lambda o=obs: helpers.encode(o), codec="npz")
        except Exception:
            _LOG.debug("Shadow inference step failed", exc_info=True)

    def _produce_human_action(
        self, obs, frame, *, step: int, now: float, base: dict,
        source: MovementSource,
    ) -> TickOutcome:
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
            # compute locally: hold pose (the gate idles toward it). The
            # browser producer solves IK itself and sends mode='targets' only
            # (ADR 0017); anything else is a stale or foreign producer.
            if frame.mode == "pose" and not self._teleop_warned:
                self._teleop_warned = True
                _LOG.warning(
                    "Teleop frame mode='pose' reached the node — the browser "
                    "producer solves IK locally and must send mode='targets' "
                    "(ADR 0017; the pod-side retarget path is retired). "
                    "Holding pose."
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
            action, actual, self._max_step, self._action_keys, step,
            source=source.value,
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

        return TickOutcome(
            source=source, action=action,
            control_source=source.value, should_record=True,
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
        actual = (
            helpers.extract(obs, self._action_keys) if self._action_keys else None
        )

        # Reject a malformed chunk before it touches the robot. Nothing else on
        # this path catches either case: the policy stream never passes through
        # the SafetyGate (that is the teleop path), and the delta clamp is
        # disabled unless --robot.max_step is set — and even when it *is* set it
        # cannot stop a NaN, because ``abs(delta) > max_step`` is False for NaN,
        # so the clamp passes it straight through to send_action.
        #
        # A rejected action is reported as a no-action tick, which is already a
        # safe, exercised state: nothing is sent and the servos hold the last
        # commanded pose, exactly as when a chunk fails to arrive.
        if not self._policy_action_ok(arr, step):
            return TickOutcome(source=MovementSource.POLICY, **base)

        # Low-pass the policy stream to damp per-tick volatility (chunk-boundary
        # / model jitter) before any safety guard.
        if self._action_filter is not None:
            if actual is not None and not getattr(self._action_filter, "primed", True):
                # Handback (or first tick): seed the smoother from the robot's
                # *measured* pose, so the first policy action is damped
                # relative to where the arm actually is — the plain warm-start
                # seeds from the policy action itself, which provides zero
                # damping on exactly the human→policy jump (ADR 0034).
                seed = getattr(self._action_filter, "seed", None)
                if seed is not None:
                    seed(actual)
            arr = self._action_filter.filter(arr)
        if self._action_keys:
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

    def _is_synchronous(self) -> bool:
        """True when the DRTC client is doing sequential chunking.

        Read off the client rather than passed in, so it cannot drift from the
        cadence actually in force — the session payload sets it (world-action
        models require it), but ``--synchronous`` and the env var set it too.
        """
        cfg = getattr(self._client, "cfg", None)
        return bool(getattr(cfg, "synchronous", False))

    def _policy_action_ok(self, arr: "np.ndarray", step: int) -> bool:
        """Validate a policy action vector; False means "treat as no-action".

        Two checks, neither of which existed anywhere on the policy path:

        * **Arity.** ``_clamp_action_delta`` operates on the shortest common
          prefix of (action_keys, action, measured) and returns the array
          untouched when that prefix is empty; ``_coerce_action_for_robot``
          falls back to passing the bare array to ``send_action`` when the
          widths disagree. So a wrong-width vector is silently *truncated* —
          it commands a different set of joints than intended, with no error.
        * **Finiteness.** There was no NaN/Inf check between the gRPC receiver
          and ``send_action``, and the delta clamp cannot serve as one:
          ``abs(delta) > max_step`` evaluates False for NaN, so a non-finite
          action passes the clamp even when the clamp is enabled.

        Rejections are counted and logged on the first few occurrences and
        then every hundredth, so a policy that has gone bad is loud once and
        does not drown the journal at 30 Hz.
        """
        n = len(self._action_keys)
        if n and arr.size != n:
            self._rejected_arity += 1
            if _should_log(self._rejected_arity):
                _LOG.error(
                    "policy action REJECTED at step %d: width %d != %d action "
                    "keys %s (holding pose; %d rejected so far). The server and "
                    "the robot disagree on the action space — check the "
                    "policy's action_dim against this robot.",
                    step, arr.size, n, self._action_keys, self._rejected_arity,
                )
            return False
        if not np.isfinite(arr).all():
            self._rejected_nonfinite += 1
            if _should_log(self._rejected_nonfinite):
                _LOG.error(
                    "policy action REJECTED at step %d: non-finite values "
                    "(holding pose; %d rejected so far). The delta clamp cannot "
                    "catch this — NaN compares False against any limit.",
                    step, self._rejected_nonfinite,
                )
            return False
        return True

    def _send(self, action: np.ndarray) -> float:
        """The single ``send_action`` sink. Every source funnels through here."""
        self._robot.send_action(self._helpers.coerce(action, self._action_keys))
        return time.perf_counter()

    def _flush_schedule(self, *, barrier: bool = False) -> None:
        """Drop queued policy chunks. With ``barrier=True``, also raise the
        schedule's merge floor to the client clock's current tick, so an Infer
        already in flight (stamped before this moment) is rejected when it
        lands instead of executing pre-takeover actions."""
        try:
            if barrier:
                clock = getattr(self._client, "clock", None)
                if clock is not None:
                    try:
                        self._client.schedule.flush(barrier_ts=clock.tick())
                        return
                    except TypeError:
                        pass  # schedule predates barrier support
            self._client.schedule.flush()
        except Exception:
            pass

    def _reset_filter(self) -> None:
        if self._action_filter is not None:
            self._action_filter.reset()

    def _reset_gate(self) -> None:
        if self._teleop_gate is not None:
            self._teleop_gate.reset()
