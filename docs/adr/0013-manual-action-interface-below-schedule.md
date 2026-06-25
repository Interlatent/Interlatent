# 0013 — Manual action interface: a final actuator below the schedule

- Status: Proposed
- Date: 2026-06-25

## Context

Until now the only way an action reached the motors was the engine path: a
**policy** on a **GPU pod** streams **action chunks** over DRTC, the client merges
them into an `ActionSchedule` (LWW), and the control loop pops one action per tick
and drives the robot via the adapter's `send_action`. There has been **no
programmatic way for a user to issue an action** — `examples/03_run_on_so101.py`
still ships an `apply_action()` *stub* that says "REPLACE THIS with your motor
write."

We want one **action interface** that both the engine path and manual/programmatic
callers converge on before physical motion, so that a cross-embodiment policy runs
on a new robot by just passing `--robot`, and a human can drive the same robot
through the same seam. The robot-adapter object that would host this already exists
in shape — `adapters/axol/robot.py::AxolNativeRobot` implements
`connect / get_observation / send_action / disconnect / action_features` — but it is
not formalized, declares no per-joint metadata, and has no manual call. The decision
below is load-bearing because it fixes *where* the manual interface sits relative to
the streaming machinery, which is expensive to move later.

## Decision

The action interface is a **final actuator that sits below the `ActionSchedule`**,
not a second action *source* that merges into it. The schedule stays engine-only.
Both paths end at the same adapter object:

```
engine:  GPU pod → DRTC chunks → schedule.pop_next() → ┐
                                                       ├─→ adapter → motors
manual:  user code → adapter.action(**named) ─────────┘
```

The adapter exposes the seam at two levels:

- **`send_action(vector)`** — non-blocking, fire-and-forget, latest-wins. The engine
  loop calls it once per control tick; each action is a *waypoint*, never a
  destination. Unchanged from today.
- **`action(**named, hold_missing=False, timeout=…)`** — the manual call. **Named
  joints are the primary contract** (positional is the internal/engine form);
  block-then-settle returns once the arm reaches the target. It is composed entirely
  from the adapter's own `send_action` + `get_observation`; it is **never** used on
  the engine path. Concrete logic lives once in `adapters/base.py` and is inherited
  by every adapter (Axol and a new thin LeRobot adapter for v1).

Contract specifics:

- **Unknown joint name → always raise** (cross-embodiment guard; no flag suppresses it).
- **Omitted known joint → raise, unless `hold_missing=True`** → autofill from the
  **measured present position** taken from one `get_observation()` snapshot (no extra
  bus read), logging which joints were held (silent embodiment mismatch must be visible).
- **Settle is per-joint by control mode:** position joints settle when
  `|measured − target| ≤ settle_tolerance`; effort/velocity joints (e.g. a gripper
  closing on an object, which never reaches its position target) settle on
  "command issued." **Timeout is mandatory; on timeout `action()` raises.**
- **Manual `action()` is human-driven motion and reuses the existing safety model.**
  It routes through the **`SafetyGate`** (workspace / velocity / deadman / staleness,
  `node/teleop/safety.py`) — whose velocity-limited `step()` also *is* the
  block-then-settle stepping mechanism — and then the source-agnostic **Delta clamp**
  inside `send_action`. Joint **ranges** for pre-validation come from the existing
  `RobotProfile.limits`; a robot kind with no profile refuses manual motion (raises)
  rather than running unguarded. Manual steps record as **`control_source="teleop"`**.

## Consequences

- The manual interface is small and safe: it adds no new control-plane state and
  inherits DRTC's "latest-wins, never replay stale commands" philosophy. The engine's
  streaming/RTC behavior is untouched — `action()` is strictly additive.
- Reuse over reinvention: the `SafetyGate` + `RobotProfile` + Delta clamp from the
  teleop work (ADR 0012) become the safety envelope for manual motion too, so there is
  one human-motion safety authority, not two. No bespoke range/velocity guard is added.
- Block-then-settle gives users the imperative mental model they expect ("go to A, then
  go to B" happens sequentially) without dragging that blocking semantics into the
  engine seam, which must never block.
- Trade-off — **rejected: a unified action *source* above the schedule**, where manual
  actions merge into the same `ActionSchedule` as engine chunks (RTC in-painting,
  cooldown, blendable mid-rollout). That would make manual and engine genuinely
  interleave, but it forces one-shot imperative calls into a streaming CRDT they do not
  fit, and inherits all of DRTC's merge/latency semantics for a feature (concurrent
  shared-autonomy control) nobody has asked for. We chose the simpler actuator seam and
  can revisit if shared autonomy becomes a real requirement.
- Trade-off — **manual control is refused on robot kinds without a `RobotProfile`.** We
  accept failing closed: a manual move with no safety envelope is more dangerous than no
  manual move. Adding a profile per kind is the cost of admission.
- Relates to [0011](0011-vendor-robot-subpackage-via-robot-kind.md) (the adapter
  registry that selects the object hosting this seam) and
  [0012](0012-teleop-receiver-stub-open-core-boundary.md) (the SafetyGate / Delta clamp
  / `control_source` this reuses).
