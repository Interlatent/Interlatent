# 0022 — The command bus owns the motion path; adapters declare guards

- Status: Accepted
- Date: 2026-07-23
- Amends: [0011](0011-vendor-robot-subpackage-via-robot-kind.md) (a native
  adapter no longer brings its own loop), [0016](0016-teleop-estop-ingress-human-only-reset.md)
  (the latch/forward now live in one place)

## Context

Four per-tick orchestrations coexisted: the bundled LeRobot loop
(`node/control.py`) plus a `loop.py` per native adapter (`yam`, `nori`,
`axol`), each a hand-maintained mirror of the first. ADR 0011 sanctioned the
copies ("reuse the wire helpers") but nothing enforced agreement, and by
2026-07 they had drifted apart on *safety* behavior:

- YAM had **no e-stop handling at all** — no `SafetyGate` latch, no schedule
  flush; the policy branch kept executing chunks after an operator e-stop.
- Nori's e-stop was **edge-triggered**: `consume_estop()` is one-shot, so the
  loop resumed driving on the tick after the latch.
- Axol swallowed `policy_enabled` into `**_` and ran inference against
  policy-less teleop-recording sessions.
- Neither native loop clamped the policy path.

A contract suite written before the fix (`tests/test_loop_contract.py`) failed
6/17 exactly on these holes. The root cause was structural, not carelessness:
every new rung in the shared loop had to be re-implemented by hand in three
other files, and a miss was silent.

`node/movement.py` (Phase 1) had already unified the *decision* — but only the
lerobot loop consulted it, and the branch bodies (gate, clamp, send, flush,
smoother reset) still lived in each loop.

## Decision

**The bus owns motion; the runner owns telemetry; an adapter owns only what is
genuinely robot-specific.**

- `CommandBus.drive(obs, step, now)` runs the entire motion path in a fixed
  order — sample teleop → observe e-stop → arbitrate (`ESTOP > TELEOP > HOLD >
  POLICY`, e-stop as a real `Arbiter` rung read from gate *state* every tick)
  → produce the winning source's action → `SafetyGate` → measured-pose delta
  clamp → the single `send_action` sink → schedule-flush / smoother-reset
  bookkeeping — and returns a `TickOutcome` the loop records and instruments
  from without re-deriving anything.
- `node/looprunner.run_control_loop` is the one robot-agnostic tick skeleton:
  observation (first and unconditionally — for daemon-driven robots it is the
  keep-alive liveness proof, ADR 0015), the optional guard, state/preview
  tees, capture, the feature report, latency accounting, profiling, pacing.
- An adapter contributes its `robot.py` plus at most two **optional**
  `RobotAdapter` members, discovered by `getattr` (never added to the Protocol
  body — that would break `isinstance` for every adapter that doesn't need
  them):
  - `pre_tick(obs) -> TickVerdict` — per-robot pre-flight (Nori: session
    death, the daemon safety FSM, telemetry staleness). Guards are **pure
    verdicts**; the interrupt hygiene they used to hand-roll (schedule flush
    on `END_EPISODE`, gate/smoother reset on `HOLD_NO_CAPTURE`) is done by
    `CommandBus.guard_interrupt`, so it cannot be forgotten.
  - `estop() -> None` — a hardware latch forward (Nori), called once per latch
    with retry-on-failure by `CommandBus.observe_estop`.
- Each former loop file is now a thin **shim**: construct the robot and the
  per-session collaborators, wire the bus, hand the tick to the runner. Robot
  construction deliberately stays with the caller.
- The wire helpers travel into the bus as an injected `WireHelpers` bundle,
  never imported (`control.py` imports `movement.py`, so the reverse would
  cycle). Crucially, `coerce` — flat vector → whatever `send_action` accepts —
  is **where the calibration frame is decided**: the engine LeRobot path
  injects `_coerce_action_for_robot` (the OLD→NEW affine; a policy commands in
  *model* frame), native robots inject the identity `movement.dict_coerce`.
  The manual `LeRobotAdapter` path is a raw robot-frame passthrough by design
  and must never carry the engine loop.
- **Two delta clamps exist and both survive.** The bus owns the
  measured-pose-anchored, source-agnostic `_clamp_action_delta`; each
  adapter's own clamp inside `send_action` (anchored to the last *accepted*
  command, gripper-exempt) stays below the Protocol boundary. Different
  anchors, different scopes — collapsing them would change behavior on every
  robot.

## Alternatives considered

- **Capability flags on the robot** (`has_estop`, `needs_hold`): cannot
  express Nori's "end the episode and free the daemon's single control-client
  slot" — a verdict, not a boolean.
- **A strategy object per driving source**: heavier than the problem; the
  sources' bodies are shared, only pre-flight and coerce vary.
- **Keep per-adapter loops, enforce with tests only**: the contract suite
  catches drift but still requires four hand-written copies to converge; the
  bus makes the invariant structural instead of policed.

## Consequences

- A new adapter cannot silently miss a safety rung: it has no loop to get
  wrong. `CONTEXT.md`'s "all motion converges on absolute target → SafetyGate
  → send_action" is now enforced by shape.
- Migration safety is *proven*, not assumed: each pre-migration loop is frozen
  verbatim under `tests/_frozen/`, and `tests/test_loop_equivalence.py` drives
  frozen-vs-migrated through nine scenarios asserting the ordered event
  stream, every sent action vector, and every captured tick match exactly.
  Delete a frozen copy only after its robot has soaked on hardware.
- Deliberate behavioral deltas, all additive, rode in with unification: YAM
  gained the teleop seq echo and the 5s latency window; Nori gained the
  per-second profiler and a uniform teardown flush; latched ticks still tee
  operator previews everywhere.
- ADR 0011's "a native adapter brings its own control loop" is superseded;
  its packaging/registry story (`--robot <kind>`, optional extras, lazy
  imports) stands. `_NATIVE_LOOPS` now points at shims and is owed a collapse
  onto `resolve_adapter` once PRs 4–6 pass hardware verification.
- `--loop module:function` remains the public escape hatch, unchanged.
