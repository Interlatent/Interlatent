# 0012 — Teleop: a thin client receiver stub, engine on the platform

- Status: Accepted (amended by [0017](0017-robot-data-ships-in-the-sdk.md))
- Date: 2026-06-24

> **Amendment (0017, 2026-07-16):** the boundary here is now read as "the SDK
> ships no IK *solver* / retargeting," not "no kinematic *data*." Robot
> embodiment data (URDF, `ik_config.json`, `kinematic_spec.json`) ships in the SDK
> wheel as `interlatent_robots/<kind>/`; the solver stays on the platform.

## Context

Interlatent supports **DAgger teleop**: a human operator can take over a robot
mid-policy (keyboard overlay or a VR/WebXR bridge), and the intervention is
recorded into the LeRobot dataset (`control_source="teleop"`) for later training.

The teleop pipeline has several stages: a **producer** (dashboard overlay or VR
bridge) emits operator intent; a **relay** carries it to the robot; the robot
turns intent into an absolute joint target and drives the arm. Historically the
node owned almost all of this — keyboard-to-target integration, WebXR
pose→joint inverse kinematics, pose retargeting, and per-robot kinematic
profiles all ran in the node's control loop. That is a lot of robotics machinery
to ship and maintain in the open-source robot SDK, and most of it is product
surface that belongs with the hosted platform.

We need teleop to keep working on the robot while drawing a clean open-core line:
what is genuinely *client-side* (must run next to the motors) versus what is
*engine* (can run on the platform).

## Decision

The open-source SDK keeps only a **thin client receiver stub plus the last-hop
safety clamp**. Everything that *computes* a joint target from operator intent
moves to the platform.

**Stays in the OSS SDK** (`interlatent/node/teleop/`):
- `channel.py` — `TeleopChannel`: a background WebSocket to the hosted teleop
  relay. Mints a node-role token (`POST …/teleop-token?role=node`), receives
  frames, and surfaces the latest one with a 250 ms staleness drop.
- `frame.py` — `TeleopFrame`: the authoritative wire-frame decoder (full schema:
  `engaged, deadman, mode, held_keys, joint_targets, ee_pos, ee_quat, pinch,
  confidence`).
- `safety.py` — `SafetyGate`: the single safety authority for human-driven
  motion — workspace + velocity + deadman + staleness clamp. It is the **last
  hop before the motors**, so it must run on the robot, not across a network.
- `robot_profile.py` — static per-robot joint limits / velocity caps / rest pose
  that `SafetyGate` needs and that the platform reads (reported via the
  robot-features endpoint) to retarget against the robot's schema.

The node consumes only `mode="targets"` frames — absolute joint vectors the
platform already computed — and routes them through the `SafetyGate` and the
adapter delta clamp before `send_action`. A `keys`/`pose` frame (which would
require local integration/IK) is held at the current pose, because the engine
that would compute it now lives on the platform.

**Leaves the OSS SDK** (runs on the closed platform): `keyboard.py` (held-key
integration), `kinematics.py` (FK/IK), `retarget.py` (WebXR pose retargeting).
The platform performs these and streams `mode="targets"`.

A second, source-agnostic guard — the **adapter delta clamp** — caps the
per-tick joint jump for *every* action (policy and teleop alike) configured per
robot (`--robot.max_step`, or `max_step_rad` for the axol adapter). See the
"layered safety" note in `CONTEXT.md`.

## Consequences

- Teleop keeps working on real robots, but the SDK no longer ships kinematics,
  IK, or retargeting — less code to maintain in the public repo, and the
  product-differentiating modality engine stays with the hosted platform.
- The wire contract (`TeleopFrame`, the relay, the token endpoint) is unchanged,
  so the existing producers and relay interoperate without modification. The
  node keeps the **full** frame schema even though it only acts on `targets`.
- Safety is not weakened by the split: the `SafetyGate` (teleop) and the delta
  clamp (all sources) both remain client-side, so a network glitch, a bad chunk,
  or a malformed teleop frame cannot drive the motors with an unbounded jump.
- Trade-off: a robot driven by `keys`/`pose` intent now depends on the platform
  to resolve targets (a round-trip), instead of computing them locally. We accept
  this — those modalities are a hosted feature, and the latency-critical inner
  loop (policy inference + safety) stays local. Third parties who want fully
  local teleop can still write a custom `--loop` that does its own integration.
- Relates to [0011](0011-vendor-robot-subpackage-via-robot-kind.md): the delta
  clamp is configured through the same per-adapter config surface.
