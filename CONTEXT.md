# Interlatent DRTC client — Context

The robot-side stack for running robot policies on cloud GPUs: the
[Interlatent dashboard](https://interlatent.com) assigns sessions and provisions a
managed **GPU pod** per session, a **Node** drives the robot and connects to the
pod, and the pod loads policies and serves action chunks over the DRTC gRPC
protocol. The thin `interlatent` CLI lists pods/nodes and starts/stops sessions
against the dashboard.

## Language

**Policy**:
A trained model (an HF repo id or local checkpoint) that maps observations to
action chunks. Identified by a **policy URI**.
_Avoid_: model (overloaded — used for the recorded-dataset "Model layer" too).

**Node**:
The long-running `interlatent-node` daemon on the robot. It pairs to the account
with an API key, polls the dashboard, and converges to whatever inference session
the dashboard assigns it. The DRTC GPU endpoint is provided per-session by the
dashboard. _Avoid_: calling this a "coordinator" — there is no self-hosted control
plane; the dashboard is the control plane.

**Session**:
A live binding of a node (or a hand-written `connect_drtc()` loop) to a policy URI
running on a managed **GPU pod**. Created from the dashboard or via
`interlatent session start --node … --gpu … --policy …`; stopping it closes the
DRTC link and triggers any recorded dataset to be built/published.

**GPU pod**:
A managed cloud GPU that loads a policy and serves action chunks over the DRTC
gRPC protocol. Pods are provisioned and warm-pooled by the dashboard, not
self-hosted. List the pods available to your account with `interlatent gpus ls`.

**Preflight**:
A non-destructive connectivity check (`interlatent-preflight`) that opens a real
**Session** against a managed **GPU pod**, streams *synthetic* observations, and
reports a PASS/WARN/FAIL verdict with the measured network-vs-compute latency. It
exercises the cloud inference path only — never the robot's cameras, joints, or
motor bus. _Avoid_: calling it a "GPU test" — there is no user-operated GPU to test;
it validates the path to a managed pod.

**Robot kind**:
The robot family a **Node** drives, set with `--robot <name>` (carried as
`robot_kind`). It selects the control loop: a registered vendor robot uses its
native loop, otherwise the bundled LeRobot wrapper runs. _Avoid_: conflating with
the `--loop module:function` override, which is a generic escape hatch, not a kind.

**Adapter**:
A **robot adapter** — a subpackage under `interlatent.adapters.<vendor>` that maps
a specific robot family to the loop the **Node** daemon drives, reusing the
LeRobot-free DRTC wire helpers so its recorded payload matches the built-in loop.
Vendor-specific and dependency-heavy, so it is optional (`interlatent[axol]`) and
imported lazily — the base install never loads it. _Avoid_: overloading "adapter"
for a server-side policy backend, a collection `--loop` adapter, or a LoRA adapter.
See [docs/adr/0011](docs/adr/0011-vendor-robot-subpackage-via-robot-kind.md).

**Action interface**:
The shared apply-an-action seam every **adapter** exposes, sitting **below** the DRTC
`ActionSchedule` — a final actuator, not a source that merges into the schedule. Two
levels on the same adapter object:
- `send_action(vector)` — non-blocking, fire-and-forget, latest-wins. The engine loop
  calls it once per control tick (each action is a waypoint, not a destination).
- `action(**named, hold_missing=False, timeout=…)` — the manual/programmatic call:
  **named joints** (positional is the internal/engine form), **block-then-settle**
  (returns once the arm reaches the target, raising on timeout). Composed from the
  adapter's own `send_action` + `get_observation`; never used on the engine path.

All actions are **joint-space** — a vector of joint targets, one per `action_feature`.
There is no inverse kinematics or Cartesian/end-effector frame in the robot-side stack;
`action(x, y, z, …)` means joint angles, not a workspace point. To support `action()`,
an adapter declares per-joint metadata (range, control mode, settle tolerance).

**Teleop receiver stub**:
The node-side half of hosted DAgger takeover (`interlatent.node.teleop`). A
`TeleopChannel` opens a WebSocket to the hosted relay and decodes `TeleopFrame`s;
the control loop applies engaged `mode="targets"` frames (absolute joint vectors
the **platform** already computed) through the **SafetyGate** before driving the
robot. _Avoid_: implying the node computes targets — the teleop *engine*
(keyboard integration, pose IK, retargeting) runs on the platform; the client is
a receiver + safety only. See [docs/adr/0012](docs/adr/0012-teleop-receiver-stub-open-core-boundary.md).

**SafetyGate**:
The node's single safety authority for human-driven motion: a workspace +
velocity + deadman + staleness clamp applied to every teleop target. The
**last hop before the motors**, so it runs on the robot, never on the platform.
Needs a static **robot profile** (limits / velocity cap / rest pose).

**Delta clamp**:
A source-agnostic execution-safety guard that caps the per-tick joint jump for
*every* action — policy and teleop alike — to a per-robot limit (`--robot.max_step`,
or `max_step_rad` for axol). Configured as part of the **adapter**. Together with
the SafetyGate this is the **layered client-side safety model**: the delta clamp
bounds single-tick slams from any source; the SafetyGate adds workspace/velocity/
deadman limits on the teleop path. Both run next to the motors.

**control_source**:
Per-tick provenance recorded into the LeRobot dataset: `"policy"` for
policy-driven steps, `"teleop"` for human DAgger interventions. Carried on the
`RecordTick` wire message and rebuilt into `annotation.interlatent.control_source`.

## Relationships

- A **Node** is paired once and may be assigned many **Sessions** over its life.
- A **Session** pins one **policy URI** on one **GPU pod** for its lifetime.
- The **dashboard** assigns sessions and provisions the GPU pod, returning the
  DRTC endpoint to the node/client per-session.

## Flagged ambiguities

- "warmup" historically meant both *pre-warm* (loading a policy before a session,
  a cloud-side latency optimization) and *correct compilation*. On the robot side
  neither is a concern — the client simply waits for the first action chunk; pod
  warm-pooling is handled by the dashboard.
