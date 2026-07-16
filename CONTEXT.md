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
`robot_kind`). It does three jobs off one string: it selects the control loop (a
registered vendor robot uses its native loop, otherwise the bundled LeRobot
wrapper runs); it is the **S3 bundle key** the platform resolves the pod's URDF +
meshes + `ik_config.json` under (`urdf/{robot_kind}/{version}/`); and it is the
**Robot data** key an operator installs with `pip install interlatent[<kind>]`.
Because all three must agree, the kind MUST equal the string the live node
reports — an early rig shipped its bundle under `so101_bimanual` while the node
reported `nori`, leaving an unreachable prefix; `nori` is canonical. _Avoid_:
conflating with the `--loop module:function` override, which is a generic escape
hatch, not a kind.

**Robot data**:
A robot kind's teleop embodiment files — the URDF, `ik_config.json`,
`kinematic_spec.json`, and `meshes.lock` — shipped as a standalone
`interlatent-robot-<kind>` distribution (top-level namespace `interlatent_robots`,
read via `interlatent.robots`). Kept out of the `interlatent` package on purpose:
the SDK and the internal engine are both the `interlatent` import package and
would collide on install. _Avoid_: calling the whole thing a "bundle" — that word
is the platform's S3 artifact; the wheel is the operator-installable mirror of the
same source.

**IK config** (`ik_config.json`):
The hand-authored half of **Robot data**: the robot-specific tuning the
retarget/IK stage reads — solver damping, per-joint `max_dq`, reach limits,
translation/rotation scales, `webxr_to_base_R`, gripper range, unit affines. The
five browser-mapper fields are surfaced to the headset as `ik_hints`. Editing it
without regenerating the **Kinematic spec** applies only half a tuning change.
_Avoid_: hand-editing the spec to tune — this is the file you tune.

**Kinematic spec** (`kinematic_spec.json`):
The **generated** half: a compact serial-chain descriptor the in-browser IK solver
walks, exported from URDF + **IK config** by the engine's MuJoCo step. A kind whose
data is missing it makes the arms do nothing (the browser can't build a solver).
On the **QUIC** teleop path the **Node** serves this spec to the browser directly
over the relay (from its installed **Robot data**), and the browser reads *both*
the solver parameters and the mapper hints from it — so it is the single source of
browser kinematics there, and no platform backend is needed (the hosted HTTP
`kinematic-spec` endpoint is only a fallback). On the WS path the pod owns IK and
the browser needs only the mapper hints, which ride in the teleop token. _Avoid_:
hand-editing — it is derived, and any edit is overwritten on regen.

**Adapter**:
A **robot adapter** — a subpackage under `interlatent.adapters.<vendor>` that maps
a specific robot family to the loop the **Node** daemon drives, reusing the
LeRobot-free DRTC wire helpers so its recorded payload matches the built-in loop.
Vendor-specific and dependency-heavy, so it is optional (`interlatent[axol]`,
`interlatent[yam]`) and imported lazily — the base install never loads it. _Avoid_:
overloading "adapter" for a server-side policy backend, a collection `--loop`
adapter, or a LoRA adapter. Vendor adapters today: **axol** (Almond Axol, native
async SDK) and **yam** (I2RT YAM bimanual arms, driven through the `i2rt` CAN driver
directly — not raiden — joint-space only, configurable left/right/both followers).
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

**Chunk scheduling — overlapping (default) vs sequential (`--synchronous`)**:
How the client paces inference against execution. The **default is overlapping
(replace-mode) chunking**: the client streams observations continuously and *never
blocks on inference* (see ARCHITECTURE.md), so a fresh **action chunk** arrives
while the previous one is still executing and overwrites its unexecuted tail in the
`ActionSchedule` (last-writer-wins). This is DRTC's whole point — it hides inference
latency and keeps motion smooth *when consecutive plans agree*. **Sequential
(request-response) chunking**, opt-in via `--synchronous`, drops the overlap: the
client sends one observation only when the schedule is fully drained, holds the
robot while it waits for the whole chunk, executes every step, then re-observes. It
deliberately reverts the "never blocks on inference" property, trading a brief
per-chunk hold (~one inference round-trip) for the elimination of mid-chunk
overwrite — the fix when a high-latency policy's successive plans *disagree* and
fight (observed as robot thrashing; MolmoAct2 on the yam). _Avoid_: conflating this
"synchronous" **mode** (an inference cadence) with the "synchronous facade"
(`DRTCClient`), which is just the blocking `step()` **API surface**; they are
independent.

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

**Nori adapter**:
Vendor adapter `interlatent.adapters.nori` (`--robot nori`, `interlatent[nori]`)
for the Nori robot: the **Node** runs on the robot's Pi and drives the on-board
daemon (`NoriCoreAgent`) over the **Nori-Protocol** v1 wire contract —
newline-delimited JSON on TCP `localhost:7777`, absolute `{"<joint>.pos": v}`
targets carried in `control` frames, 12 arm joints (left-then-right) in the
daemon-normalized `range_m100_100` units. v1 is arms-only (base/lifts:
FUTURE.md #15). It depends on the Nori-Protocol schema repo only (vendored for
conformance tests at `tests/fixtures/nori_protocol/`); `@nori/sdk` is a browser
WebRTC client with no reusable logic and is not a dependency. Nori keeps all
safety enforcement robot-side (range clamping, e-stop hard latch, watchdog
safe-stop); the adapter discloses that state, never re-enforces it, and
fail-closes at connect if the live ack descriptor disagrees with the static
`nori` **robot profile** — accumulating every mismatch into one raise. A
daemon-reported latch/safe-stop is a hard episode boundary: the native loop
ends the session, freeing the daemon's single control-client slot for
`interlatent-act --robot nori --reset-latch`. While the Node holds that slot,
Nori's own browser/VR teleop cannot connect — interlatent teleop rides the
interlatent QUIC/WS relay instead. _Avoid_: "Nori teleop" for interlatent
DAgger takeover — Nori's own teleop stack is a separate system that is
displaced, not reused, during a session. See ADR 0015/0016.

**Keep-alive pump (Nori)**:
Nori's daemon has no heartbeat message — the control-frame stream *is* the
watchdog heartbeat, and silence beyond `t_stop_ms` safe-stops the robot. The
Nori adapter therefore runs an internal ~50 Hz pump sending motion-free
`control` frames, but only while the control loop proves liveness (a
`get_observation` call within ~`t_warn_ms`). If the loop stalls, the pump stops
and the daemon safe-stops as designed. Deliberately conditional — an
unconditional pump would defeat the daemon's watchdog. Distinct from the
**SafetyGate** staleness hold (200 ms), which guards *human-input* liveness;
the daemon watchdog guards *client* liveness. Lives entirely inside the
adapter's session client; the control loop and DRTC client never see it.

**E-stop ingress (teleop)**:
An additive `estop: true` field on the teleop wire frame — the operator's hard
stop. On receipt the control loop latches the **SafetyGate** (all robots); the
Nori loop additionally sends the daemon's `command{name:"estop"}`, which
hard-latches robot-side. Clearing is never automatic and never the control
loop's job: for Nori it is an explicit `--reset-latch` act on `--robot nori`,
which sends the daemon's token-gated `reset_latch` (token from
`/etc/nori/agent.token` on the Pi) and then clears the gate latch — daemon
first, gate second. _Avoid_: conflating with deadman release, which is a soft
hold, not a stop. Universal adapter-level e-stop is future work (FUTURE.md #14).

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

- **Sequential (`--synchronous`) chunking is currently a Node-level flag, but the
  concept is per-Session.** Whether sequential vs overlapping chunking is right is
  a *per-policy* fact (MolmoAct2 needs it, SmolVLA doesn't), and a **Session** pins
  one policy — so the natural home is the dashboard session payload (like
  `chunk_size` / `num_inference_steps`, read in `daemon.py`). For now it's only the
  `--synchronous` CLI flag on the Node (applies to every session that node runs),
  because it shipped as a diagnostic. Promoting it to a per-session payload field is
  the intended evolution.
