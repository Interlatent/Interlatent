# Interlatent — Context

The project's vocabulary. Structure is in [ARCHITECTURE.md](ARCHITECTURE.md); the
mental model is in [docs/concepts.md](docs/concepts.md).

The shape in one paragraph: one **adapter** per robot family puts every supported
robot behind the same **action interface**, which is driven either directly and
offline by a **Robot** handle, or on a **Session**'s behalf by a **Node** — the
[Interlatent dashboard](https://interlatent.com) assigns the session and provisions
a **GPU pod**, the node dials the pod, and the pod loads policies and serves action
chunks over the DRTC gRPC protocol. Both ends of that protocol live in this repo:
`packages/sdk` is the client, `packages/server` is the pod.

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
plane; the dashboard is the control plane (the *compute* may be self-hosted — see
**GPU pod** — but session assignment always comes from the dashboard).

**Session**:
A live binding of a node (or a hand-written `connect_drtc()` loop) to a policy URI
running on a managed **GPU pod**. Created from the dashboard or via
`interlatent session start --node … --gpu … --policy …`; stopping it closes the
DRTC link and triggers any recorded dataset to be built/published.

**GPU pod**:
A GPU box that loads a policy and serves action chunks over the DRTC gRPC
protocol. Two flavors, one protocol: **managed** pods the dashboard provisions
and warm-pools, and **self-hosted** pods — your own hardware running
`interlatent-serve` from the `interlatent-server` dist (`packages/server/`),
registered to your account with your API key (see `docs/self-hosting.md`).
Either way the dashboard assigns sessions and the node dials the pod directly.
List the pods available to your account with `interlatent gpus ls`.

**Preflight**:
A non-destructive connectivity check (`interlatent-preflight`) that opens a real
**Session** against a managed **GPU pod**, streams *synthetic* observations, and
reports a PASS/WARN/FAIL verdict with the measured network-vs-compute latency. It
exercises the cloud inference path only — never the robot's cameras, joints, or
motor bus. _Avoid_: calling it a "GPU test" — it validates the *path* to a pod, not
the GPU.

**Robot kind**:
The robot family a **Node** drives, set with `--robot <name>` (carried as
`robot_kind`). It does three jobs off one string: it selects the **adapter**
(registered vendor kinds resolve to their own, everything else to the bundled
LeRobot one); it is the **S3 bundle key** the platform resolves the pod's URDF +
meshes + `ik_config.json` under (`urdf/{robot_kind}/{version}/`); and it is the
**Robot data** key an operator installs with `pip install interlatent[<kind>]`.
Because all three must agree, the kind MUST equal the string the live node
reports — an early rig shipped its bundle under `so101_bimanual` while the node
reported `nori`, leaving an unreachable prefix; `nori` is canonical. _Avoid_:
conflating with the `--loop module:function` override, which is a generic escape
hatch, not a kind.

**Robot data**:
A robot kind's teleop embodiment files — a kinematics-only URDF, `ik_config.json`,
and `kinematic_spec.json` — under the top-level namespace `interlatent_robots`
(one subpackage per kind, read via `interlatent.robots`). The wheel ships the URDF
+ `kinematic_spec.json`; `ik_config.json` is a repo-only curation source, excluded
from wheels (ADR 0017, amended 2026-07-18). A namespace of its own, not
`interlatent`, because the SDK and the internal engine are both the `interlatent`
import package and would collide on install. Every kind ships with every install
(~18 KB each); the per-kind extras carry that robot's **driver** deps, not its
data. Meshes are **not** part of it — IK needs no geometry — though a `meshes.lock`
may be added for a kind that later needs STLs (viewer/sim). _Avoid_: calling it a
"bundle" — that word is the platform's S3 artifact, a path being retired.

**IK config** (`ik_config.json`):
The hand-authored half of **Robot data**, kept in the repo but not in the wheel:
the robot-specific tuning the
retarget/IK stage reads — solver damping, per-joint `max_dq`, reach limits,
translation/rotation scales, `webxr_to_base_R`, gripper range, unit affines. The
five browser-mapper fields are surfaced to the headset as `ik_hints`. Editing it
without regenerating the **Kinematic spec** applies only half a tuning change.
_Avoid_: hand-editing the spec to tune — this is the file you tune.

**Kinematic spec** (`kinematic_spec.json`):
The **generated** half: a compact serial-chain descriptor the in-browser IK solver
walks, exported from URDF + **IK config** by the engine's MuJoCo step. A kind whose
data is missing it makes the arms do nothing (the browser can't build a solver).
The **Node** serves this spec to the browser over the relay (from its installed
**Robot data**), and the browser reads *both* the solver parameters and the mapper
hints from it — the single and *only* source of browser kinematics: no platform
backend is involved and there is no fallback, by design. A node that cannot serve
its spec fails teleop loudly rather than letting the browser solve against a hosted
copy of kinematics it isn't driving. _Avoid_: hand-editing — it is derived, and
any edit is overwritten on regen.

**Adapter**:
A **robot adapter** — a subpackage under `interlatent.adapters.<vendor>` implementing
the `RobotAdapter` Protocol (`adapters/base.py`) for one robot family, driven by the
shared **tick runner**. Vendor-specific and dependency-heavy, so it is optional
(`interlatent[axol]`, `interlatent[yam]`, …) and imported lazily — the base install
never loads it; `--robot <kind>` resolves it through the one registry in
`adapters/__init__.py`. _Avoid_: overloading "adapter" for a server-side policy
backend, a collection `--loop` adapter, or a LoRA adapter. Vendor adapters today:
**axol** (Almond Axol, native async SDK), **yam** (I2RT YAM bimanual arms, driven
through the `i2rt` CAN driver directly — not raiden — joint-space only, configurable
left/right/both followers), **nori** and **dimos** (own entries below); everything
else runs through the bundled **lerobot** adapter.
See [docs/adr/0011](docs/adr/0011-vendor-robot-subpackage-via-robot-kind.md),
amended by [0022](docs/adr/0022-command-bus-owns-the-motion-path.md).

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

**Behavior**:
A named, deterministic move — `home`, `hello`, or your own — authored in TOML
(`~/.interlatent/behaviors.toml`, or a file passed explicitly) or as a Python function
decorated `@il.behavior`, and resolved through a layered per-robot registry. Every
**robot kind** ships `home`. Behaviors are **joint-space only** and run *offline*: the
`TrajectoryExecutor` samples a validated min-jerk trajectory through the adapter's
ordinary **action interface**, so the **delta clamp** still applies. `speed` time-scales
a behavior rather than re-planning it. Validate one without hardware attached
(`interlatent behavior validate`). See [docs/behaviors.md](docs/behaviors.md).
_Avoid_: calling a behavior an "action" (that is one vector at the action interface) or
an **action chunk** (that is a **policy** output); a behavior is authored, not inferred.
_Avoid_: implying a Cartesian frame — there is no IK on this path.

**Robot** (`il.Robot`):
The manual, no-cloud handle on one robot: it resolves a **robot kind** to an **adapter**
exactly as `interlatent-act` does, opens it, loads the behavior registry, and exposes
`act()`, `move()`, `pose()`, `behaviors()`, `close()`. It never runs a **policy** and
needs no API key or network. _Avoid_: conflating with **Node** — the node is the
dashboard-connected daemon that serves **Sessions**; `Robot` is in-process and offline.
The two are *arbitrated, not coexistent*: the constructor raises `RobotBusyError` if a
node (or another `Robot`) already holds the robot, and `force=True` overrides that at the
risk of corrupting a live inference session. Note the arbitration is **best-effort** — a
client-side lockfile plus an OS serial lock, not a guarantee
(`interlatent.behaviors.arbitration`).

**Chunk scheduling — overlapping (default) vs sequential (`--synchronous`)**:
How the client paces inference against execution. The **default is overlapping
(replace-mode) chunking**: the client never blocks on inference, so a fresh
**action chunk** arrives while the previous one is still executing and overwrites
its unexecuted tail in the `ActionSchedule` (last-writer-wins). This is DRTC's
whole point — it hides inference latency *when consecutive plans agree*.
**Sequential (request-response) chunking** drops the overlap: the client sends one
observation only when the schedule is fully drained, holds the robot while it waits
for the whole chunk, executes every step, then re-observes. It trades a brief
per-chunk hold (~one inference round-trip) for the elimination of mid-chunk
overwrite — the fix when a high-latency policy's successive plans *disagree* and
fight (robot thrashing; MolmoAct2 on the yam), and mandatory for world-action
models whose inference outlasts a chunk. Selected two ways, ORed in `daemon.py`:
the node-wide `--synchronous` flag (or `INTERLATENT_SYNCHRONOUS`), and a
per-session `synchronous` field the backend sets for policies that require it
(ADR 0037, platform repo). _Avoid_: conflating this "synchronous" **mode** (an
inference cadence) with the "synchronous facade" (`DRTCClient`), which is just the
blocking `step()` **API surface**; they are independent.

**Teleop receiver stub**:
The node-side half of VR teleop (`interlatent.node.teleop`) — remote human
demonstration, and mid-policy takeover (live **intervention**: engaging teleop
while a policy session runs preempts the policy and records
`control_source="intervention"`; the node shadow-steps the client so handback
is ≈1 control tick — ADR 0034 in the platform repo). `make_teleop_channel`
builds a `QuicTeleopChannel` against the hosted WebTransport/QUIC relay and
decodes `TeleopFrame`s; the **command bus** applies engaged `mode="targets"`
frames (absolute joint vectors the *browser* already IK-solved) through the
**SafetyGate** before driving the robot. _Avoid_: implying the node computes
targets — the teleop *engine* (pose mapping, IK) runs in the browser producer;
the node is a receiver + safety only, plus the **Kinematic spec** it serves the
browser. See [docs/adr/0012](docs/adr/0012-teleop-receiver-stub-open-core-boundary.md)
and [docs/adr/0021](docs/adr/0021-quic-teleop-child-process.md). QUIC is the only
transport; the WebSocket/hosted-IK path was removed.

**SafetyGate**:
The node's single safety authority for human-driven motion: a workspace +
velocity + deadman + staleness clamp applied to every teleop target. The
**last hop before the motors**, so it runs on the robot, never on the platform.
Needs a static **robot profile** (limits / velocity cap / rest pose).

**Delta clamp**:
A source-agnostic execution-safety guard that caps the per-tick joint jump for
*every* action — policy and teleop alike — to a per-robot limit (`--robot-arg max_step=…`,
or `max_step_rad` for axol/yam/dimos). Configured as part of the **adapter**. Together with
the SafetyGate this is the **layered client-side safety model**: the delta clamp
bounds single-tick slams from any source; the SafetyGate adds workspace/velocity/
deadman limits on the teleop path. Both run next to the motors. Two distinct
clamps carry the name and both are load-bearing: the shared, measured-pose-anchored
clamp the **command bus** runs before every send, and each adapter's own clamp
inside `send_action` (anchored to the last *accepted* command, gripper-exempt).
_Avoid_: "simplifying" them into one — different anchors, different scopes.

**Command bus**:
`interlatent.node.movement.CommandBus` — the one point of access where every
physical movement is decided *and produced* (ADR 0022). Its **Arbiter** picks
who drives each tick on a fixed ladder — `ESTOP > INTERVENTION|TELEOP > HOLD >
POLICY` (a human driving *while a policy is loaded* is INTERVENTION, otherwise
TELEOP), the e-stop rung read from **SafetyGate** *state* (level), never from
the arriving event (edge) — and `drive()` then runs the whole motion path in a
fixed order: produce → SafetyGate → delta clamp → the single `send_action` sink
→ flush/smoother bookkeeping, returning a **TickOutcome** (what was commanded,
what to record, what to instrument). **MovementSource** is the vocabulary:
str-valued so a member's value doubles as the dataset `control_source` label;
`ESTOP` is the one member that is deliberately never a recorded label.
_Avoid_: adding a movement decision anywhere else — an `if` above the bus is
how the pre-2026-07 loop drift started.

**Tick runner / pre-tick guard**:
`interlatent.node.looprunner.run_control_loop` — the one robot-agnostic tick
skeleton every loop shares: observation first (for a daemon-driven robot it is
the keep-alive liveness proof), then the optional guard, then `bus.drive()`,
then capture/tees/reporting/profiling/pacing. An adapter with per-robot
pre-flight conditions implements `pre_tick(obs) -> TickVerdict` (`PROCEED`,
`HOLD_NO_CAPTURE` — no motion *and no capture*, e.g. stale telemetry, worse to
record than to gap — or `END_EPISODE`), discovered via `getattr`, never
declared on the `RobotAdapter` Protocol body. Guards are pure verdicts: the
bus's `guard_interrupt` does the interrupt's hygiene so an adapter cannot
forget it. The former per-adapter `loop.py` files are thin shims over this
runner; the shared behaviour they must all exhibit is pinned by
`tests/test_loop_contract.py`.

**Nori adapter**:
Vendor adapter `interlatent.adapters.nori` (`--robot nori`, `interlatent[nori]`)
for the Nori robot: the **Node** runs on the robot's Pi and drives the on-board
daemon (`NoriCoreAgent`) over the **Nori-Protocol** v1 wire contract —
newline-delimited JSON on TCP `localhost:7777`, absolute `{"<joint>.pos": v}`
targets carried in `control` frames, 12 arm joints (left-then-right) in the
daemon-normalized `range_m100_100` units. v1 is arms-only (base/lift support is
future work). It depends on the Nori-Protocol schema repo only (vendored for
conformance tests at `tests/fixtures/nori_protocol/`); `@nori/sdk` is a browser
WebRTC client with no reusable logic and is not a dependency. Nori keeps all
safety enforcement robot-side (range clamping, e-stop hard latch, watchdog
safe-stop); the adapter discloses that state, never re-enforces it, and
fail-closes at connect if the live ack descriptor disagrees with the static
`nori` **robot profile** — accumulating every mismatch into one raise. A
daemon-reported latch/safe-stop is a hard episode boundary: the adapter's
**pre-tick guard** ends the episode, freeing the daemon's single control-client
slot for the reset act (see **E-stop ingress**). While the Node holds that slot,
Nori's own browser/VR teleop cannot connect — interlatent teleop rides the
interlatent relay instead. _Avoid_: "Nori teleop" for interlatent teleop — Nori's
own teleop stack is a separate system, displaced rather than reused during a
session. See ADR 0015/0016.

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
stop. On receipt the **command bus** (`observe_estop`) latches the
**SafetyGate**, and arbitration thereafter re-reads that **latch** (not the
frame flag) every tick: the flag is transient and the channel's sticky latch is
one-shot, so branching on the event would resume driving on the next tick.
This lives in exactly one place now (ADR 0022); Axol has no `RobotProfile`
yet, so it has no gate to latch (ADR 0011). For robots exposing a hardware
latch (`robot.estop()` — Nori's daemon `command{name:"estop"}`), the bus forwards
it once per latch, retrying on failure without ending the episode. Clearing is
never automatic and never the control loop's job: for Nori it is an explicit
`interlatent-act --robot nori --reset-latch`, which sends the daemon's token-gated
`reset_latch` (token from `/etc/nori/agent.token` on the Pi) and then clears the
gate latch — daemon first, gate second. _Avoid_: conflating with deadman release,
which is a soft hold, not a stop. Universal adapter-level e-stop is future work.

**control_source**:
Per-tick provenance recorded into the LeRobot dataset. A **four-value
contract** — `{"policy", "teleop", "hold", "intervention"}`: `"policy"` for
autonomous inference chunks, `"teleop"` for human-driven ticks in a
policy-less recording session, `"intervention"` for a human override *of a
running policy*, and `"hold"` for disengaged hold ticks. Carried on the
`RecordTick` wire message and rebuilt into
`annotation.interlatent.control_source`.

`"teleop"` and `"intervention"` are deliberately **not** the same label: an
intervention is a correction against a policy's behaviour and carries the
training signal DAgger-style methods consume, so collapsing the two is silent
training-data corruption (`proto/messages.proto`). `MovementSource.ESTOP` is the
one arbitration source with no `control_source` — e-stop ticks are never
captured, which is what keeps the contract at four values.

**Dimos adapter**:
Vendor adapter `interlatent.adapters.dimos` (`--robot dimos`,
`interlatent[dimos]`, python 3.12 — 3.11 resolves only off linux/x86_64, where
`dimos[manipulation]`'s `a750-control` is cp312-wheel-only; see ROBOT.md) for
robots managed by a **running
[dimos](https://github.com/dimensionalOS/dimos) stack**. Unlike every other
adapter there is no motor driver: the adapter joins dimos's LCM/Zenoh bus as an
**external peer** — `coordinator_joint_state` + camera topics in,
`joint_command` out (consumed by a dimos servo task), the **gripper riding
the same stream as a claimed joint** (dimos re-sends its last-commanded
gripper value every tick while streaming, so out-of-band gripper RPCs are
stomped; the RPC is read-only here). Identity is **declare-then-verify**: the operator
states `--robot-arg kind=<embodiment>` (per-embodiment kinds — `xarm7` →
profile `dimos_xarm7`) and `connect()` fail-closes against live evidence,
accumulating every mismatch (Nori pattern) — including the trap that a stock
dimos blueprint has **no servo task and silently ignores** streamed commands,
and **strict exclusivity** (v1): no other dimos task may claim the session's
joints. The dimos side runs a **session blueprint** satisfying that contract;
the SDK ships references via dimos's entry-point registry (`dimos run
interlatent.xarm7`) and documents the contract for custom stacks. Recording is
role-partitioned: the node records the episode of record as usual, and the
adapter publishes **episode markers** on the bus so an optional dimos-side
recorder can segment local low-level data — never recorder-to-recorder
forwarding. _Avoid_: calling dimos's RPC "gRPC" — dimos RPC is request/reply
over its own LCM/Zenoh bus; gRPC in this stack is only ever the robot↔pod DRTC
link. See [docs/adr/0018](docs/adr/0018-dimos-adapter-external-bus-peer.md).

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

- **Sequential chunking has two homes.** It is a *per-policy* fact (MolmoAct2 needs
  it, SmolVLA doesn't), and a **Session** pins one policy — so the session payload's
  `synchronous` field is the right home. The node-wide `--synchronous` flag survives
  as an operator override and is ORed with it, so a node launched with the flag runs
  every session sequentially regardless of the policy.
