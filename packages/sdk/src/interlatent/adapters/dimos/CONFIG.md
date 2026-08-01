# Dimos adapter configuration

Robot kind `dimos` (`interlatent[dimos]`, python 3.11–3.12). Drives a robot
managed by a **running dimos stack**, as an external LCM/Zenoh bus peer: the
adapter streams `joint_command` to a dimos servo task, reads
`coordinator_joint_state` + camera `Image` topics, and calls the coordinator's
gripper RPC. See ADR 0018.

Two embodiments (`--robot-arg kind=...`) are supported today: `xarm7` (used in
the walkthrough below) and `a1z` (Galaxea A1Z — same shape and same
hardware-free-by-default UX; read
["Units and conventions (a1z)"](#units-and-conventions-a1z) for its gripper
unit divergence before using it).

```bash
# Terminal 1 — the dimos side (reference session blueprint, shipped by this SDK):
dimos run interlatent.xarm7

# Terminal 2 — the interlatent node:
interlatent-node run --robot dimos \
  --robot-arg kind=xarm7 \
  --camera wrist=/color_image
```

Global dimos-process flags (`--simulation`, `--can-port`, `--xarm7-ip`, ...)
are options on `dimos` itself, not on `dimos run` — they go **before** `run`:

```bash
dimos --xarm7-ip 192.168.1.185 run interlatent.xarm7   # correct
dimos run interlatent.xarm7 --xarm7-ip 192.168.1.185   # fails: "No such option"
```

(`--can-port` is `piper`'s knob and does **not** reach A1Z motors — see the
A1Z notes below.)

The reference blueprint includes DIMOS's `ManipulationModule` with **Viser as
its default visualization backend**. Viser serves its browser UI at
`http://127.0.0.1:8095` by default (the DIMOS log also prints the URL).

With no `xarm7_ip` configured, DIMOS selects its in-memory mock xArm adapter.
This is the recommended hardware-free path: run the same two commands, omit
`--camera` if the test does not need images, and watch policy/manual commands
move the mock robot in Viser. This exercises the real
Interlatent → `joint_command` → servo task → coordinator path without relying
on the MuJoCo simulation.

The manipulation module is intentionally **preview-only** in this session
blueprint. It provides the robot model, collision world, planning, trajectory
preview, and live Viser state, but does not execute through a second DIMOS
trajectory task. Adding that task would make it compete with Interlatent's
servo task for the same joints, violating the strict-exclusivity contract
below.

## `--robot-arg` reference

| key | default | meaning |
|---|---|---|
| `kind` | **required** | Declared embodiment (`xarm7` \| `a1z`). Verified against the live stack at connect — a mismatch fail-closes with every problem listed. |
| `transport` | follow `DIMOS_TRANSPORT`/.env | `lcm` \| `zenoh`. Both processes MUST agree or they silently cannot see each other. |
| `joint_state_topic` | `/coordinator_joint_state` | Joint state subscription. |
| `joint_command_topic` | `/joint_command` | Servo command publish. |
| `episode_topic` | `/interlatent/episode` | Episode-marker publish (pickled `EpisodeMarker`). |
| `staleness_ms` | `200` | Joint-state freshness gate; the loop holds (no motion, no capture) when stale. |
| `camera_staleness_ms` | `500` | Stale camera serves the last frame + a one-shot warning (frozen image, never a dead session). |
| `camera_warmup_s` | `10` | connect() blocks until each camera topic delivered one frame. |
| `max_step_rad` | `0.05` | Per-tick delta clamp on arm joints (gripper exempt). **The ONLY clamp in the whole path** — dimos applies no limits to streamed joint commands. |
| `connect_timeout_s` | `10` | Budget for the coordinator ping and first joint state. |

`--camera <name>=<topic>` maps an observation key to a dimos bus topic
(`wrist=/color_image`). **`<name>` must match the policy's training camera
keys.** Cameras are optional for manual moves.

There is deliberately **no `verify=false`**: connect-time verification is
fail-closed by design.

## Units and conventions (xarm7)

- Arm joints: **radians**, dimos names `arm/joint1..arm/joint7` mapped to
  feature keys `arm_joint1.pos..arm_joint7.pos` (`/`→`_`), gripper last.
- Gripper: dimos maps the xArm SDK's 0–850 pulse scale (~85 mm stroke) ×0.001
  into its "meters" convention → range `[0, 0.85]`, 0 closed. The gripper
  **rides the `joint_command` stream** like any other joint (and the servo
  task must claim it): dimos's per-tick hardware write re-sends its
  last-commanded gripper value whenever any task streams to the hardware, so
  an out-of-band `set_gripper_position` RPC is stomped at tick rate — the RPC
  is safe only on an idle stack and this adapter uses it read-only. If the
  running stack does not fold the gripper into `coordinator_joint_state`,
  observations serve the last commanded value (disclosed here on purpose).
- Wire commands are **always full vectors**: dimos's servo task rejects a
  `joint_command` missing even one claimed joint (`set_target_by_name`
  returns False without updating), so every message the adapter publishes
  carries all declared joints — an action without the gripper key holds the
  last commanded gripper value.
- Timestamps: staleness is gated on local arrival time, not the producer `ts`
  (same-host loopback; immune to clock skew).

## Units and conventions (a1z)

Galaxea A1Z, `dimos run interlatent.a1z`:

- Arm joints: **radians**, dimos names `arm/joint1..arm/joint6` mapped to
  feature keys `arm_joint1.pos..arm_joint6.pos` (`/`→`_`), gripper last.
- Gripper: the SDK-side range in `robots/a1z.toml` is the contract. An earlier
  version of this section described `gripper_open_position`/
  `gripper_closed_position` fields activating a dimos normalization layer;
  `HardwareComponent` (dimos 0.0.14b1, `dimos/control/components.py`) has no
  such fields. Verify the open/closed direction once against a live or mocked
  stack before trusting it in a policy.
- **This blueprint is mock-only, and that is a dimos limitation, not a
  choice.** dimos 0.0.14b1 — the newest release; nothing newer exists on PyPI —
  ships A1Z as a *planning model* only: `dimos/robot/manipulators/a1z/config.py`
  is URDF paths, joint names, and collision pairs, and its only hardware
  helper, `make_a1z_hardware`, is a raw builder that defaults to
  `adapter_type="mock", address=None`. There is no Galaxea driver to bind to:
  the manipulator registry
  (`dimos.hardware.manipulators.registry.adapter_registry.available()`) holds
  exactly `a750, mock, openarm, piper, sim_mujoco, xarm`. Both of dimos's own
  A1Z blueprints (`a1z/blueprints/basic.py`) call `make_a1z_hardware("arm")`
  bare for the same reason. `--can-port` does NOT reach A1Z motors through this
  blueprint — that flag is `piper`'s knob, and nothing in the A1Z path reads
  it; a previous version of this blueprint gated a "real hardware" branch on it
  and produced a mock component wearing a real-hardware label.
  **Driving real A1Z hardware does not require this blueprint.** The adapter is
  a bus peer: point `--robot dimos --robot-arg kind=a1z` at any running stack
  that satisfies the ADR 0018 session contract (servo task, non-zero timeout,
  exclusive joint claim) — a dimos-side or vendor-side blueprint is equally
  valid. If your dimos build *does* expose a Galaxea driver, wiring it here is
  one line: pass that registry name as `adapter_type=` plus the CAN port as
  `address=` in `blueprints.py`'s `a1z` block.
- **No MuJoCo visualization path.** dimos ships no MuJoCo scene for A1Z, so
  `--simulation` here selects the generic in-memory mock adapter (not a
  physics sim). Viser (via the `ManipulationModule`) still renders the mock
  adapter's live state, so this remains a useful hardware-free path — it is
  just not a physically simulated one.
- Position limits are transcribed verbatim from dimos's A1Z URDF, which
  matches the vendor SDK's own joint-limit table exactly. Velocity caps are
  NOT taken verbatim from either the URDF's `<limit velocity=...>` tags or the
  vendor adapter's own cap (they disagree with each other by roughly 2x); see
  the comment above `DIMOS_A1Z_PROFILE` in `robot_profile.py` for the derivation.

## Adding a kind

A dimos kind's declaration (`DimosKind`: dimos wire joint names, gripper
joint/hardware id) and its teleop safety profile (`RobotProfile`: position
limits, velocity caps, rest pose) both load at import time from
[`adapters/dimos/robots/<kind>.toml`](robots/) — one file per kind, not
Python literals. `kinds.py` scans that directory; `robot_profile.py` lazily
loads a TOML's `[profile]` table the first time `get_profile("dimos_<kind>")`
is called (cached after). Adding a kind's declaration is "add a TOML file,"
not "edit `kinds.py`/`robot_profile.py`."

A `[profile]` section is optional: if a kind's TOML omits it,
`robot_profile.py` synthesizes a conservative default (±2π position bound, a
small fixed velocity fraction, zero rest pose) and logs a loud warning every
time it's used. This unblocks a kind moving at all without hand-tuning being
a hard prerequisite — but an auto-derived profile is unaudited by
construction; tune it before trusting it in a policy.

Two scripts make the transcription step mechanical instead of manual, both
reading the vendor's URDF directly (dimos bundles it as local package data —
no live RPC exists to fetch limits/URDF from a *running* stack):

- `packages/sdk/scripts/dimos_profile_gen.py` — joint position limits, for a
  `[profile]` section.
- `packages/sdk/scripts/dimos_kinematic_spec_gen.py` — the full joint chain
  (origin/axis/limits), for VR teleop's `kinematic_spec.json` (see below).

Neither is imported by any runtime path (`robot_profile.py`/`kinds.py` stay
dimos-import-free either way); their output is reviewed and committed like
any other robot's transcribed literals, not generated fresh each run.
Velocity caps and rest pose are never auto-derived from these scripts — a
URDF's velocity tag is typically a motor max, not a safe per-tick streaming
cap, and a rest pose isn't a URDF property at all.

The blueprint side (`adapters/dimos/blueprints.py`) is a shared, generic
composition (`_streaming_blueprint`/`_mock_hardware`/`_resolve_hardware`) plus
a few kind-specific lines binding that vendor's own hardware/model factory via
`functools.partial`. Every kind gets the same hardware-free dev path this way,
regardless of whether the vendor's own factory happens to support one
directly.

Full recipe: [`ROBOT.md`](../../../../../../ROBOT.md#special-case-the-dimos-adapter-the-robot-is-a-running-stack).

## VR/QUIC teleoperation

Driving a dimos kind from the VR/QUIC path (not just manual/policy joint
actions) needs a separate data bundle the adapter's own kind/profile TOML does
not supply: `interlatent_robots/<kind>/` — a browser-side IK descriptor
(`kinematic_spec.json`) and its tuning surface (`ik_config.json`), see
[`interlatent_robots/README.md`](../../../interlatent_robots/README.md).
`a1z` ships one; `xarm7` does not yet.

The lookup is keyed by the specific embodiment, not `--robot`: for
`--robot dimos` sessions, `node/daemon.py` resolves `--robot-arg kind=` (since
`--robot dimos` alone can't say which kinematically distinct arm is live) —
without this a QUIC session logs `no local kinematic_spec for
robot_kind='dimos'` and teleop simply doesn't start (everything else about the
session is unaffected; this only disables the VR channel).

`a1z`'s bundle was built without the SDK's canonical generator (a pod-side
MuJoCo tool this SDK doesn't carry) — the joint-chain geometry is independently
verified (`packaging/verify_urdf.py` passes: FK parity to ~1e-16 m across 256
random configs), but the IK-solver tuning fields (`damping`, `webxr_to_base_R`,
reach limits) are copied from YAM as an unverified starting point, not tuned
for A1Z specifically. Verify against real VR hardware before trusting motion
feel/direction.

## The blueprint contract

`dimos run interlatent.xarm7` ships a known-good session stack. An
operator-authored blueprint is equally valid iff it provides:

1. a `ControlCoordinator` with the kind's hardware (and `publish_joint_state`
   left on — the default),
2. optionally, a manipulation planner/visualizer that does not add another
   task claiming the session joints (the reference blueprint uses
   `ManipulationModule` + Viser),
3. a **servo task** claiming **exactly** the kind's joints — arm joints AND
   gripper — with a **non-zero timeout**. Without a servo task, dimos
   **silently ignores** streamed `joint_command` (stock dimos coordinator
   blueprints configure only a trajectory task, which is the trap); with the
   gripper unclaimed, the per-tick hardware write stomps it back to its
   startup value the moment streaming starts,
4. **no other task claiming any of those joints** (strict exclusivity, v1):
   dimos arbitration is priority-based with first-writer-wins ties, and a
   competing claimant would fight the policy invisibly — including corrupting
   the recorded `control_source`. Embedding the session into agentic/teleop
   dimos blueprints is unsupported in v1.

Connect-time verification enforces this contract either way, accumulating
every violation into one error.

## Recording (role partition)

The interlatent node records the **episode of record** (policy-visible
observations + commanded actions + `control_source`; session-stop triggers the
LeRobot build) — exactly as on every other robot. For low-level dimos streams
(lidar/odom/tf), add a dimos-side memory2 `Recorder` to the blueprint and
segment it by the adapter's episode markers; do **not** forward between
recorders. Sketch (go2_base pattern, v1.5):

```python
from dimos.memory2.module import Recorder
from dimos.msgs.sensor_msgs import PointCloud2
from interlatent.adapters.dimos.episode import EpisodeMarker

class AuxRecorder(Recorder):
    lidar: In[PointCloud2]
    episode: In[EpisodeMarker]   # segment local data by episode id + timestamps
```

## Failure modes worth knowing

- **Silent no-motion with a healthy-looking stack** → no servo task (see
  contract above); verification catches this at connect.
- **`no running dimos stack answered Coordinator/ping`** → stack not up, or a
  `DIMOS_TRANSPORT` mismatch between the two processes.
- **Gripper snaps back to its startup position during a session** → the
  running blueprint's servo task does not claim the gripper joint (see
  contract point 2); verification catches this when task introspection is
  available.
- **Sporadic 120s RPC hangs on hosts without working UDP multicast** (some
  VPNs/locked-down networks): dimos's zenoh peers discover each other via
  multicast scouting, and its RPC publishes each request exactly once. If the
  bus is affected, fix the network or route zenoh through an explicit local
  endpoint.
- **Arm holds its last setpoint after a session dies** → that is the servo
  task's `timeout` hold-last semantics; the reference blueprint sets 0.5 s.
- **E-stop**: `robot.estop()` deactivates the coordinator's tick output
  (best-effort). Human reset: dimos-side `reset_runtime_state` +
  `set_activated(True)`.
