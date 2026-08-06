# Dimos adapter configuration

Robot kind `dimos` (`interlatent[dimos]`, python 3.11–3.12). Drives a robot
managed by a **running dimos stack**, as an external LCM/Zenoh bus peer: the
adapter streams `joint_command` to a dimos servo task, reads
`coordinator_joint_state` + camera `Image` topics, and calls the coordinator's
gripper RPC. See ADR 0018.

Three embodiments (`--robot-arg kind=...`) are supported today: `xarm7` (used
in the walkthrough below), `xarm6` (UFACTORY xArm6 — same vendor, same units
and same gripper as `xarm7`; substitute `xarm6` for `xarm7` everywhere below
and use `--xarm6-ip`), and `a1z` (Galaxea A1Z — same shape and same
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

Global dimos-process flags (`--simulation`, `--can-port`, `--xarm7-ip`,
`--xarm6-ip`, ...)
are options on `dimos` itself, not on `dimos run` — they go **before** `run`:

```bash
dimos --can-port can0 run interlatent.a1z     # correct
dimos run interlatent.a1z --can-port can0     # fails: "No such option: --can-port"
```

(`--can-port` only reaches A1Z motors on a Galaxea-enabled dimos build; on a
published release it is inert. See the A1Z notes below.)

The reference blueprint includes DIMOS's `ManipulationModule` with **Viser as
its default visualization backend**. Viser serves its browser UI at
`http://127.0.0.1:8095` by default (the DIMOS log also prints the URL).

With no `xarm7_ip` configured (`xarm6_ip` for `xarm6`), DIMOS selects its
in-memory mock xArm adapter.
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
| `kind` | **required** | Declared embodiment (`xarm7` \| `xarm6` \| `a1z`). Verified against the live stack at connect — a mismatch fail-closes with every problem listed. |
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

## Units and conventions (xarm7 / xarm6)

Both UFACTORY kinds share this section — same vendor, same dimos `XArmAdapter`,
same gripper, same units. `xarm6` differs only in joint count and in its
per-joint position limits (see the note at the end).

- Arm joints: **radians**, dimos names `arm/joint1..arm/joint7` mapped to
  feature keys `arm_joint1.pos..arm_joint7.pos` (`/`→`_`), gripper last —
  `arm/joint1..arm/joint6` → `arm_joint1.pos..arm_joint6.pos` for `xarm6`.
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
- **`xarm6` is not `xarm7` with a joint removed.** The 6-DOF wrist is a
  different arrangement, so `robots/xarm6.toml` carries the xArm6's own limits
  transcribed from dimos's bundled `xarm6.urdf`. Most visibly, xarm6's J3 is
  `[-3.927, 0.19198]` (−225°..11°) — the *negative mirror* of xarm7's J4
  `[-0.19198, 3.927]`. Its rest pose is also a real elbow-up stance
  (`[0, −40°, −50°, 0, 90°, 0]`, dimos's own `_XARM6_INITIAL_JOINTS_DEG`),
  not xarm7's near-zero pose.

## Units and conventions (a1z)

Galaxea A1Z, `dimos run interlatent.a1z`:

- Arm joints: **radians**, dimos names `arm/joint1..arm/joint6` mapped to
  feature keys `arm_joint1.pos..arm_joint6.pos` (`/`→`_`), gripper last.
- **A1Z needs a Galaxea-enabled dimos for real hardware — check which build you
  have.** dimos ships A1Z two different ways, and the blueprint adapts to
  whichever you installed:

  ```bash
  python -c "from dimos.robot.manipulators.a1z import config; \
    print('real-capable' if hasattr(config, 'a1z_hardware') else 'mock-only')"
  ```

  - **Galaxea-enabled branch** — exports `a1z_hardware`, a vendor *policy
    wrapper* like xarm7's: `dimos --can-port can0 run interlatent.a1z` binds the
    real CAN adapter, and omitting `--can-port` gives the hardware-free mock.
    This is the build A1Z was developed and hardware-verified against.
  - **Any published release** (0.0.14b1 and earlier) — A1Z is a *planning model*
    only. `a1z/config.py` is URDF paths, joint names, and collision pairs; its
    lone hardware helper `make_a1z_hardware` is a raw builder defaulting to
    `adapter_type="mock", address=None`, and there is no Galaxea entry in
    `dimos.hardware.manipulators.registry.adapter_registry.available()`
    (`a750, mock, openarm, piper, sim_mujoco, xarm`). No flag reaches real
    motors here, and `--can-port` is inert — it is `piper`'s knob. dimos's own
    A1Z blueprints call the builder bare for the same reason.

  The blueprint feature-detects `a1z_hardware` rather than pinning either
  lineage, so one source tree serves both. **And you never need this blueprint
  to drive real hardware**: the adapter is a bus peer, so `--robot dimos
  --robot-arg kind=a1z` binds to any running stack satisfying the ADR 0018
  contract (servo task, non-zero timeout, exclusive joint claim) — a dimos-side
  or vendor-side blueprint is equally valid.
- Gripper: **a normalized `[0, 1]` fraction, NOT meters** — the opposite
  convention from xarm7. On a build whose `HardwareComponent` declares
  `gripper_open_position`/`gripper_closed_position`, the blueprint sets them
  (activating dimos's hardware-normalization layer) so the wire value is
  already 0 (closed) .. 1 (open); on a build without those fields they are
  omitted rather than passed blindly, which was a `TypeError` on every
  hardware-free start. Either way the SDK-side range in `robots/a1z.toml` is
  what the adapter clamps against. Verify the open/closed direction once
  against a live or mocked stack before trusting it in a policy.
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
`a1z`, `xarm7` and `xarm6` all ship one.

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

`xarm7`'s bundle carries the same caveat, and one more. Its joint chain is
generated by `scripts/dimos_kinematic_spec_gen.py` from the xArm7 URDF that
`dimos[manipulation]` ships as package data
(`data/xarm_description/urdf/xarm7/xarm7.urdf`, stripped to kinematics-only),
and `packaging/verify_urdf.py` passes (FK parity ~7e-16 m over 256 configs) —
but the tuning fields (`damping`, `w_rot`, `webxr_to_base_R`, reach limits) are
copied from A1Z, i.e. from YAM two hops back, and are **not** tuned for an
xArm7. In particular `webxr_to_base_R` encodes how the arm is physically
mounted relative to the operator; if hand motion drives the arm along the wrong
axis, that matrix is the first thing to fix.

Note also that the two limit sources disagree, and the shipped spec takes the
tighter of the two per joint: the URDF is wider than the datasheet-derived
`RobotProfile` on `joint2` (±2.18 vs -2.059..2.0944) and on `joint6`'s lower
bound (-1.75 vs -1.69297), while being far narrower on the four
full-rotation joints (±3.110 vs ±2π). Intersecting keeps the browser solver
from converging on targets the node-side `SafetyGate` would then clamp — the
gate is the authority, and a solver allowed to run past it just fights it.
The `solver_type` is recorded as `weighted_dls` rather than A1Z's
`decoupled_6dof`, which is a 6-DOF specialization and wrong for a redundant
7-DOF arm; the browser ignores the field and always runs its generic weighted
DLS (`dlsSolver.ts`), so this is a labelling fix, not a behavior change.

`xarm6`'s bundle is built the same way — `dimos_kinematic_spec_gen.py` over the
xArm6 URDF `dimos[manipulation]` ships (`data/xarm_description/urdf/xarm6/
xarm6.urdf`, stripped to kinematics-only), `verify_urdf.py` passing at FK
parity ~5e-16 m over 512 configs — and inherits the same **untuned solver
fields**, copied here from `xarm7` (same vendor, same wrist family, same
gripper and units) rather than from A1Z. `webxr_to_base_R` is again the first
thing to suspect if hand motion drives the wrong axis on real hardware.

Two things differ from `xarm7`, both in xArm6's favour:

- **No limit intersection was needed.** dimos ships xArm6's URDF with the full
  ±2π ranges rather than the `limited=true` ±3.110 variant it ships for xArm7,
  and those limits already equal `robots/xarm6.toml`'s datasheet-derived
  profile on all six joints. The spec therefore carries the URDF's limits
  unmodified, and `tests/test_robots.py` pins spec-vs-profile equality so a
  future divergence fails loudly instead of silently needing the xarm7
  treatment.
- **`weighted_dls` is right for a positive reason here**, not just by
  elimination. xArm6 *is* a 6-DOF arm, so `decoupled_6dof` looks applicable —
  but that specialization assumes a spherical wrist, and xArm6's is offset
  (`joint6` sits at `[0.076, 0.097, 0]` off the `joint4`/`joint5` crossing, the
  same offset wrist xArm7 has). The last three axes never meet at a point, so
  the decoupling does not hold.

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
