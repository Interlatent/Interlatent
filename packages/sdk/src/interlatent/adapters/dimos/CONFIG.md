# Dimos adapter configuration

Robot kind `dimos` (`interlatent[dimos]`, python 3.11–3.12). Drives a robot
managed by a **running dimos stack**, as an external LCM/Zenoh bus peer: the
adapter streams `joint_command` to a dimos servo task, reads
`coordinator_joint_state` + camera `Image` topics, and calls the coordinator's
gripper RPC. See ADR 0018.

Three embodiments (`--robot-arg kind=…`): `xarm7` (used below), `xarm6`
(UFACTORY xArm6 — same vendor, units and gripper as `xarm7`; substitute
`xarm6` everywhere and use `--xarm6-ip`), and `a1z` (Galaxea A1Z — read
["Units and conventions (a1z)"](#units-and-conventions-a1z) for its gripper
unit divergence first).

```bash
# Terminal 1 — the dimos side (reference session blueprint, shipped by this SDK):
dimos run interlatent.xarm7

# Terminal 2 — the interlatent node:
interlatent-node run --robot dimos \
  --robot-arg kind=xarm7 \
  --camera wrist=/color_image
```

`--robot dimos_<kind>` (e.g. `interlatent-act --robot dimos_xarm7`) is sugar for
`--robot dimos --robot-arg kind=<kind>` on the manual path; a driving session
uses the canonical `dimos` kind.

Global dimos-process flags (`--simulation`, `--can-port`, `--xarm7-ip`,
`--xarm6-ip`, …) belong to `dimos` itself, so they go **before** `run`:

```bash
dimos --can-port can0 run interlatent.a1z     # correct
dimos run interlatent.a1z --can-port can0     # fails: "No such option: --can-port"
```

The reference blueprint includes DIMOS's `ManipulationModule` with Viser as its
visualization backend (browser UI at `http://127.0.0.1:8095` by default; the
DIMOS log prints the URL). It is deliberately **preview-only**: it provides the
robot model, collision world, planning, trajectory preview, and live Viser
state, but executes nothing — a second trajectory task would compete with
Interlatent's servo task for the same joints, violating the exclusivity
contract below.

With no `xarm7_ip` configured (`xarm6_ip` for `xarm6`), DIMOS selects its
in-memory mock xArm adapter. That is the recommended hardware-free path: run the
same two commands, omit `--camera` if you don't need images, and watch commands
move the mock robot in Viser — the real Interlatent → `joint_command` → servo
task → coordinator path, without MuJoCo.

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
| `max_step_rad` | `0.05` | Per-tick delta clamp on arm joints (gripper exempt); must be > 0. **The ONLY adapter-side clamp** — dimos applies no limits to streamed joint commands. |
| `connect_timeout_s` | `10` | Budget for the coordinator ping and first joint state. |

Two more keys are read by the control loop rather than by
`build_adapter_config`, so `interlatent-node run` logs them as "unrecognized"
and still honors them: `max_step` (loop-level delta clamp in radians, off unless
set) and `action_filter_hz` (Butterworth low-pass on the policy action stream,
default `3.0`; `0`/`none`/`off` disables).

`--camera <name>=<topic>` maps an observation key to a dimos bus topic
(`wrist=/color_image`). **`<name>` must match the policy's training camera
keys.** Cameras are optional for manual moves.

There is deliberately **no `verify=false`**: connect-time verification is
fail-closed by design.

## Units and conventions (xarm7 / xarm6)

Both UFACTORY kinds share this section — same vendor, same dimos `XArmAdapter`,
same gripper, same units.

- Arm joints: **radians**, dimos names `arm/joint1..arm/joint7` mapped to
  feature keys `arm_joint1.pos..arm_joint7.pos` (`/`→`_`), gripper last —
  `joint1..joint6` for `xarm6`.
- Gripper: dimos maps the xArm SDK's 0–850 pulse scale (~85 mm stroke) ×0.001
  into its "meters" convention → range `[0, 0.85]`, 0 closed. The gripper
  **rides the `joint_command` stream** like any other joint (and the servo task
  must claim it): dimos's per-tick hardware write re-sends its last-commanded
  gripper value whenever any task streams, so an out-of-band
  `set_gripper_position` RPC is stomped at tick rate — this adapter uses that
  RPC read-only. If the running stack does not fold the gripper into
  `coordinator_joint_state`, observations serve the last commanded value.
- Wire commands are **always full vectors**: dimos's servo task rejects a
  `joint_command` missing even one claimed joint (`set_target_by_name` returns
  False without updating), so every published message carries all declared
  joints — an action without the gripper key holds the last commanded gripper.
- Timestamps: staleness is gated on local arrival time, not the producer `ts`
  (same-host loopback; immune to clock skew).
- **`xarm6` is not `xarm7` with a joint removed.** The 6-DOF wrist is a
  different arrangement, so `robots/xarm6.toml` carries the xArm6's own limits
  from dimos's bundled `xarm6.urdf`. Most visibly, xarm6's J3 is
  `[-3.927, 0.19198]` (−225°..11°) — the *negative mirror* of xarm7's J4
  `[-0.19198, 3.927]`. Its rest pose is a real elbow-up stance
  (`[0, −40°, −50°, 0, 90°, 0]`, dimos's `_XARM6_INITIAL_JOINTS_DEG`), not
  xarm7's near-zero pose.

## Units and conventions (a1z)

Galaxea A1Z, `dimos run interlatent.a1z`:

- Arm joints: **radians**, dimos names `arm/joint1..arm/joint6` → feature keys
  `arm_joint1.pos..arm_joint6.pos`, gripper last.
- Gripper: **a normalized `[0, 1]` fraction, NOT meters** — the opposite
  convention from xarm7. On a build whose `HardwareComponent` declares
  `gripper_open_position`/`gripper_closed_position`, the blueprint sets them
  (activating dimos's hardware-normalization layer) so the wire value is already
  0 (closed) .. 1 (open); on a build without those fields they are omitted
  rather than passed blindly. Either way the SDK-side range in `robots/a1z.toml`
  is what the adapter clamps against. Verify the open/closed direction once
  against a live or mocked stack.
- **A1Z needs a Galaxea-enabled dimos for real hardware.** dimos ships A1Z two
  ways and the blueprint feature-detects which you have:

  ```bash
  python -c "from dimos.robot.manipulators.a1z import config; \
    print('real-capable' if hasattr(config, 'a1z_hardware') else 'mock-only')"
  ```

  - **Galaxea-enabled branch** — exports `a1z_hardware`, a vendor policy wrapper
    like xarm7's: `dimos --can-port can0 run interlatent.a1z` binds the real CAN
    adapter, and omitting `--can-port` gives the hardware-free mock. This is the
    build A1Z was hardware-verified against.
  - **Any published release** (0.0.14b1 and earlier) — A1Z is a *planning model*
    only: `a1z/config.py` is URDF paths, joint names and collision pairs, its
    lone helper `make_a1z_hardware` defaults to `adapter_type="mock",
    address=None`, and there is no Galaxea entry in
    `dimos.hardware.manipulators.registry.adapter_registry.available()`
    (`a750, mock, openarm, piper, sim_mujoco, xarm`). No flag reaches real
    motors here, and `--can-port` is inert — it is `piper`'s knob.

  You never need this blueprint to drive real hardware: the adapter is a bus
  peer, so `--robot-arg kind=a1z` binds to any running stack satisfying the
  ADR 0018 contract (servo task, non-zero timeout, exclusive joint claim).
- **No MuJoCo visualization path.** dimos ships no MuJoCo scene for A1Z, so
  `--simulation` selects the generic in-memory mock adapter, not a physics sim.
  Viser still renders the mock's live state.
- Position limits are transcribed verbatim from dimos's A1Z URDF, which matches
  the vendor SDK's joint-limit table. Velocity caps are not taken from either
  the URDF's `<limit velocity=…>` tags or the vendor adapter's cap (they
  disagree by ~2x); see the comment above `DIMOS_A1Z_PROFILE` in
  `robot_profile.py`.

## Adding a kind

A kind's declaration (`DimosKind`: dimos wire joint names, gripper
joint/hardware id) and its teleop safety profile (`RobotProfile`: position
limits, velocity caps, rest pose) both load from
[`adapters/dimos/robots/<kind>.toml`](robots/) — one file per kind. `kinds.py`
scans that directory at import; `robot_profile.py` lazily loads a TOML's
`[profile]` table the first time `get_profile("dimos_<kind>")` is called. Adding
a kind is "add a TOML file," not "edit `kinds.py`/`robot_profile.py`."

`[profile]` is optional: without it `robot_profile.py` synthesizes a
conservative default (±2π position bound, a small fixed velocity fraction, zero
rest pose) and logs a loud warning on every use. That unblocks a new kind, but
an auto-derived profile is unaudited — tune it before trusting it in a policy.

Two scripts make transcription mechanical, both reading the vendor's URDF that
dimos bundles as package data (no live RPC exists to fetch limits from a
*running* stack):

- `packages/sdk/scripts/dimos_profile_gen.py` — joint position limits, for a
  `[profile]` section.
- `packages/sdk/scripts/dimos_kinematic_spec_gen.py` — the full joint chain
  (origin/axis/limits), for VR teleop's `kinematic_spec.json`.

Neither is imported by any runtime path; their output is reviewed and committed
like any other transcribed literals. Velocity caps and rest pose are never
auto-derived — a URDF's velocity tag is a motor max, not a safe per-tick
streaming cap, and a rest pose isn't a URDF property at all.

`blueprints.py` is a shared composition
(`_streaming_blueprint`/`_mock_hardware`/`_resolve_hardware`) plus a few
kind-specific lines binding that vendor's hardware/model factory via
`functools.partial`, so every kind gets the same hardware-free dev path.

Full recipe: [`ROBOT.md`](../../../../../../ROBOT.md#special-case-the-dimos-adapter-the-robot-is-a-running-stack).

## VR/QUIC teleoperation

Driving a dimos kind from the VR/QUIC path needs a separate data bundle the
kind/profile TOML does not supply: `interlatent_robots/<kind>/` — a browser-side
IK descriptor (`kinematic_spec.json`) and its tuning surface (`ik_config.json`),
see [`interlatent_robots/README.md`](../../../interlatent_robots/README.md).
`a1z`, `xarm7` and `xarm6` all ship one.

The lookup is keyed by the embodiment, not `--robot`: for `--robot dimos`
sessions `node/daemon.py` resolves `--robot-arg kind=`. Without a bundle a QUIC
session logs `no local kinematic_spec for robot_kind='dimos'` and teleop simply
doesn't start (nothing else about the session is affected).

All three bundles share one caveat: their joint-chain geometry is independently
verified by `packaging/verify_urdf.py` (FK parity ~1e-16 m / 256 configs for
a1z, ~7e-16 m / 256 for xarm7, ~5e-16 m / 512 for xarm6), but the IK-solver
tuning fields (`damping`, `w_rot`, `webxr_to_base_R`, reach limits) are copied —
a1z from YAM, xarm7 from a1z, xarm6 from xarm7 — and are **not** tuned per arm.
`webxr_to_base_R` encodes how the arm is mounted relative to the operator; if
hand motion drives the wrong axis, fix that matrix first.

Two xarm7-specific notes:

- Its two limit sources disagree and the shipped spec takes the tighter per
  joint: the URDF is wider than the datasheet-derived `RobotProfile` on `joint2`
  (±2.18 vs -2.059..2.0944) and on `joint6`'s lower bound (-1.75 vs -1.69297),
  and far narrower on the four full-rotation joints (±3.110 vs ±2π).
  Intersecting keeps the browser solver off targets the node-side `SafetyGate`
  would then clamp. xarm6 needed no intersection — dimos ships its URDF with
  full ±2π ranges that already equal `robots/xarm6.toml`, and
  `tests/test_robots.py` pins spec-vs-profile equality.
- `solver_type` is `weighted_dls` on both xArms rather than `decoupled_6dof`.
  That specialization assumes a spherical wrist, and both xArm wrists are offset
  (xArm6's `joint6` sits at `[0.076, 0.097, 0]` off the `joint4`/`joint5`
  crossing), so it does not hold even on the 6-DOF arm. The browser always runs
  its generic weighted DLS (`dlsSolver.ts`) anyway, so the field is a label.

## The blueprint contract

`dimos run interlatent.xarm7` ships a known-good session stack. An
operator-authored blueprint is equally valid iff it provides:

1. a `ControlCoordinator` with the kind's hardware (and `publish_joint_state`
   left on — the default),
2. optionally, a manipulation planner/visualizer that adds no task claiming the
   session joints (the reference blueprint uses `ManipulationModule` + Viser),
3. a **servo task** claiming **exactly** the kind's joints — arm joints AND
   gripper — with a **non-zero timeout**. Without a servo task, dimos
   **silently ignores** streamed `joint_command` (stock dimos coordinator
   blueprints configure only a trajectory task, which is the trap); with the
   gripper unclaimed, the per-tick hardware write stomps it back to its startup
   value the moment streaming starts,
4. **no other task claiming any of those joints** (strict exclusivity, v1):
   dimos arbitration is priority-based with first-writer-wins ties, so a
   competing claimant fights the policy invisibly — including corrupting the
   recorded `control_source`. Embedding the session into agentic/teleop dimos
   blueprints is unsupported in v1.

Connect-time verification enforces this contract either way, accumulating every
violation into one error.

## Recording (role partition)

The interlatent node records the **episode of record** (policy-visible
observations + commanded actions + `control_source`; session-stop triggers the
LeRobot build), exactly as on every other robot. For low-level dimos streams
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
- **Gripper snaps back to its startup position during a session** → the running
  blueprint's servo task does not claim the gripper joint (contract point 3);
  verification catches this when task introspection is available.
- **Sporadic 120s RPC hangs on hosts without working UDP multicast** (some
  VPNs/locked-down networks): dimos's zenoh peers discover each other via
  multicast scouting, and its RPC publishes each request exactly once. Fix the
  network or route zenoh through an explicit local endpoint.
- **Arm holds its last setpoint after a session dies** → that is the servo
  task's `timeout` hold-last semantics; the reference blueprint sets 0.5 s.
- **E-stop**: `robot.estop()` deactivates the coordinator's tick output
  (best-effort). Human reset: dimos-side `reset_runtime_state` +
  `set_activated(True)`.
