# Dimos adapter configuration

Robot kind `dimos` (`interlatent[dimos]`, python 3.11–3.12). Drives a robot
managed by a **running dimos stack**, as an external LCM/Zenoh bus peer: the
adapter streams `joint_command` to a dimos servo task, reads
`coordinator_joint_state` + camera `Image` topics, and calls the coordinator's
gripper RPC. See ADR 0018.

```bash
# Terminal 1 — the dimos side (reference session blueprint, shipped by this SDK):
dimos run interlatent.xarm7

# Terminal 2 — the interlatent node:
interlatent-node run --robot dimos \
  --robot-arg kind=xarm7 \
  --camera wrist=/color_image
```

## `--robot-arg` reference

| key | default | meaning |
|---|---|---|
| `kind` | **required** | Declared embodiment (`xarm7`). Verified against the live stack at connect — a mismatch fail-closes with every problem listed. |
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

## The blueprint contract

`dimos run interlatent.xarm7` ships a known-good session stack. An
operator-authored blueprint is equally valid iff it provides:

1. a `ControlCoordinator` with the kind's hardware (and `publish_joint_state`
   left on — the default),
2. a **servo task** claiming **exactly** the kind's joints — arm joints AND
   gripper — with a **non-zero timeout**. Without a servo task, dimos
   **silently ignores** streamed `joint_command` (stock dimos coordinator
   blueprints configure only a trajectory task, which is the trap); with the
   gripper unclaimed, the per-tick hardware write stomps it back to its
   startup value the moment streaming starts,
3. **no other task claiming any of those joints** (strict exclusivity, v1):
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
