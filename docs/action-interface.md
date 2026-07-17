# The action interface

Every robot exposes one **action interface** that both the cloud policy
(engine path) and your own code (manual path) drive — a final actuator sitting
*below* the DRTC action schedule. Actions are **joint-space**: a vector of joint
targets, one per joint. There is no inverse kinematics or Cartesian frame; `action(x,
y, z, …)` means joint angles, not a workspace point.

There are two levels on the same robot object:

| Call | Used by | Semantics |
|---|---|---|
| `send_action(action)` | the engine loop | non-blocking, fire-and-forget, latest-wins — one waypoint per control tick |
| `action(**named, …)` | your code (manual) | **named joints**, **blocks until the arm settles** (raises on timeout) |

See [ADR 0013](adr/0013-manual-action-interface-below-schedule.md) for why it sits
below the schedule and reuses the teleop safety model.

## Manual control: `action()`

```python
from interlatent.adapters.lerobot.robot import LeRobotAdapter

robot = LeRobotAdapter("so101", port="/dev/ttyACM0")
robot.connect()
try:
    # Absolute joint targets, in the robot's own frame (degrees for SO-101).
    # Blocks until the arm settles at the target.
    robot.action(
        shoulder_pan=0.0, shoulder_lift=0.0, elbow_flex=0.0,
        wrist_flex=0.0, wrist_roll=0.0, gripper=50.0,
        timeout=8.0,
    )

    # Move one joint, hold the rest where they are.
    robot.action(shoulder_pan=30.0, hold_missing=True)
finally:
    robot.disconnect()
```

Runnable version: [examples/04_manual_action.py](../examples/04_manual_action.py).

### From the command line: `interlatent-act`

For a one-shot move without writing Python, `interlatent-act` drives the same seam —
name the joints as `name=value`, it connects, blocks until the arm settles, and exits:

```bash
# move two joints, hold the rest where they are
interlatent-act --robot so101 --port /dev/ttyACM0 shoulder_pan=30 gripper=80 --hold-missing

# just read and print the current joint pose (no motion)
interlatent-act --robot so101 --port /dev/ttyACM0 --show
```

Same safety as the Python path: it refuses a robot kind with no `RobotProfile`, and the
contract errors (unknown/missing joint, out-of-range) exit non-zero **before** any motion.
`--timeout`/`--rate-hz` tune the settle loop. This is the manual path only — for a cloud
policy use `interlatent-node run`.

### The contract

`action(**named, hold_missing=False, timeout=10.0, rate_hz=30.0)`:

- **Named joints are the contract.** Pass joints by name (`shoulder_pan=…`). Use the
  names in `robot.action_features` (without the `.pos` suffix). Positional vectors
  are the *internal* form used by the engine path — not how you call `action()`.
- **Unknown joint name → `ValueError`.** A name the robot doesn't have (typo, or a
  policy/robot mismatch) is always an error; no flag suppresses it.
- **Omitted joint → `ValueError`, unless `hold_missing=True`.** With the flag, any joint
  you don't name is held at its **measured present position** (read once, up front),
  and the held joints are logged. Without it, you must name every joint.
- **Out-of-range target → `ValueError` before any motion.** Targets are validated
  against the robot's joint limits up front, so a bad target never moves the arm.
- **Blocks until settled, or raises `TimeoutError`.** A *position* joint settles when it
  is within its tolerance of the target. A *gripper* (or any non-position joint) settles
  as soon as it is commanded — a gripper closing on an object never reaches its position
  target, so the call does not wait on it. `timeout` is mandatory and has a default.

### Safety

Manual motion is human-driven, so it routes through the same client-side safety model
as teleop:

- The **SafetyGate** velocity/workspace/deadman-clamps every commanded step, walking
  the arm to the target at a safe speed (this is also what drives "settle").
- On the native adapters (YAM, Axol, Nori) the robot's **delta clamp** additionally
  caps the per-tick joint jump inside `send_action` itself. The LeRobot adapters
  (SO-101) have no in-adapter clamp — there the manual path is guarded by the
  SafetyGate alone (the engine loop's opt-in `max_step` clamp applies only to
  engine-driven sessions).

The SafetyGate requires a
[`RobotProfile`](../packages/sdk/src/interlatent/node/teleop/robot_profile.py)
for the robot kind (joint limits + velocity caps). **If there is no profile for the
kind, `action()` refuses to run** (raises) rather than driving the arm unguarded.
LeRobot-side profile: `so101`; YAM and Nori ship their own profiles — see the
full registry in `_PROFILES`. The engine path (`send_action`) does not need a profile.

### Smoothing the engine stream

The policy's per-tick action stream (`send_action`, engine path) is low-pass
filtered on the node before it reaches the motors, to damp chunk-boundary and model
jitter. It is a **2nd-order Butterworth** low-pass designed at the control rate,
default cutoff **3 Hz** — deliberate arm motion sits well below it, per-tick wobble
above. It runs *before* the delta clamp, so the clamp stays the final execution-safety
guard, and is warm-started from the live pose (no zero-ramp) and reset across teleop
engagements. Tune or disable it via the robot smoothing arg:

```
interlatent-node run --robot so101 --robot-arg action_filter_hz=3     # default
interlatent-node run --robot so101 --robot-arg action_filter_hz=none  # disable
```

Smoothing applies only to the **engine path**. The manual `action()` path is already
velocity-limited and settles to a target, so it is not filtered. See
[`node/smoothing.py`](../packages/sdk/src/interlatent/node/smoothing.py).

### Caveats

- **Don't hand-roll a tight loop of partial `action()` calls.** Holding joints at their
  *measured* position every tick re-injects measurement (and gravity sag) as the next
  setpoint, which can slowly droop a gravity-loaded joint. One-shot manual calls are
  safe; for streaming, that's the engine path's job.
- **`action()` is not for the engine path.** The engine streams waypoints through
  `send_action`; blocking-to-settle there would break DRTC. Never call `action()` from a
  control loop.

## Adding a robot

A robot kind becomes manually drivable in two steps:

1. **A `RobotProfile`** in
   [`robot_profile.py`](../packages/sdk/src/interlatent/node/teleop/robot_profile.py):
   ordered `joint_names`, per-joint `joint_limits` and `max_velocity`, and a `rest_pose`,
   registered in `_PROFILES` under the `--robot` kind(s). The `joint_names` **must match
   the order** of the robot's `action_features` (bare names) — `action()` raises a
   clear mismatch error otherwise. Start conservative (tighter limits, lower velocities)
   and widen only after checking the `DRTC-DEBUG joints` log on real hardware.
2. **A robot driver.** For a LeRobot-supported arm, `LeRobotAdapter("<kind>", …)` already
   works once the profile exists. For non-LeRobot hardware, implement the
   [`RobotAdapter`](../packages/sdk/src/interlatent/adapters/base.py) duck type
   (`connect` / `get_observation` / `send_action` / `disconnect`, `action_features`,
   `joint_specs`) and inherit `ManualActionInterface` to get `action()` for free — see
   [`adapters/axol/robot.py`](../packages/sdk/src/interlatent/adapters/axol/robot.py)
   and [`adapters/yam/robot.py`](../packages/sdk/src/interlatent/adapters/yam/robot.py).

`joint_specs` declares only per-joint `control_mode` (`"position"` vs gripper/effort) and
`settle_tolerance`; ranges come from the profile, so limits live in exactly one place.

### Example: I2RT YAM bimanual arms (`--robot yam`)

The [`yam` robot](../packages/sdk/src/interlatent/adapters/yam/) drives I2RT's YAM arms
through the `i2rt` CAN driver directly (no raiden dependency). Each follower is 7-DOF
(6 revolute joints in radians + a gripper in `[0, 1]`); topology is configurable
(`--robot-arg arms=both|left|right`), and bimanual order is left arm then right. It ships
three profiles (`yam` / `yam_left` / `yam_right`) selected by the robot's per-instance
`robot_kind`. `connect()` preflights the CAN buses, opens each arm, sets the follower PD
gains, opens any RGB cameras (`--camera wrist=realsense:1234`), and — unless
`--robot-arg auto_home=false` — homes to the rest pose. Install with
`pip install 'interlatent[yam]'` (Linux + SocketCAN; the ZED SDK is host-installed).

```bash
# one-shot manual move of the left arm's base joint (radians), holding the rest
interlatent-act --robot yam --robot-arg arms=left left_joint_0=0.2 --hold-missing
```

> The YAM arm `joint_limits` are the exact hardware limits transcribed from the i2rt
> YAM URDF (`i2rt/robot_models/arm/yam/yam.urdf`). The `max_velocity` is capped
> conservatively (2 rad/s, well below the URDF's 10) and the gripper `[0, 1]` range is
> still a placeholder — verify both on hardware (`DRTC-DEBUG joints`) before widening.
