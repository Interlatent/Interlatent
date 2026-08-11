# The action interface

Every robot exposes one **action interface** that both the cloud policy (engine path)
and your own code (manual path) drive — a final actuator sitting *below* the DRTC action
schedule. Actions are **joint-space**: a vector of joint targets, one per joint. There is
no IK or Cartesian frame — the arguments are joint angles, not a workspace point.

There are two levels on the same robot object:

| Call | Used by | Semantics |
|---|---|---|
| `send_action(action)` | the engine loop | non-blocking, fire-and-forget, latest-wins — one waypoint per control tick |
| `action(**joints)` | your code (manual) | **named joints**, **blocks until the arm settles** (raises on timeout) |

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

`interlatent-act` drives the same seam without Python — name the joints as `name=value`;
it connects, blocks until the arm settles, and exits:

```bash
# move two joints, hold the rest where they are
interlatent-act --robot so101 --port /dev/ttyACM0 shoulder_pan=30 gripper=80 --hold-missing

# just read and print the current joint pose (no motion)
interlatent-act --robot so101 --port /dev/ttyACM0 --show
```

Same safety as the Python path: it refuses a robot kind with no `RobotProfile`, and the
contract errors (unknown/missing joint, out-of-range) exit non-zero **before** any motion.
`--timeout`/`--rate-hz` tune the settle loop, `--robot-arg key=value` (repeatable) passes
adapter config, `-v` turns on debug logging, and `--reset-latch`/`--token` are the
Nori-only e-stop recovery (never moves the robot; see
[ADR 0016](adr/0016-teleop-estop-ingress-human-only-reset.md)). This is the manual path only — for a cloud
policy use `interlatent-node run`.

### The contract

`action(*, hold_missing=False, timeout=10.0, rate_hz=30.0, **joints)`:

- **Named joints are the contract.** Pass joints by name (`shoulder_pan=…`), using the
  names in `robot.action_features` without the `.pos` suffix. Positional vectors are the
  *internal* form used by the engine path.
- **Unknown joint name → `ValueError`.** A name the robot doesn't have (typo, or a
  policy/robot mismatch) is always an error; no flag suppresses it.
- **Omitted joint → `ValueError`, unless `hold_missing=True`.** With the flag, any joint
  you don't name is held at its **measured present position** (read once, up front) and
  logged. Without it, you must name every joint.
- **Out-of-range target → `ValueError` before any motion.** Every joint you name is
  validated against the robot's joint limits up front, so a bad target never moves the
  arm.
- **Blocks until settled, or raises `TimeoutError`** (`timeout` defaults to 10 s). A
  *position* joint settles when its *measured* position is within its
  `settle_tolerance` of the target. A *gripper* (or any non-position joint) settles once
  the *commanded* trajectory has reached the target — a gripper closing on an object
  never reaches its position target, so the call does not wait on a measured value.

### Safety

Manual motion routes through the same client-side safety model as teleop:

- The **SafetyGate** velocity/workspace/deadman-clamps every commanded step, walking
  the arm to the target at a safe speed (this is also what drives "settle").
- On the native adapters (YAM, Axol, Nori, dimos) the robot's **delta clamp**
  additionally caps the per-tick joint jump inside `send_action` itself. The LeRobot
  adapters (SO-101, Koch) have no in-adapter clamp — there the manual path is guarded by
  the SafetyGate alone (the engine loop's opt-in `max_step` clamp applies only to
  engine-driven sessions).

The SafetyGate requires a
[`RobotProfile`](../packages/sdk/src/interlatent/node/teleop/robot_profile.py)
for the robot kind (joint limits + velocity caps). **With no profile, `action()` refuses
to run** rather than driving the arm unguarded; it also raises if the profile's
`joint_names` do not match the adapter's `action_features` order. Shipped kinds:
`so101`, `koch`/`koch_follower`, `yam`/`yam_bimanual`/`yam_left`/`yam_right`, `nori`, and
`dimos_<embodiment>` from robot data — see `_PROFILES`. The engine path needs no profile.

### Smoothing the engine stream

The policy's per-tick action stream (`send_action`, engine path) is low-pass filtered on
the node before it reaches the motors, damping chunk-boundary and model jitter. It is a
**2nd-order Butterworth** designed at the control rate, default cutoff **3 Hz** —
deliberate arm motion sits below it, per-tick wobble above. It runs *before* the delta
clamp (so the clamp stays the final guard), warm-starts from the live pose, and resets
across teleop engagements. Tune or disable it:

```
interlatent-node run --robot so101 --robot-arg action_filter_hz=3     # default
interlatent-node run --robot so101 --robot-arg action_filter_hz=none  # disable
```

Smoothing applies only to the **engine path**; the manual `action()` path is already
velocity-limited and settles to a target. See
[`node/smoothing.py`](../packages/sdk/src/interlatent/node/smoothing.py).

### Caveats

- **Don't hand-roll a tight loop of partial `action()` calls.** Holding joints at their
  *measured* position every tick re-injects measurement (and gravity sag) as the next
  setpoint, slowly drooping a gravity-loaded joint. One-shot calls are safe; streaming is
  the engine path's job.
- **`action()` is not for the engine path.** Blocking-to-settle in a control loop would
  break DRTC. The engine streams waypoints through `send_action`.

## Adding a robot

Two steps make a robot kind manually drivable:

1. **A `RobotProfile`** in
   [`robot_profile.py`](../packages/sdk/src/interlatent/node/teleop/robot_profile.py):
   ordered `joint_names`, per-joint `joint_limits` and `max_velocity`, and a `rest_pose`,
   registered in `_PROFILES` under the `--robot` kind(s). The `joint_names` **must match
   the order** of the robot's `action_features` (bare names) or `action()` raises. Start
   conservative and widen only after checking the `DRTC-DEBUG joints` log on hardware.
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
(`--robot-arg arms=both|left|right`), and bimanual order is left arm then right. Three
profiles (`yam`/`yam_bimanual`, `yam_left`, `yam_right`) are selected by the robot's
per-instance `robot_kind`. `connect()` preflights the CAN buses, opens each arm, sets the
follower PD gains, opens any RGB cameras (`--camera wrist=realsense:1234`), and — unless
`--robot-arg auto_home=false` — homes to the rest pose. Install with
`pip install 'interlatent[yam]'` (Linux + SocketCAN; the ZED SDK is host-installed).

```bash
# one-shot manual move of the left arm's base joint (radians), holding the rest
interlatent-act --robot yam --robot-arg arms=left left_joint_0=0.2 --hold-missing
```

> YAM `joint_limits` are transcribed exactly from the i2rt YAM URDF
> (`i2rt/robot_models/arm/yam/yam.urdf`). `max_velocity` is capped conservatively at
> 2 rad/s (the URDF says 10) and the gripper `[0, 1]` range is still a placeholder —
> verify both on hardware (`DRTC-DEBUG joints`) before widening.
