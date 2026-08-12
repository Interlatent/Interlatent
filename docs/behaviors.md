# Named behaviors

Named, deterministic behaviors — `home`, `hello`, or your own — drive a robot through
canned moves and trajectories from Python or the CLI. They run offline on the robot
side, through the same adapter action path the policy loop uses (so the native
adapters' delta clamp still applies). No API key, no GPU, no policy.

```python
import interlatent as il

with il.Robot("so101", port="/dev/ttyACM0") as robot:
    robot.act("home")                  # move to the rest pose, block until reached
    robot.act("hello")                 # play the built-in wave
    robot.act("home", speed=0.5)       # same behavior, time-scaled gentler
    print(robot.pose())                # {'shoulder_pan': 0.0, ...}
    robot.move(wrist_roll=30, duration=0.5)   # ad-hoc single-joint move
```

## Units and conventions

Joint values are in the **robot's own units** — the same ones the adapter and the
robot's [`RobotProfile`](../packages/sdk/src/interlatent/node/teleop/robot_profile.py)
use. Behaviors are **joint-space only**; there is no IK / Cartesian frame.

| Robot | Arm joints | Gripper | Velocity caps |
|---|---|---|---|
| SO-101 | degrees | `0` (closed) … `100` (open) | deg/s, from the profile |
| I2RT YAM | radians | `0` (closed) … `1` (open) | rad/s, from the profile |

## The Python API

### `il.Robot(robot_type, *, port=None, behaviors=None, robot_arg=None, cameras=None, control_hz=30.0, realtime=True, force=False, connect=True)`

Resolves the kind to an adapter (as `interlatent-act` does), opens it, loads the
behavior registry, and gives you:

| Method | What it does |
|---|---|
| `act(name, *, speed=1.0, wait=True)` | Run a named behavior. Blocks by default, returning an `ActResult`; `wait=False` returns an `ActHandle`. |
| `move(*, duration=0.5, speed=1.0, wait=True, **joints)` | Move named joints to absolute targets; unnamed joints hold. |
| `pose()` | Live joint positions as `{name: value}`. |
| `behaviors()` | List available behavior names. |
| `close()` | Disconnect and release the bus lock (also via `with`). |

`speed` **time-scales** a behavior: `0.5` is half-speed (gentler), `2.0` is twice as
fast. A `speed` (or an explicit `duration`) that would break a velocity cap **raises
`BehaviorValidationError` before any motion** — it never silently clamps. Procedural
behaviors ignore `speed`.

`act(..., wait=False)` returns an **`ActHandle`**:

```python
handle = robot.act("hello", wait=False)
# ... do other work ...
handle.cancel()          # decelerate smoothly to a stop (never a hard freeze)
result = handle.wait()   # ActResult(behavior, reached, aborted, elapsed, joint_error, reason)
```

### Programmatic behaviors: the `@il.behavior` decorator

`@il.behavior(name, *, robot=None, description="")` registers a Python function for
motions awkward to express as static keyframes (loops, counts). It receives the `Robot`
and drives it through the public API:

```python
@il.behavior("nod", robot="so101")   # robot= scopes it; omit for any robot
def nod(robot):
    for _ in range(2):
        robot.move(wrist_flex=-20, duration=0.3)
        robot.move(wrist_flex=0, duration=0.3)

robot.act("nod")
```

`description` defaults to the function's first docstring line.

## The TOML format

Behaviors are data. Each top-level table is one named behavior and must set `type`.
`description` is optional on any behavior; every other non-reserved key is a joint
target.

### `type = "pose"` — one target, reached over a duration

`duration` is optional and must be `> 0` — omit it and the move is auto-sized to the
velocity caps (with headroom), which is how the built-in `home` is always feasible from
any start pose. A pose must name at least one joint.

```toml
[open_gripper]
type = "pose"
duration = 0.5      # seconds; omit for an auto-sized, cap-respecting move
gripper = 100.0

[tuck]
type = "pose"
description = "Fold the arm in."   # optional, any behavior
interpolation = "min_jerk"   # min_jerk (default) | linear | trapezoidal
shoulder_lift = -20.0
elbow_flex = -40.0
```

### `type = "trajectory"` — timed keyframes

Each keyframe is an inline table with a time `t` (seconds) plus joint targets. Joints
omitted from a keyframe **hold their previous value**. Keyframe times must strictly
increase, and the first must be at `t = 0` and name **every joint the trajectory ever
moves** (so the whole path can be velocity-checked up front).

```toml
[wave]
type = "trajectory"
interpolation = "min_jerk"
keyframes = [
    { t = 0.0, wrist_roll = 0.0 },
    { t = 0.6, wrist_roll = 35.0 },
    { t = 1.2, wrist_roll = -35.0 },
    { t = 1.8, wrist_roll = 0.0 },
]
```

**Interpolation profiles.** `min_jerk` (default) is a quintic with zero velocity *and*
acceleration at each keyframe — the smoothest join. `linear` is constant-velocity.
`trapezoidal` accelerates, cruises, then decelerates. All three reach every keyframe
exactly, and each segment is validated so its **peak** velocity
(`shape_factor × |Δ| / segment_seconds`) stays inside the joint's cap.

### Where behaviors come from (load order)

Declarative layers override earlier ones **by name**:

1. **Built-in defaults** — `home` (generated from the profile's rest pose) plus any
   packaged data (SO-101 ships `hello`).
2. **User file** — `~/.interlatent/behaviors.toml`, if it exists.
3. **Explicit file** — the path passed to `Robot(behaviors=...)` / `--behaviors`.

Procedural behaviors (`@il.behavior`) are a separate set, consulted only when no
declarative behavior claims the name — a TOML behavior always wins the name.

## The CLI

All three subcommands are offline (no API key):

```bash
interlatent behavior ls --robot so101 [--json]                  # list behaviors + type/duration
interlatent behavior validate [my_behaviors.toml] --robot so101 # validate, no hardware
interlatent behavior run hello --robot so101 --port /dev/ttyACM0 --speed 0.5
```

`--robot` defaults to `so101` on all three. `run` also takes `--behaviors <file.toml>`,
`--robot-arg key=value` (repeatable), `--control-hz 30`, and `--force`.

`validate` is the fast feedback loop while authoring a TOML — it checks joint names,
limits, and velocity caps against the robot profile without touching hardware.

## Arbitration with the node daemon

The [node daemon](concepts.md) holds a robot's bus while running an inference session.
`Robot` detects the conflict where it can — a client-side lockfile under
`~/.interlatent/locks/` plus the OS serial lock — and **raises `RobotBusyError`** rather
than corrupting a live session. Override with `force=True` (dangerous):

```python
robot = il.Robot("so101", port="/dev/ttyACM0", force=True)  # last resort
```

Detection is best-effort: it catches another Interlatent `Robot`/`interlatent-act` on
the same machine, not a remote one. Daemon-session preemption is not implemented.

## Safety

Behaviors reuse the existing client-side safety model:

- Every target is validated against the profile's **joint limits** and **velocity
  caps** — at load time for trajectories, at plan time for poses — and a violation
  raises *before* any motion.
- The plan is velocity-safe by construction and goes out through the adapter's ordinary
  `send_action` — never the motor bus directly. On the native adapters (YAM, Axol, Nori)
  the in-adapter **delta clamp** is the final backstop; on the LeRobot adapters the
  validated plan is.
- If a joint stalls (the commanded target outruns the measured position past a margin
  for several consecutive ticks) or the adapter errors, the run **aborts cleanly and
  raises** `BehaviorExecutionError`.
- Cancellation **decelerates to a stop**, it does not freeze mid-command.

Record-by-demonstration (`interlatent behavior record`) is not implemented.
