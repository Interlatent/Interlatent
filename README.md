<div align="center">

<img src="assets/Final Logo pt6.png" alt="Interlatent" width="420"/>

### One open interface to control every robot.

The open-source SDK and protocol for controlling robots. Read joint state and command
motion the **same way on every supported arm** - whether you're driving it by hand,
playing a named behavior, running a cloud VLA policy, or recording a dataset. Add a
robot once (an adapter + a profile) and every capability above it comes for free.

[![PyPI](https://img.shields.io/pypi/v/interlatent?color=7C5CFF&label=interlatent)](https://pypi.org/project/interlatent/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LeRobot](https://img.shields.io/badge/works%20with-%F0%9F%A4%97%20LeRobot-FFD21E)](https://github.com/huggingface/lerobot)
[![GitHub stars](https://img.shields.io/github/stars/interlatent/interlatent?style=social)](https://github.com/interlatent/interlatent)

[The idea](#-one-robot-class-many-adapters) · [The map](#-what-actually-defines-a-robot) · [How it works](#-how-the-sdk-works) · [Quickstart](#-quickstart) · [Robots](#-supported-robots) · [Docs](docs/)

</div>

---

Robotics tooling is fragmented: every arm ships its own SDK, its own joint conventions, its
own scripts. Interlatent is **one interface across robots** - a single way to read state and
command joints, with a shared safety model underneath. Write against it once and you can
teleoperate, run models, and collect data on any supported arm.

This README leads with the architecture, because the architecture *is* the pitch. If you
just want to move an arm, skip to the [Quickstart](#-quickstart).

## 🧩 One robot class, many adapters

Everything in this SDK rests on a single idea: **a robot is one object with five methods.**

```python
robot.connect()
obs = robot.get_observation()   # {'shoulder_pan.pos': 12.4, 'observation.images.front': <HxWx3 uint8>, ...}
robot.send_action(action)       # {'shoulder_pan.pos': 30.0, ...}  fire-and-forget, latest-wins
robot.disconnect()
robot.action_features           # ordered joint keys; defines what the action vector means
```

That contract lives in [`adapters/base.py`](packages/sdk/src/interlatent/adapters/base.py)
and it is the only thing the layers above a robot are allowed to know about. Behaviors, VLA
policies, teleop, and dataset recording are all written against those five methods, which is
exactly why adding a robot gives you all of them at once.

`base.py` defines two things:

- **`RobotAdapter`** - a `Protocol` (a duck type, not a base class you must subclass).
  Lifecycle plus observe/act, plus the metadata the manual path needs (`action_features`,
  `joint_specs`).
- **`ManualActionInterface`** - a mixin carrying the *one* piece of shared behavior:
  `action(shoulder_pan=30, gripper=80)`, a named-joint, block-then-settle move composed
  entirely out of the adapter's own `send_action` + `get_observation`. Every adapter
  inherits it; none of them implement it.

Two invariants worth stating up front, because they shape everything else:

- **All actions are joint-space.** A vector of absolute joint targets, one per
  `action_feature`. There is no IK or Cartesian frame anywhere in the robot-side stack.
- **Each action is a waypoint, not a destination.** `send_action` is non-blocking and
  latest-wins; the control loop calls it once per tick.

### What an adapter actually is today

An adapter is a directory under [`adapters/`](packages/sdk/src/interlatent/adapters/):

| File | Role |
|---|---|
| `robot.py` | **The robot.** Implements the five methods above. Owns the vendor driver (CAN bus, serial, motor SDK) and the cameras. This is the only file that has to exist. |
| `config.py` | Turns the daemon's flat CLI passthrough (`--robot-arg key=value`, `--camera name=device`) into a typed config dataclass. Deliberately import-light, so importing the adapter never drags in its heavy extra. |
| `cameras.py` | Frame capture, normalized to `uint8 HxWx3` RGB. Vendor SDKs are imported lazily inside methods. |
| `loop.py` | A per-robot control loop, registered so `--robot <kind>` resolves to it. |

A useful way to read the tree: `robot.py` is the *leaf*, `base.py` is the *contract*, and
the rest is plumbing that exists because a robot needs configuring and looking at.

### Where this is going: fold the adapters into the robot class

The contract above is real and it works. The layer around it is **not finished**, and we'd
rather say so here than let you discover it in the source. Today:

- **The robot is a clean abstraction.** `base.py` + `robot.py` genuinely is one interface
  across arms. This part is done.
- **Cameras are only partly abstracted.** The YAM adapter defines a proper `Camera` Protocol
  (`connect` / `read() -> RGB` / `disconnect`) with RealSense, ZED, and UVC backends behind
  it. That is the right shape. But it is *local to that adapter* - others open their cameras
  inside `robot.py` instead, so there is no single camera seam across the SDK.
- **The control loop is copy-pasted, not factored.** There are three of them
  ([`node/control.py`](packages/sdk/src/interlatent/node/control.py) for LeRobot robots, plus
  a `loop.py` per native adapter). They all have the same shape - observe, decide who is
  driving, clamp, `send_action`, record - and they all reuse the same wire helpers. They
  differ only in small ways: whether teleop is wired, which safety composition applies, which
  calibration preset is active. Those differences are *configuration wearing the costume of
  code*.

**The direction:** collapse those seams into the robot class.

1. **One `Camera` protocol** for the whole SDK (`connect` / `read() -> uint8 HxWx3 RGB` /
   `disconnect`) that every adapter implements rather than reinvents. YAM's is already the
   template.
2. **One universal control loop**, parametrized instead of duplicated. The per-adapter
   variation becomes explicit capabilities the robot *declares* (does it support teleop? does
   it have an e-stop latch? which calibration applies?) rather than a forked copy of the loop.
3. **A smaller adapter.** Once cameras and the loop are shared, a new robot is `robot.py`
   plus a `RobotProfile`. `config.py` shrinks to a schema; `loop.py` disappears.

The test for whether we've done this right: **adding an arm should be one file and a
profile.** Anything more is a seam we haven't closed yet. Tracked in
[Future directions](#-future-directions).

## 🗺️ What actually defines a robot

The section above says what a robot *does*. This one shows the files that say what your
robot **is** - the whole map, before you write a line of code.

### Robot kinds that work today

`--robot <kind>` on the CLI, `il.Robot("<kind>")` in Python:

| `--robot` | Joints | Units | Control loop | Extra |
|---|---|---|---|---|
| `so101`, `so101_follower` | 6 | degrees; gripper 0-100 | bundled LeRobot | `[lerobot]` |
| `koch`, `koch_follower` | 6 | degrees; gripper 0-100 | bundled LeRobot | `[lerobot]` |
| `yam`, `yam_bimanual` | 14 (left block, then right) | radians; gripper 0-1 | native | `[yam]` |
| `yam_left`, `yam_right` | 7 | radians; gripper 0-1 | native | `[yam]` |
| any other LeRobot robot | its own | LeRobot's | bundled LeRobot | `[lerobot]` |
| `--loop module:fn` | yours | yours | yours | - |

The first four rows are the kinds with a **`RobotProfile`** (the full list lives in
`_PROFILES` in [`robot_profile.py`](packages/sdk/src/interlatent/node/teleop/robot_profile.py)).
That distinction is the one rule worth internalizing:

> **No profile, no human-driven motion.** Any other LeRobot robot still runs a cloud policy
> fine. But `action()`, behaviors (including `home`), and teleop **refuse to run** without a
> profile, rather than move an arm with no safety envelope. This fails closed on purpose.

Koch is wired and has a profile, but its envelope is a conservative starting guess rather
than hardware-measured. Treat it as unverified.

### 1. The profile: what your robot physically is

The file people don't expect, and the most important one here. A `RobotProfile` is the
kinematic truth about an arm: joint names **and their order**, software limits, per-joint
velocity caps, and the rest pose. The `SafetyGate` enforces it, `home` is generated from it,
`action()` validates against it, and behaviors are checked against it at load.

It exists because no vendor gives you all of it. A driver (or LeRobot) hands you joint names
and live positions; a URDF hands you mechanical limits. Neither declares a *safe per-tick
velocity cap* or a *home pose*, and those are precisely what you need to move an arm without
breaking it.

YAM is the instructive case. SO-101 is a 6-joint arm in degrees and LeRobot already carries
most of its tooling; YAM is a 14-DOF bimanual robot on raw CAN, where this file is doing all
the work:

```python
# packages/sdk/src/interlatent/node/teleop/robot_profile.py   (comments abridged)

# Units are i2rt/MuJoCo native: RADIANS for revolute joints, gripper [0, 1].
# (SO-101 and Koch are in degrees. The profile is where that difference lives.)
_YAM_ARM_JOINT_NAMES = ("joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5")

# EXACT hardware limits, transcribed from i2rt's yam.urdf <limit lower=… upper=…>.
_YAM_ARM_LIMITS = (
    (-2.61799, 3.13),     # joint_0 / URDF joint1
    (0.0, 3.65),          # joint_1 / URDF joint2   lower=0: home sits at this edge
    (0.0, 3.13),          # joint_2 / URDF joint3   lower=0: home sits at this edge
    (-1.5708, 1.5708),    # joint_3 / URDF joint4
    (-1.5708, 1.5708),    # joint_4 / URDF joint5
    (-2.0944, 2.0944),    # joint_5 / URDF joint6
)
_YAM_GRIPPER_LIMIT = (0.0, 1.0)      # 0 closed, 1 open. Not in the URDF; placeholder.

# The URDF declares velocity=10 rad/s on every joint (the motor max). That is far too
# fast for the per-tick SafetyGate clamp, so we cap 5x below it and widen only after
# reading the DRTC-DEBUG joints log on real hardware.
_YAM_ARM_MAX_VELOCITY = tuple(2.0 for _ in _YAM_ARM_JOINT_NAMES)
_YAM_GRIPPER_MAX_VELOCITY = 4.0

_YAM_ARM_REST = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
_YAM_GRIPPER_REST = 1.0              # open. So `home` = 6 zeros + open gripper, per arm.

# One 7-joint block per side (6 revolute + gripper last), composed left-then-right
# into the three topologies the adapter can report:
YAM_PROFILE       = _yam_profile("yam",       ("left", "right"))   # 14 joints
YAM_LEFT_PROFILE  = _yam_profile("yam_left",  ("left",))           #  7 joints
YAM_RIGHT_PROFILE = _yam_profile("yam_right", ("right",))          #  7 joints
```

Three decisions in there are worth calling out, because each is judgement a "just import the
URDF" approach would throw away:

- **The limits are the URDF's, unmodified** - and deliberately *not* inset below the
  mechanical range, the way SO-101's are. YAM's all-zeros home sits exactly on the lower
  edge of `joint_1` and `joint_2` (URDF `lower=0`), so insetting them would make the robot
  unable to home.
- **The velocity caps are ours, not the URDF's.** The URDF says 10 rad/s because that's the
  motor's maximum. We ship 2.0. A velocity cap here is a per-tick safety clamp, not a spec
  sheet.
- **The gripper isn't in the URDF at all** (i2rt combines it in separately from the
  LINEAR_4310 model), so its `[0, 1]` range is still a placeholder pending hardware.

That tension - the URDF is authoritative for *limits* but wrong for *caps*, and silent about
the gripper - is exactly why profiles are still hand-written, and what makes
[deriving them from URDFs](#-future-directions) a real design problem rather than a chore.

One more load-bearing detail: `joint_names` order here **equals**
`YAMNativeRobot.action_features`, and `base.py` raises if they ever diverge. A policy binds
to joint *order*, not names, so this alignment is a correctness property, not a convention.

**Adding a robot to the safety/behaviors/teleop world is: write one of these, register it in
`_PROFILES`.**

### 2. The adapter: what talks to the motors

One directory under [`adapters/`](packages/sdk/src/interlatent/adapters/), implementing the
five methods. `robot.py` is the only required file; see
[the anatomy table above](#what-an-adapter-actually-is-today).

```python
class YAMNativeRobot(ManualActionInterface):   # adapters/yam/robot.py

    def __init__(self, config: YAMAdapterConfig) -> None:
        # robot_kind is per-instance, and it selects the profile topology:
        # --robot-arg arms=left  ->  "yam_left"  ->  YAM_LEFT_PROFILE (7 joints)
        self.robot_kind = "yam" if self._sides == ("left", "right") \
                          else f"yam_{self._sides[0]}"

    @property
    def action_features(self) -> list[str]:    # ordered: defines the action vector
        ...                                    # ['left_joint_0.pos', ..., 'right_gripper.pos']

    def connect(self) -> None: ...             # opens the CAN buses + cameras
    def get_observation(self) -> dict: ...     # joints + camera frames
    def send_action(self, action: dict): ...   # non-blocking, latest-wins, delta-clamped
    def disconnect(self) -> None: ...
```

That `robot_kind` line is worth a second look: it's how one adapter serves three profiles.
Ask for one arm and the robot reports itself as `yam_left`, so the SafetyGate, `home`, and
`action()` all bind to the 7-joint envelope automatically.

The observation and action are plain dicts keyed by `action_features`, plus camera frames:

```python
{
  "shoulder_pan.pos": 12.4,                        # degrees (SO-101)
  "shoulder_lift.pos": -30.0,
  ...
  "gripper.pos": 80.0,
  "observation.images.front": <ndarray (H, W, 3) uint8 RGB>,
}
```

### 3. Runtime knobs: `--robot-arg` and `--camera`

The adapter's `config.py` turns these flat CLI pairs into its typed config. They are
per-robot, and each robot's config doc is authoritative. YAM's are the most illustrative:

```bash
interlatent-node run --robot yam \
  --robot-arg arms=both \                 # both | left | right  (sets 14-DOF vs 7-DOF)
  --robot-arg left_channel=can_follower_l \
  --robot-arg right_channel=can_follower_r \
  --robot-arg max_step_rad=0.05 \         # per-tick delta clamp (execution safety)
  --robot-arg auto_home=false \           # true MOVES THE ARM the instant you connect
  --camera front=/dev/video0              # -> observation.images.front
```

SO-101's surface is much smaller (`--robot-arg id=<calibration-id>` is the one you'll
usually set, forwarded to LeRobot's `SO101FollowerConfig`). `--camera <name>=<device>`
works the same everywhere: **`<name>` must match the policy's training camera keys**, since
it becomes the `observation.images.<name>` the model sees.

Full references: [SO-101 config](packages/sdk/src/interlatent/adapters/lerobot/CONFIG.md) ·
[YAM config](packages/sdk/src/interlatent/adapters/yam/CONFIG.md).

### 4. `node.toml`: who this machine is

Written for you by `interlatent-node pair`; you rarely touch it. It holds a long-lived
credential, so it's created `0600`:

```toml
# ~/.interlatent/node.toml
node_id  = "..."                        # assigned by the dashboard at pair time
token    = "ilnode_..."                 # long-lived node credential
api_base = "https://interlatent.com"
name     = "my-arm"
```

### Where everything lives

```
~/.interlatent/
  node.toml            # this machine's identity + node token (written by `pair`, 0600)
  behaviors.toml       # your named behaviors (optional; overrides built-ins by name)

packages/sdk/src/interlatent/
  adapters/base.py                    # THE CONTRACT: RobotAdapter + ManualActionInterface
  adapters/lerobot/robot.py           # SO-101 / Koch / any LeRobot robot
  adapters/yam/{robot,cameras,config,loop}.py   # a native vendor adapter, in full
  node/teleop/robot_profile.py        # every RobotProfile: limits, velocity, rest pose
  node/teleop/safety.py               # SafetyGate: workspace / velocity / deadman
  node/control.py                     # the bundled LeRobot control loop
  behaviors/data/so101.toml           # built-in behaviors (`hello`)
```

If you read only two files to understand this SDK, read `adapters/base.py` (what a robot is)
and `node/teleop/robot_profile.py` (what your robot is).

## 🧠 How the SDK works

Four layers, bottom to top. Each only knows about the layer directly beneath it.

```
   behaviors        VLA policy (cloud)        teleop            collection
   act("home")      DRTC action chunks        joint targets     watch()/tick()
        │                    │                     │                  │
        └────────────────────┴──────────┬──────────┴──────────────────┘
                                        │
                              ┌─────────▼─────────┐
                              │   control loop    │  observe → decide → clamp
                              │  (once per tick)  │  → send_action → record
                              └─────────┬─────────┘
                                        │
                              ┌─────────▼─────────┐
                              │  robot interface  │  adapters/base.py
                              │    five methods   │
                              └─────────┬─────────┘
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
              lerobot adapter      yam adapter         your adapter
              SO-101, Koch         I2RT YAM            --loop module:fn
```

**The control loop is the heart.** Once per tick it reads an observation, decides which
source is driving the robot this tick (a human on teleop, the policy, or nothing at all),
produces a joint vector, clamps it, calls `send_action`, and records the tick. Everything
above the robot interface is just a different answer to "who is driving."

**Safety is layered, and always local to the motors** - never across the network:

- The **per-adapter delta clamp** (`--robot-arg max_step=…`) caps the per-tick joint jump for
  *every* action, policy and human alike. It is the last thing to touch an action before the
  motors.
- The **`SafetyGate`** adds workspace, velocity, and deadman limits on human-driven motion.
  Its velocity-limited stepping is also what makes the manual `action()` call
  block-then-settle rather than slam.
- Limits come from a per-robot **`RobotProfile`** (joint names, order, limits, velocity caps,
  rest pose). A robot kind with no profile **refuses** manual motion rather than run
  unguarded.

**Running a policy** means talking to a GPU, and big VLA models are too slow for naive
request/response - the arm would stutter. So the client and the pod speak **DRTC**
(Distributed Real-Time Chunking): the robot streams observations continuously and never
blocks, the pod returns *overlapping action chunks*, and the client merges them
last-writer-wins while estimating network-vs-compute latency so it knows how far ahead to
schedule. The result is smooth 30 Hz control on top of a multi-second model. Details in
[docs/concepts.md](docs/concepts.md) and [ARCHITECTURE.md](ARCHITECTURE.md).

**The node daemon** (`interlatent-node`) is how a robot stays online: pair it once, and it
polls the dashboard and converges to whatever inference session is assigned to it. It
resolves the control loop for your `--robot` kind, opens the DRTC client, and runs.

**Collection is local-first.** `watch()` / `tick()` / `collect()` stage per-step
observations, actions, and rewards into local SQLite plus JPEGs. Building a LeRobot v3.0
dataset from that works fully offline with no account; uploading it is a separate, optional
step.

## ⚡ Quickstart

### 1. Install

```bash
pip install interlatent
```

> **Per-robot extras.** The base package is robot-agnostic. Driving real hardware needs the
> extra for your robot - install **one** of:
> ```bash
> pip install 'interlatent[lerobot]'   # SO-101 and other LeRobot robots
> pip install 'interlatent[yam]'       # I2RT YAM (Linux + SocketCAN)
> ```
> SO-101's Feetech servos additionally need the Feetech servo SDK; if the serial bus won't
> open, `pip install feetech-servo-sdk`. See each robot's config doc under
> [Supported robots](#-supported-robots) for full host requirements.

**Requires Python 3.11+.**

### 2. Drive a robot directly (no cloud, no account)

The fastest thing you can do needs no GPU and no policy - just an arm. This is the robot
interface with nothing on top of it:

```python
import interlatent as il

with il.Robot("so101", port="/dev/ttyACM0") as robot:
    print(robot.pose())                     # read joint state: {'shoulder_pan': 0.0, ...}
    robot.act("home")                       # go to the robot's rest pose, block until reached
    robot.act("hello")                      # play the packaged SO-101 wave
    robot.act("hello", speed=0.5)           # the same wave, at half speed
    robot.move(wrist_roll=30, duration=0.5) # ad-hoc joint move, no behavior needed
```

**Where do `home` and `hello` come from? You don't set them up.** That's the point, so it's
worth being precise about what each one is:

- **`home` is generated, never authored.** It is built from your robot's
  `RobotProfile.rest_pose`, so it cannot drift from the hardware: change the profile and
  `home` changes with it. On SO-101 the rest pose is all six joints at 0°. **Every robot
  kind with a profile gets `home` for free**, including ones that ship no behavior file at
  all. It is the one behavior you can always assume exists.
- **`hello` is a packaged example**, and only for SO-101
  ([`behaviors/data/so101.toml`](packages/sdk/src/interlatent/behaviors/data/so101.toml) is
  the only built-in file today). It exists to show what a hand-authored behavior looks like.
  Ask for it on an arm that doesn't define it and you get an error naming the behavior and
  listing what that arm *does* have, rather than a surprise movement.

A behavior is just **data**. `hello` in full is a keyframed wrist wave, and this is the
entire definition:

```toml
[hello]
type = "trajectory"
interpolation = "min_jerk"
description = "Raise the arm and wave the wrist."
keyframes = [
    { t = 0.0, shoulder_lift = 0.0, elbow_flex = 0.0, wrist_flex = 0.0, wrist_roll = 0.0 },
    { t = 1.5, shoulder_lift = -30.0, elbow_flex = -40.0 },   # raise the forearm
    { t = 2.1, wrist_roll = 35.0 },                           # wave
    { t = 2.7, wrist_roll = -35.0 },
    { t = 3.3, wrist_roll = 35.0 },
    { t = 3.9, wrist_roll = -35.0 },
    { t = 4.5, wrist_roll = 0.0 },                            # straighten the wrist
    { t = 6.0, shoulder_lift = 0.0, elbow_flex = 0.0 },       # lower the forearm
]
```

Times are seconds, arm joints are degrees, and `min_jerk` smooths between keyframes. The
amplitudes are deliberately conservative: the wrist swings peak at ~219°/s against a 240°/s
cap, and the shoulder raise at ~38°/s against a 50°/s cap. Those caps come from the same
profile that generates `home`.

Your own behaviors resolve through four layers, each overriding the previous **by name** -
so you can redefine `home` or `hello` without touching the package:

1. **Built-in** - generated `home`, plus any packaged `data/<robot>.toml`.
2. **User file** - `~/.interlatent/behaviors.toml`.
3. **Explicit file** - `Robot(behaviors=...)` or `--behaviors`.
4. **Procedural** - Python functions registered with `@il.behavior`.

Nothing moves before it is checked. Declarative behaviors are validated against the profile
**as they load**: unknown joint names, out-of-limit targets, and velocity-cap violations all
raise an error naming the behavior, joint, value, and limit. That is why `behavior validate`
below needs no hardware. Full format reference: [docs/behaviors.md](docs/behaviors.md).

The same commands work from the terminal:

```bash
interlatent behavior ls --robot so101
interlatent behavior validate my_behaviors.toml --robot so101   # validate, no hardware
interlatent behavior run hello --robot so101 --port /dev/ttyACM0 --speed 0.5
```

No arm handy? [`examples/07_named_behaviors.py`](examples/07_named_behaviors.py) runs the
whole thing against a fake adapter and prints the action stream.

### 3. Run a cloud policy on it

Sign in at [interlatent.com](https://interlatent.com), create an API key, and export it:

```bash
export INTERLATENT_API_KEY=ilat_...
```

Pair the machine on your robot once, then run the node daemon:

```bash
interlatent-node pair --name my-arm --api-key ilat_...
interlatent-node run  --robot so101 --port /dev/ttyACM0 --camera front=/dev/video0
```

Then start a session against it, from the CLI or the dashboard:

```bash
interlatent gpus ls          # GPU pods available to your account
interlatent nodes ls         # robot nodes paired to your account
interlatent session start --node my-arm --gpu a100-0 --policy lerobot/smolvla_base
interlatent session stop <session-id>
```

The node picks up the assigned session and the arm starts moving. To test the cloud path
with no robot attached:

```bash
interlatent-preflight --environment my-arm --policy lerobot/smolvla_base
```

That opens a real session against a managed GPU pod, streams synthetic observations, and
prints a **PASS / WARN / FAIL** verdict with measured network-vs-compute latency. It
exercises the cloud inference path only, not your cameras, joints, or motor bus.

### 4. Or drive the loop yourself

If you'd rather own the control loop instead of running the daemon:

```python
from interlatent.inference.integration import connect_drtc

client = connect_drtc(
    environment="my-arm",
    policy_uri="lerobot/smolvla_base",
    api_key="ilat_...",                # or rely on INTERLATENT_API_KEY
    task="pick up the red cube",
    fps=30,
)
while running:
    action = client.step(observation_npz_bytes, codec="npz")  # None while the first chunk loads
    if action is not None:
        robot.send_action(action)
client.close()
```

An observation is just an `np.savez` blob whose keys mirror LeRobot features
(`observation.images.<camera>`, `observation.state`, `task`). See
[`examples/03_run_on_so101.py`](examples/03_run_on_so101.py) for a complete SO-101 loop, or
[`examples/06_connect_hosted.py`](examples/06_connect_hosted.py) for the minimal connect.

### Configuration

Only `INTERLATENT_API_KEY` is required; the rest are optional tuning knobs.

| Env var | What it does |
|---|---|
| `INTERLATENT_API_KEY` | Your account API key (`ilat_…`). Authenticates the CLI and DRTC inference. **Required.** |
| `INTERLATENT_DRTC_URL` | Pin the DRTC inference endpoint (operator/dev override; normally provided per-session). |
| `INTERLATENT_NUM_INFERENCE_STEPS` | Flow-matching denoising steps for VLA policies (e.g. MolmoAct2). Range 3-10; default 5. |
| `INTERLATENT_IMAGE_RESIZE` | Resize camera frames to this square edge (px) before JPEG-encoding. `256` suits MolmoAct2. |
| `INTERLATENT_NODE_CONFIG` | Path to the node config TOML (default `~/.interlatent/node.toml`). |
| `INTERLATENT_CALIB_PRESET` | Force or disable a joint-calibration preset (e.g. `so101_pre777`, or `none`). |

## 🦾 Supported robots

Each robot has its own config doc covering host requirements, `--robot-arg` knobs, camera
declarations, joint names/units, and worked examples. For the joint counts, units, and
which profile each kind binds to, see
[What actually defines a robot](#-what-actually-defines-a-robot).

| Robot | `--robot` | Extra | Config doc |
|---|---|---|---|
| **SO-101** (reference) | `so101` | `[lerobot]` (+ `feetech-servo-sdk`) | [config](packages/sdk/src/interlatent/adapters/lerobot/CONFIG.md) |
| I2RT YAM (bimanual) | `yam` | `[yam]` | [config](packages/sdk/src/interlatent/adapters/yam/CONFIG.md) |
| Any LeRobot robot | `<type>` | `[lerobot]` | cameras attach as `observation.images.<name>` |
| Custom hardware | `--loop module:fn` | - | bring your own I/O loop |

For the policy side (SmolVLA, Pi0, ACT, MolmoAct2, your fine-tunes), see
[docs/robots-and-policies.md](docs/robots-and-policies.md).

**Missing your arm?** Adding robots is the contribution we most want, and per
[the section above](#-one-robot-class-many-adapters) it should cost you one `robot.py` and a
profile. See [CONTRIBUTING.md](CONTRIBUTING.md).

## 🖥️ Using the dashboard

The [Interlatent dashboard](https://interlatent.com) owns the cloud side: the GPU pods, and
which policy each robot is running. The core objects:

- **Environments** - a robot setup and its task, the unit everything else hangs off. The `environment` slug you pass to `connect_drtc` matches one here.
- **GPU boxes** - managed, warm cloud GPUs that serve the policy. You don't rent or boot the hardware. (`interlatent gpus ls`)
- **Nodes** - your paired robots, created by `interlatent-node pair`. The running daemon heartbeats and reports status. (`interlatent nodes ls`)
- **Sessions** - a policy running on a GPU box, bound to a node. Start one and the node converges to it; stop it and the arm idles. (`interlatent session start | ls | stop`)

Create an environment, configure its policy, start a GPU box, pair and run your node, then
start a session. The node picks it up and the arm starts moving under the policy.

## 📚 Examples

| Example | Hardware needed |
|---|---|
| [`03_run_on_so101.py`](examples/03_run_on_so101.py) - drive an SO-101 against a cloud pod | SO-101 (or none - synthesizes obs) |
| [`04_manual_action.py`](examples/04_manual_action.py) - one-shot manual joint move | a supported arm |
| [`06_connect_hosted.py`](examples/06_connect_hosted.py) - the minimal cloud connect | none |
| [`07_named_behaviors.py`](examples/07_named_behaviors.py) - named behaviors offline | none (fake arm) or a supported arm |

## ☁️ Open source vs. Interlatent Cloud

This SDK is open source and yours to run, but it's built to plug into the
[dashboard](https://interlatent.com), which runs inference on managed GPUs and orchestrates
your pods, nodes, and sessions - so you never operate GPUs, warm pools, or storage.

| Capability | Open source | [Interlatent](https://interlatent.com) |
|---|:---:|:---:|
| One interface + safety model across robots | ✅ | ✅ |
| Drive robots directly (behaviors, manual moves) | ✅ | ✅ |
| Robot node daemon + DRTC client | ✅ | ✅ |
| Run a VLA policy on your robot | - (needs a GPU pod) | ✅ managed warm GPUs, no cold starts |
| CLI for pods / nodes / sessions | ✅ | ✅ + full dashboard |
| Hosted, versioned datasets | DIY | ✅ managed, shareable |
| Auto policy analysis & reports | ❌ | ✅ |
| GPU autoscaling & warm pools | ❌ | ✅ |
| Support / SLA | community | ✅ |

## 📖 Documentation

- [Getting started](docs/getting-started.md) - robot → first rollout
- [Named behaviors](docs/behaviors.md) - drive robots directly (Python + CLI + TOML), no cloud
- [The action interface](docs/action-interface.md) - the robot contract in depth
- [Concepts](docs/concepts.md) - DRTC, sessions, chunks, the node
- [Supported robots & policies](docs/robots-and-policies.md)
- [Going to cloud](docs/going-to-cloud.md)
- [Architecture](ARCHITECTURE.md) - for contributors

## 🤝 Contributing

We'd love your help - especially **adding robots**, which is how this project gets breadth.
Start with [CONTRIBUTING.md](CONTRIBUTING.md) and the
[`good first issue`](https://github.com/interlatent/interlatent/labels/good%20first%20issue) label.

This project uses the [Developer Certificate of Origin](https://developercertificate.org/)
(`git commit -s`). Questions, demos, robot pics: team@interlatent.com.

## 📄 License

[Apache-2.0](LICENSE) © Interlatent Contributors.

"Interlatent Cloud" and the hosted service at interlatent.com are operated separately from
this open-source project.

## 🔭 Future directions

Forward-looking work that isn't scheduled yet. Each item is a direction, not a spec.

### Fold the adapters into the robot class

The [opening section](#-one-robot-class-many-adapters) states the goal; this is the shape of
the work. Today an adapter is up to four files (`robot.py`, `config.py`, `cameras.py`,
`loop.py`), and two of them exist only because we haven't finished the abstraction.

**Direction:** a new robot should be one `robot.py` plus a `RobotProfile`.

**What we know already:**
- The robot contract (`adapters/base.py`) is settled and does not need to change. This work
  sits entirely above it.
- YAM's `cameras.py` already has the right shape - a `Camera` Protocol with lazily-imported
  vendor backends behind it. Promoting it to a shared module is mostly a move, not a design.
- The control loops are near-duplicates. They share the observe → decide → clamp →
  `send_action` → record skeleton and the same wire helpers; they diverge on whether teleop
  is wired, which safety composition applies, and which calibration preset is active.

**Open design questions (resolve before building):**
- What is the unit of variation for the universal loop - capability flags the robot declares,
  a strategy object per driving source, or hooks the adapter can override? Flags are simplest
  until a robot needs a genuinely different tick shape.
- Some robots need per-tick work that isn't "send an action" (liveness proofs, keep-alive
  pumps, watchdog feeds for arms driven through a daemon). Does that belong in
  `get_observation`, in an explicit `tick()` on the contract, or outside the loop entirely?
- Cameras behind a network transport rather than a local SDK still have to satisfy
  `read() -> uint8 HxWx3 RGB`. Does the shared Camera protocol need a staleness/async story,
  or is latest-wins-plus-decode enough?
- Does `config.py` survive as a schema the daemon validates against, or does the robot
  declare its own knobs and the daemon stay generic?

### Robots should consume URDFs directly

Today a robot's kinematic facts - joint names, order, limits, velocity caps, rest
pose - are hand-transcribed into static `RobotProfile` literals in
[`robot_profile.py`](packages/sdk/src/interlatent/node/teleop/robot_profile.py). That
is a transcription step that drifts from the hardware: the YAM profile shipped with a
conservative placeholder envelope, and the real limits only landed once we pulled the
joint `<limit>` values out of the i2rt YAM URDF by hand. The URDF is the manufacturer's
source of truth; the robot should read it rather than restate it.

**Direction:** let a robot derive its profile (and eventually FK/collision data)
from the robot's URDF, so limits/order/rest-pose come from one authoritative file.

**What we know already:**
- I2RT ships a real YAM URDF at `i2rt/robot_models/arm/yam/yam.urdf` (joints listed
  reversed vs i2rt command order; `joint1..joint6` map to our `joint_0..joint_5`).
  The arm `joint_limits` in our profile are now transcribed from it; `max_velocity`
  and the gripper range are still hand-chosen (the gripper is combined in separately
  from the `LINEAR_4310` model, so it is not in `yam.urdf`).

**Open design questions (resolve before building):**
- Parse the URDF at build time into a static profile (keeps the current convention,
  no runtime parse-dep) vs. at `connect()` (always matches the installed driver, adds
  a `yourdfpy`-style dependency on the import path)?
- Vendor the URDF + meshes into the robot package, or read it from the installed vendor
  package (e.g. i2rt's `ARM_YAM_XML_PATH`)? Meshes/asset paths complicate vendoring.
- How does URDF joint order reconcile with `action_features` ordering (the policy
  binds to order, not names)? The reversed YAM ordering shows this needs an explicit
  mapping, not a blind import.
- Keep the static literal as a hand-verified fallback / safety-tightened override, or
  treat the URDF as canonical? URDF limits are mechanical max - we currently inset
  velocity below them on purpose, which a naive import would lose.
