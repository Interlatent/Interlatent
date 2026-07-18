# Defining a robot

Everything this SDK does to an arm goes through one contract: four methods plus three
pieces of metadata (`action_features`, `joint_specs`, `robot_kind`) on a robot object
([`adapters/base.py`](packages/sdk/src/interlatent/adapters/base.py)). The
[README](README.md#robot-class) explains that idea, and
[Supported robots](README.md#supported-robots) lists the arms that work today. This document
is the reference for the files underneath: what each one is, what it decides, and what you
would write to add an arm of your own.

Four files define a robot:

| File | What it decides |
|---|---|
| [1. The profile](#1-the-profile-what-your-robot-physically-is) | joint names and order, software limits, velocity caps, rest pose |
| [2. The adapter](#2-the-adapter-what-talks-to-the-motors) | what talks to the motors and the cameras |
| [3. `--robot-arg` / `--camera`](#3-runtime-knobs---robot-arg-and---camera) | per-run configuration |
| [4. `node.toml`](#4-nodetoml-who-this-machine-is) | which machine this is, and its credential |

Only the first two are yours to write. The third is a CLI surface your adapter declares, and
the fourth is generated for you.

---

## 1. The profile: what your robot physically is

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
# (SO-101 is in degrees. The profile is where that difference lives.)
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
[deriving them from URDFs](README.md#future-directions) a real design problem rather than a
chore.

One more load-bearing detail: `joint_names` order here **equals**
`YAMNativeRobot.action_features`, and `base.py` raises if they ever diverge. A policy binds
to joint *order*, not names, so this alignment is a correctness property, not a convention.

**Adding a robot to the safety/behaviors/teleop world is: write one of these, register it in
`_PROFILES`.**

## 2. The adapter: what talks to the motors

[`adapters/base.py`](packages/sdk/src/interlatent/adapters/base.py) declares the contract as
two pieces:

- **`RobotAdapter`** - a `Protocol` (a duck type, not a base class you must subclass).
  Lifecycle plus observe/act, plus the metadata the manual path needs (`action_features`,
  `joint_specs`).
- **`ManualActionInterface`** - a mixin carrying the *one* piece of shared behavior:
  `action(shoulder_pan=30, gripper=80)`, a named-joint, block-then-settle move composed
  entirely out of the adapter's own `send_action` + `get_observation`. Every adapter
  inherits it; none of them implement it.

An adapter is one directory under
[`adapters/`](packages/sdk/src/interlatent/adapters/), implementing the contract:

| File | Role |
|---|---|
| `robot.py` | **The robot.** Implements the contract. Owns the vendor driver (CAN bus, serial, motor SDK) and the cameras. The only file that has to exist. |
| `config.py` | Turns the daemon's flat CLI passthrough (`--robot-arg key=value`, `--camera name=device`) into a typed config dataclass. Deliberately import-light, so importing the adapter never drags in its heavy extra. |
| `cameras.py` | Frame capture, normalized to `uint8 HxWx3` RGB. Vendor SDKs are imported lazily inside methods. |
| `loop.py` | A per-robot control loop, registered so `--robot <kind>` resolves to it. |

A useful way to read the tree: `robot.py` is the *leaf*, `base.py` is the *contract*, and
the rest is plumbing that exists because a robot needs configuring and looking at. Two of
those four exist only because the abstraction isn't closed yet, which is why folding them
into the robot class is a [future direction](README.md#fold-the-adapters-into-the-robot-class).

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

    @property
    def joint_specs(self) -> Sequence[JointSpec]:  # ordered, aligned with action_features
        ...

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

## 3. Runtime knobs: `--robot-arg` and `--camera`

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

## 4. `node.toml`: who this machine is

Written for you by `interlatent-node pair`; you rarely touch it. It holds a long-lived
credential, so it's created `0600`:

```toml
# ~/.interlatent/node.toml
node_id  = "..."                        # assigned by the dashboard at pair time
token    = "ilnode_..."                 # long-lived node credential
api_base = "https://interlatent.com"
name     = "my-arm"
```

## Special case: the dimos adapter (the robot is a running stack)

`--robot dimos` ([`adapters/dimos/`](packages/sdk/src/interlatent/adapters/dimos/),
`interlatent[dimos]`, python 3.11–3.12) inverts the usual shape: there is no motor
driver, because the "vendor SDK" is a **running dimos process**. The adapter joins
dimos's LCM/Zenoh bus as a peer — `coordinator_joint_state` + camera `Image`
topics in, `joint_command` out (consumed by a dimos **servo task** that claims
every joint *including the gripper* — dimos stomps unclaimed grippers back to
their startup value while streaming). The four files still exist, but two get
reinterpreted:

- **The profile** works exactly as above (`dimos_xarm7` in `robot_profile.py`,
  radians) — and matters *more* here: dimos applies **no limits** to streamed
  joint commands, so the profile + the adapter's `max_step_rad` clamp are the
  only safety envelope in the entire path.
- **The adapter** declares an embodiment per *kind* (`--robot-arg kind=xarm7`),
  and `connect()` **verifies the declaration against the live stack** —
  coordinator present, joints, a servo task claiming exactly the kind's joints
  with a non-zero timeout and no competing claimant, joint-state order — failing
  closed with every mismatch listed. The trap this exists for: a stock dimos
  coordinator blueprint has no servo task and *silently ignores* streamed
  commands.
- `--camera <name>=<topic>` maps observation keys to bus topics instead of
  devices.
- The dimos side needs a **session blueprint** satisfying that contract; this
  SDK ships one per kind via dimos's entry-point registry:

```bash
dimos run interlatent.xarm7          # terminal 1: the dimos session stack
interlatent-node run --robot dimos \
  --robot-arg kind=xarm7 \
  --camera wrist=/color_image        # terminal 2: the node
```

See [`adapters/dimos/CONFIG.md`](packages/sdk/src/interlatent/adapters/dimos/CONFIG.md)
for the full knob table and blueprint contract, and
[docs/adr/0018](docs/adr/0018-dimos-adapter-external-bus-peer.md) for the design.

## Adding a new robot

Putting the four files together, the whole job for a new arm is:

1. **Write the profile.** A `RobotProfile` in
   [`robot_profile.py`](packages/sdk/src/interlatent/node/teleop/robot_profile.py): joint
   names in the adapter's order, software limits, per-joint velocity caps, rest pose.
   Register it in `_PROFILES` under your robot kind. Start conservative; the `SafetyGate`
   fails safe when limits are too tight, not too loose.
2. **Write the adapter.** A `robot.py` implementing the contract, inheriting
   `ManualActionInterface` so `action()` comes free. Keep vendor SDK imports lazy (inside
   methods) so importing the package never requires your extra.
3. **Make sure `action_features` order matches the profile's `joint_names`.** `base.py`
   raises if they diverge, and a policy binds to order.
4. **Register a control loop** if your robot cannot use the bundled LeRobot one: add an
   entry to `_NATIVE_LOOPS` in [`node/daemon.py`](packages/sdk/src/interlatent/node/daemon.py),
   or pass `--loop module:fn`. (This step is the one we are trying to
   [delete](README.md#future-directions).)
5. **Optionally ship behaviors** as `behaviors/data/<robot>.toml`. You get `home` for free
   from the profile either way.

Adding robots is the contribution we most want. See
[CONTRIBUTING.md](CONTRIBUTING.md), and note that steps 3-4 exist only because the adapter
layer isn't finished; the goal is for a new arm to cost you steps 1 and 2 and nothing else.
