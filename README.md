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

[Robot class](#robot-class) · [How it works](#how-the-sdk-works) · [Quickstart](#quickstart) · [Robots](#supported-robots) · [Defining a robot](ROBOT.md) · [Docs](docs/)

</div>

---

Robotics tooling is fragmented: every arm ships its own SDK, its own joint conventions, its
own scripts. Interlatent is **one interface across robots** - a single way to read state and
command joints, with a shared safety model underneath. Write against it once and you can
teleoperate, run models, and collect data on any supported arm.

This README leads with the architecture, because the architecture *is* the pitch. If you
just want to move an arm, skip to the [Quickstart](#quickstart).

## Robot Class

Everything in this SDK rests on one idea: **a robot is a single object with four methods
and the metadata that gives them meaning.**

```python
robot.connect()
obs = robot.get_observation()   # joint positions + camera frames
robot.send_action(action)       # absolute joint targets
robot.disconnect()
robot.action_features           # the ordered joint names; defines what an action means
robot.joint_specs               # per-joint specs, aligned with action_features
robot.robot_kind                # the kind string the platform keys config on
```

That is the whole contract, and it is the only thing the layers above a robot are allowed to
know about. Behaviors, VLA policies, teleop, and dataset recording are all written against
this contract, so adding a robot gives you all of them at once.

Four words carry the rest of the docs:

| Term | What it means |
|---|---|
| **contract** | the interface above. Nothing above a robot knows anything else about it. |
| **adapter** | the code that implements them for one robot: its motors, cameras, and units. |
| **profile** | your arm's physical facts: joint names and their order, limits, speed caps, and the home pose. |
| **kind** | the name you ask for (`--robot yam`, `il.Robot("so101")`). It selects the adapter and the profile. |

Two rules shape everything above:

- **Actions are joint-space.** Absolute joint targets, one per joint. There is no IK and no
  Cartesian frame anywhere in the robot-side stack.
- **An action is a waypoint, not a destination.** Sending one never blocks, and the newest
  one wins. The control loop sends one per tick.

**Adding a robot is an adapter plus a profile.** [ROBOT.md](ROBOT.md) is the file-by-file
reference; [Supported robots](#supported-robots) lists the arms that work today.

## How the SDK works

Four layers, bottom to top. Each only knows about the layer directly beneath it.

```
   behaviors        VLA policy (cloud)        teleop            recording
   act("home")      DRTC action chunks        joint targets     RecordTicks
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
                              │   robot contract  │
                              └─────────┬─────────┘
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
              lerobot adapter      yam adapter         your adapter
              SO-101               I2RT YAM            --loop module:fn
```

**The control loop is the heart.** Once per tick it reads an observation, decides which
source is driving the robot this tick (the policy, a VR teleoperator recording a
demonstration, or nothing at all), produces a joint vector, clamps it, calls
`send_action`, and records the tick. Everything above the robot interface is just a
different answer to "who is driving."

**Safety is layered, and always local to the motors** - never across the network:

- The **per-adapter delta clamp** (`--robot-arg max_step=…`) caps the per-tick joint jump for
  *every* action, policy and human alike. It is the last thing to touch an action before the
  motors.
- The **`SafetyGate`** adds workspace, velocity, and deadman limits on human-driven motion.
  Its velocity-limited stepping is also what makes the manual `action()` call
  block-then-settle rather than slam.
- Both read their limits from the robot's **`RobotProfile`** (joint names, order, limits,
  velocity caps, rest pose), which is why a robot without one gets no human-driven motion at
  all.

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

**Collection is streaming-first.** The control loop JPEG-encodes each camera frame per
tick and streams `RecordTick`s to the hosted recorder, which builds and uploads the
LeRobot v3.0 dataset server-side. A node-side disk spool with delete-after-ack keeps the
uplink lossless — a link drop never silently thins an episode. (The old on-device
`watch()`/`tick()`/`upload()` build was removed in 2.0.0; existing datasets enter via
the dashboard's HF import.)

## Quickstart

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
> [Supported robots](#supported-robots) for full host requirements.

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
| `INTERLATENT_JPEG_BACKEND` | Force the frame encoder (`auto`\|`nvjpeg`\|`gpujpeg`\|`turbojpeg`\|`cv2`\|`pil`). See [node encoding](docs/node-encoding.md). |
| `INTERLATENT_PREVIEW_HZ` | Live teleop preview push rate (1-30, default 10). Competes with recording/teleop for uplink — see [node encoding](docs/node-encoding.md). |

## Supported robots

`--robot <kind>` on the CLI, `il.Robot("<kind>")` in Python. Each robot's config doc covers
host requirements, `--robot-arg` knobs, camera declarations, and worked examples.

| Robot | `--robot` | Joints and units | Extra | Config doc |
|---|---|---|---|---|
| **SO-101** (reference) | `so101`, `so101_follower` | 6; degrees, gripper 0-100 | `[lerobot]` (+ `feetech-servo-sdk`) | [config](packages/sdk/src/interlatent/adapters/lerobot/CONFIG.md) |
| **I2RT YAM** (bimanual) | `yam`, `yam_bimanual` | 14 (left block, then right); radians, gripper 0-1 | `[yam]` | [config](packages/sdk/src/interlatent/adapters/yam/CONFIG.md) |
| **I2RT YAM** (single arm) | `yam_left`, `yam_right` | 7; radians, gripper 0-1 | `[yam]` | [config](packages/sdk/src/interlatent/adapters/yam/CONFIG.md) |
| **Nori** (dual-SO-101 rig, **unstable beta**) | `nori` | 12 (left block, then right); daemon-normalized ±100 | `[nori]` | [config](packages/sdk/src/interlatent/adapters/nori/CONFIG.md) |
| **Almond Axol** (dual arm, **unstable beta**) | `axol` | 16 (7 + gripper per side); radians, gripper 0-1 | `[axol]` | [config](packages/sdk/src/interlatent/adapters/axol/CONFIG.md) |
| Any other LeRobot robot | `<type>` | LeRobot's | `[lerobot]` | policy only, see below |
| Custom hardware | `--loop module:fn` | yours | - | bring your own I/O loop |

The SO-101, YAM, and Nori rows are the kinds that ship a **`RobotProfile`** (the full
list lives in
`_PROFILES` in [`robot_profile.py`](packages/sdk/src/interlatent/node/teleop/robot_profile.py));
Axol does not yet, so it is policy-only.
That distinction is the one rule worth internalizing:

> **No profile, no human-driven motion.** Any other LeRobot robot still runs a cloud policy
> fine. But `action()`, behaviors (including `home`), and teleop **refuse to run** without a
> profile, rather than move an arm with no safety envelope. This fails closed on purpose.

For the policy side (SmolVLA, Pi0, ACT, MolmoAct2, your fine-tunes), see
[docs/robots-and-policies.md](docs/robots-and-policies.md).

**Missing your arm?** Adding robots is the contribution we most want, and it should cost you
one `robot.py` and a profile. [ROBOT.md](ROBOT.md#adding-a-new-robot) is the walkthrough;
[CONTRIBUTING.md](CONTRIBUTING.md) is the process.

## Using the dashboard

The [Interlatent dashboard](https://interlatent.com) owns the cloud side: the GPU pods, and
which policy each robot is running. The core objects:

- **Environments** - a robot setup and its task, the unit everything else hangs off. The
  `environment` slug you pass to `connect_drtc` matches one here.
- **GPU boxes** - managed, warm cloud GPUs that serve the policy. You don't rent or boot the
  hardware. (`interlatent gpus ls`)
- **Nodes** - your paired robots, created by `interlatent-node pair`. The running daemon
  heartbeats and reports status. (`interlatent nodes ls`)
- **Sessions** - a policy running on a GPU box, bound to a node. Start one and the node
  converges to it; stop it and the arm idles. (`interlatent session start | ls | stop`)

Create an environment, configure its policy, start a GPU box, pair and run your node, then
start a session. The node picks it up and the arm starts moving under the policy.

## Examples

| Example | Hardware needed |
|---|---|
| [`03_run_on_so101.py`](examples/03_run_on_so101.py) - drive an SO-101 against a cloud pod | SO-101 (or none - synthesizes obs) |
| [`04_manual_action.py`](examples/04_manual_action.py) - one-shot manual joint move | a supported arm |
| [`06_connect_hosted.py`](examples/06_connect_hosted.py) - the minimal cloud connect | none |
| [`07_named_behaviors.py`](examples/07_named_behaviors.py) - named behaviors offline | none (fake arm) or a supported arm |

## Open source vs. Interlatent Cloud

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

## Documentation

- [Getting started](docs/getting-started.md) - robot → first rollout
- [Defining a robot](ROBOT.md) - the profile, the adapter, and adding your own arm
- [Named behaviors](docs/behaviors.md) - drive robots directly (Python + CLI + TOML), no cloud
- [The action interface](docs/action-interface.md) - the robot contract in depth
- [Concepts](docs/concepts.md) - DRTC, sessions, chunks, the node
- [Supported robots & policies](docs/robots-and-policies.md)
- [Teleoperation](docs/teleop.md) - drive the robot in VR to collect demonstrations, safety, recordings
- [Node encoding & GPU acceleration](docs/node-encoding.md) - the JPEG backend chain, Jetson GPUJPEG setup, bandwidth budgeting
- [Going to cloud](docs/going-to-cloud.md)
- [Architecture](ARCHITECTURE.md) - for contributors

## Contributing

We'd love your help - especially **adding robots**, which is how this project gets breadth.
Start with [CONTRIBUTING.md](CONTRIBUTING.md) and the
[`good first issue`](https://github.com/interlatent/interlatent/labels/good%20first%20issue) label.

This project uses the [Developer Certificate of Origin](https://developercertificate.org/)
(`git commit -s`). Questions, demos, robot pics: team@interlatent.com.

## License

[Apache-2.0](LICENSE) © Interlatent Contributors.

"Interlatent Cloud" and the hosted service at interlatent.com are operated separately from
this open-source project.

## Future directions

Forward-looking work that isn't scheduled yet. Each item is a direction, not a spec.

### Fold the adapters into the robot class

The [robot contract](#robot-class) is real and it works. The adapter layer *around* it is
not finished, and we would rather say so here than let you discover it in the source. Today
an adapter is up to four files (`robot.py`, `config.py`, `cameras.py`, `loop.py`), and two
of them exist only because we haven't closed the abstraction.

**Where it stands:**
- **The robot is a clean abstraction.** `base.py` + `robot.py` genuinely is one interface
  across arms. This part is done, and the work below sits entirely above it.
- **Cameras are only partly abstracted.** The YAM adapter defines a proper `Camera` Protocol
  (`connect` / `read() -> RGB` / `disconnect`) with lazily-imported RealSense, ZED, and UVC
  backends behind it. That is the right shape, but it is *local to that adapter* - others
  open their cameras inside `robot.py`, so there is no single camera seam across the SDK.
- **The control loop is copy-pasted, not factored.** There are three
  ([`node/control.py`](packages/sdk/src/interlatent/node/control.py) for LeRobot robots,
  plus a `loop.py` per native adapter). They share the observe → decide → clamp →
  `send_action` → record skeleton and the same wire helpers, and diverge only on whether
  teleop is wired, which safety composition applies, and which calibration preset is
  active. Those differences are *configuration wearing the costume of code*.

**Direction:** a new robot should be one `robot.py` plus a `RobotProfile`.

1. **One `Camera` protocol** for the whole SDK (`connect` / `read() -> uint8 HxWx3 RGB` /
   `disconnect`) that every adapter implements rather than reinvents. YAM's is already the
   template, so promoting it to a shared module is mostly a move, not a design.
2. **One universal control loop**, parametrized instead of duplicated. The per-adapter
   variation becomes explicit capabilities the robot *declares* (does it support teleop? does
   it have an e-stop latch? which calibration applies?) rather than a forked copy of the loop.
3. **A smaller adapter.** Once cameras and the loop are shared, `config.py` shrinks to a
   schema and `loop.py` disappears.

The test for whether we've done this right: **adding an arm should be one file and a
profile.** Anything more is a seam we haven't closed yet.

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
