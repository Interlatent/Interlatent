<div align="center">

<img src="assets/Final Logo pt6.png" alt="Interlatent" width="420"/>

### One open interface to control every robot.

The open-source SDK and protocol for controlling robots. Read joint state and command
motion the **same way on every supported arm** — whether you're driving it by hand,
playing a named behavior, running a cloud VLA policy, or recording a dataset. Add a robot
once (an adapter + a profile) and every capability above it comes for free.

[![PyPI](https://img.shields.io/pypi/v/interlatent?color=7C5CFF&label=interlatent)](https://pypi.org/project/interlatent/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LeRobot](https://img.shields.io/badge/works%20with-%F0%9F%A4%97%20LeRobot-FFD21E)](https://github.com/huggingface/lerobot)
[![GitHub stars](https://img.shields.io/github/stars/interlatent/interlatent?style=social)](https://github.com/interlatent/interlatent)

[About](#about) · [Features](#features) · [Installation](#installation) · [Usage](#usage) · [API docs](#api-docs) · [Contributing](#contributing)

</div>

---

## About

Robotics tooling is fragmented: every arm ships its own SDK, its own joint conventions, its
own scripts. Interlatent is **one interface across robots** — a single way to read state and
command joints, with a shared safety model underneath. Write against it once and you can
teleoperate, run models, and collect data on any supported arm.

Everything rests on one idea: **a robot is a single object with four methods and the
metadata that gives them meaning.** That is the `RobotAdapter` contract, and it is the only
thing the layers above a robot are allowed to know about:

```python
robot.connect()
obs = robot.get_observation()   # joint positions + camera frames
robot.send_action(action)       # absolute joint targets
robot.disconnect()
robot.action_features           # the ordered joint names; defines what an action means
robot.joint_specs               # per-joint specs, aligned with action_features
robot.robot_kind                # the kind string the platform keys config on
```

Behaviors, VLA policies, teleop, and dataset recording are all written against this
contract, so adding a robot gives you all of them at once.

Four words carry the rest of the docs:

| Term | What it means |
|---|---|
| **contract** | the interface above. Nothing above a robot knows anything else about it. |
| **adapter** | the code that implements it for one robot: its motors, cameras, and units. |
| **profile** | your arm's physical facts: joint names and their order, limits, speed caps, and the rest pose. |
| **kind** | the name you ask for (`--robot yam`, `il.Robot("so101")`). It selects the adapter and the profile. |

Two rules shape everything above:

- **Actions are joint-space.** Absolute joint targets, one per joint. There is no IK and no
  Cartesian frame anywhere in the robot-side stack. (VR teleop solves IK *in the browser*,
  and sends joint targets down.)
- **An action is a waypoint, not a destination.** Sending one never blocks, and the newest
  one wins. The control loop sends one per tick.

### How it fits together

Four layers, bottom to top. Each only knows about the layer directly beneath it.

```
   behaviors        VLA policy (GPU box)      teleop            recording
   act("home")      DRTC action chunks        joint targets     RecordTicks
        │                    │                     │                  │
        └────────────────────┴──────────┬──────────┴──────────────────┘
                                        │
                              ┌─────────▼─────────┐
                              │   control loop    │  observe → arbitrate → gate
                              │  (once per tick)  │  → clamp → send_action → record
                              └─────────┬─────────┘
                                        │
                              ┌─────────▼─────────┐
                              │  robot interface  │  adapters/base.py
                              │   robot contract  │
                              └─────────┬─────────┘
              ┌──────────┬──────────────┼──────────────┬──────────┐
              ▼          ▼              ▼              ▼          ▼
          so101       yam            nori           dimos     your adapter
        (lerobot)   (i2rt CAN)     (Pi daemon)    (LCM/zenoh)  --loop module:fn
```

**The control loop is the heart.** Once per tick it reads an observation, decides which
source is driving the robot (`ESTOP > INTERVENTION/TELEOP > HOLD > POLICY`), produces a
joint vector, gates it, clamps it, calls `send_action`, and records the tick. Everything
above the robot interface is just a different answer to "who is driving."

### Open source and the dashboard

Everything that touches the robot is open source and runs with **no account**: the robot
contract and its adapters, named behaviors, manual moves, the control loop and its safety
model, the DRTC client, and the policy server itself. Driving an arm needs no account, and
neither does running a policy on a GPU you own if you drive the loop yourself.

The [Interlatent dashboard](https://interlatent.com) is the control plane on top of that.
It pairs nodes, assigns sessions, brokers VR teleop, and holds datasets — so the node
daemon, the `interlatent` CLI, teleop, and dataset recording all go through it, and a
self-hosted GPU box registers with it rather than replacing it. See
[ADR 0023](docs/adr/0023-self-hosted-policy-server-returns.md) for why the shape is "hosted
control plane, bring your own compute" rather than a fully standalone mode.

## Features

### Inference

Running a VLA policy means talking to a GPU, and big models are too slow for naive
request/response — the arm would stutter. The client and the box speak **DRTC**
(Distributed Real-Time Chunking): the robot streams observations continuously and never
blocks, the box returns *overlapping action chunks*, and the client merges them
last-writer-wins while estimating network-vs-compute latency so it knows how far ahead to
schedule. The result is smooth 30 Hz control on top of a multi-second model.

- **Client** — `interlatent.inference.client` (`DRTCClient`), entry point
  [`connect_drtc()`](packages/sdk/src/interlatent/inference/integration/connect.py).
  Last-writer-wins `ActionSchedule`, Jacobson–Karels latency estimation, a cooldown that
  doubles as drop recovery, and a `synchronous=True` mode for sequential chunking.
- **Server** — `interlatent-serve` from the [`interlatent-server`](packages/server) package:
  the same code that runs Interlatent's hosted GPU boxes, so you can run one on your own
  hardware. Backends: `lerobot` (SmolVLA, ACT, Diffusion Policy, Pi0/Pi0.5 — anything
  `make_policy` loads), `molmoact2`, `dreamzero`, plus `echo`/`tiny_torch` for tests. A
  policy is named by `policy_backend` + `policy_uri` (an HF repo id or a local checkpoint
  dir).
- **Seam handling** — RTC in-painting for flow-matching policies, and a backend-agnostic
  crossfade for the ones with no in-painting hook.

By default the client talks to Interlatent's hosted endpoint, which requires an API key.
Point `server_address=` at your own box and no key is involved.

### Teleoperation

[`teleop/teleop-web/`](teleop/teleop-web) is a **WebXR PWA**: open it in a Quest browser,
grip to clutch, trigger for the gripper, and the right controller pose drives the end
effector. IK runs **in the browser** (damped least-squares over a compact `kinematic_spec`
shipped with each robot kind); the node only ever receives joint targets.

- **Transport** is QUIC/WebTransport through Interlatent's relay, with the session token
  minted by the dashboard — teleop is a dashboard-brokered capability.
- **Intervention / DAgger is real** — take over mid-rollout and the bus flushes the policy
  schedule behind a merge barrier, keeps shadow inference warm so handback costs about one
  control tick, and labels every tick `policy` / `teleop` / `hold` / `intervention` in the
  dataset.
- **Safety** is enforced node-side by `SafetyGate`: staleness, deadman, confidence,
  workspace and velocity clamps, plus a latched e-stop that only a human can clear.

One practical caveat: **nothing in this repo serves the web app.** Run `npm run build` and
host `dist/` over HTTPS yourself — WebXR needs a secure context, and a Quest is not
`localhost`. Full walkthrough in [docs/teleop.md](docs/teleop.md).

### Control

The layer everything else is written against.

- **`il.Robot`** — the high-level facade: `pose()`, `act()`, `move()`, `behaviors()`,
  `close()`, and a context manager. Named **behaviors** resolve through four layers that
  override by name: generated built-ins → `~/.interlatent/behaviors.toml` → an explicit
  file → `@il.behavior` Python functions. `home` is *generated* from the profile's rest
  pose, so every kind with a profile has it and it cannot drift from the hardware.
  Declarative behaviors are validated against the profile **as they load**, which is why
  `behavior validate` needs no hardware.
- **`RobotAdapter`** — the seven-member contract in
  [`adapters/base.py`](packages/sdk/src/interlatent/adapters/base.py), plus two optional
  hooks discovered off the robot (`pre_tick` for per-tick pre-flight, `estop`).
- **One motion path.** `CommandBus.drive()` arbitrates → gates → clamps → sends, and one
  shared loop runner owns the tick ([ADR 0022](docs/adr/0022-command-bus-owns-the-motion-path.md)).
  A new adapter has no per-tick code, so it cannot silently miss a safety rung.
- **Layered safety, always local to the motors** — never across the network. The per-adapter
  delta clamp (`--robot-arg max_step=…`) caps the per-tick joint jump for *every* action,
  policy and human alike. `SafetyGate` adds workspace, velocity, and deadman limits on
  human-driven motion. Both read their limits from the robot's `RobotProfile`.

> **No profile, no human-driven motion.** `act()`, behaviors (including `home`), manual
> moves, and teleop **will not run** for a kind without a `RobotProfile` for safety reasons.
> Such a kind can still run a policy.

#### Supported robots

`--robot <kind>` on the CLI, or `il.Robot("<kind>")` in Python. Each config doc covers host
requirements, `--robot-arg` knobs, camera declarations, and worked examples.

| Robot | `--robot` | Joints and units | Profile | Extra | Config doc |
|---|---|---|---|:---:|---|
| **SO-101** (reference) | `so101` | 6; degrees, gripper 0–100 | ✅ | `[lerobot]` | [config](packages/sdk/src/interlatent/adapters/lerobot/CONFIG.md) |
| **I2RT YAM** (bimanual) | `yam`, `yam_bimanual` | 14 (left block, then right); radians, gripper 0–1 | ✅ | `[yam]` | [config](packages/sdk/src/interlatent/adapters/yam/CONFIG.md) |
| **I2RT YAM** (single arm) | `yam_left`, `yam_right` | 7; radians, gripper 0–1 | ✅ | `[yam]` | [config](packages/sdk/src/interlatent/adapters/yam/CONFIG.md) |
| **Nori** (dual-SO-101 rig, **unstable beta**) | `nori` | 12 (left block, then right); daemon-normalized ±100 | ✅ | `[nori]` | [config](packages/sdk/src/interlatent/adapters/nori/CONFIG.md) |
| **UFACTORY xArm7 / xArm6 / Galaxea A1Z** (via a running dimos stack) | `dimos`, `--robot-arg kind=xarm7\|xarm6\|a1z` | 8 / 7 / 7 incl. gripper; radians | ✅ | `[dimos]` | [config](packages/sdk/src/interlatent/adapters/dimos/CONFIG.md) |
| **Almond Axol** (dual arm, **unstable beta**) | `axol` | 16 (7 + gripper per side); radians, gripper 0–1 | ❌ **policy only** | `[axol]` | [config](packages/sdk/src/interlatent/adapters/axol/CONFIG.md) |
| Custom hardware | `--loop module:fn` | yours | — | — | bring your own I/O loop |

Only `so101` is constructed through LeRobot; other LeRobot robot types are not wired up and
reach the stack via `--loop module:fn`. For the policy side (SmolVLA, Pi0, ACT, MolmoAct2,
your fine-tunes), see [docs/robots-and-policies.md](docs/robots-and-policies.md).

**Missing your arm?** Adding robots is the contribution we most want, and it should cost you
one `robot.py` and a profile. [ROBOT.md](ROBOT.md#adding-a-new-robot) is the walkthrough.

### `interlatent-act`

A one-shot manual move: drive a robot to absolute joint targets, block until it settles,
exit. No cloud, no policy, no config file — the smallest possible thing on top of the robot
contract, and the fastest way to prove an arm is wired correctly.

```bash
interlatent-act --robot so101 --port /dev/ttyACM0 shoulder_pan=30 wrist_roll=-15
interlatent-act --robot so101 --port /dev/ttyACM0 --show      # print live pose, don't move
```

It goes through the same `SafetyGate` and profile validation as teleop, so out-of-limit
targets, unknown joint names, and omitted joints are all rejected **before anything moves**
(pass `--hold-missing` to hold unnamed joints where they are). Exit codes: `2` contract or
usage error and nothing moved, `1` connect failure or settle timeout, `0` reached.

## Installation

**Requires Python 3.11+.** The base package is robot-agnostic; driving real hardware needs
the extra for your robot.

```bash
pip install interlatent
```

| Extra | For | Notes |
|---|---|---|
| `[lerobot]` | SO-101 | Feetech servos may also need `pip install feetech-servo-sdk` if the serial bus won't open. |
| `[yam]` | I2RT YAM | **Linux + SocketCAN.** Needs a build constraint under plain pip — see below. |
| `[nori]` | Nori | LAN/on-Pi only; the node runs on the robot's Pi next to the daemon. |
| `[dimos]` | xArm7 / xArm6 / A1Z | Python 3.11–3.12 (3.12 on linux/x86_64). **Heavy** — pulls open3d, rerun-sdk, numba, pinocchio; expect a multi-GB env. Needs a *running* dimos stack. |
| `[axol]` | Almond Axol | **Python ≥ 3.13.** ZED SDK + `pyzed` must be installed out of band (not on PyPI). |
| `[turbo]` | Faster CPU JPEG encoding | Also needs the host `libturbojpeg` library, or it silently falls back. |
| `[teleop-quic]` | QUIC teleop on the node | The default WebSocket path needs nothing extra. |

Install **one** robot extra per environment: `axol`, `yam`, `dimos`, and `lerobot` are
mutually exclusive (conflicting `python-can`/`rerun-sdk` pins and disjoint interpreter
ranges). uv enforces this; under pip the combination is simply unsatisfiable.

**`[yam]` under plain pip** — i2rt pins `ruckig==0.15.3`, whose sdist-only build fails
against scikit-build-core ≥ 0.10. Either use `uv pip install 'interlatent[yam]'` (uv honors
i2rt's own constraint) or supply one yourself:

```bash
printf 'scikit-build-core<0.10\n' > /tmp/c.txt
PIP_CONSTRAINT=/tmp/c.txt pip install 'interlatent[yam]'
```

**The policy server** installs separately, on the GPU machine — the two packages run on
different hardware and are versioned independently:

```bash
pip install 'interlatent-server[lerobot]'     # CUDA torch from the default index
```

Docker is the recommended path for a GPU box (`docker build -f docker/Dockerfile .`,
linux/amd64 only); `docker/install-bare-metal.sh` provisions one without Docker. See
[docs/self-hosting.md](docs/self-hosting.md) and [docker/README.md](docker/README.md).

**The VR teleop app** is a static PWA — build it and serve `dist/` over HTTPS from any host:

```bash
cd teleop/teleop-web && npm install && npm run build
```

## Usage

### Drive an arm directly — no cloud, no account

The fastest thing you can do needs no GPU and no policy. This is the robot interface with
nothing on top of it:

```python
import interlatent as il

with il.Robot("so101", port="/dev/ttyACM0") as robot:
    print(robot.pose())                     # {'shoulder_pan': 0.0, ...}
    robot.act("home")                       # go to the profile's rest pose, block until reached
    robot.act("hello")                      # play the packaged SO-101 wave
    robot.act("hello", speed=0.5)           # the same wave, at half speed
    robot.move(wrist_roll=30, duration=0.5) # ad-hoc joint move, no behavior needed
```

`home` is generated from the profile, so every kind with one has it. `hello` is a packaged
example and **only for SO-101** —
[`behaviors/data/so101.toml`](packages/sdk/src/interlatent/behaviors/data/so101.toml) is the
only built-in behavior file. Ask for it on an arm that doesn't define it and you get an
error naming the behavior and listing what that arm *does* have.

A behavior is just data — this is `hello` in full:

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
    { t = 4.5, wrist_roll = 0.0 },                            # straighten the wrist
    { t = 6.0, shoulder_lift = 0.0, elbow_flex = 0.0 },       # lower the forearm
]
```

The same thing from the terminal:

```bash
interlatent behavior ls --robot so101
interlatent behavior validate my_behaviors.toml --robot so101   # validate, no hardware
interlatent behavior run hello --robot so101 --port /dev/ttyACM0 --speed 0.5
```

No arm handy? [`examples/07_named_behaviors.py`](examples/07_named_behaviors.py) runs the
whole thing against a fake adapter and prints the action stream. Format reference:
[docs/behaviors.md](docs/behaviors.md).

### Run a policy on it

Sign in at [interlatent.com](https://interlatent.com), create an API key, and export it:

```bash
export INTERLATENT_API_KEY=ilat_...
```

Pair the machine on your robot once, then run the node daemon. It long-polls the dashboard
and converges to whatever session is assigned to it:

```bash
interlatent-node pair --name my-arm --api-key ilat_...
interlatent-node run  --robot so101 --port /dev/ttyACM0 --camera front=/dev/video0
```

Then start a session against it, from the CLI or the dashboard:

```bash
interlatent gpus ls          # GPU boxes available to your account
interlatent nodes ls         # robot nodes paired to your account
interlatent session start --node my-arm --gpu a100-0 --policy lerobot/smolvla_base
interlatent session stop <session-id>
```

The node picks up the assigned session and the arm starts moving. To test the inference path
with no robot attached:

```bash
interlatent-preflight --environment my-arm --policy lerobot/smolvla_base
```

That opens a real session against a managed GPU pod, streams synthetic observations, and
prints a **PASS / WARN / FAIL** verdict with measured network-vs-compute latency. It
exercises the inference path only — not your cameras, joints, or motor bus.

Running the policy on your own GPU instead is
[`interlatent-serve`](docs/self-hosting.md): the box registers with the dashboard and your
nodes connect to it exactly like a managed pod.

### Or drive the loop yourself

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

This is also the one path that runs with **no account at all**: start a box with
`interlatent-serve --no-register` and pass `server_address="your-box:50051"`, and the key
check never applies. You own the loop in exchange — there is no session management, no node
daemon, and no recording — and `--no-register` forces `--insecure`, which drops the
owner-key check on the gRPC port, so keep it off routable networks.

### Teleoperate and collect data

Recording is **streaming-first**: the control loop JPEG-encodes each camera frame per tick
and streams `RecordTick`s to the GPU box, which builds the LeRobot v3.0 dataset server-side
and publishes it to the dashboard's episode inbox. A node-side disk spool with
delete-after-ack keeps the uplink lossless, so a link drop never silently thins an episode.

Both teleop and recording are brokered by the dashboard: it mints the teleop session token
and owns the dataset destination. Build and serve the VR app, open it in the headset, point
it at your account, and start a recording or join a live session to intervene. Full
walkthrough, including safety and what lands in the dataset:
[docs/teleop.md](docs/teleop.md).

### Examples

| Example | Hardware needed |
|---|---|
| [`03_run_on_so101.py`](examples/03_run_on_so101.py) — drive an SO-101 against a GPU pod | SO-101, or none (synthesizes obs) |
| [`04_manual_action.py`](examples/04_manual_action.py) — one-shot manual joint move | a supported arm |
| [`06_connect_hosted.py`](examples/06_connect_hosted.py) — the minimal cloud connect | none |
| [`07_named_behaviors.py`](examples/07_named_behaviors.py) — named behaviors offline | none (fake arm), or a supported arm |

### Configuration

Only `INTERLATENT_API_KEY` is required; the rest are optional tuning knobs.

| Env var | What it does |
|---|---|
| `INTERLATENT_API_KEY` | Your account API key (`ilat_…`). Authenticates the CLI and DRTC inference. **Required.** |
| `INTERLATENT_API_BASE` | Dashboard base URL (default `https://interlatent.com`). |
| `INTERLATENT_DRTC_URL` | Pin the DRTC inference endpoint (operator/dev override; normally provided per-session). |
| `INTERLATENT_NODE_CONFIG` | Path to the node config TOML (default `~/.interlatent/node.toml`). |
| `INTERLATENT_NUM_INFERENCE_STEPS` | Flow-matching denoising steps for VLA policies. Range 3–10; default 5. |
| `INTERLATENT_IMAGE_RESIZE` | Resize camera frames to this square edge before JPEG-encoding. `256` suits MolmoAct2. |
| `INTERLATENT_JPEG_BACKEND` | Force the frame encoder (`auto`\|`nvjpeg`\|`gpujpeg`\|`turbojpeg`\|`cv2`\|`pil`). |
| `INTERLATENT_PREVIEW_HZ` | Live teleop preview push rate (1–30, default 30). Competes with recording for uplink. |
| `INTERLATENT_SPOOL_DIR` / `_MAX_MB` | Node record spool location and cap (default `~/.interlatent/spool`, 6 GiB). |
| `INTERLATENT_CALIB_PRESET` | Force or disable a joint-calibration preset (e.g. `so101_pre777`, or `none`). |

Encoder and bandwidth tuning: [docs/node-encoding.md](docs/node-encoding.md). Teleop and
preview knobs: [docs/teleop.md](docs/teleop.md).

## API docs

### Where the code lives

Three deliverables, released independently — the whole path from a robot's motors to a
policy's actions.

| Path | Ships as | What it is |
|---|---|---|
| [`packages/sdk/`](packages/sdk) | `pip install interlatent` | The `Robot` contract, adapters, behaviors, the node daemon, the DRTC client, and the CLI. |
| [`packages/server/`](packages/server) | `pip install interlatent-server` | The **DRTC policy server** (`interlatent-serve`) — the same code that runs Interlatent's hosted GPU boxes. |
| [`teleop/teleop-web/`](teleop/teleop-web) | static PWA | The **WebXR VR teleop producer**. |

Two shared pieces sit above them: [`proto/messages.proto`](proto/README.md) is the single
source of truth for the DRTC wire protocol (both Python packages mirror it), and
[`docker/`](docker/README.md) builds the server image. Nothing in the SDK imports the
server or vice versa; they meet only at the protocol.

### Reference by component

| Component | Where the docs are |
|---|---|
| Robot contract, adapters, adding an arm | [ROBOT.md](ROBOT.md) + the per-kind `CONFIG.md` linked in [Supported robots](#supported-robots) |
| The action interface (`send_action` vs `action`) | [docs/action-interface.md](docs/action-interface.md) |
| Named behaviors (Python + CLI + TOML) | [docs/behaviors.md](docs/behaviors.md) |
| DRTC, sessions, chunks, the node | [docs/concepts.md](docs/concepts.md) |
| Robots & policies support matrix | [docs/robots-and-policies.md](docs/robots-and-policies.md) |
| VR teleoperation, safety, recordings | [docs/teleop.md](docs/teleop.md) + [teleop-web/README.md](teleop/teleop-web/README.md) |
| Node encoding & GPU acceleration | [docs/node-encoding.md](docs/node-encoding.md) |
| The DRTC wire protocol | [proto/README.md](proto/README.md) |
| Self-hosting the policy server | [docs/self-hosting.md](docs/self-hosting.md) + [docker/README.md](docker/README.md) |
| Using the hosted dashboard | [docs/going-to-cloud.md](docs/going-to-cloud.md) |
| Robot embodiment data (URDF, kinematic spec) | [interlatent_robots/README.md](packages/sdk/src/interlatent_robots/README.md) |
| First rollout, end to end | [docs/getting-started.md](docs/getting-started.md) |
| Domain glossary | [CONTEXT.md](CONTEXT.md) |
| System architecture, for contributors | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Design decisions and their reversals | [docs/adr/](docs/adr/) |
| Release notes | [CHANGELOG.md](CHANGELOG.md) |

There is no dedicated reference yet for the `interlatent` CLI surface, the node daemon, or
the safety model — each is documented in fragments across the files above. Filling those
gaps is a welcome contribution.

## Contributing

We'd love your help — especially **adding robots**, which is how this project gets breadth.
The other high-value areas are anything that breaks the path from `pip install` to a first
rollout, plus docs, examples, and latency work.

```bash
git clone https://github.com/interlatent/interlatent && cd interlatent
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ./packages/sdk pytest pytest-timeout ruff jsonschema

pytest tests/ packages/sdk/tests/ -v --timeout=120   # both roots; not a subset of each other
ruff check .
```

Everything runs with **no GPU and no robot**. The server and web suites are separate:
`pytest packages/server/tests/`, and `npm test` in `teleop/teleop-web`. Full plan,
including what CI does *not* cover: [TESTING.md](TESTING.md).

- **Adding a robot** — [ROBOT.md](ROBOT.md#adding-a-new-robot) is the file-by-file
  walkthrough; the process is in [CONTRIBUTING.md](CONTRIBUTING.md).
- **Changing the wire protocol** — [`proto/messages.proto`](proto/README.md) is the source
  of truth; regenerate with `./proto/gen_proto.sh`. Changes must be additive.
- **Pull requests** — keep them focused, separate refactors from behavior changes, and
  update docs and examples alongside any public-surface change.
- **Sign off your commits** (`git commit -s`) — this project uses the
  [Developer Certificate of Origin](https://developercertificate.org/), and CI checks it.
- **Good first issues** are labeled
  [`good first issue`](https://github.com/interlatent/interlatent/labels/good%20first%20issue).
- **Security** — don't open a public issue; email team@interlatent.com.

Questions, demos, robot pics: team@interlatent.com.

## License

[Apache-2.0](LICENSE) © Interlatent Contributors.

"Interlatent Cloud" and the hosted service at interlatent.com are operated separately from
this open-source project.
