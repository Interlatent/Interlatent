<div align="center">

<img src="assets/Final Logo pt6.png" alt="Interlatent" width="420"/>

### One open interface to control every robot.

Read joint state and command motion the **same way on every supported arm** — by hand, via
a named behavior, under a cloud VLA policy, or in VR. Add a robot once (an adapter + a
profile) and every capability above it comes for free.

[![PyPI](https://img.shields.io/pypi/v/interlatent?color=7C5CFF&label=interlatent)](https://pypi.org/project/interlatent/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LeRobot](https://img.shields.io/badge/works%20with-%F0%9F%A4%97%20LeRobot-FFD21E)](https://github.com/huggingface/lerobot)

[About](#about) · [Features](#features) · [Installation](#installation) · [Usage](#usage) · [API docs](#api-docs) · [Contributing](#contributing)

</div>

---

## About

Every arm ships its own SDK, joint conventions, and scripts. Interlatent is one interface
across robots, with a shared safety model underneath.

It rests on one idea: **a robot is a single object with four methods and the metadata that
gives them meaning** — `connect()`, `get_observation()`, `send_action()`, `disconnect()`,
plus `action_features`, `joint_specs`, and `robot_kind`. That is the `RobotAdapter`
contract, and it's all the layers above a robot may know. Behaviors, policies, teleop, and
recording are written against it, so adding a robot gives you all of them at once.

| Term | What it means |
|---|---|
| **contract** | the interface above. Nothing above a robot knows anything else about it. |
| **adapter** | the code implementing it for one robot: motors, cameras, units. |
| **profile** | your arm's physical facts: joint names and order, limits, speed caps, rest pose. |
| **kind** | the name you ask for (`--robot yam`). It selects the adapter and the profile. |

Two rules shape everything: **actions are joint-space** (absolute targets, no IK or
Cartesian frame anywhere in the robot-side stack — VR teleop solves IK in the browser), and
**an action is a waypoint, not a destination** (sending never blocks, newest wins). Once
per tick the control loop observes, decides who is driving
(`ESTOP > INTERVENTION/TELEOP > HOLD > POLICY`), then gates, clamps, sends, and records.

**Open source and the dashboard.** Everything touching the robot is open source and runs
with no account: the contract, adapters, behaviors, manual moves, the control loop and its
safety model, the DRTC client, and the policy server itself. The
[dashboard](https://interlatent.com) is the control plane on top — it pairs nodes, assigns
sessions, brokers teleop, and holds datasets, so the node daemon, the `interlatent` CLI,
teleop, and recording go through it. See
[ADR 0023](docs/adr/0023-self-hosted-policy-server-returns.md).

## Features

### Inference

Big VLA models are too slow for request/response — the arm would stutter. The client and
GPU box speak **DRTC** (Distributed Real-Time Chunking): the robot streams observations and
never blocks, the box returns *overlapping* action chunks, and the client merges them
last-writer-wins while estimating latency to know how far ahead to schedule. Smooth 30 Hz
control on top of a multi-second model.

`interlatent-serve` ([`packages/server`](packages/server)) is the same code that runs the
hosted GPU boxes. Backends: `lerobot` (SmolVLA, ACT, Diffusion Policy, Pi0/Pi0.5),
`molmoact2`, `dreamzero`. The client defaults to the hosted endpoint, which needs an API
key; point `server_address=` at your own box and no key is involved.

### Teleoperation

[`teleop/teleop-web/`](teleop/teleop-web) is a WebXR PWA — grip to clutch, trigger for the
gripper, right controller pose drives the end effector. IK runs **in the browser**; the node
only ever receives joint targets. Transport is QUIC/WebTransport, with the session token
minted by the dashboard.

**Intervention is real:** take over mid-rollout and the bus flushes the policy schedule
behind a merge barrier, keeps shadow inference warm so handback costs ~one tick, and labels
every tick `policy`/`teleop`/`hold`/`intervention` in the dataset. `SafetyGate` enforces
staleness, deadman, confidence, workspace and velocity limits, plus a latched e-stop only a
human can clear.

Note: **nothing here serves the web app** — build it and host `dist/` over HTTPS yourself.
See [docs/teleop.md](docs/teleop.md).

### Control

- **`il.Robot`** — `pose()`, `act()`, `move()`, `behaviors()`, `close()`, context manager.
  Behaviors resolve through four layers overriding by name: built-ins →
  `~/.interlatent/behaviors.toml` → an explicit file → `@il.behavior` functions. `home` is
  *generated* from the profile's rest pose, so it can't drift from the hardware.
  Declarative behaviors validate against the profile **as they load** — no hardware needed.
- **One motion path.** `CommandBus.drive()` arbitrates → gates → clamps → sends, and one
  shared runner owns the tick ([ADR 0022](docs/adr/0022-command-bus-owns-the-motion-path.md)).
  A new adapter has no per-tick code, so it can't miss a safety rung.
- **Layered safety, always local to the motors.** The delta clamp
  (`--robot-arg max_step=…`) caps the per-tick jump for *every* action, policy and human
  alike; `SafetyGate` adds workspace, velocity, and deadman limits on human-driven motion.
  Both read their limits from the `RobotProfile`.

> **No profile, no human-driven motion.** `act()`, behaviors (including `home`), manual
> moves, and teleop **will not run** for a kind without a `RobotProfile` for safety reasons.
> Such a kind can still run a policy.

| Robot | `--robot` | Joints and units | Profile | Extra | Config |
|---|---|---|:---:|---|---|
| **SO-101** (reference) | `so101` | 6; degrees, gripper 0–100 | ✅ | `[lerobot]` | [doc](packages/sdk/src/interlatent/adapters/lerobot/CONFIG.md) |
| **I2RT YAM** (bimanual) | `yam`, `yam_bimanual` | 14; radians, gripper 0–1 | ✅ | `[yam]` | [doc](packages/sdk/src/interlatent/adapters/yam/CONFIG.md) |
| **I2RT YAM** (single) | `yam_left`, `yam_right` | 7; radians, gripper 0–1 | ✅ | `[yam]` | [doc](packages/sdk/src/interlatent/adapters/yam/CONFIG.md) |
| **Nori** (**beta**) | `nori` | 12; normalized ±100 | ✅ | `[nori]` | [doc](packages/sdk/src/interlatent/adapters/nori/CONFIG.md) |
| **xArm7 / xArm6 / A1Z** (via dimos) | `dimos`, `--robot-arg kind=…` | 8 / 7 / 7 incl. gripper; radians | ✅ | `[dimos]` | [doc](packages/sdk/src/interlatent/adapters/dimos/CONFIG.md) |
| **Almond Axol** (**beta**) | `axol` | 16; radians, gripper 0–1 | ❌ policy only | `[axol]` | [doc](packages/sdk/src/interlatent/adapters/axol/CONFIG.md) |
| Custom | `--loop module:fn` | yours | — | — | your own I/O loop |

Only `so101` is constructed through LeRobot; other LeRobot types reach the stack via
`--loop`. Policies: [docs/robots-and-policies.md](docs/robots-and-policies.md). Adding an
arm should cost one `robot.py` and a profile — [ROBOT.md](ROBOT.md#adding-a-new-robot).

### `interlatent-act`

One-shot manual move: drive to absolute joint targets, block until settled, exit. No cloud,
no config — the fastest way to prove an arm is wired correctly.

```bash
interlatent-act --robot so101 --port /dev/ttyACM0 shoulder_pan=30 wrist_roll=-15
interlatent-act --robot so101 --port /dev/ttyACM0 --show      # print pose, don't move
```

Same `SafetyGate` and profile validation as teleop, so bad targets, unknown joints, and
omitted joints are rejected **before anything moves** (`--hold-missing` holds the rest).

## Installation

**Python 3.11+.** The base package is robot-agnostic; real hardware needs your robot's extra.

```bash
pip install interlatent
```

| Extra | For | Notes |
|---|---|---|
| `[lerobot]` | SO-101 | May also need `pip install feetech-servo-sdk`. |
| `[yam]` | I2RT YAM | **Linux + SocketCAN.** Build constraint needed under pip — below. |
| `[nori]` | Nori | LAN/on-Pi only; the node runs next to the daemon. |
| `[dimos]` | xArm7 / xArm6 / A1Z | Python 3.11–3.12. **Heavy** (open3d, rerun-sdk, numba, pinocchio). Needs a running dimos stack. |
| `[axol]` | Almond Axol | **Python ≥ 3.13.** ZED SDK + `pyzed` installed out of band. |
| `[turbo]` | Fast CPU JPEG | Also needs host `libturbojpeg`, or it falls back silently. |
| `[teleop-quic]` | QUIC teleop on the node | The default WebSocket path needs nothing extra. |

Install **one** robot extra per environment — `axol`, `yam`, `dimos`, and `lerobot` are
mutually exclusive (conflicting pins, disjoint interpreters).

`[yam]` under plain pip needs a build constraint (i2rt pins a `ruckig` whose sdist build
fails on scikit-build-core ≥ 0.10). Use `uv pip install`, or:

```bash
printf 'scikit-build-core<0.10\n' > /tmp/c.txt
PIP_CONSTRAINT=/tmp/c.txt pip install 'interlatent[yam]'
```

The policy server installs separately on the GPU machine:
`pip install 'interlatent-server[lerobot]'`. Docker is recommended — see
[docs/self-hosting.md](docs/self-hosting.md). The VR app is a static PWA:
`cd teleop/teleop-web && npm install && npm run build`.

## Usage

### Drive an arm — no cloud, no account

```python
import interlatent as il

with il.Robot("so101", port="/dev/ttyACM0") as robot:
    print(robot.pose())                     # {'shoulder_pan': 0.0, ...}
    robot.act("home")                       # profile's rest pose, blocks until reached
    robot.act("hello", speed=0.5)           # the packaged SO-101 wave, half speed
    robot.move(wrist_roll=30, duration=0.5) # ad-hoc joint move
```

`home` exists for every kind with a profile. `hello` is packaged for SO-101 only. Behaviors
are plain TOML keyframes — format in [docs/behaviors.md](docs/behaviors.md). Same from the
terminal:

```bash
interlatent behavior ls --robot so101
interlatent behavior validate my_behaviors.toml --robot so101   # no hardware
interlatent behavior run hello --robot so101 --port /dev/ttyACM0
```

### Run a policy on it

```bash
export INTERLATENT_API_KEY=ilat_...                    # from interlatent.com

interlatent-node pair --name my-arm                    # once per machine
interlatent-node run  --robot so101 --port /dev/ttyACM0 --camera front=/dev/video0

interlatent gpus ls
interlatent session start --node my-arm --gpu a100-0 --policy lerobot/smolvla_base
interlatent session stop <session-id>
```

The node long-polls, converges to the assigned session, and the arm moves.
`interlatent-preflight --environment my-arm --policy lerobot/smolvla_base` tests the
inference path with no robot attached and prints a PASS/WARN/FAIL latency verdict.

### Or drive the loop yourself

```python
from interlatent.inference.integration import connect_drtc

client = connect_drtc(environment="my-arm", policy_uri="lerobot/smolvla_base",
                      api_key="ilat_...", task="pick up the red cube", fps=30)
while running:
    action = client.step(observation_npz_bytes, codec="npz")  # None until the first chunk
    if action is not None:
        robot.send_action(action)
client.close()
```

An observation is an `np.savez` blob with LeRobot-style keys
(`observation.images.<camera>`, `observation.state`, `task`). This is also the one path
needing **no account**: run a box with `interlatent-serve --no-register` and pass
`server_address="your-box:50051"`. You own the loop in exchange — no session management, no
node daemon, no recording — and `--no-register` forces `--insecure`, which drops the gRPC
owner-key check, so keep it off routable networks.

### Teleoperate and collect data

Recording is streaming-first: the loop JPEG-encodes each frame per tick and streams
`RecordTick`s to the GPU box, which builds the LeRobot v3.0 dataset server-side and
publishes it to the dashboard's inbox. A node-side spool with delete-after-ack keeps the
uplink lossless. Both teleop and recording are dashboard-brokered — see
[docs/teleop.md](docs/teleop.md).

Runnable examples, ordered by hardware required: [`examples/`](examples/README.md).

### Configuration

Only `INTERLATENT_API_KEY` is required.

| Env var | What it does |
|---|---|
| `INTERLATENT_API_KEY` | Account API key (`ilat_…`). **Required.** |
| `INTERLATENT_API_BASE` | Dashboard base URL (default `https://interlatent.com`). |
| `INTERLATENT_NODE_CONFIG` | Node config TOML (default `~/.interlatent/node.toml`). |
| `INTERLATENT_IMAGE_RESIZE` | Resize frames to this square edge before encoding. `256` suits MolmoAct2. |
| `INTERLATENT_JPEG_BACKEND` | Force the encoder (`auto`\|`nvjpeg`\|`gpujpeg`\|`turbojpeg`\|`cv2`\|`pil`). |
| `INTERLATENT_SPOOL_DIR` / `_MAX_MB` | Record spool location and cap (default `~/.interlatent/spool`, 6 GiB). |

Encoder and bandwidth tuning: [docs/node-encoding.md](docs/node-encoding.md).

## API docs

Three deliverables, released independently:
[`packages/sdk/`](packages/sdk) (`pip install interlatent` — contract, adapters, behaviors,
node daemon, DRTC client, CLI), [`packages/server/`](packages/server)
(`pip install interlatent-server` — the DRTC policy server), and
[`teleop/teleop-web/`](teleop/teleop-web) (the WebXR producer). Neither Python package
imports the other; they meet at [`proto/messages.proto`](proto/README.md).

| Component | Docs |
|---|---|
| Robot contract, adapters, adding an arm | [ROBOT.md](ROBOT.md) + the per-kind config docs above |
| The action interface (`send_action` vs `action`) | [docs/action-interface.md](docs/action-interface.md) |
| Named behaviors (Python + CLI + TOML) | [docs/behaviors.md](docs/behaviors.md) |
| DRTC, sessions, chunks, the node | [docs/concepts.md](docs/concepts.md) |
| Robots & policies support matrix | [docs/robots-and-policies.md](docs/robots-and-policies.md) |
| VR teleoperation, safety, recordings | [docs/teleop.md](docs/teleop.md) |
| Node encoding & GPU acceleration | [docs/node-encoding.md](docs/node-encoding.md) |
| The DRTC wire protocol | [proto/README.md](proto/README.md) |
| Self-hosting the policy server | [docs/self-hosting.md](docs/self-hosting.md) |
| Using the hosted dashboard | [docs/going-to-cloud.md](docs/going-to-cloud.md) |
| First rollout, end to end | [docs/getting-started.md](docs/getting-started.md) |
| Domain glossary · Architecture · Decisions | [CONTEXT.md](CONTEXT.md) · [ARCHITECTURE.md](ARCHITECTURE.md) · [docs/adr/](docs/adr/) |

The `interlatent` CLI surface, the node daemon, and the safety model have no dedicated
reference yet — each is documented in fragments above. Filling those gaps is welcome.

## Contributing

Most wanted: **adding robots**, and anything that breaks `pip install` → first rollout.

```bash
pip install -e ./packages/sdk pytest pytest-timeout ruff jsonschema
pytest tests/ packages/sdk/tests/ -v --timeout=120   # both roots; neither is a subset
ruff check .
```

Everything runs with no GPU and no robot. Server and web suites are separate
(`pytest packages/server/tests/`; `npm test` in `teleop/teleop-web`) — see
[TESTING.md](TESTING.md). `proto/messages.proto` is the source of truth; regenerate with
`./proto/gen_proto.sh` and keep changes additive. **Sign off your commits**
(`git commit -s`) — CI checks [DCO](https://developercertificate.org/). Process:
[CONTRIBUTING.md](CONTRIBUTING.md). Security issues: team@interlatent.com, never a public
issue.

## License

[Apache-2.0](LICENSE) © Interlatent Contributors. "Interlatent Cloud" and the hosted service
at interlatent.com are operated separately from this open-source project.
