<div align="center">

<img src="assets/Final Logo pt6.png" alt="Interlatent" width="420"/>

### One open interface to control every robot.

Rollout policies, teleoperate, and collect data the **same way on every supported robot** via
Python, the command line, a VR headset, or a learned policy.

[![PyPI](https://img.shields.io/pypi/v/interlatent?color=7C5CFF&label=interlatent)](https://pypi.org/project/interlatent/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LeRobot](https://img.shields.io/badge/works%20with-%F0%9F%A4%97%20LeRobot-FFD21E)](https://github.com/huggingface/lerobot)

[Install](#install) · [Quickstart](#quickstart) · [Robots](#supported-robots) · [Run a policy](#run-an-ai-policy) · [Docs](#docs) · [Contributing](#contributing)

</div>

---

## Install

Python 3.11+. The base package works without hardware, and each robot gets its own extra.

```bash
pip install interlatent
```

| Extra | For | Notes |
|---|---|---|
| `interlatent[lerobot]` | SO-101 | May also need `pip install feetech-servo-sdk`. |
| `interlatent[yam]` | I2RT YAM | Linux + SocketCAN. See the build note below. |
| `interlatent[nori]` | Nori | Runs on the robot's own network or on the Pi. |
| `interlatent[dimos]` | xArm7 / xArm6 / A1Z | Python 3.11–3.12. Large install; needs a running dimos stack. |
| `interlatent[axol]` | Almond Axol | Python 3.13+. ZED SDK and `pyzed` installed separately. |
| `interlatent[turbo]` | Faster camera encoding | Also needs `libturbojpeg` on the host. |
| `interlatent[teleop-quic]` | Lower-latency VR teleop | Optional; the default path needs nothing extra. |

> `[yam]` needs a build constraint under plain pip. Use `uv pip install`, or:
>```bash
>printf 'scikit-build-core<0.10\n' > /tmp/c.txt
>PIP_CONSTRAINT=/tmp/c.txt pip install 'interlatent[yam]'
>```

## Quickstart

Plug in an arm and move it. This doesn't rely on an account, cloud, or config file.

```python
import interlatent as il

with il.Robot("so101", port="/dev/ttyACM0") as robot:
    print(robot.pose())                     # {'shoulder_pan': 0.0, ...}
    robot.act("home")                       # go to the rest pose, wait until it arrives
    robot.act("hello", speed=0.5)           # a packaged SO-101 wave, at half speed
    robot.move(wrist_roll=30, duration=0.5) # move one joint
```

`Robot` gives you `pose()`, `move()`, `act()`, `behaviors()`, and `close()`. Positions/actions are
currently always **joint angles**.

You can also do the same thing from the terminal:

```bash
interlatent-act --robot so101 --port /dev/ttyACM0 --show          # print the pose and joint names
interlatent-act --robot so101 --port /dev/ttyACM0 shoulder_pan=30 wrist_roll=-15
```

`interlatent-act` is the quickest way to check an arm is wired up correctly. Bad or unknown
joint values are rejected before the robot moves.

### Named moves

`act("hello")` runs a **behavior** — a named sequence of joint poses written in TOML. Every
robot gets `home`.

```bash
interlatent behavior ls --robot so101
interlatent behavior validate my_behaviors.toml --robot so101   # no hardware needed
interlatent behavior run hello --robot so101 --port /dev/ttyACM0
```

Write your own in `~/.interlatent/behaviors.toml`, in a file you pass explicitly, or as a
Python function decorated with `@il.behavior`. See the format at:
[docs/behaviors.md](docs/behaviors.md).

## Supported robots

| Robot | `--robot` | Joints and units | Manual moves | Extra | Setup |
|---|---|---|:---:|---|---|
| **SO-101** (reference) | `so101` | 6; degrees, gripper 0–100 | ✅ | `[lerobot]` | [doc](packages/sdk/src/interlatent/adapters/lerobot/CONFIG.md) |
| **I2RT YAM** (two arms) | `yam`, `yam_bimanual` | 14; radians, gripper 0–1 | ✅ | `[yam]` | [doc](packages/sdk/src/interlatent/adapters/yam/CONFIG.md) |
| **I2RT YAM** (one arm) | `yam_left`, `yam_right` | 7; radians, gripper 0–1 | ✅ | `[yam]` | [doc](packages/sdk/src/interlatent/adapters/yam/CONFIG.md) |
| **Nori** (beta) | `nori` | 12; normalized ±100 | ✅ | `[nori]` | [doc](packages/sdk/src/interlatent/adapters/nori/CONFIG.md) |
| **xArm7 / xArm6 / A1Z** | `dimos`, `--robot-arg kind=…` | 8 / 7 / 7 incl. gripper; radians | ✅ | `[dimos]` | [doc](packages/sdk/src/interlatent/adapters/dimos/CONFIG.md) |
| **Almond Axol** (beta) | `axol` | 16; radians, gripper 0–1 | ❌ policy only | `[axol]` | [doc](packages/sdk/src/interlatent/adapters/axol/CONFIG.md) |

Adding a robot takes one adapter file and one profile of its physical limits —
[ROBOT.md](ROBOT.md#adding-a-new-robot). Everything else (behaviors, teleop, policies,
recording) then works on it automatically.

## Run an AI policy

Policies run on a GPU machine and stream actions back to the robot at 30 Hz.

To set up a receiver node, which controls the robot:

```bash
export INTERLATENT_API_KEY=ilat_...

interlatent-node pair --name my-arm                    # this will setup connection and naming details.
interlatent-node run  --robot so101 --port /dev/ttyACM0 --camera front=/dev/video0
```
You can run servers off of your own GPUs or on cloud providers such as runpod or modal. Install the server there
(`pip install 'interlatent-server[lerobot]'`, or use Docker — see
[docs/self-hosting.md](docs/self-hosting.md)), and start it with `interlatent-serve`. You can then start a session:

```bash
interlatent gpus ls
interlatent session start --node my-arm --gpu <gpu-addr> --policy lerobot/smolvla_base
interlatent session stop <session-id>
```

Supported policies: SmolVLA, ACT, Diffusion Policy, Pi0/Pi0.5 (via LeRobot), MolmoAct2. Full info: [docs/robots-and-policies.md](docs/robots-and-policies.md).

WAM inference is still WIP.

## VR teleoperation and data collection

[`teleop/teleop-web/`](teleop/teleop-web) is a web app you open in a VR headset: grip to
clutch, trigger for the gripper. By default, the right controller drives the end effector. You can also take over a
running policy at any time. Every frame is
labelled with who was driving, so the recording is usable intervention data.

Build and host it yourself over HTTPS:

```bash
cd teleop/teleop-web && npm install && npm run build   # serve dist/
```

Teleop and recording both go through the dashboard, which mints the session token and stores
the resulting LeRobot v3.0 datasets. Setup: [docs/teleop.md](docs/teleop.md).

## Safety

Two limits apply on every tick, both read from the robot's own profile and both enforced on
the machine holding the motors:

- A **per step delta clamp** caps how far any joint can jump in one tick. This applies to both policies and humans
  alike. Tune it with `--robot-arg max_step=…`.
- **Workspace, speed, staleness, and deadman limits** apply to human-driven motion, plus an
  emergency stop that only a person can clear.

## Configuration

`INTERLATENT_API_KEY` is the only required setting, and only for hosted features — the
Python and CLI paths above need nothing.

| Env var | What it does |
|---|---|
| `INTERLATENT_API_KEY` | Your account key (`ilat_…`), from interlatent.com. |
| `INTERLATENT_API_BASE` | Dashboard URL (default `https://interlatent.com`). |
| `INTERLATENT_NODE_CONFIG` | Node config file (default `~/.interlatent/node.toml`). |
| `INTERLATENT_IMAGE_RESIZE` | Shrink camera frames to this square size. `256` suits MolmoAct2. |
| `INTERLATENT_JPEG_BACKEND` | Pick the image encoder (`auto`\|`nvjpeg`\|`gpujpeg`\|`turbojpeg`\|`cv2`\|`pil`). |
| `INTERLATENT_SPOOL_DIR` / `_MAX_MB` | Where recordings buffer, and the cap (default `~/.interlatent/spool`, 6 GiB). |

> The JPEG encoding backend is automatically configured based on available hardware, but can be manually set.

## Docs

Runnable examples, ordered by how much hardware they need:
[`examples/`](examples/README.md).

| Topic | Where |
|---|---|
| First rollout, end to end | [docs/getting-started.md](docs/getting-started.md) |
| Adding a robot | [ROBOT.md](ROBOT.md) |
| Named behaviors | [docs/behaviors.md](docs/behaviors.md) |
| Sending actions | [docs/action-interface.md](docs/action-interface.md) |
| How the pieces fit together | [docs/concepts.md](docs/concepts.md) |
| Robot and policy support | [docs/robots-and-policies.md](docs/robots-and-policies.md) |
| VR teleop and recording | [docs/teleop.md](docs/teleop.md) |
| Camera encoding and bandwidth | [docs/node-encoding.md](docs/node-encoding.md) |
| Running your own policy server | [docs/self-hosting.md](docs/self-hosting.md) |
| Using the hosted dashboard | [docs/going-to-cloud.md](docs/going-to-cloud.md) |
| Wire protocol | [proto/README.md](proto/README.md) |
| Docs for Agents | [CONTEXT.md](CONTEXT.md) · [ARCHITECTURE.md](ARCHITECTURE.md) · [docs/adr/](docs/adr/) |

Everything that touches the robot is open source and runs without an account. The
[dashboard](https://interlatent.com) is the hosted layer on top: it pairs machines, assigns
GPUs, brokers VR teleop, and stores datasets.

## Contributing

Adding robot profiles and additional features to this repository helps make it more useful for others. We appreciate any contributions from the community.

To run tests:

```bash
pip install -e ./packages/sdk pytest pytest-timeout ruff jsonschema
pytest tests/ packages/sdk/tests/ -v --timeout=120   # both roots; neither is a subset
pytest tests/ packages/server/tests/ -v --timeout=120  
ruff check .
```

Exisiting pytests run with no GPU and no robot, but you should **test your code on real hardware** before pushing.

Server and web tests are separate
(`pytest packages/server/tests/`; `npm test` in `teleop/teleop-web`) — see
[TESTING.md](TESTING.md). 

If you change `proto/messages.proto`, regenerate with
`./proto/gen_proto.sh` and keep changes additive. **Sign off your commits**
(`git commit -s`) — CI checks [DCO](https://developercertificate.org/). 

Details:
[CONTRIBUTING.md](CONTRIBUTING.md). 

Security issues should go to team@interlatent.com, never a
public issue.

## License

[Apache-2.0](LICENSE) © Interlatent Contributors. "Interlatent Cloud" and the hosted service
at interlatent.com are operated separately from this open-source project.
