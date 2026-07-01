<div align="center">

<img src="assets/logo.svg" alt="Interlatent" width="420"/>

### Run any VLA policy on your robot — open source.

The robot-side SDK for the [Interlatent dashboard](https://interlatent.com). The dashboard
runs the policy on managed cloud GPUs and orchestrates your pods, nodes, and sessions; this
SDK streams your robot's observations up and drives the arm with the action chunks that come
back, allowing for smooth real-time control on top of big, slow models.

[![PyPI](https://img.shields.io/pypi/v/interlatent?color=7C5CFF&label=interlatent)](https://pypi.org/project/interlatent/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LeRobot](https://img.shields.io/badge/works%20with-%F0%9F%A4%97%20LeRobot-FFD21E)](https://github.com/huggingface/lerobot)
[![GitHub stars](https://img.shields.io/github/stars/interlatent/interlatent?style=social)](https://github.com/interlatent/interlatent)

[Quickstart](#-quickstart) · [Dashboard](#-using-the-dashboard) · [Robots](#-supported-robots) · [Roadmap](#-roadmap) · [Docs](docs/)

</div>

---

Modern robot policies (VLAs) are too big to run on the robot. Interlatent splits the problem:
the **[dashboard](https://interlatent.com)** runs the policy on managed cloud GPUs; this
**open-source SDK** is the robot-side client that streams observations up and actions back
with real-time chunking, so the arm never stutters while the model thinks.

- 🚀 **Run any policy the GPU pod can serve** — if LeRobot's policy factory can load it (SmolVLA, Pi0/Pi0.5, ACT, Diffusion Policy, VQ-BeT, TDMPC, your fine-tune), or it's a supported native backend (MolmoAct2), a pod can serve it
- 🦾 **Drive real hardware** — SO-101 (reference platform), plus native robots like YAM and Axol — over LAN, a VPN, or the internet
- ⚡ **Real-time action chunking (DRTC)** — pipelined inference, latency estimation, and chunk merging keep control smooth at 30 Hz even with multi-second model latency
- 🛰️ **Robot node daemon** — pair a Pi (or any always-on machine) to your account; it converges to whatever inference session the dashboard assigns
- 🖥️ **Dashboard CLI** — list your pods and nodes and start/stop inference sessions from the terminal

## ⚡ Quickstart

### 1. Install & set your API key

```bash
pip install interlatent
```

Sign in at [interlatent.com](https://interlatent.com) and create an API key (`ilat_…`).
Export it so the SDK and CLI can find it:

```bash
export INTERLATENT_API_KEY=ilat_...
```

> **Per-robot extras.** The base package is robot-agnostic. Driving real hardware needs the
> extra for your robot — install **one** of:
> ```bash
> pip install 'interlatent[lerobot]'   # SO-101
> pip install 'interlatent[yam]'       # I2RT YAM (Linux + SocketCAN)
> pip install 'interlatent[axol]'      # Almond Axol (install with uv; Python ≥ 3.13)
> ```
> SO-101's Feetech servos additionally need the Feetech servo SDK; if the serial bus won't
> open, `pip install feetech-servo-sdk`. See each robot's config doc under
> [Supported robots](#-supported-robots) for full host requirements.

### 2. Sync your robot to the dashboard

Pair the machine on your robot once, then run the node daemon. It polls the dashboard and
converges to whatever inference session is assigned to it.

```bash
interlatent-node pair --name my-arm --api-key ilat_...      # register this robot once
interlatent-node run  --robot so101 --port /dev/ttyACM0     # converge to assigned sessions
```

### 3. Run a policy

Spin up a GPU pod and start an inference session against your node — from the
[dashboard](#-using-the-dashboard) or the CLI:

```bash
interlatent gpus ls          # GPU pods available to your account
interlatent nodes ls         # robot nodes paired to your account
interlatent session start --node my-arm --gpu a100-0 --policy lerobot/smolvla_base
interlatent session ls       # active inference sessions
interlatent session stop <session-id>
```

The node picks up the assigned session and the arm starts moving under the policy. To test
the cloud path with no robot attached:

```bash
interlatent-preflight --environment my-arm --policy lerobot/smolvla_base
```

This opens a real session against a managed GPU pod, streams synthetic observations, and
prints a **PASS / WARN / FAIL** verdict with the measured network-vs-compute latency. It
exercises the cloud inference path only — not your cameras, joints, or motor bus.

### 4. Drive a policy from your own code

If you'd rather run the control loop yourself instead of the node daemon:

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
[`examples/03_run_on_so101.py`](examples/03_run_on_so101.py) for a complete SO-101 loop that
synthesizes observations until you wire real hardware, or
[`examples/06_connect_hosted.py`](examples/06_connect_hosted.py) for the minimal connect.

### Configuration

Only `INTERLATENT_API_KEY` is required; the rest are optional tuning knobs.

| Env var | What it does |
|---|---|
| `INTERLATENT_API_KEY` | Your account API key (`ilat_…`). Authenticates the CLI and DRTC inference. **Required.** |
| `INTERLATENT_DRTC_URL` | Pin the DRTC inference endpoint (operator/dev override; normally provided per-session). |
| `INTERLATENT_NUM_INFERENCE_STEPS` | Flow-matching denoising steps for VLA policies (e.g. MolmoAct2). Range 3–10; default 5. |
| `INTERLATENT_IMAGE_RESIZE` | Resize camera frames to this square edge (px) before JPEG-encoding. `256` suits MolmoAct2. |
| `INTERLATENT_NODE_CONFIG` | Path to the node config TOML (default `~/.interlatent/node.toml`). |
| `INTERLATENT_CALIB_PRESET` | Force or disable a joint-calibration preset (e.g. `so101_pre777`, or `none`). |

## 🖥️ Using the dashboard

The [Interlatent dashboard](https://interlatent.com) is where the cloud side lives — it owns
the GPU pods and decides which policy each of your robots is running. The SDK is the
robot-side counterpart to it: everything you do with `interlatent …` on the CLI you can also
do in the dashboard UI.

The core objects:

- **Environments** — a robot setup and its task, the unit everything else hangs off. The `environment` slug you pass to `connect_drtc` / `--env-slug` matches one here.
- **GPU boxes** — managed, warm cloud GPUs that serve the policy. You don't rent or boot the hardware; you start a box for an environment. (`interlatent gpus ls`)
- **Nodes** — your paired robots. A node is created by `interlatent-node pair` and shows up here; the running daemon heartbeats and reports status. (`interlatent nodes ls`)
- **Sessions** — a policy running on a GPU box, bound to a node. Start one and the node converges to it; stop it and the arm idles. (`interlatent session start | ls | stop`)

The end-to-end workflow:

1. **Create an environment** from the Environments page.
2. **Configure the policy** for that environment.
3. **Start a GPU box** for that environment.
4. **Pair and run your node** with the SDK — `interlatent-node pair` then `interlatent-node run` (see [step 2](#2-sync-your-robot-to-the-dashboard)).
5. **Start an inference session** — from the CLI (`interlatent session start …`) or on the dashboard through the environment.

The node daemon picks up the assigned session and the arm starts moving under the policy.

## 🤖 Example: SO-101

SO-101 is the reference platform. The fastest path:

```bash
pip install 'interlatent[lerobot]'
interlatent-node pair --name so101 --api-key ilat_...
interlatent-node run  --robot so101 --port /dev/ttyACM0 --camera front=/dev/video0
# then start a session (CLI or dashboard):
interlatent session start --node so101 --gpu a100-0 --policy lerobot/smolvla_base
```

One-shot manual moves (no policy, no GPU) are handy for bring-up:

```bash
interlatent-act --robot so101 --port /dev/ttyACM0 --show          # read current pose
interlatent-act --robot so101 --port /dev/ttyACM0 shoulder_pan=30 gripper=80 --hold-missing
```

Full SO-101 setup, arguments, and calibration:
**[SO-101 config doc](packages/sdk/src/interlatent/adapters/lerobot/CONFIG.md)**.

## 🦾 Supported robots

Each robot has its own config doc covering host requirements, `--robot-arg` knobs, camera
declarations, joint names/units, and worked examples.

| Robot | `--robot` | Extra | Config doc |
|---|---|---|---|
| **SO-101** (reference) | `so101` | `[lerobot]` (+ `feetech-servo-sdk`) | [config](packages/sdk/src/interlatent/adapters/lerobot/CONFIG.md) |
| I2RT YAM (bimanual) | `yam` | `[yam]` | [config](packages/sdk/src/interlatent/adapters/yam/CONFIG.md) |
| Almond Axol (dual-arm) | `axol` | `[axol]` | [config](packages/sdk/src/interlatent/adapters/axol/CONFIG.md) |
| Any LeRobot robot | `<type>` | `[lerobot]` | cameras attach as `observation.images.<name>` |
| Custom hardware | `--loop module:fn` | — | bring your own I/O loop |

For the policy side (SmolVLA, Pi0, ACT, MolmoAct2, your fine-tunes), see
[docs/robots-and-policies.md](docs/robots-and-policies.md).

**Missing your arm?** Adding robots is the contribution we most want — see
[CONTRIBUTING.md](CONTRIBUTING.md).

## 🛣️ Roadmap

Directions we're excited about. Each is a direction, not a dated commitment — and
contributions are welcome on any of them.

- **VR teleoperation** — drive and demonstrate on your arm from a VR headset, streaming teleop straight into the same DRTC path and dataset recording.
- **More first-class robots** — finish and test Koch (wired but unverified), and broaden tested support beyond SO-101/YAM/Axol. This is where the project grows; bring yours.
- **URDF-derived robot profiles** — read joint names/limits/rest-pose straight from a robot's URDF instead of hand-transcribed `RobotProfile` literals, so limits track the hardware. Design notes in [Future directions](#-future-directions) below.

## 🧠 How it works

The client and the GPU pod speak **DRTC** (Distributed Real-Time Chunking): the robot streams
observations continuously, the pod returns overlapping *action chunks*, and the client merges
them with last-writer-wins semantics while estimating network vs. compute latency. The result
is smooth high-rate control on top of slow, big models. The dashboard sits alongside,
assigning each node a session and a warm GPU pod. Read more in
[docs/concepts.md](docs/concepts.md).

## 📚 Examples

| Example | Hardware needed |
|---|---|
| [`03_run_on_so101.py`](examples/03_run_on_so101.py) — drive an SO-101 against a cloud pod | SO-101 (or none — synthesizes obs) |
| [`04_manual_action.py`](examples/04_manual_action.py) — one-shot manual joint move | a supported arm |
| [`06_connect_hosted.py`](examples/06_connect_hosted.py) — the minimal cloud connect | none |

## ☁️ Open source vs. Interlatent Cloud

This SDK is open source and yours to run, but it's built to plug into the
[dashboard](https://interlatent.com), which runs inference on managed GPUs and orchestrates
your pods, nodes, and sessions — so you never operate GPUs, warm pools, or storage.

| Capability | Open source | [Interlatent](https://interlatent.com) |
|---|:---:|:---:|
| Robot node daemon + DRTC client | ✅ | ✅ |
| Run a VLA policy on your robot | — (needs a GPU pod) | ✅ managed warm GPUs, no cold starts |
| CLI for pods / nodes / sessions | ✅ | ✅ + full dashboard |
| Hosted, versioned datasets | DIY | ✅ managed, shareable |
| Auto policy analysis & reports | ❌ | ✅ |
| GPU autoscaling & warm pools | ❌ | ✅ |
| Support / SLA | community | ✅ |

## 📖 Documentation

- [Getting started](docs/getting-started.md) — robot → first rollout
- [Concepts](docs/concepts.md) — DRTC, sessions, chunks, the node
- [Supported robots & policies](docs/robots-and-policies.md)
- [Going to cloud](docs/going-to-cloud.md)
- [Architecture](ARCHITECTURE.md) — for contributors

## 🤝 Contributing

We'd love your help — especially **adding robots**, which is how this project gets breadth.
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

### Robots should consume URDFs directly

Today a robot's kinematic facts — joint names, order, limits, velocity caps, rest
pose — are hand-transcribed into static `RobotProfile` literals in
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
- Axol has no URDF in the picture yet — needs investigation before this generalizes.

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
  treat the URDF as canonical? URDF limits are mechanical max — we currently inset
  velocity below them on purpose, which a naive import would lose.
