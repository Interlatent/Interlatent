# interlatent (Python SDK)

The robot-side half of [Interlatent](https://github.com/interlatent/interlatent): run VLA
policies on real hardware against a GPU box you run, drive robots directly, and record
episodes into storage you own.

What's in this package:

1. **DRTC inference client** (`interlatent.inference`) — `connect_drtc(server_address=..., …)`
   opens a real-time action-chunking session against a GPU box running
   [`interlatent-server`](https://pypi.org/project/interlatent-server/). You dial the box
   directly, or let a session hand your node its address.
2. **Robot node daemon** (`interlatent-node`) — a long-running daemon for always-on robots,
   with camera capture. Pair it once against your coordinator, then `interlatent-node run`;
   it polls the coordinator and converges to whatever inference session is assigned to it.
3. **Coordinator + fleet CLI** (`interlatent`) — run a coordinator of your own
   (`interlatent up`), register GPU boxes and nodes against it, create environments, and
   drive sessions.
4. **Offline control** — named behaviors (`interlatent behavior run`, `interlatent.Robot`) and
   one-shot joint moves (`interlatent-act`), both without a key or a network.

> **Recording is server-side.** The old client-side collection verbs (`watch()`, `tick()`,
> `collect()`, `upload()`, `checkpoint()`, `sb3_callback()`, `register_cameras()`) were
> removed in ADR 0018 and now raise `RuntimeError` with a pointer. A robot node streams
> per-tick observations to the GPU box's recorder, which builds the LeRobot dataset and
> publishes it to the session's destination — a local directory or an S3 bucket you own.
> `Interlatent(db_path=..., fps=...)` is accepted but ignored.

## Install

Requires **Python >= 3.11**. The SDK runs robot-/edge-side and uses torch only for CPU-side
work (tensor marshalling, model type detection) — it never touches CUDA, so **only CPU torch
wheels are installed** and installs stay small.

```bash
# uv — the CPU wheel index is pinned in pyproject.toml, so this resolves CPU-only torch
uv pip install interlatent

# pip — cannot read the index pin from metadata, so install CPU torch first
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install interlatent
```

Optional extras, one per driver stack:

| Extra | For |
|---|---|
| `lerobot` | The LeRobot robot/teleop stack (SO-101, Koch, ALOHA, …) |
| `so101` | Alias of `lerobot` for a single SO-101 follower |
| `yam` | I2RT YAM bimanual arms over CAN (Linux + SocketCAN) |
| `nori` | Nori robot via its on-Pi NDJSON daemon |
| `axol` | Almond Axol dual-arm (Python >= 3.13) |
| `dimos` | Bind to a running dimos stack (Python 3.11/3.12; 3.12 on linux/x86_64) |
| `turbo` | libjpeg-turbo JPEG encoding for the node capture path |
| `teleop-quic` | QUIC/WebTransport in-browser teleop transport |

```bash
uv pip install 'interlatent[lerobot]'

# pip: install CPU torch + torchvision first, then the extra — lerobot pins them
# to a CUDA index in its own packaging, and an already-satisfied torch is not replaced.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install 'interlatent[lerobot]'
```

`[yam]`, `[axol]` and `[dimos]` are mutually exclusive and need host-side pieces that pip
cannot install (CAN tooling, the ZED SDK, a running dimos stack). See the per-extra notes in
`pyproject.toml` and
[ROBOT.md](https://github.com/interlatent/interlatent/blob/main/ROBOT.md).

## Quickstart — DRTC inference

```python
from interlatent.inference.integration import connect_drtc

client = connect_drtc(
    server_address="10.0.0.7:50051",      # your GPU box, or rely on INTERLATENT_DRTC_URL
    api_key="ilop_...",                   # or rely on INTERLATENT_API_KEY
    environment="my-arm",                 # environment slug (must exist)
    policy_uri="lerobot/smolvla_base",
    task="pick up the red cube",
    fps=30,
    record=True,                          # the box records the episode server-side
)

while running:
    payload = pack_observation(frame, joints, "pick up the red cube")
    action = client.step(payload, codec="npz")  # None while the first chunk is in flight
    if action is not None:
        robot.send_action(action)
client.close()
```

An observation is just an `np.savez` blob — keys mirror LeRobot features:

| Key | Type |
|---|---|
| `observation.images.<camera>` | uint8 `(H, W, 3)` |
| `observation.state` | float32 `(state_dim,)` |
| `task` | str (the instruction) |

## Command-line tools

Auth is `--api-key` or `INTERLATENT_API_KEY` — the `ilop_` operator key your coordinator
prints. The coordinator address is `--coordinator` or `INTERLATENT_COORDINATOR`; it has **no
default**, so nothing ever talks to a control plane you did not name.

```bash
# Run a coordinator on this machine
interlatent up                                 # prints the ilop_ operator key
interlatent config --output-dir ~/episodes     # where finished datasets are published
interlatent status
interlatent logs -f
interlatent down

# Fleet
interlatent gpu add --name box0 --url 10.0.0.7:50051   # register a box by address
interlatent gpus ls                        # GPU boxes this coordinator knows about
interlatent nodes ls                       # paired robot nodes
interlatent env create --slug my-arm --robot-type so101 --task "pick and place"
interlatent session ls
interlatent session start --node my-arm --gpu box0 --policy lerobot/smolvla_base
interlatent session stop <session-id>

# Named behaviors — offline, no key
interlatent behavior ls --robot so101
interlatent behavior validate my_behaviors.toml --robot so101
interlatent behavior run home --robot so101 --port /dev/ttyACM0

# Node daemon
interlatent-node pair --name my-arm --coordinator http://10.0.0.2:8900 --api-key ilop_...
interlatent-node run --robot so101 --port /dev/ttyACM0 --camera top=/dev/video0

# One-shot manual move (needs [lerobot])
interlatent-act --robot so101 --port /dev/ttyACM0 shoulder_pan=30
```

### `interlatent-preflight` — DRTC connectivity check

Opens a real DRTC session against a GPU box and pushes synthetic observations — no robot or
cameras needed. Reports the network-vs-compute latency split and a PASS/WARN/FAIL verdict:

```bash
export INTERLATENT_COORDINATOR=http://10.0.0.2:8900
export INTERLATENT_API_KEY=ilop_...
interlatent-preflight \
    --environment my-arm \
    --policy lerobot/smolvla_base \
    --fps 30 \
    --steps 300
```

The coordinator is used only to fetch the environment's observation schema; `--server
<host>:50051` dials a box directly. A green preflight means the inference path is healthy —
it does not exercise cameras, joints, or the motor bus.

## Offline behaviors

```python
from interlatent import Robot

robot = Robot("so101", port="/dev/ttyACM0")
result = robot.act("home", speed=0.5)      # blocks; wait=False returns a handle
robot.close()
```

## HTTP resources

`Interlatent` is the HTTP surface for the environments on your coordinator:

```python
from interlatent import Interlatent

client = Interlatent(
    api_key="ilop_...",                # or INTERLATENT_API_KEY
    base_url="http://10.0.0.2:8900",   # or INTERLATENT_COORDINATOR
    timeout=30.0,
)

envs = client.environments.list()
env = client.environments.create(slug="ant-v5", display_name="Ant-v5")
```

| Resource | Methods |
|----------|---------|
| `client.environments` | `list()`, `get()`, `create()` (see [docs/coordinator-protocol.md](../../docs/coordinator-protocol.md)) |

All constructor arguments are keyword-only, and the client is a context manager
(`with Interlatent(...) as client:` closes the HTTP session on exit).

`create_environment()` is a convenience wrapper over `environments.create()`:

```python
client.create_environment(
    env_id="my-robot-env",
    slug="my-robot-env",
    display_name="My Robot Environment",
    robot_type="so101",
    num_cameras=2,
    camera_names=["front", "wrist"],
    action_dim=7,
    observation_keys=["observation.state"],
    task_description="Pick and place task",
    preset=None,
    notes=None,
    environment_type="robotics",
)
```

## More

[Repo README](https://github.com/interlatent/interlatent) ·
[getting-started](https://github.com/interlatent/interlatent/blob/main/docs/getting-started.md) ·
[self-hosting](https://github.com/interlatent/interlatent/blob/main/docs/self-hosting.md) ·
[robots-and-policies](https://github.com/interlatent/interlatent/blob/main/docs/robots-and-policies.md) ·
[behaviors](https://github.com/interlatent/interlatent/blob/main/docs/behaviors.md) ·
[examples](https://github.com/interlatent/interlatent/tree/main/examples)
