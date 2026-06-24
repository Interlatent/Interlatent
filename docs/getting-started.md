# Getting started

Goal: from `pip install` to a policy driving a robot. Inference runs on managed cloud GPU
pods through the [Interlatent dashboard](https://interlatent.com), so you bring the robot and
an API key — not a GPU.

## 1. Install

```bash
git clone https://github.com/interlatent/interlatent && cd interlatent   # for the examples
pip install interlatent
```

Python 3.11+ required.

## 2. Get an API key

Sign in at [interlatent.com](https://interlatent.com) and create an API key (`ilat_…`).
Export it so the SDK and CLI can find it:

```bash
export INTERLATENT_API_KEY=ilat_...
```

## 3. Connect from the robot

```python
from interlatent.inference.integration import connect_drtc

client = connect_drtc(
    environment="my-arm",                 # a label for this robot/collection
    policy_uri="lerobot/smolvla_base",
    api_key="ilat_...",                   # or rely on INTERLATENT_API_KEY
    task="pick up the red cube",          # the language instruction
    fps=30,
)

while running:
    payload = pack_observation(camera_frame, joint_state, "pick up the red cube")
    action = client.step(payload, codec="npz")   # returns None while the first chunk is in flight
    if action is not None:
        robot.send_action(action)
client.close()
```

`api_key=` resolves your account and the GPU pod the dashboard attaches to the session — you
never dial a pod endpoint yourself.

An observation is just a `np.savez` blob — no custom types:

```python
import io
import numpy as np

def pack_observation(frame: np.ndarray, joints: np.ndarray, task: str) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **{
        "observation.images.front": frame,            # uint8 (H, W, 3) from your camera
        "observation.state": joints.astype(np.float32),
        "task": np.array(task),
    })
    return buf.getvalue()
```

[`examples/03_run_on_so101.py`](../examples/03_run_on_so101.py) is a complete version of
this loop that introspects the policy's expected observation keys/shapes automatically and
synthesizes observations until you wire real hardware.

Observation payload convention (npz keys mirror LeRobot features):

| Key | Type |
|---|---|
| `observation.images.<camera>` | uint8 `(H, W, 3)` |
| `observation.state` | float32 `(action_dim,)` |
| `task` | str (the instruction) |

## 4. Or run a robot node + CLI

For an always-on robot, pair it once and let the dashboard assign sessions to it:

```bash
interlatent-node pair --name my-arm --api-key ilat_...   # register the robot once
interlatent-node run  --robot so101 --port /dev/ttyACM0  # converge to assigned sessions

interlatent gpus ls          # GPU pods available to your account
interlatent nodes ls         # robot nodes paired to your account
interlatent session ls       # active inference sessions
interlatent session start --node my-arm --gpu a100-0 --policy lerobot/smolvla_base
interlatent session stop <session-id>
```

The node polls the dashboard and converges to whatever session is assigned; the GPU endpoint
is provided per-session by the dashboard.

## 5. Collect data (no account)

```bash
pip install 'interlatent[lerobot]' gymnasium
python examples/05_collect_dataset.py
```

`watch()` / `tick()` stage every step locally (SQLite + JPEGs); `LeRobotRebuilder` turns
the staging cache into a standard LeRobot v3.0 dataset on disk. No account involved.

## 6. Troubleshooting

**`client.step()` keeps returning `None`.** Normal for the first ~0.5–2 s of a session:
the first observation has to reach the pod, run inference, and the chunk has to come back.
If it never returns an action, check that your API key is valid and that the session shows
as running with `interlatent session ls`. To isolate the cloud path from your robot, run
`interlatent-preflight --environment <slug> --policy <uri>` — it drives synthetic
observations and reports a PASS/WARN/FAIL verdict with the network-vs-compute latency split.

**Connect fails / hangs.** Confirm `INTERLATENT_API_KEY` is set (or `api_key=` is passed)
and reachable: `interlatent gpus ls` should list pods. Across networks,
[Tailscale](https://tailscale.com) helps the per-session GPU link if direct routing is
blocked.

**Which robots/policies work:** [robots-and-policies.md](robots-and-policies.md).

## 7. Where next

- Concepts (DRTC, sessions, chunks, the node): [concepts.md](concepts.md)
- Which robots/policies work: [robots-and-policies.md](robots-and-policies.md)
- Managed GPUs + hosted datasets: [going-to-cloud.md](going-to-cloud.md)
