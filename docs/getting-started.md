# Getting started

Goal: from `pip install` to a policy driving a (real or simulated) robot.

## 1. Install

```bash
git clone https://github.com/interlatent/interlatent && cd interlatent   # for the examples
pip install interlatent interlatent-server
```

Python 3.11+ recommended (the `interlatent` SDK requires 3.11; the server alone runs on 3.10).

Three packages, three roles:

- `interlatent` — runs on the **robot side** (Pi, laptop): DRTC client, dataset collection
- `interlatent-server` — runs on the **GPU side**: policy inference server
- `interlatent-teleop` — optional laptop ↔ Pi teleoperation

They can all live on one machine while you're developing.

## 2. Sanity-check the loop (no GPU, no robot)

```bash
python examples/01_loopback_no_hardware.py
```

The example spawns its own local `interlatent-serve` (on port 50123) and drives it.
You should see action chunks flowing within a second. This is the exact loop a real robot
runs — only the policy backend (`echo`) and the observation source are fake.

## 3. Serve a real policy

On a machine with a CUDA GPU (same machine is fine):

```bash
pip install 'interlatent-server[lerobot]'
interlatent-serve --policy lerobot/smolvla_base
```

First start compiles the policy (minutes, once — the cache persists under `~/.cache`).
For rented GPUs and Docker, see [self-hosting.md](self-hosting.md).

## 4. Connect from the robot

```python
from interlatent.inference.integration import connect_drtc

client = connect_drtc(
    environment="my-arm",                 # a label for this robot/collection
    policy_uri="lerobot/smolvla_base",
    server_address="gpu-box:50051",       # or "localhost:50051"
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

## 5. Collect data

```bash
pip install 'interlatent[lerobot]' gymnasium
python examples/05_collect_dataset.py
```

`watch()` / `tick()` stage every step locally (SQLite + JPEGs); `LeRobotRebuilder` turns
the staging cache into a standard LeRobot v3.0 dataset on disk. No account involved.

## 6. Troubleshooting

**`client.step()` keeps returning `None`.** Normal for the first ~0.5–2 s of a session:
the first observation has to reach the server, run inference, and the chunk has to come
back. If it never returns an action, check the server log — the policy may still be
loading, or the policy backend may have failed to import.

**`connection refused` / the client hangs on connect.** `interlatent-serve` isn't
running, is bound to a different port (default 50051), or a firewall is blocking the
port. From the robot, `nc -vz gpu-box 50051` should say "succeeded" before anything
else will work. Across networks, [Tailscale](https://tailscale.com) is the easiest path.

**First real-policy session takes minutes.** That's `torch.compile` warming up. Start
the server with `--policy <uri>` so it happens at boot instead of on the first robot
connection. The cache persists under `~/.cache`, so restarts are fast.

**`ModuleNotFoundError: No module named 'lerobot'` on the server.** Real policies need
the extra: `pip install 'interlatent-server[lerobot]'`. The built-in `echo` and
`tiny_torch` test backends work without it.

**"requested recording but no x-api-key present" in the server log.** Server-side
episode recording uploads to a hosted dataset inbox and needs an API key. For fully
local, account-free data collection use the client-side path instead
(`examples/05_collect_dataset.py`).

**The teleop arm freezes and the ack says `estop_latched(...)`.** A driver write failed
(usually a serial hiccup or the motor bus power-cycling). Release the deadman and press
it again to clear the latch; if the fault persists the latch re-engages immediately —
check the cable and power before retrying.

## 7. Where next

- Teleoperation: [examples/04_teleop_record.md](../examples/04_teleop_record.md)
- Concepts (DRTC, sessions, chunks): [concepts.md](concepts.md)
- Which robots/policies work: [robots-and-policies.md](robots-and-policies.md)
- Managed GPUs + hosted datasets: [going-to-cloud.md](going-to-cloud.md)
