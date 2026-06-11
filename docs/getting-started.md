# Getting started

Goal: from `pip install` to a policy driving a (real or simulated) robot.

## 1. Install

```bash
pip install interlatent interlatent-server
```

Three packages, three roles:

- `interlatent` — runs on the **robot side** (Pi, laptop): DRTC client, dataset collection
- `interlatent-server` — runs on the **GPU side**: policy inference server
- `interlatent-teleop` — optional laptop ↔ Pi teleoperation

They can all live on one machine while you're developing.

## 2. Sanity-check the loop (no GPU, no robot)

```bash
interlatent-serve                              # terminal 1
python examples/01_loopback_no_hardware.py     # terminal 2
```

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
    payload = pack_observation_npz(cameras, joint_state)   # np.savez bytes
    action = client.step(payload, codec="npz")
    if action is not None:
        robot.send_action(action)
client.close()
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
pip install 'interlatent[lerobot]'
python examples/05_collect_dataset.py
```

`watch()` / `tick()` stage every step locally (SQLite + JPEGs); `LeRobotRebuilder` turns
the staging cache into a standard LeRobot v3.0 dataset on disk. No account involved.

## 6. Where next

- Teleoperation: [examples/04_teleop_record.md](../examples/04_teleop_record.md)
- Concepts (DRTC, sessions, chunks): [concepts.md](concepts.md)
- Which robots/policies work: [robots-and-policies.md](robots-and-policies.md)
- Managed GPUs + hosted datasets: [going-to-cloud.md](going-to-cloud.md)
