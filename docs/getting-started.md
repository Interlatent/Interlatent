# Getting started

From `pip install` to a policy driving a robot. You run the whole stack: a **coordinator**
that assigns work, a **GPU machine** that serves the policy, and the **robot** that executes
the actions. Three processes, all yours.

## 1. Install

```bash
git clone https://github.com/interlatent/interlatent && cd interlatent   # for the examples
pip install interlatent
```

Python 3.11+ required. The base package is robot-agnostic; driving real hardware needs the
extra for your robot — `[lerobot]` (or `[so101]`) for SO-101, `[yam]`, `[axol]`, `[nori]`,
`[dimos]`. SO-101 also needs `pip install feetech-servo-sdk` if the serial bus won't open.
See [robots-and-policies.md](robots-and-policies.md).

## 2. Start a coordinator

Anywhere your robots and GPU can reach — a laptop on the same LAN is fine:

```bash
interlatent up               # background daemon on :8900; prints an operator key
interlatent status
```

The first `up` mints an operator key (`ilop_…`), prints it, and stores it under
`~/.interlatent/`; later runs print that path instead of the key. CLI commands on that same
machine read the key from there, so only the address is left to name:

```bash
export INTERLATENT_COORDINATOR=http://10.0.0.2:8900   # or http://127.0.0.1:8900 locally
interlatent gpus ls
```

From any other machine, pass the key as well — `--api-key ilop_…` or `INTERLATENT_API_KEY`.

There is **no default** coordinator address: name it with `--coordinator` or
`INTERLATENT_COORDINATOR`, or the command errors rather than guessing. Nothing in the stack
dials a service you didn't name.

## 3. Serve a policy on a GPU

On a Linux machine with an NVIDIA GPU — under your desk, or one you rent from RunPod /
Lambda / Vast:

```bash
pip install 'interlatent-server[lerobot]'    # CUDA torch; install on the GPU machine

interlatent-serve --name rig \
  --coordinator http://10.0.0.2:8900 \
  --api-key ilop_... \
  --advertise-address <IP-your-robots-can-reach> --port 50051
```

The box registers itself with the coordinator under `--name` (default: its hostname) and
reports `ready` once its policy is warm; `interlatent gpus ls` will show it. Docker image,
systemd unit, warmup targets and the full flag list: [self-hosting.md](self-hosting.md).

## 4. Run the robot node

For an always-on robot, pair the machine once and leave the daemon running — it converges to
whatever session the coordinator assigns it:

```bash
interlatent-node pair --name my-arm \
  --coordinator http://10.0.0.2:8900 --api-key ilop_...
interlatent-node run --robot so101 --port /dev/ttyACM0 --camera front=/dev/video0
```

## 5. Start a session

With `INTERLATENT_COORDINATOR` set (step 2), from anywhere that can reach it:

```bash
interlatent gpus ls          # GPU boxes registered with your coordinator
interlatent nodes ls         # robot nodes paired to it
interlatent session start --node my-arm --gpu rig \
  --policy lerobot/smolvla_base --task "pick up the red cube"
interlatent session ls
interlatent session stop <session-id>
```

The node dials the GPU box directly over gRPC; the coordinator only hands out the address.
Stopping the session is what triggers the dataset build, so stop it rather than killing the
node. Set once where those datasets land — with no destination a session still runs, but
nothing is saved:

```bash
interlatent config --output-dir /data/lerobot        # or --s3-uri s3://my-bucket/prefix
```

## 6. Or drive the loop yourself

You don't need the node daemon — a script can open its own DRTC session against the box:

```python
from interlatent.inference.integration import connect_drtc

client = connect_drtc(
    environment="my-arm",                 # a label for this robot/collection
    policy_uri="lerobot/smolvla_base",
    server_address="10.0.0.7:50051",      # your GPU box, host:port
    api_key="ilop_...",                   # the operator key from step 2
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

There is no default GPU endpoint either: name the box with `server_address=` (or
`INTERLATENT_DRTC_URL`), or let a coordinator hand it to a node per session.

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

npz keys mirror LeRobot features:

| Key | Type |
|---|---|
| `observation.images.<camera>` | uint8 `(H, W, 3)` |
| `observation.state` | float32 `(state_dim,)` |
| `task` | str (the instruction) |

[`examples/03_run_on_so101.py`](../examples/03_run_on_so101.py) is a complete version of
this loop that introspects the policy's expected observation keys/shapes automatically and
synthesizes observations until you wire real hardware.

## 7. Troubleshooting

**`client.step()` keeps returning `None`.** Normal for the first ~0.5–2 s of a session: the
first observation has to reach the GPU box, run inference, and the chunk has to come back. If
it never returns an action, check that the session shows as running with
`interlatent session ls`. To isolate the network path from your robot, dial the box with
synthetic observations:

```bash
interlatent-preflight --server <box-ip>:50051 --api-key ilop_... \
  --environment my-arm --policy lerobot/smolvla_base
```

It reports a PASS/WARN/FAIL verdict with the network-vs-compute latency split.

**Connect fails / hangs.** Confirm the coordinator is reachable (`interlatent status` on its
machine, `interlatent gpus ls` from yours) and that the box is listed as `ready`. The node
dials the box's advertised `host:port` directly, so if routing between them is blocked, a VPN
or SSH tunnel can bridge it — and `--advertise-address` must be the address the *robot* can
reach, not the one you SSH'd in on.

**"needs a coordinator".** Whatever you ran couldn't resolve an address: pass
`--coordinator <url>`, set `INTERLATENT_COORDINATOR`, or run `interlatent up` to start one
locally. The stack never falls back to an address you didn't choose.

## 8. Where next

- Concepts (DRTC, sessions, chunks, the node): [concepts.md](concepts.md)
- Which robots/policies work: [robots-and-policies.md](robots-and-policies.md)
- Running the policy server on your own GPU: [self-hosting.md](self-hosting.md)
- Frame encoding, Jetson GPU setup, bandwidth: [node-encoding.md](node-encoding.md)
