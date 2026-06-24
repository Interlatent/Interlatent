# Going to cloud

Interlatent runs inference on managed cloud GPU pods through the
[Interlatent dashboard](https://interlatent.com). The robot-side stack in this repo is
open source and yours to run; the dashboard brings the compute, storage, and labeling.

- **"I don't own a GPU that serves Pi0 at low latency."** Managed pods run on warm pools —
  no box to rent, no cold starts, no torch.compile babysitting.
- **"I want my data stored, versioned, and viewable."** Episodes record into a hosted
  canonical LeRobot dataset per environment, with a dashboard episode viewer.
- **"I want my data labeled with rewards so I can actually train on it."** **Robometer**
  labels your episodes with dense rewards / value estimates — this is the part that doesn't
  exist in the OSS and isn't DIY-able from it.
- **"I have several robots and a team."** Nodes, session assignment, sharing, access
  control.

## Connect with an API key

```python
client = connect_drtc(
    environment="my-arm",
    policy_uri="lerobot/smolvla_base",
    api_key="ilat_...",                # or set INTERLATENT_API_KEY
    task="pick up the red cube",
)
```

`api_key=` resolves your account and the GPU pod the dashboard attaches to the session —
you never dial a pod endpoint. Your robot code, observation packing, and control loop are the
same; the hosted endpoint speaks the exact gRPC contract in [`proto/`](../proto). Datasets
collected offline are standard LeRobot v3.0 and can be imported later; datasets recorded
hosted are exportable — no lock-in in either direction.

Steps:

1. Sign up at [interlatent.com](https://interlatent.com) and create an API key (`ilat_…`).
2. Create an **environment** in the dashboard (one per robot/policy collection).
3. Pass `api_key=` (or set `INTERLATENT_API_KEY`) — see
   [examples/06_connect_hosted.py](../examples/06_connect_hosted.py).
4. Optional: pair always-on robots with `interlatent-node pair` so the dashboard can assign
   sessions to them.

## The `interlatent` CLI

The CLI is a thin client over the dashboard API. Authenticate with `--api-key` or
`INTERLATENT_API_KEY`; the base URL defaults to https://interlatent.com (override with
`--api-base` / `INTERLATENT_API_BASE`).

```bash
interlatent gpus ls          # GPU pods available to your account
interlatent nodes ls         # robot nodes paired to your account
interlatent session ls       # current sessions
interlatent session start --node my-arm --gpu a100-0 --policy lerobot/smolvla_base
interlatent session stop  <session-id>
```

## What stays true either way

- The client, node, CLI, and wire protocol in this repo are Apache-2.0.
- Local LeRobot dataset collection runs with zero account.
- The cloud consumes these same packages from PyPI — the OSS is the product, not a demo.
