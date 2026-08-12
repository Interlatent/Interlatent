# Going to cloud

Interlatent runs inference on GPU pods coordinated through the
[Interlatent dashboard](https://interlatent.com) — managed pods the dashboard provisions,
or **your own GPU** running the open-source [`interlatent-server`](self-hosting.md)
(it registers with the dashboard and behaves identically).

- **"I don't own a GPU that serves Pi0 at low latency."** Managed pods run on warm pools —
  no box to rent, no cold starts, no torch.compile babysitting. (Own a GPU after all?
  See [self-hosting](self-hosting.md).)
- **"I want my data stored, versioned, and viewable."** Episodes record into a hosted
  canonical LeRobot dataset per environment, with a dashboard episode viewer. (Recording
  itself is not hosted-only — `interlatent config --output-dir` publishes to your own disk
  with no account.)
- **"I want automatic policy analysis."** Hosted sessions get policy analysis and reports
  off the recorded rollouts — the part that isn't DIY-able from the OSS alone.

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
you never dial a pod endpoint. Your robot code, observation packing, and control loop are
the same; the hosted endpoint speaks the exact gRPC contract in [`proto/`](../proto).

Steps:

1. Sign up at [interlatent.com](https://interlatent.com) and create an API key (`ilat_…`).
2. Create an **environment** (one per robot/policy collection), in the dashboard or with
   `interlatent env create --slug my-arm`.
3. Pass `api_key=` (or set `INTERLATENT_API_KEY`) — see
   [examples/06_connect_hosted.py](../examples/06_connect_hosted.py).
4. Optional: pair always-on robots with `interlatent-node pair` so the dashboard can assign
   sessions to them.

## The `interlatent` CLI

The dashboard is a **coordinator** — one implementation of the contract in
[coordinator-protocol.md](coordinator-protocol.md), which the CLI also implements itself
(`interlatent up`). Point the same commands at either; there is no second set of verbs.

Authenticate with `--api-key` or `INTERLATENT_API_KEY`, and name the coordinator with
`--coordinator` or `INTERLATENT_COORDINATOR`. **There is no default** — defaulting to a
hosted control plane is how a self-hosted fleet ends up quietly phoning home.

```bash
export INTERLATENT_COORDINATOR=https://interlatent.com

interlatent gpus ls          # GPU boxes available to your account
interlatent nodes ls         # robot nodes paired to it
interlatent session ls       # current sessions
interlatent session start --node my-arm --gpu a100-0 --policy lerobot/smolvla_base
interlatent session stop  <session-id>
```

(`interlatent behavior …` also exists and is fully offline — no API key, no cloud.)

## What stays true either way

- The client, node, CLI, **coordinator**, **policy server** (`packages/server/`), and both
  wire protocols in this repo are Apache-2.0. Inference, collection and VR teleop all run
  with no account; what the cloud adds is the canonical dataset store, the merge pipeline,
  offline policy improvement, and the annotation stack.
- Datasets are standard LeRobot v3.0 in both directions: hosted recordings are
  exportable, and datasets you collected elsewhere can be imported — no lock-in.

- The cloud consumes these same packages from PyPI — the OSS is the product, not a demo.
