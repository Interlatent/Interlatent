# Going to cloud

Interlatent runs inference on GPU pods coordinated through the
[Interlatent dashboard](https://interlatent.com) — managed pods the dashboard provisions,
or **your own GPU** running the open-source [`interlatent-server`](self-hosting.md)
(it registers with the dashboard and behaves identically).

What the dashboard adds on top of the OSS robot-side stack:

- **Managed warm pools** — no box to rent, no cold starts, no torch.compile babysitting.
- **Hosted datasets** — episodes record into a canonical LeRobot dataset per environment,
  with a dashboard episode viewer.
- **Policy analysis** — hosted sessions get automatic analysis and reports off the
  recorded rollouts.

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

Authenticate with `--api-key` or `INTERLATENT_API_KEY`; the base URL defaults to
https://interlatent.com (override with `--api-base` / `INTERLATENT_API_BASE`).

```bash
interlatent gpus ls          # GPU pods available to your account
interlatent nodes ls         # robot nodes paired to your account
interlatent env create --slug my-arm --robot-type so101
interlatent session ls       # current sessions
interlatent session start --node my-arm --gpu a100-0 --policy lerobot/smolvla_base
interlatent session stop  <session-id>
```

(`interlatent behavior …` also exists and is fully offline — no API key, no cloud.)

## What stays true either way

- The client, node, CLI, **policy server** (`packages/server/`), and wire protocol in
  this repo are Apache-2.0. The cloud consumes these same packages from PyPI.
- Datasets are standard LeRobot v3.0 in both directions: hosted recordings are
  exportable, and datasets you collected elsewhere can be imported — no lock-in.
  (Recording itself happens through a hosted session, so it needs an account.)
