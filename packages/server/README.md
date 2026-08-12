# interlatent-server

The Interlatent DRTC policy server, packaged to run on **your own GPU
machine**. The box registers itself with your coordinator — the one
`interlatent up` starts — and your robot nodes dial it directly over native
gRPC.

```
robot node ──native gRPC──▶ your GPU box (this package)
                                 │  dials out only
                                 ▼
                            your coordinator
              (register · warmup target · status · authz)
```

## Quick start

```bash
# On the GPU machine (Python >= 3.11). Install with the default PyPI index so
# the CUDA torch wheels resolve — unlike the SDK, this package wants a GPU.
pip install 'interlatent-server[lerobot]'

export INTERLATENT_COORDINATOR=http://10.0.0.2:8900   # where you ran `interlatent up`
export INTERLATENT_API_KEY=ilop_...                   # the operator key it printed
interlatent-serve --advertise-address <IP-your-robots-can-reach> --port 50051
```

The box shows up in `interlatent gpu ls` as `ready`. Attach it to an
environment and start sessions with `interlatent session start`.

- `--advertise-address` is the `host[:port]` your **nodes** dial (public IP,
  LAN IP, or VPN address). Required to register; the coordinator hands it to
  nodes verbatim.
- `--coordinator` (alias `--api-base`; env `INTERLATENT_COORDINATOR`) is
  required and has **no default** — the box never talks to a control plane you
  did not name.
- Other flags: `--name` (default: hostname), `--host` (default `0.0.0.0`),
  `--box-id`, `--no-register`.
- The box only dials **out** to the coordinator — it needs no inbound route
  except the gRPC port from your nodes.
- Finished episodes land wherever the session's recording destination points:
  a local directory or an S3 bucket you own (`interlatent config`).

## Security

Every RPC on the gRPC port is gated on an authorization check: the caller's
`x-api-key` must be one the coordinator vouches for — the `ilop_` operator key
or a node token that coordinator issued — probed at
`GET /api/v1/compute/boxes/{box_id}/authz` and cached 60 s. `--insecure`
disables this for air-gapped networks — never expose an insecure box to the
public internet: it is an open GPU.

## Warmup

Attach the box to an environment and it fetches its warmup target (policy +
camera keys) at boot, paying the multi-minute `torch.compile` once before it
reports `ready`. Outside that flow, `--warmup-policy <hf-repo>` pre-warms a
checkpoint; pass the camera names too with `--warmup-image-keys top,wrist`
(required for MolmoAct2, and they must match the node's `--camera` names).

## Lifecycle

- Restarting `interlatent-serve` re-registers the same box (a UUID persisted
  at `~/.interlatent/box-id`) — no orphan rows.
- Reported statuses are `ready`, `running`, `uploading`, and `stopped` on
  graceful exit, so the coordinator doesn't show a ghost box.
- Drop the box row with `interlatent gpu rm <name>` when you're done with the
  machine.

See [docs/self-hosting.md](https://github.com/Interlatent/Interlatent/blob/main/docs/self-hosting.md)
for the full guide, including the Docker image.
