# interlatent-server

The open-source Interlatent DRTC policy server — the same code that powers
Interlatent's hosted GPU boxes, packaged so you can run it on **your own GPU
machine**. The Interlatent dashboard stays the control plane: the box
registers itself with your API key, shows up on the Compute page as a
self-hosted box, and your robot nodes connect to it exactly like a hosted
one.

```
robot node ──native gRPC──▶ your GPU box (this package)
                                 │  dials out only
                                 ▼
                     Interlatent dashboard/backend
              (register · warmup target · status · episode inbox)
```

## Quick start

```bash
# On the GPU machine (Python >= 3.11, CUDA torch):
pip install 'interlatent-server[lerobot]'

INTERLATENT_API_KEY=ilat_...   # from the dashboard
interlatent-serve --advertise-address <IP-your-robots-can-reach> --port 50051
```

That's it — the box appears on the dashboard Compute page as **Self-hosted**,
`ready`. Attach it to an environment and launch sessions as usual.

- `--advertise-address` is the address your **nodes** dial (public IP, LAN
  IP, or VPN address). The dashboard hands it to nodes verbatim.
- The box only dials **out** to the backend — it needs no inbound route
  except the gRPC port from your nodes.
- Episode recordings upload through backend-issued presigned URLs; no cloud
  credentials ever live on the box.

## Security

By default every RPC on the gRPC port is gated on an owner check: the
caller's `x-api-key` must belong to the same account that registered the
box (validated against the backend, cached 60 s). `--insecure` disables
this for air-gapped networks — never expose an insecure box to the public
internet: it is an open GPU.

## Warmup

Attach the box to an environment in the dashboard and it fetches its warmup
target (policy + camera keys) at boot, paying the multi-minute
`torch.compile` once before it reports `ready`. Outside the dashboard flow,
`--warmup-policy <hf-repo>` pre-warms a self-describing checkpoint.

## Lifecycle

- Restarting `interlatent-serve` re-registers the same box (a UUID persisted
  at `~/.interlatent/box-id`) — no orphan rows.
- Graceful exit reports `stopped` so the dashboard doesn't show a ghost box;
  the next start flips it back to `ready`.
- Remove the box from the dashboard when you're done with the machine.

See `docs/self-hosting.md` at the repo root for the full guide, including
the Docker image.
