# Self-hosting the policy server

Run the Interlatent DRTC policy server on **your own GPU machine**. The
Interlatent dashboard stays the control plane — your box registers itself
with your API key, appears on the Compute page as a **self-hosted** box, and
your robot nodes connect to it exactly like a managed pod. Same server code,
same wire protocol, same episode recording into your environments.

```
robot node ──native gRPC (50051)──▶ your GPU box (interlatent-server)
                                        │ dials out only
                                        ▼
                            Interlatent dashboard/backend
                (register · warmup target · status · episode inbox)
```

## What you need

- A Linux machine with an NVIDIA GPU (a rented RunPod/Lambda/Vast box works
  too — the point is it's *yours*), Python ≥ 3.11 or Docker.
- An Interlatent account and an API key (`ilat_…`) from the dashboard.
- An address your robot nodes can reach the machine at — public IP, LAN IP,
  or VPN address. The dashboard hands it to nodes verbatim.

## Docker (recommended)

```bash
docker build -f docker/Dockerfile -t interlatent-server:latest .

docker run --rm --gpus all -p 50051:50051 \
  -v interlatent-cache:/root/.cache \
  -e INTERLATENT_API_KEY=ilat_xxx \
  -e INTERLATENT_ADVERTISE_ADDRESS=<IP-your-robots-can-reach> \
  -e HF_TOKEN=hf_xxx \
  interlatent-server:latest
```

The `interlatent-cache` volume persists torch.compile artifacts and HF model
downloads — without it every restart re-pays the multi-minute compile for
VLA-class policies.

## pip

```bash
pip install 'interlatent-server[lerobot]'   # CUDA torch; install on the GPU machine

INTERLATENT_API_KEY=ilat_xxx interlatent-serve \
  --advertise-address <IP-your-robots-can-reach> --port 50051
```

## What happens

1. `interlatent-serve` mints a box UUID once (persisted at
   `~/.interlatent/box-id`) and registers with
   `POST /api/v1/compute/boxes/register`. Restarting re-registers the same
   box — no orphan rows, and a stopped box comes back `ready`.
2. Attach the box to an environment on the Compute page; at boot it fetches
   its warmup target (policy + camera keys) and pre-warms before reporting
   `ready`.
3. Launch sessions from the dashboard as usual. Recordings stream to the box
   and upload through backend-issued presigned URLs — **no cloud credentials
   ever live on your machine**, only your own revocable API key.
4. Graceful exit (Ctrl-C / `docker stop`) reports `stopped` so the dashboard
   doesn't show a ghost box.

## Security

The gRPC port is guarded **by default**: every RPC's `x-api-key` must belong
to the account that registered the box (checked against the backend, cached
60 s). Your nodes already send it. `--insecure` (or
`INTERLATENT_INSECURE=1`) disables the check for air-gapped networks — never
expose an insecure box to the public internet; it is an open GPU.

Also expose **only** the gRPC port. The box needs no inbound access from
anything but your nodes.

## Limits

- The dashboard shows a self-hosted box but cannot stop/restart it — the
  machine is yours ("managed by its operator"). Remove the box row from the
  dashboard when you retire the machine.
- Teleop is unaffected: the browser↔node path goes through the hosted QUIC
  relay and never touches the GPU box.
- GPU sizing follows the same rules as managed boxes — ~24 GB VRAM covers the
  common families (SmolVLA/ACT/Diffusion); Pi0/MolmoAct2-class VLAs want
  more. See [`docker/README.md`](../docker/README.md) for the family/dep
  matrix, the build args, and every environment variable the image reads.
