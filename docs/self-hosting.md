# Self-hosting the policy server

Run the Interlatent DRTC policy server on **your own GPU machine**. The
dashboard stays the control plane — your box registers itself with your
API key, appears on the Compute page as a **self-hosted** box, and your
robot nodes connect to it exactly like a managed pod. Same server code,
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
  too), and either Docker or Python ≥ 3.12 (lerobot's floor).
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

## Bare metal (no Docker)

From a checkout on the GPU machine:

```bash
sudo ./docker/install-bare-metal.sh          # system deps, venv, torch, lerobot
sudo ./docker/install-bare-metal.sh --systemd \
  --api-key ilat_xxx --advertise-address 203.0.113.7
```

It provisions what the image otherwise gives you: a Python ≥ 3.12, a torch
matched to the driver's CUDA (auto-detected from `nvidia-smi`), ffmpeg for
the dataset writers, lerobot at the commit `docker/Dockerfile` pins (read out
of that file, so the two can't drift), and protobuf stubs regenerated against
the installed runtime. `--systemd` writes a unit that restarts on failure and
stops with `SIGINT`, so the box reports `stopped` rather than leaving a ghost
`ready` row. `--no-system` skips apt if you don't have root. Re-running is safe.

## pip

If you'd rather do it by hand:

```bash
pip install 'interlatent-server[lerobot]'   # CUDA torch; install on the GPU machine

INTERLATENT_API_KEY=ilat_xxx interlatent-serve \
  --advertise-address <IP-your-robots-can-reach> --port 50051
```

Two caveats the script exists to absorb: PyPI lerobot has no MolmoAct2 (the
image pins a git ref for it), and `pyproject.toml` declares `protobuf>=4.25`
while the checked-in stubs assert a ≥ 6.31.1 runtime — fine on a fresh
resolve, an import error if anything holds protobuf lower. `./proto/gen_proto.sh`
regenerates them.

## What happens

1. `interlatent-serve` mints a box UUID once (persisted at
   `~/.interlatent/box-id`) and registers with
   `POST /api/v1/compute/boxes/register`. Restarting re-registers the same
   box — no orphan rows, and a stopped box comes back `ready`.
2. Attach the box to an environment on the Compute page; at boot it fetches
   its warmup target (policy + camera keys) and pre-warms before reporting
   `ready`. Until you do, the fetch 404s and the box falls back to your own
   `--warmup-policy` / `--warmup-image-keys` if you passed them — MolmoAct2
   needs both, since it can't load without camera keys. Once an env is
   attached, its target wins.
3. Launch sessions from the dashboard as usual. Recordings stream to the box
   and upload through backend-issued presigned URLs — **no cloud credentials
   ever live on your machine**, only your own revocable API key.
4. Graceful exit (Ctrl-C / `docker stop`) reports `stopped` so the dashboard
   doesn't show a ghost box.

## Security

The gRPC port is guarded **by default**: every RPC's `x-api-key` must belong
to the account that registered the box (checked against the backend, cached
60 s). Your nodes already send it. `--insecure` (or `INTERLATENT_INSECURE=1`)
disables the check for air-gapped networks — never expose an insecure box to
the public internet; it is an open GPU.

Expose **only** the gRPC port. The box needs no inbound access from anything
but your nodes.

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
