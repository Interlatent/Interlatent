# Self-hosting the policy server

Run the Interlatent DRTC policy server on **your own GPU machine**. The box
registers with a **coordinator** — the hosted dashboard, or one you run
yourself with `interlatent up` — and your robot nodes connect to it exactly
like a managed pod. Same server code, same wire protocol.

Nothing here requires an account. A box registered with a self-hosted
coordinator and publishing to `--output-dir` never contacts interlatent.com.

```
robot node ──native gRPC (50051)──▶ your GPU box (interlatent-server)
                                        │ dials out only
                                        ▼
                                  a coordinator
                    (register · warmup target · status · authz)

          `interlatent up` on your own LAN, or the hosted dashboard —
             two deployments of one protocol, and the box cannot
                           tell which it registered with
```

Where a finished dataset lands is a separate question, answered by the
session's destination rather than by the coordinator: a local directory, an
S3 bucket you own, or the hosted episode inbox. See
[coordinator-protocol.md](coordinator-protocol.md) for the full contract.

## What you need

- A Linux machine with an NVIDIA GPU (a rented RunPod/Lambda/Vast box works
  too), Python ≥ 3.11 or Docker.
- A coordinator to register with, and a key for it: the `ilop_` operator key
  `interlatent up` prints, or an `ilat_` key from the dashboard.
- An address your robot nodes can reach the machine at — public IP, LAN IP,
  or VPN address. The coordinator hands it to nodes verbatim.

## Docker (recommended)

```bash
docker build -f docker/Dockerfile -t interlatent-server:latest .

docker run --rm --gpus all -p 50051:50051 \
  -v interlatent-cache:/root/.cache \
  -e INTERLATENT_COORDINATOR=http://10.0.0.2:8900 \
  -e INTERLATENT_API_KEY=ilop_xxx \
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

INTERLATENT_API_KEY=ilop_xxx interlatent-serve \
  --coordinator http://10.0.0.2:8900 \
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
   attached its target wins, because it feeds the node's cameras and the
   warm from one source and so can't disagree with itself.
3. Launch sessions from your coordinator (`interlatent session start`, or the
   dashboard). Recordings stream to the box and land wherever the session's
   destination points: a local directory, an S3 bucket you own, or the hosted
   inbox via backend-issued presigned URLs — **no cloud credentials ever live
   on your machine** in the hosted case, only your own revocable key.
4. Graceful exit (Ctrl-C / `docker stop`) reports `stopped` so the coordinator
   doesn't show a ghost box.

## Security

The gRPC port is guarded **by default**: every RPC's `x-api-key` must be one
the coordinator will vouch for, probed at
`GET /api/v1/compute/boxes/{box_id}/authz` and cached 60 s. Against the hosted
dashboard that means the account that registered the box; against
`interlatent up` it means the `ilop_` operator key or a node token that
coordinator issued. Your nodes already send it. `--insecure` (or
`INTERLATENT_INSECURE=1`) disables the check for air-gapped networks — never
expose an insecure box to the public internet; it is an open GPU.

Expose **only** the gRPC port. The box needs no inbound access from anything
but your nodes.

## Limits

- A coordinator shows a self-hosted box but cannot stop/restart it — the
  machine is yours ("managed by its operator"). Drop the box row when you
  retire the machine: `interlatent gpu rm <name>`, or the equivalent on the
  dashboard.
- Teleop never touches the GPU box: the browser↔node path goes through a QUIC
  relay. Against the hosted dashboard that is the hosted relay; a self-hosted
  coordinator runs its own (`pip install 'interlatent[teleop-relay]'`).
- GPU sizing follows the same rules as managed boxes — ~24 GB VRAM covers the
  common families (SmolVLA/ACT/Diffusion); Pi0/MolmoAct2-class VLAs want
  more. See [`docker/README.md`](../docker/README.md) for the family/dep
  matrix, the build args, and every environment variable the image reads.
