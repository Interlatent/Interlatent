# Self-hosting the policy server

Run the Interlatent DRTC policy server on **your own GPU machine**. The box
registers with your **coordinator** — the one you start with `interlatent up`
— and your robot nodes dial it directly over gRPC.

Nothing here reaches a service you don't run: a box registered with your
coordinator and publishing to `--output-dir` talks only to your own machines.

```
robot node ──native gRPC (50051)──▶ your GPU box (interlatent-server)
                                        │ dials out only
                                        ▼
                              your coordinator (`interlatent up`)
                    (register · warmup target · status · authz)
```

Where a finished dataset lands is a separate question, answered by the
session's destination rather than by the coordinator: a local directory or an
S3 bucket you own. See
[coordinator-protocol.md](coordinator-protocol.md) for the full contract.

## What you need

- A Linux machine with an NVIDIA GPU (a rented RunPod/Lambda/Vast box works
  too), Python ≥ 3.11 or Docker.
- A coordinator to register with, and its key: the `ilop_` operator key
  `interlatent up` prints.
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
  --api-key ilop_xxx --advertise-address 203.0.113.7 \
  --coordinator http://10.0.0.2:8900
```

It provisions what the image otherwise gives you: a Python ≥ 3.12, a torch
matched to the driver's CUDA (auto-detected from `nvidia-smi`), ffmpeg for
the dataset writers, lerobot at the commit `docker/Dockerfile` pins (read out
of that file, so the two can't drift), and protobuf stubs regenerated against
the installed runtime. `--systemd` writes a unit that restarts on failure and
stops with `SIGINT`, so the box reports `stopped` rather than leaving a ghost
`ready` row; it bakes in `--api-key`, `--coordinator` and
`--advertise-address`, and refuses to write a unit without all three, since a
service has no shell environment to inherit them from. `--no-system` skips apt
if you don't have root. Re-running is safe.

## pip

If you'd rather do it by hand:

```bash
pip install 'interlatent-server[lerobot]'   # CUDA torch; install on the GPU machine

interlatent-serve --api-key ilop_xxx \
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
2. At boot it asks the coordinator what to pre-warm (policy + camera keys) and
   warms before reporting `ready`. With no target set for the box the fetch
   404s and it falls back to your own `--warmup-policy` /
   `--warmup-image-keys` if you passed them — MolmoAct2 needs both, since it
   can't load without camera keys. A coordinator-side target wins when there
   is one, because it feeds the node's cameras and the warm from one source
   and so can't disagree with itself. (`interlatent gpu add --name rig --url
   <host>:50051 --warm-policy <uri>` registers a box by address with its
   target already set, for machines that don't register themselves.)
3. Launch sessions with `interlatent session start`. Recordings stream to the
   box and land wherever the session's destination points: a local directory
   or an S3 bucket you own. Set it once with
   `interlatent config --output-dir /data/lerobot` (or `--s3-uri`) and the
   coordinator stamps it onto every session it issues.
4. Graceful exit (Ctrl-C / `docker stop`) reports `stopped` so the coordinator
   doesn't show a ghost box.

## The `interlatent` CLI

The CLI both **runs** a coordinator and is a **client** of one: `interlatent
up` starts the service, and every other verb speaks the contract in
[coordinator-protocol.md](coordinator-protocol.md) to it over HTTP — the same
routes `interlatent-serve` and the node use.

Name the coordinator with `--coordinator` or `INTERLATENT_COORDINATOR`.
**There is no default** — a control plane you never named is how a fleet ends
up quietly phoning home, so an unset address is an error, on the coordinator's
own host as much as anywhere else. The key is the one exception: run on the
same machine as `interlatent up` and the CLI reads the `ilop_` operator key out
of `~/.interlatent/` for you. Elsewhere, pass `--api-key` (or set
`INTERLATENT_API_KEY`).

```bash
export INTERLATENT_COORDINATOR=http://10.0.0.2:8900

interlatent gpus ls          # GPU boxes registered with it
interlatent nodes ls         # robot nodes paired to it
interlatent session ls       # current sessions
interlatent session start --node my-arm --gpu rig --policy lerobot/smolvla_base
interlatent session stop  <session-id>
```

(`interlatent behavior …` also exists and is fully offline — no key, no
network.)

Datasets are standard LeRobot v3.0 in both directions: what you record is
readable by anything that reads LeRobot, and datasets collected elsewhere
import unchanged. No lock-in, and no format of ours to learn.

## Security

The gRPC port is guarded **by default**: every RPC's `x-api-key` must be one
the coordinator will vouch for, probed at
`GET /api/v1/compute/boxes/{box_id}/authz` and cached 60 s. That means the
`ilop_` operator key or a node token your coordinator issued. Your nodes
already send it. `--insecure` (or `INTERLATENT_INSECURE=1`) disables the check
for air-gapped networks — never expose an insecure box to the public
internet; it is an open GPU.

Expose **only** the gRPC port. The box needs no inbound access from anything
but your nodes.

## Limits

- The coordinator shows the box but cannot stop/restart it — the machine is
  yours ("managed by its operator"). Drop the box row when you retire the
  machine: `interlatent gpu rm <name>`.
- Teleop never touches the GPU box: the browser↔node path goes through a QUIC
  relay, which the coordinator runs itself
  (`pip install 'interlatent[teleop-relay]'`).
- GPU sizing: ~24 GB VRAM covers the common families (SmolVLA/ACT/Diffusion);
  Pi0/MolmoAct2-class VLAs want more. See
  [`docker/README.md`](../docker/README.md) for the family/dep matrix, the
  build args, and every environment variable the image reads.
