# Self-hosting the inference server

Everything here runs on infrastructure you control. No account, no callbacks to us.

## Requirements

- Linux host with a CUDA GPU (see sizing below) — or any machine for the CPU test backends
- Python ≥ 3.10, or Docker with the NVIDIA container runtime
- Network path from the robot to the server: same machine, LAN, or a tailnet

| Policy class | GPU |
|---|---|
| ACT / Diffusion / VQ-BeT / TDMPC | any modern CUDA GPU |
| SmolVLA | A10G / RTX 3090 class+ (~50–150 ms per inference) |
| Pi0 / Pi0.5 / MolmoAct2 | ≥24 GB VRAM |

## Option A — pip

```bash
python3 -m venv venv && source venv/bin/activate
pip install 'interlatent-server[lerobot]'
interlatent-serve --policy lerobot/smolvla_base --host 0.0.0.0 --port 50051
```

Notes:

- `--policy` pre-warms (weights + torch.compile) before serving. Skip it and the first
  session per policy pays that cost instead.
- The torch.compile + Hugging Face caches live under `~/.cache`. Keep them on persistent
  disk — they're what makes restarts fast.
- `HF_TOKEN` env var for private checkpoints.

## Option B — Docker (rented GPUs: RunPod, Lambda, Vast, Prime Intellect)

The [`docker/`](../docker) directory ships a self-contained CUDA 12.8 image (torch +
lerobot + the server, ~8–10 GB). Build and run:

```bash
docker build -t interlatent-server docker/
docker run --gpus all \
  -p 50051:50051 -p 50052:50052 \
  -v $HOME/.cache:/root/.cache \
  -e DRTC_WARMUP_POLICY=lerobot/smolvla_base \
  -e HF_TOKEN=$HF_TOKEN \
  interlatent-server
```

The image can optionally join your Tailscale network at boot (`TS_AUTHKEY=...`), which is
the easiest way to reach a cloud GPU from a robot at home/lab — no public ports. See
[docker/README.md](../docker/README.md) for provider-specific notes and the full env var
list.

## Running sessions offline — the coordinator

To drive an always-on robot **node** without the hosted dashboard, run a local
**coordinator** (ships with the `interlatent` SDK). It assigns sessions to nodes over the
same API the node already polls — no account, no dashboard.

```bash
# 1. On the GPU box: serve a policy (no API key needed).
interlatent-serve --policy lerobot/smolvla_base

# 2. On your laptop/control host: start the coordinator and tell it where to
#    save recorded episodes (local dir here; or --s3-uri for S3/R2/MinIO).
interlatent up --port 8900 --output-dir /data/so101-kitchen
interlatent gpu add gpu0 100.x.y.z:50051          # the GPU box (LAN/tailnet URL)

# 3. On the robot (Pi): pair the node at the coordinator (no key required).
interlatent-node pair --name arm0 --api-base http://<coordinator-host>:8900
interlatent-node run --robot so101 --port /dev/ttyACM0 --camera top=/dev/video0 \
    --api-base http://<coordinator-host>:8900

# 4. Back on the control host: assign a session, then stop it when done.
interlatent node ls
interlatent session start --node arm0 --gpu gpu0 --policy lerobot/smolvla_base \
    --task "pick up the cube"
interlatent session stop <session-id>     # node winds down; dataset is published
interlatent down                           # refuses while a session is active (use --force)
```

`session stop` **unassigns** the session — the node closes the DRTC session gracefully,
which is what makes the GPU box build and publish the episode. The coordinator is only a
control plane: the inference link is direct node↔GPU and keeps running even if the
coordinator is down (`interlatent down` therefore guards against orphaning a moving robot).
See [ADR-0001](adr/0001-offline-coordinator-control-plane.md).

### Recording destinations

When a node session records, the **GPU box** builds a LeRobot v3 dataset at session close and
publishes it. The destination is configured on the coordinator and passed to the server
per-session (so you don't set it on `interlatent-serve`):

| `interlatent up …` | Result |
|---|---|
| `--output-dir /data/run` | one flat LeRobot dataset on the GPU box's disk, **merged across sessions** |
| `--s3-uri s3://bucket/prefix [--s3-endpoint-url … --s3-access-key … --s3-secret-key …]` | same, uploaded to S3 / R2 / MinIO (needs `interlatent-server[s3]` on the GPU box) |
| _(neither)_ | inference-only; sessions are **not** recorded (`session start` warns) |

Merge-on-stop appends each session into one training-ready dataset via lerobot's
`aggregate_datasets`; point one destination at one robot/policy collection. A local
`--output-dir` lands on the **GPU box's** filesystem (recording is server-side) — with
everything on one machine that's the same disk; for a remote GPU box use `--s3-uri`. See
[ADR-0002](adr/0002-recording-destination-via-session-metadata.md).

## Ports

| Port | What | Expose to |
|---|---|---|
| 50051 | gRPC inference (DRTC) | your robots |
| 50052 | WebSocket teleop relay (DAgger takeover) | your operator laptops/browsers |
| 8900 | coordinator control plane (`interlatent up`) | your robot nodes + control host |

## Networking

You supply the address the robot uses to reach the server — any reachable `host:port` works
(LAN IP, tailnet `100.x`, public DNS, a tunnel URL); there's no lock-in to a particular
method, and the gRPC vs gRPC-web transport is inferred from the address (an `https://…` URL
uses gRPC-web). On startup `interlatent-serve` **logs the addresses it's reachable at** (and
`interlatent-node` logs its host addresses), so you can copy one straight into
`interlatent gpu add <name> <addr>`.

Each GPU registration also carries a **routing method** (`interlatent gpu add … --method`,
default `direct` = dial the address as-is). This is a forward-looking seam: future methods
(e.g. a NAT-traversal relay both sides dial out to, or MagicDNS resolution) register a
resolver + connector in `interlatent/routing.py` without changing the coordinator or node —
see that module's docstring.

- **Same machine / LAN:** point the client at `host:50051`. Done.
- **Tailscale (recommended for remote GPUs):** join both ends to the tailnet, use the
  100.x address. gRPC handles roaming fine.
- **Public internet:** put your own TLS/auth in front (or use the API-key gate below).
  DRTC tolerates WAN latency — that's what the chunk scheduling is for — but plan for the
  robot to keep `min_execution_horizon` actions buffered.

## Auth

By default the server trusts its network — there is no auth, which is right for LAN and
tailnets. For exposed deployments, `interlatent_server/server/auth.py` provides an optional
API-key gate that validates keys against an Interlatent backend (`INTERLATENT_API_BASE`);
or front the port with your own mTLS/reverse proxy.

## Browser teleop across hostile networks

If operators' browsers can't reach the relay directly (corporate Wi-Fi, CGNAT),
[`teleop-proxy/`](../teleop-proxy) is a ~100-line WebSocket relay you can deploy on any
public host (a Fly.io config is included) that forwards to the box over your tailnet.

## Troubleshooting

- **First session hangs for minutes** — that's torch.compile on a cold cache. Use
  `--policy` at startup and persist `~/.cache`.
- **`no actions received`** — check the server log for the OpenSession line; usually a
  wrong `server_address` or a firewall on 50051.
- **MolmoAct2 fails to load at warmup** — released MolmoAct2 checkpoints need per-session
  camera metadata (`image_keys`); they load on first session instead of at `--policy` time.
- **Choppy control** — raise `min_execution_horizon` / lower `fps` on the client;
  inspect `client.estimated_latency_s` to see whether the bottleneck is network or compute.
