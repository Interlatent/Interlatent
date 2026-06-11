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

## Ports

| Port | What | Expose to |
|---|---|---|
| 50051 | gRPC inference (DRTC) | your robots |
| 50052 | WebSocket teleop relay (DAgger takeover) | your operator laptops/browsers |

## Networking

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
