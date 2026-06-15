# interlatent-server

Self-hosted, low-latency inference server for VLA and action-chunking policies. Serve
SmolVLA, Pi0, ACT, MolmoAct2 and friends on your own GPU and drive a real robot over the
network with the [`interlatent`](../sdk) client.

```bash
pip install 'interlatent-server[lerobot]'
interlatent-serve --policy lerobot/smolvla_base
# DRTC gRPC server listening on 0.0.0.0:50051
```

`--policy` pre-warms the policy (weights + torch.compile) before the server accepts
traffic, so the first robot session starts instantly. Without it, the first session per
policy pays the load/compile cost once per process.

No GPU handy? The base install (`pip install interlatent-server`) runs with built-in test
backends (`echo`, `tiny_torch`) on CPU — enough to develop clients and exercise the full
protocol.

## Supported policies

Any policy loadable by LeRobot's policy factory, plus a dedicated MolmoAct2 backend:

| Policy | Backend | Notes |
|---|---|---|
| SmolVLA | `lerobot` | torch.compile warm-up ~minutes once per process; ~50–150 ms/infer on A10G+ |
| Pi0 / Pi0.5 | `lerobot` | needs ≥24 GB VRAM |
| ACT, Diffusion Policy, VQ-BeT, TDMPC | `lerobot` | light; fine on small GPUs |
| MolmoAct2 | `molmoact2` (auto-routed) | transformers-native; needs per-session camera `image_keys` metadata |
| your own | `register_backend("name")` | see [CONTRIBUTING.md](../../CONTRIBUTING.md) |

Policy URIs are Hugging Face repo ids (`lerobot/smolvla_base`, your fine-tune repo) or
local checkpoint paths. Private repos: set `HF_TOKEN`.

## How it serves

The server implements the GPU half of **DRTC** (Distributed Real-Time Chunking): clients
stream observations; the server runs the policy and streams back overlapping action chunks,
conditioning each inference on the actions already scheduled on the robot (RTC
in-painting), so control stays smooth at 30 Hz despite multi-hundred-ms inference.

gRPC API (`proto/messages.proto`):

| RPC | Purpose |
|---|---|
| `OpenSession` | bind a session to a policy URI + metadata (task, fps, recording) |
| `Stream` | bidirectional observation → action-chunk streaming (preferred) |
| `Infer` | unary fallback (gRPC-Web friendly) |
| `RecordTick` | optional per-control-tick recording, decoupled from inference |
| `CloseSession` | teardown (+ finalize recording) |

Ports: `50051` gRPC inference · `50052` WebSocket teleop relay (DAgger takeover).

## Deployment

- **Docker (recommended for cloud GPUs):** see [`docker/`](../../docker) — CUDA 12.8 image
  for RunPod / Lambda / Vast / Prime Intellect / bare metal, with optional Tailscale join
  and warm-load via `DRTC_WARMUP_POLICY`.
- **Bare metal:** `interlatent-serve` inside any Python ≥3.10 env with CUDA torch.
- Mount/persist `~/.cache` to keep torch.compile + HF weights across restarts.

| Env var | Purpose |
|---|---|
| `HF_TOKEN` | private Hugging Face checkpoints |
| `INTERLATENT_API_BASE` | backend used for optional API-key auth + recording upload (cloud-connected setups only) |

Auth is **off by default** for self-hosting — the server trusts its network (use LAN,
Tailscale, or your own mTLS). The optional API-key gate (`server/auth.py`) validates
Interlatent Cloud keys for internet-exposed deployments.

## Extras

| Install | Adds |
|---|---|
| `interlatent-server` | protocol + test backends (CPU-only OK) |
| `interlatent-server[lerobot]` | torch + lerobot — real policies |
| `interlatent-server[recording]` | pyarrow — server-side episode recording to LeRobot datasets |
| `interlatent-server[s3]` | boto3 — upload recorded datasets to an S3-compatible bucket |

## Recording episodes

When a session opts into recording (the [node](../sdk) and DRTC client do this
automatically), the server captures every control tick and, at `CloseSession`, builds a
LeRobot v3 dataset and **publishes it to a destination**:

| Destination | How to set it |
|---|---|
| Hosted inbox (default) | requires an Interlatent API key; the backend merges episodes into the environment's dataset |
| Local directory | `interlatent-serve --output-dir /data/run` (or `INTERLATENT_OUTPUT_DIR`) — no account needed |
| S3-compatible bucket | `interlatent-serve --s3-uri s3://bucket/prefix [--s3-endpoint-url … --s3-access-key … --s3-secret-key … --s3-region …]` (AWS / R2 / MinIO) |

Local and S3 destinations **merge-on-stop**: each session is appended into one flat,
training-ready LeRobot dataset (via lerobot's `aggregate_datasets`). Point one destination
at one robot/policy collection. When driven by the [coordinator](../sdk) the destination is
configured there and passed per-session, so you don't set it on `interlatent-serve` at all —
see [docs/self-hosting.md](../../docs/self-hosting.md).

Apache-2.0. Part of [interlatent/interlatent](https://github.com/interlatent/interlatent).
