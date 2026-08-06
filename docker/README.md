# `interlatent-server` — GPU image

A self-contained CUDA image of the Interlatent DRTC policy server. Deploy it to any
GPU provider (RunPod, Lambda Labs, Vast.ai, Prime Intellect, bare metal) without
touching the host Python environment.

It runs [`interlatent-serve`](../docs/self-hosting.md): the box registers itself with
the Interlatent dashboard using **your** API key, then serves policies to your robot
nodes over native gRPC. The dashboard stays the control plane; the GPU is yours.

`linux/amd64` only — there is no CUDA on arm64 hosts.

## Supported policies

The backend loads any policy LeRobot's `PreTrainedConfig.from_pretrained(policy_uri)`
can decode, plus any third-party policy registered through
`lerobot.utils.import_utils.register_third_party_plugins`. The default image bakes in
runtime deps for the popular families:

| Family | Examples | Extra deps baked in |
|---|---|---|
| ACT | `lerobot/act_aloha_*` | base lerobot only |
| Diffusion Policy | `lerobot/diffusion_pusht` | `diffusers` |
| VQ-BeT | `lerobot/vqbet_*` | `einops` |
| TDMPC | `lerobot/tdmpc_*` | base |
| SmolVLA | `lerobot/smolvla_base` | `transformers`, `accelerate`, `num2words` |
| Pi0 / Pi0.5 | `lerobot/pi0_*` | `transformers`, `sentencepiece` |
| MolmoAct2 | `allenai/MolmoAct-*` | lerobot `molmoact2` extra |
| Out-of-tree (OpenVLA, custom heads) | any plugin registered at import time | install via `EXTRA_PIP_PACKAGES` |

Pick the policy per session from the dashboard (or `connect_drtc(policy_uri=...)` on
the manual path). The server lazy-loads it on the first `OpenSession`, or warm-loads it
up front when `DRTC_WARMUP_POLICY` is set.

**GPU sizing** follows the same rules as Interlatent's managed boxes: ~24 GB VRAM covers
SmolVLA / ACT / Diffusion; Pi0- and MolmoAct2-class VLAs want more.

## Build

The build context must be the repo root, so the `packages/server` COPY resolves:

```bash
docker build -f docker/Dockerfile -t interlatent-server:latest .
```

Multi-arch push to a registry:

```bash
docker buildx build --platform linux/amd64 \
  -f docker/Dockerfile \
  -t ghcr.io/<you>/interlatent-server:latest \
  --push .
```

### Build args

| Build arg | Default | Effect |
|---|---|---|
| `LEROBOT_EXTRAS` | `dataset,smolvla,pi0,molmoact2` | Comma-list of `lerobot` pip extras. If the pinned lerobot doesn't have one, the build falls back to plain `lerobot` — every model still works, just without the convenience extra. |
| `LEROBOT_REF` | pinned git SHA | The lerobot commit to install. MolmoAct2 needs lerobot main (post-0.5.1), which is why this is a git ref rather than a release. |
| `EXTRA_PIP_PACKAGES` | *(empty)* | Space-separated pip args installed after lerobot. Use for out-of-tree policies. |

```bash
# Slim: SmolVLA only
docker build -f docker/Dockerfile \
  --build-arg LEROBOT_EXTRAS=dataset,smolvla \
  -t interlatent-server:smolvla .

# Add an out-of-tree policy
docker build -f docker/Dockerfile \
  --build-arg EXTRA_PIP_PACKAGES="openvla flash-attn==2.5.8" \
  -t interlatent-server:openvla .
```

The stubs for the DRTC protocol are regenerated during the build
(`docker/gen_proto.sh`) against the image's own protobuf runtime, so generated code and
runtime can never disagree. See [`proto/README.md`](../proto/README.md).

## Run

```bash
docker run -d --name interlatent-server --gpus all \
  -p 50051:50051 \
  -v interlatent-cache:/root/.cache \
  -e INTERLATENT_API_KEY=ilat_xxx \
  -e INTERLATENT_ADVERTISE_ADDRESS=<IP-your-robots-can-reach> \
  -e DRTC_WARMUP_POLICY=lerobot/smolvla_base \
  -e HF_TOKEN=hf_xxx \
  interlatent-server:latest
```

Watch the logs for `DRTC server listening on 0.0.0.0:50051`, then attach the box to an
environment on the Compute page.

### Environment variables

| Var | Default | Purpose |
|---|---|---|
| `INTERLATENT_API_KEY` | *(unset)* | Your `ilat_` key. **Set it** — the entrypoint registers the box with the dashboard and turns on owner-checked RPC auth. Without it the container serves locally and unregistered. |
| `INTERLATENT_ADVERTISE_ADDRESS` | *(unset)* | `host[:port]` your robot nodes can reach this machine at. Required to register — handed to nodes verbatim. |
| `INTERLATENT_PORT` | `50051` | Port the gRPC server listens on. |
| `INTERLATENT_API_BASE` | `https://interlatent.com` | Backend base URL. |
| `INTERLATENT_BOX_NAME` | hostname | Display name on the Compute page. |
| `INTERLATENT_BOX_ID` | minted once | Stable box UUID; persisted at `~/.interlatent/box-id`. |
| `INTERLATENT_INSECURE` | *(unset)* | `1` disables the owner check on the gRPC port. Air-gapped networks only. |
| `DRTC_WARMUP_POLICY` | *(unset)* | HF repo / local path to load + compile at startup. Used only when the box has no env attached in the dashboard — an attached env's warmup target always wins. |
| `DRTC_WARMUP_IMAGE_KEYS` | *(unset)* | Comma-separated camera names (`cam_high,cam_left_wrist`) to pre-warm `DRTC_WARMUP_POLICY` with. **Required for MolmoAct2**, which can't build its feature dict without them. Must match the node's `--camera` names — the runtime cache is keyed on `(backend, policy_uri)`, so a mismatched warm is inherited by the first real session rather than discarded. |
| `HF_TOKEN` | *(unset)* | HF token for private policies — read by `huggingface_hub` directly. |

Anything after the image name is forwarded to the server, so CLI flags still work:

```bash
docker run --rm --gpus all -p 50052:50052 interlatent-server:latest --port 50052
```

Teleop needs no box-side config: control runs browser → hosted QUIC relay → node, and
never touches the GPU box.

### Persistent cache

Always mount a volume at `/root/.cache`. It holds:

- `torchinductor/` — torch.compile artifacts. Without this, every restart re-pays the
  multi-minute compile for VLA-class policies.
- `triton/` — Triton kernel cache.
- `huggingface/` — model weights. Without this, every restart re-downloads the policy.

A throwaway run loses all three.

### Reachability and security

The node is always the connection initiator — observations up, actions down, recording,
and `CloseSession` all ride the one node-opened gRPC channel. The box therefore needs no
inbound route back to the node; only the node needs to reach the box.

Publish **only** `:50051`, and set `INTERLATENT_ADVERTISE_ADDRESS` to the address that
reaches it. On the manual (unregistered) path, point the node at it directly:

```bash
# on the node:
export INTERLATENT_DRTC_URL=<box-public-ip>:50051   # host:port, NO scheme
interlatent-node run --robot so101 --port /dev/ttyACM0 ...
```

With `INTERLATENT_API_KEY` set, every RPC's `x-api-key` is validated against the backend
and must belong to the account that registered the box (cached 60 s). Your nodes already
send it. With `INTERLATENT_INSECURE=1`, or with no API key at all, the port is **open**:
anyone who can reach it can consume your GPU and record into your inbox. Firewall it to
your nodes' egress IPs.

## Provider notes

### RunPod

1. Create a Pod → **GPU Cloud** → RTX 4090 / A100 / H100.
2. **Container Image**: your pushed tag (e.g. `ghcr.io/you/interlatent-server:latest`).
3. **Expose TCP port** `50051`. RunPod proxies it at a *different* external port —
   e.g. `202.181.159.212:10608` → container `50051`. Put the **full external
   `host:port`** in `INTERLATENT_ADVERTISE_ADDRESS`; leave `INTERLATENT_PORT` at
   `50051` (what the server binds inside the container). A bare host gets the
   container port appended, and nodes then dial a port nothing listens on —
   `UNAVAILABLE: Connection refused`, while the box logs look perfectly healthy.
4. **Volume Mount** → `/root/.cache` (≥ 50 GB for SmolVLA).
5. **Environment**: `INTERLATENT_API_KEY`, `INTERLATENT_ADVERTISE_ADDRESS`, optionally
   `DRTC_WARMUP_POLICY` and `HF_TOKEN`.
6. Launch, then attach the box to an environment on the Compute page.

### Bare metal / LAN

Set `INTERLATENT_ADVERTISE_ADDRESS` to the LAN or VPN address your robots use. No public
IP is needed — the box only ever dials *out* to the backend.
