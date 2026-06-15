# Interlatent DRTC GPU server — Docker image

Self-contained image of the Interlatent DRTC inference server. Deploy
to any GPU provider (RunPod, Lambda Labs, Vast.ai, Prime Intellect,
bare metal) without touching the host Python environment.

The image runs `interlatent-serve` (from `packages/server`), the
persistent-GPU policy server that robot-side clients connect to over
native gRPC.

## Supported policies

The DRTC backend loads any policy LeRobot's
`PreTrainedConfig.from_pretrained(policy_uri)` can decode, plus any
third-party policy registered through
`lerobot.utils.import_utils.register_third_party_plugins`. The default
image bakes in runtime deps for the popular families:

| Family            | Examples                                | Extra deps baked in |
|-------------------|-----------------------------------------|---------------------|
| ACT               | `lerobot/act_aloha_*`                   | base lerobot only |
| Diffusion Policy  | `lerobot/diffusion_pusht`               | `diffusers` |
| VQ-BeT            | `lerobot/vqbet_*`                       | `einops` |
| TDMPC             | `lerobot/tdmpc_*`                       | base |
| SmolVLA           | `lerobot/smolvla_base`                  | `transformers`, `accelerate`, `num2words` |
| Pi0 / Pi0.5       | `lerobot/pi0_*`                         | `transformers`, `sentencepiece` |
| Out-of-tree (e.g. OpenVLA, custom heads) | any plugin registered at import time | install via `EXTRA_PIP_PACKAGES` |

Specify the policy at session-create time from the dashboard (or in
`connect_drtc(policy_uri=...)` for the manual path). The server lazy-
loads it on the first `OpenSession` — or warm-loads it up-front when
you set `DRTC_WARMUP_POLICY`.

> **Auth note.** `interlatent-serve` does **not** enforce API-key auth — the
> assumption is that the GPU box sits on a private network (Tailscale,
> a VPC, etc.) and the network is the trust boundary. **Do not expose
> port 50051 to the public internet.**

## Build

From the repo root (the build context must be the repo root so the
`packages/server/` COPY paths resolve):

```bash
docker build -f docker/Dockerfile -t interlatent-drtc:latest .
```

For a multi-arch image you can push to a registry:

```bash
docker buildx build --platform linux/amd64 \
  -f docker/Dockerfile \
  -t ghcr.io/<you>/interlatent-drtc:latest \
  --push .
```

The image is `linux/amd64` only — there is no CUDA on arm64 hosts.

### Build args

Two `--build-arg` knobs control which policy stacks are bundled. The
defaults give you a broad image that runs every popular LeRobot
family; pass them to slim down or extend.

| Build arg            | Default              | Effect |
|----------------------|----------------------|--------|
| `LEROBOT_EXTRAS`     | `smolvla,pi0`        | Comma-list of `lerobot` pip extras. If the published lerobot version doesn't have one of them, the build falls back to plain `lerobot` — every model still works, just without the convenience extra. |
| `EXTRA_PIP_PACKAGES` | *(empty)*            | Space-separated pip args installed after lerobot. Use this for out-of-tree policies (`openvla`, `flash-attn`, custom plugin packages). |

Slim variant (SmolVLA-only, no Diffusion / Pi0 stack):

```bash
docker build -f docker/Dockerfile \
  --build-arg LEROBOT_EXTRAS=smolvla \
  --build-arg EXTRA_PIP_PACKAGES="" \
  -t interlatent-drtc:smolvla .
```

Add an out-of-tree policy:

```bash
docker build -f docker/Dockerfile \
  --build-arg EXTRA_PIP_PACKAGES="openvla flash-attn==2.5.8" \
  -t interlatent-drtc:openvla .
```

## Run

Minimum viable run on a single-GPU box:

```bash
docker run --rm --gpus all \
  -p 50051:50051 \
  -v interlatent-cache:/root/.cache \
  interlatent-drtc:latest
```

With warmup so the first robot session is fast (recommended for
SmolVLA-class policies):

```bash
docker run -d --name drtc --gpus all \
  -p 50051:50051 \
  -v interlatent-cache:/root/.cache \
  -e DRTC_WARMUP_POLICY=lerobot/smolvla_base \
  -e HF_TOKEN=hf_xxx \
  interlatent-drtc:latest
```

### Environment variables

| Var                  | Default              | Purpose |
|----------------------|----------------------|---------|
| `DRTC_PORT`          | `50051`              | Port the gRPC server listens on. |
| `DRTC_HOST`          | `0.0.0.0`            | Bind host. |
| `DRTC_WARMUP_POLICY` | *(unset)*            | HF repo / local path to load+compile at startup. |
| `HF_TOKEN`           | *(unset)*            | HF token for private policies. Aliased to `HUGGING_FACE_HUB_TOKEN`. |
| `TS_AUTHKEY`         | *(unset)*            | Tailscale auth key. When set, the container joins your tailnet on startup and exposes `DRTC_PORT` to tailnet peers. See [Tailscale integration](#tailscale-integration). |
| `TS_HOSTNAME`        | `interlatent-drtc`   | Tailnet hostname for this container. |
| `TS_EXTRA_ARGS`      | *(empty)*            | Extra flags forwarded to `tailscale up` (e.g. `--advertise-tags=tag:gpu --ssh`). |
| `TS_STATE_DIR`       | `/var/lib/tailscale` | Where tailscaled persists state. Mount a volume here to avoid re-auth on restart. |
| `INTERLATENT_API_KEY` | *(unset)*           | Your Interlatent API key. When set, the box self-registers with the hosted dashboard and reports status. Unset = no outbound calls. See [Make a personal box discoverable](../docs/self-hosting.md#make-a-personal-box-discoverable-in-the-dashboard). |
| `INTERLATENT_ADVERTISE_ADDRESS` | *(detected)* | `host:port` robots dial to reach this box, reported to the dashboard. **Set this** behind NAT — e.g. your tailnet `100.x:50051` or `$TS_HOSTNAME:50051`. |
| `INTERLATENT_BOX_ID` | *(persisted)*        | Pins the box's dashboard identity. Otherwise a UUID is minted once at `~/.interlatent/box-id`; **mount `/root/.interlatent` on a volume** (or set this) so a restart re-attaches to the same box instead of creating a new one. |

You can still pass CLI flags directly — anything after the image name
is forwarded to `interlatent-serve`:

```bash
docker run --rm --gpus all -p 50051:50051 interlatent-drtc:latest \
  --policy lerobot/smolvla_base --port 50051
```

### Tailscale integration

The image ships with the Tailscale binaries baked in. If you set
`TS_AUTHKEY`, the container will:

1. Start `tailscaled` in **userspace networking** mode (no
   `NET_ADMIN`, no `/dev/net/tun` — works on every GPU provider).
2. Run `tailscale up --authkey ... --hostname <TS_HOSTNAME>`.
3. Run `tailscale serve --bg --tcp ${DRTC_PORT} tcp://127.0.0.1:${DRTC_PORT}`
   so tailnet peers can actually reach the gRPC port (userspace mode
   blocks inbound TCP otherwise).
4. Print the tailnet address the Pi should connect to.

Generate an auth key in the Tailscale admin console
(<https://login.tailscale.com/admin/settings/keys>). Reusable keys are
fine for fleets; ephemeral keys auto-expire when the container exits.

```bash
docker run -d --name drtc --gpus all \
  -v interlatent-cache:/root/.cache \
  -v interlatent-ts:/var/lib/tailscale \
  -e TS_AUTHKEY=tskey-auth-xxxxx \
  -e TS_HOSTNAME=lab-gpu-01 \
  -e DRTC_WARMUP_POLICY=lerobot/smolvla_base \
  -e HF_TOKEN=hf_xxx \
  interlatent-drtc:latest
```

Notes:

- **No `-p 50051:50051`.** With Tailscale you don't need to publish
  the port to the host — peers reach the container directly over the
  tailnet. Publish only if you also want host-local access.
- **Mount `/var/lib/tailscale`** to a named volume so the node's
  identity persists across container restarts. Without that, every
  restart consumes a fresh slot from your reusable key.
- **Pi side.** After the container starts you'll see:
  ```
  [entrypoint] reachable on tailnet at  100.x.y.z:50051
  ```
  Use that on the Pi:
  ```bash
  export INTERLATENT_DRTC_URL=100.x.y.z:50051
  interlatent-node run --robot so101 --port /dev/ttyACM0 ...
  ```
- **Userspace mode is intentional.** Real-TUN mode would be slightly
  faster but requires `--cap-add NET_ADMIN --device /dev/net/tun`,
  which many managed GPU providers (RunPod, Vast, etc.) don't allow.
  Userspace + `serve` works everywhere.

If `TS_AUTHKEY` is unset, the Tailscale layer is skipped entirely and
the container behaves exactly like before — publish the port with
`-p 50051:50051` and reach it however you usually would.

### Persistent cache

Always mount a volume at `/root/.cache`. It holds:

- `torchinductor/` — torch.compile artifacts. Without this, every
  restart re-pays the multi-minute SmolVLA compile.
- `triton/` — Triton kernel cache.
- `huggingface/` — model weights. Without this, every restart
  re-downloads the policy.

A throwaway run loses all three.

## Provider quickstarts

### RunPod

1. Create a Pod → **GPU Cloud** → pick an RTX 4090 / A100 / H100 template.
2. Set **Container Image** to your pushed tag (e.g. `ghcr.io/you/interlatent-drtc:latest`).
3. **Expose TCP port** `50051`. RunPod gives you a public TCP proxy
   address — use that as the daemon's `INTERLATENT_DRTC_URL` if you
   are not on Tailscale. (Prefer Tailscale.)
4. **Volume Mount** → `/root/.cache` (any size ≥ 50&nbsp;GB for SmolVLA).
5. **Environment**:
   - `DRTC_WARMUP_POLICY=lerobot/smolvla_base`
   - `HF_TOKEN=hf_xxx` (if the policy is private)
6. Launch. Watch logs for `DRTC server listening on 0.0.0.0:50051`.

### Lambda Labs

1. Launch an instance with a CUDA-capable GPU and the **Lambda Stack** AMI.
2. SSH in, install Docker + the NVIDIA container toolkit if not present:
   ```bash
   sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```
3. Pull and run:
   ```bash
   docker run -d --name drtc --gpus all \
     -p 50051:50051 \
     -v /home/ubuntu/.interlatent-cache:/root/.cache \
     -e DRTC_WARMUP_POLICY=lerobot/smolvla_base \
     ghcr.io/<you>/interlatent-drtc:latest
   ```
4. Join the instance to your tailnet so the Pi can reach it privately.

### Vast.ai

1. Search for offers with the GPU you want; pick a template that
   supports Docker (most do).
2. **Docker Image** → `ghcr.io/<you>/interlatent-drtc:latest`.
3. **Docker Options** → `-p 50051:50051 -v /root/.cache:/root/.cache`.
4. Pass env via the instance's "On-start script" or template env.

### Prime Intellect / generic Linux GPU box

```bash
# Install nvidia-container-toolkit if needed.
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

docker run -d --name drtc --gpus all \
  --restart unless-stopped \
  -p 50051:50051 \
  -v /opt/interlatent-cache:/root/.cache \
  -e DRTC_WARMUP_POLICY=lerobot/smolvla_base \
  -e HF_TOKEN=$HF_TOKEN \
  interlatent-drtc:latest
```

## Pointing the Pi at the container

On the Pi (after `interlatent-node pair`):

```bash
# host:port — NO http:// or https:// prefix. That tells the SDK to
# use native gRPC. http(s):// would select gRPC-Web instead.
export INTERLATENT_DRTC_URL=<gpu-tailscale-ip>:50051

interlatent-node run --robot so101 --port /dev/ttyACM0 \
  --robot-arg id=lab_so101_01 \
  --camera top=/dev/video0
```

## Operations

### Logs
```bash
docker logs -f drtc
```

### Health
The image declares an HTTP-less TCP healthcheck on port `50051`.
`docker ps` shows the container as `(healthy)` once the server is
listening (~60s after start with a warmup policy).

### Update / redeploy
```bash
docker pull ghcr.io/<you>/interlatent-drtc:latest
docker stop drtc && docker rm drtc
docker run -d --name drtc ... interlatent-drtc:latest
```
Mount the same cache volume to skip warmup on the new container.

### Image size
Roughly **8–10&nbsp;GB** depending on the lerobot / torch versions
pinned. The base `pytorch/pytorch` image is ~5&nbsp;GB on its own —
shrinking further requires building torch with only the kernels you
need, which is rarely worth the operational pain.

## Troubleshooting

**`nvidia-smi` warning in logs, then crash.**
The container started without `--gpus all`. On Docker ≥ 19.03 with
`nvidia-container-toolkit` installed, always pass `--gpus all`.

**First session takes 5+ minutes even with warmup.**
The cache volume is empty or wasn't mounted. Confirm
`-v interlatent-cache:/root/.cache` is present and the warmup log line
prints `Pre-warm complete`. Note this only affects VLA-class policies
(SmolVLA, Pi0…) — ACT / Diffusion / VQ-BeT load in seconds and don't
benefit much from warmup.

**"OSError: HEAD /api/models/... 401" during warmup.**
The policy is private and `HF_TOKEN` wasn't passed.

**"ImportError: cannot import name '...' from 'lerobot.policies.<family>'".**
The policy class isn't available in the installed lerobot version, or
its runtime dep is missing. Rebuild with the relevant pip in
`EXTRA_PIP_PACKAGES`, or pin a newer lerobot via
`EXTRA_PIP_PACKAGES="lerobot==<version>"` (it gets installed after the
base layer and replaces it).

**Pi reports "failed to connect to all addresses".**
- The container isn't listening: `docker ps` should show `(healthy)`.
- The Pi can't reach the box: `tailscale ping <gpu-ip>` from the Pi.
- The URL has a scheme: `INTERLATENT_DRTC_URL` must be `host:port`, no `http(s)://`.

**Container joined the tailnet but the Pi still can't reach the port.**
With userspace networking, inbound TCP only works through
`tailscale serve`. The entrypoint sets this up automatically, but if
you exec into the container check:
```bash
tailscale --socket=/tmp/tailscale/tailscaled.sock serve status
```
You should see a `tcp://127.0.0.1:50051` mapping. If not, restart the
container — `tailscaled` likely wasn't ready when `serve` ran.

**Tailscale node consumes a new slot on every restart.**
You forgot to mount `/var/lib/tailscale` to a named volume. Without
that the node identity is wiped on container recreate and you'll
churn through reusable-key slots in the admin console.

## Files

```
docker/
  Dockerfile      CUDA + torch + lerobot + interlatent-server.
  entrypoint.sh   Translates env vars to interlatent-serve CLI; checks GPU visibility.
  .dockerignore   Trims the build context (no site/, no SDK, no node_modules).
  README.md       This file.
```
