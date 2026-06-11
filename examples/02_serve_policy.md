# Serve a real policy on your GPU

One process, one command. The server loads the policy once per process (including
torch.compile for policies that use it) and keeps it warm for every robot session.

## On a machine with a CUDA GPU

```bash
pip install 'interlatent-server[lerobot]'
interlatent-serve --policy lerobot/smolvla_base
```

- `--policy` accepts any LeRobot-loadable Hugging Face repo id or a local checkpoint path
  (your fine-tune works the same way). Private repos: `export HF_TOKEN=...`.
- First start with SmolVLA takes minutes (one-time torch.compile); the cache persists under
  `~/.cache`, so restarts are fast. Keep that directory on a volume if containerized.
- Listens on `:50051` (gRPC) and `:50052` (teleop relay).

## GPU sizing

| Policy | Works on |
|---|---|
| ACT / Diffusion / VQ-BeT / TDMPC | almost any CUDA GPU |
| SmolVLA | A10G / RTX 3090 class and up (~50–150 ms per inference) |
| Pi0 / Pi0.5, MolmoAct2 | ≥24 GB VRAM |

## On a rented GPU (RunPod / Lambda / Vast / bare metal)

Use the prebuilt Docker image — it bundles CUDA, torch, and lerobot, can auto-join your
Tailscale network, and warm-loads a policy at boot:

```bash
docker run --gpus all -p 50051:50051 -p 50052:50052 \
  -v $HOME/.cache:/root/.cache \
  -e DRTC_WARMUP_POLICY=lerobot/smolvla_base \
  ghcr.io/interlatent/interlatent-server   # or build from docker/ in this repo
```

See [docker/README.md](../docker/README.md) for Tailscale setup, cache volumes, and
provider-specific notes, and [docs/self-hosting.md](../docs/self-hosting.md) for the full
guide.

## Point a robot at it

From any machine that can reach the server:

```python
from interlatent.inference.integration import connect_drtc

client = connect_drtc(
    environment="my-arm",
    policy_uri="lerobot/smolvla_base",
    server_address="gpu-box:50051",   # LAN hostname, Tailscale IP, ...
    task="pick up the red cube",
    fps=30,
)
```

Continue with [03_run_on_so101.py](03_run_on_so101.py).
