# Supported robots & policies

## Policies (served by `interlatent-server`)

| Policy | Backend | Status | Notes |
|---|---|---|---|
| SmolVLA (`lerobot/smolvla_base`, fine-tunes) | `lerobot` | ✅ | torch.compile warm-up once per process; ~50–150 ms/infer on A10G+ |
| Pi0 / Pi0.5 | `lerobot` | ✅ | ≥24 GB VRAM |
| ACT | `lerobot` | ✅ | light, great first policy |
| Diffusion Policy | `lerobot` | ✅ | |
| VQ-BeT | `lerobot` | ✅ | |
| TDMPC | `lerobot` | ✅ | |
| MolmoAct2 (released checkpoints) | `molmoact2` | ✅ | auto-routed; needs camera `image_keys` session metadata |
| SpatialVLA (`IPEC-COMMUNITY/spatialvla-*`) | `spatialvla` | 🧪 | Shanghai AI Lab, MIT; transformers-native, auto-routed; image+instruction → 7-DoF chunk (no proprio). GPU-only, untested in CI |
| RDT-1B (`robotics-diffusion-transformer/rdt-1b`) | `rdt` | 🧪 | Tsinghua THU-ML, MIT; diffusion, proprio-aware, 64-step chunk. Needs the RDT repo on `PYTHONPATH` + T5-XXL; `action_indices` metadata for the unified→robot DoF map. GPU-only, untested in CI |
| Your fine-tune | `lerobot` | ✅ | any HF repo id or local checkpoint path |
| Anything else | custom | 🔌 | `register_backend()` — see [CONTRIBUTING.md](../CONTRIBUTING.md) |

If LeRobot's policy factory can load it, `interlatent-server` can serve it. Model families
that live *outside* LeRobot (transformers-native or custom-repo VLAs) plug in as their own
backend and are auto-routed from `policy_backend="lerobot"` by their checkpoint URI — see
`molmoact2_backend` / `spatialvla_backend` / `rdt_backend` and the `register_router()` seam.
🧪 = backend wired + unit-tested for routing/observation handling, but the real GPU
inference path has not yet been validated on hardware.

## Robots (client side)

The DRTC client is robot-agnostic — if you can read observations and write actions in
Python, you can drive it (that's the whole of
[examples/03](../examples/03_run_on_so101.py)). Tested/first-class paths:

| Robot | Path | Notes |
|---|---|---|
| SO-101 | `interlatent-node` / teleop driver / examples | reference platform |
| Koch v1.1 | `interlatent-node --robot koch` | via LeRobot robot classes |
| ALOHA | `interlatent-node --robot aloha` | via LeRobot robot classes |
| Any LeRobot-supported robot | `interlatent-node --robot <type>` | cameras attach as `observation.images.<name>` |
| Custom hardware | `--loop module:fn` or hand-written loop | bring your own I/O |

## Simulators (data collection)

Any gym-style environment — including simulators — can be recorded with the generic
`client.watch()` / `client.tick()` API; see [examples/05](../examples/05_collect_dataset.py).
There are no simulator-specific wrappers at the moment.

**Missing your arm or your policy family?** That's the contribution we most want — open an
issue and we'll help you land it.
