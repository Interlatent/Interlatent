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
| Your fine-tune | `lerobot` | ✅ | any HF repo id or local checkpoint path |
| Anything else | custom | 🔌 | `register_backend()` — see [CONTRIBUTING.md](../CONTRIBUTING.md) |

If LeRobot's policy factory can load it, `interlatent-server` can serve it.

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

| Sim | Integration |
|---|---|
| Isaac Lab | [`interlatent.isaaclab.IsaacSimCollectionEnv`](../packages/sdk/src/interlatent/isaaclab/README.md) |
| MuJoCo (mjlab) | [`interlatent.mjlab.CollectionEnv`](../packages/sdk/src/interlatent/mjlab/README.md) |
| Gym-style anything | `client.watch()` / `client.tick()` — see [examples/05](../examples/05_collect_dataset.py) |

**Missing your arm or your policy family?** That's the contribution we most want — open an
issue and we'll help you land it.
