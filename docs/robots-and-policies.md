# Supported robots & policies

## Policies (run on managed cloud GPU pods)

| Policy | Backend | Status | Notes |
|---|---|---|---|
| SmolVLA (`lerobot/smolvla_base`, fine-tunes) | `lerobot` | ✅ | ~50–150 ms/infer on A10G+ |
| Pi0 / Pi0.5 | `lerobot` | ✅ | ≥24 GB VRAM pod |
| ACT | `lerobot` | ✅ | light, great first policy |
| Diffusion Policy | `lerobot` | ✅ | |
| VQ-BeT | `lerobot` | ✅ | |
| TDMPC | `lerobot` | ✅ | |
| MolmoAct2 (released checkpoints) | `molmoact2` | ✅ | auto-routed; needs camera `image_keys` session metadata |
| Your fine-tune | `lerobot` | ✅ | any HF repo id or local checkpoint path |

Pick a policy by passing its URI to `connect_drtc(policy_uri=…)` or
`interlatent session start --policy …`. If LeRobot's policy factory can load it, a pod can
serve it.

## Robots (client side)

The DRTC client is robot-agnostic — if you can read observations and write actions in
Python, you can drive it (that's the whole of
[examples/03](../examples/03_run_on_so101.py)). Tested/first-class paths:

| Robot | Path | Notes |
|---|---|---|
| SO-101 | `interlatent-node --robot so101` / examples | reference platform; manual [`action()`](action-interface.md) |
| Koch v1.1 | `interlatent-node --robot koch` | via LeRobot robot classes; manual [`action()`](action-interface.md) |
| ALOHA | `interlatent-node --robot aloha` | via LeRobot robot classes |
| Any LeRobot-supported robot | `interlatent-node --robot <type>` | cameras attach as `observation.images.<name>` |
| Custom hardware | `--loop module:fn` or hand-written loop | bring your own I/O |

## Simulators (data collection)

Any gym-style environment — including simulators — can be recorded with the generic
`client.watch()` / `client.tick()` API; see [examples/05](../examples/05_collect_dataset.py).
There are no simulator-specific wrappers at the moment.

**Missing your arm or your policy family?** That's the contribution we most want — open an
issue and we'll help you land it.
