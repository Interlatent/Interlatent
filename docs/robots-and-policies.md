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
[examples/03](../examples/03_run_on_so101.py)). Each robot has a config doc with host
requirements, `--robot-arg` knobs, camera declarations, and joint names/units. Tested
paths:

| Robot | `--robot` | Extra | Config doc | Notes |
|---|---|---|---|---|
| SO-101 | `so101` | `[lerobot]` (+ `feetech-servo-sdk`) | [config](../packages/sdk/src/interlatent/adapters/lerobot/CONFIG.md) | reference platform; manual [`action()`](action-interface.md) |
| Koch v1.1 | `koch` | `[lerobot]` | [config](../packages/sdk/src/interlatent/adapters/lerobot/CONFIG.md) | via LeRobot classes; no `RobotProfile` yet, so manual `action()` fails closed |
| I2RT YAM | `yam` | `[yam]` | [config](../packages/sdk/src/interlatent/adapters/yam/CONFIG.md) | bimanual, native CAN via i2rt |
| Almond Axol | `axol` | `[axol]` | [config](../packages/sdk/src/interlatent/adapters/axol/CONFIG.md) | dual-arm, native async SDK |
| Any LeRobot robot | `<type>` | `[lerobot]` | — | cameras attach as `observation.images.<name>` |
| Custom hardware | `--loop module:fn` | — | — | bring your own I/O |

Each robot needs its own extra installed (`pip install 'interlatent[<extra>]'`). SO-101's
Feetech servos additionally need `feetech-servo-sdk` if the serial bus won't open.

**Missing your arm or your policy family?** That's the contribution we most want — open an
issue and we'll help you land it.
