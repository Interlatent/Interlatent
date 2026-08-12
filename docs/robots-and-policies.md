# Supported robots & policies

## Policies

Policies run on a GPU box you run ([self-hosting.md](self-hosting.md)) — a
workstation with an NVIDIA card, or a machine you rent from RunPod, Lambda or
Vast. The list is the same either way.

| Policy | Backend | Status | Notes |
|---|---|---|---|
| SmolVLA (`lerobot/smolvla_base`, fine-tunes) | `lerobot` | ✅ | ~50–150 ms/infer on A10G+ |
| Pi0 / Pi0.5 | `lerobot` | ✅ | ≥24 GB VRAM |
| ACT | `lerobot` | ✅ | light, great first policy |
| Diffusion Policy | `lerobot` | ✅ | |
| VQ-BeT | `lerobot` | ✅ | |
| TDMPC | `lerobot` | ✅ | |
| MolmoAct2 (released checkpoints) | `molmoact2` | ✅ | auto-routed off `config.json`; needs camera `image_keys` session metadata |
| DreamZero (world-action) | `dreamzero` | ⚠️ | auto-routed off `config.json`; needs ≥2 GPUs and a separate DreamZero env (`DREAMZERO_PYTHON`), not in the default image |
| Your fine-tune | `lerobot` | ✅ | any HF repo id or local checkpoint path |

Pick a policy by passing its URI to `connect_drtc(policy_uri=…)` or
`interlatent session start --policy …`; override the backend with
`--backend` (default `lerobot`). If LeRobot's policy factory can load it, a box
can serve it.

## Robots (client side)

The DRTC client is robot-agnostic — if you can read observations and write actions in
Python, you can drive it (that's the whole of
[examples/03](../examples/03_run_on_so101.py)). Each robot has a config doc with host
requirements, `--robot-arg` knobs, camera declarations, and joint names/units. Tested
paths:

| Robot | `--robot` | Extra | Config doc | Notes |
|---|---|---|---|---|
| SO-101 | `so101` | `[lerobot]` or `[so101]` (+ `feetech-servo-sdk`) | [config](../packages/sdk/src/interlatent/adapters/lerobot/CONFIG.md) | reference platform; manual [`action()`](action-interface.md) |
| I2RT YAM | `yam` (`yam_left`, `yam_right`, `yam_bimanual`) | `[yam]` | [config](../packages/sdk/src/interlatent/adapters/yam/CONFIG.md) | bimanual, native CAN via i2rt |
| Almond Axol | `axol` | `[axol]` | [config](../packages/sdk/src/interlatent/adapters/axol/CONFIG.md) | dual-arm, native async SDK — **unstable beta** |
| Nori | `nori` | `[nori]` | [config](../packages/sdk/src/interlatent/adapters/nori/CONFIG.md) | dual-SO-101 rig, on-Pi daemon over NDJSON — **unstable beta** |
| dimos stack | `dimos --robot-arg kind=<k>` (or `dimos_<k>`) | `[dimos]` | [config](../packages/sdk/src/interlatent/adapters/dimos/CONFIG.md) | `kind` = `xarm7`, `xarm6`, `a1z`; binds to a running dimos stack as a bus peer ([ROBOT.md](../ROBOT.md)) |
| Any LeRobot robot | `<type>` | `[lerobot]` | — | cameras attach as `observation.images.<name>` |
| Custom hardware | `--loop module:fn` | — | — | bring your own I/O |

Each robot needs its own extra installed (`pip install 'interlatent[<extra>]'`). SO-101's
Feetech servos additionally need `feetech-servo-sdk` if the serial bus won't open.
`[axol]` requires Python ≥ 3.13; `[dimos]` requires 3.12 (3.11 resolves only off
linux/x86_64).

**Missing your arm or your policy family?** Open an issue and we'll help you land it.
