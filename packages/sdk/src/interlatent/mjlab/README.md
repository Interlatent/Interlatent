# interlatent.mjlab

MuJoCo / Isaac Lab (mjlab) integration for the Interlatent SDK.

## Installation

```bash
pip install 'interlatent[mjlab]'
```

> **Note:** `mjlab` itself must be installed separately from source — it is not available on PyPI.

## Overview

`CollectionEnv` is a drop-in replacement for `RslRlVecEnvWrapper` that layers Interlatent trajectory and activation collection on top of a `ManagerBasedRlEnv` without changing the training contract.

Two usage modes are supported:

| Mode | Description |
|------|-------------|
| **Passive** (`collecting()`) | Hooks into an existing training / eval loop via a context manager |
| **Standalone** (`collect()`) | Drives its own rollout loop |

## Passive Collection (Mode 2)

This is the primary mode for use with `play_collect.py`-style scripts. Wrap your env, then use `collecting()` as a context manager around your runner loop.

```python
from pathlib import Path
from datetime import datetime
from dataclasses import asdict

import torch
from interlatent import Interlatent
from interlatent.mjlab import CollectionEnv

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

task_id = "Mjlab-Velocity-Flat-Unitree-G1-v0"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

env_cfg = load_env_cfg(task_id, play=True)
agent_cfg = load_rl_cfg(task_id)

# Build the base env (render_mode="rgb_array" required for frame capture)
env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")

# Set up Interlatent client — one DB per run
run_id = f"{task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
db_path = str(Path("interlatent_runs") / run_id / "raw.db")
Path(db_path).parent.mkdir(parents=True, exist_ok=True)
client = Interlatent(db_path=db_path)

# Wrap with CollectionEnv
env = CollectionEnv(
    env,
    interlatent_client=client,
    max_layers=3,
    clip_actions=agent_cfg.clip_actions,
    env_name=task_id,
)

# Optional: log episode boundaries
env.on_episode_end(lambda ep: print(f"Episode {ep} ended (step {env._global_step})"))

# Load checkpoint
runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device=device)
runner.load("path/to/checkpoint.pt", load_cfg={"actor": True}, strict=True)

# Collect while the runner evaluates
with env.collecting(runner.alg.actor):
    # Any loop that calls env.step() internally will be captured automatically.
    # Replace this with your viewer / eval loop, e.g.:
    #   NativeMujocoViewer(env, policy).run()
    #   ViserPlayViewer(env, policy).run()
    pass

# Upload results and run SAE checkpoint
env.upload()
client.checkpoint(sae_k=64)
env.close()
client.close()
```

The `collecting()` context manager:
- Calls `watch()` on the Interlatent client to attach forward hooks to the actor
- Calls `tick()` automatically on every `env.step()` to record obs / reward / done / frames
- Calls `upload()` at context exit (you can also call `env.upload()` manually after)

## Standalone Collection (Mode 1)

Use `collect()` to drive the rollout loop directly — useful when you don't have an external runner loop.

```python
result = col_env.collect(
    actor_model=runner.alg.actor,
    steps=2000,
    tags={"task": task_id, "checkpoint": "model_1000.pt"},
)
print("run_id:", result["run_id"])
```

Requires `num_envs=1` (only env index 0 is observed).

## API Reference

### `CollectionEnv(env, *, interlatent_client, ...)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `env` | `ManagerBasedRlEnv` | The unwrapped base environment |
| `interlatent_client` | `Interlatent` | SDK client instance (owns the DB and upload) |
| `max_layers` | `int` | Max actor layers to auto-detect and hook (default: `3`) |
| `actor_obs_key` | `str` | TensorDict key for actor observations (default: `"actor"`) |
| `clip_actions` | `float \| None` | Forwarded to `RslRlVecEnvWrapper` |
| `env_name` | `str` | Human-readable task identifier written to the run record |

### Key methods

```python
env.attach(runner.alg.actor)          # Pre-register actor (avoids passing it each time)
env.on_episode_end(callback)          # Register episode-end callback (receives episode index)

with env.collecting(actor) as run_id: # Passive collection context manager
    runner.learn(...)

result = env.collect(steps=2000)      # Standalone rollout collection

env.upload(run_id=..., tags={...})    # Upload DB to Interlatent platform
```

## Notes

- Only env index 0 is recorded. For multi-env collection, run separate processes with separate DB files and merge them.
- The actor model can be pre-registered with `attach()` so `collect()` / `collecting()` need no argument.
- Frame capture (`render_mode="rgb_array"`) should be set on the base env before wrapping.
