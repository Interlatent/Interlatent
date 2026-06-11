# interlatent.isaaclab

Isaac Lab integration for the Interlatent SDK.

## Installation

```bash
pip install 'interlatent[isaaclab]'
```

> **Note:** `isaaclab` and `isaaclab_rl` must be installed separately — they are not available on PyPI.
> Follow the [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) before using this module.

## Overview

`IsaacSimCollectionEnv` is a drop-in replacement for Isaac Lab's `RslRlVecEnvWrapper` that layers Interlatent trajectory and activation collection on top of any Isaac Lab environment without changing the training contract. It works with both `ManagerBasedRLEnv` and `DirectRLEnv`.

Two usage modes are supported:

| Mode | Description |
|------|-------------|
| **Passive** (`collecting()`) | Hooks into an existing training / eval loop via a context manager |
| **Standalone** (`collect()`) | Drives its own rollout loop |

## Passive Collection (Mode 2)

This is the primary mode for use alongside existing training or play scripts. Wrap your env, then use `collecting()` as a context manager around your runner loop.

```python
from pathlib import Path
from datetime import datetime

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from interlatent import Interlatent
from interlatent.isaaclab import IsaacSimCollectionEnv

import isaaclab_tasks  # noqa: F401 — registers Isaac Lab task IDs
from isaaclab_tasks.utils.hydra import hydra_task_config

task_id = "Isaac-Velocity-Flat-Spot-v0"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Build the base env (render_mode="rgb_array" required for frame capture)
env_cfg, agent_cfg = ...  # load via hydra_task_config or directly
base_env = gym.make(task_id, cfg=env_cfg, render_mode="rgb_array")

# Set up Interlatent client — one DB per run
run_label = f"{task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
db_path = str(Path("interlatent_runs") / run_label / "raw.db")
Path(db_path).parent.mkdir(parents=True, exist_ok=True)
client = Interlatent(db_path=db_path)

# Wrap with IsaacSimCollectionEnv
env = IsaacSimCollectionEnv(
    base_env,
    interlatent_client=client,
    max_layers=3,
    clip_actions=agent_cfg.clip_actions,
    env_name=task_id,
    # actor_obs_key defaults to "policy" (Isaac Lab convention).
    # Set to "actor" for asymmetric actor-critic setups.
)

# Optional: log episode boundaries
env.on_episode_end(lambda ep: print(f"Episode {ep} ended (step {env._global_step})"))

# Load checkpoint
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
runner.load("path/to/checkpoint.pt")

# Collect while the runner evaluates
with env.collecting(runner.alg.actor) as run_id:
    # Any loop that calls env.step() internally will be captured automatically.
    # For example, run the Isaac Lab play loop here:
    #   policy = runner.get_inference_policy(device=device)
    #   obs, _ = env.reset()
    #   while True:
    #       actions = policy(obs)
    #       obs, _, _, _ = env.step(actions)
    pass

print("Collected run:", run_id)

env.close()
client.close()
```

The `collecting()` context manager:
- Calls `watch()` on the Interlatent client to attach forward hooks to the actor
- Calls `tick()` automatically on every `env.step()` to record obs / reward / done / frames
- Calls `upload()` automatically at context exit

## Standalone Collection (Mode 1)

Use `collect()` to drive the rollout loop directly — useful for offline evaluation without an external runner loop. Requires `num_envs=1`.

```python
from pathlib import Path
from datetime import datetime

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from interlatent import Interlatent
from interlatent.isaaclab import IsaacSimCollectionEnv

import isaaclab_tasks  # noqa: F401

task_id = "Isaac-Velocity-Flat-Spot-v0"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

env_cfg, agent_cfg = ...  # num_envs=1 required for Mode 1
base_env = gym.make(task_id, cfg=env_cfg, render_mode="rgb_array")

client = Interlatent(db_path="run.db")

env = IsaacSimCollectionEnv(
    base_env,
    interlatent_client=client,
    env_name=task_id,
)

runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
runner.load("path/to/checkpoint.pt")

result = env.collect(
    actor_model=runner.alg.actor,
    steps=2000,
    tags={"task": task_id, "checkpoint": "model_1000.pt"},
)
print("run_id:", result["run_id"])

env.close()
client.close()
```

## DirectRLEnv

`IsaacSimCollectionEnv` works with `DirectRLEnv` as well, with one limitation: observation labels cannot be derived automatically (there is no `ObservationManager`), so the Interlatent platform will fall back to `obs_0`, `obs_1`, ... naming.

```python
from isaaclab.envs import DirectRLEnv
# Usage is identical — just pass your DirectRLEnv instead of ManagerBasedRLEnv.
env = IsaacSimCollectionEnv(direct_env, interlatent_client=client, env_name="my-task")
```

## API Reference

### `IsaacSimCollectionEnv(env, *, interlatent_client, ...)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `env` | `ManagerBasedRLEnv \| DirectRLEnv` | — | The base Isaac Lab environment |
| `interlatent_client` | `Interlatent` | — | SDK client instance (owns the DB and upload) |
| `max_layers` | `int` | `3` | Max actor layers to auto-detect and hook |
| `actor_obs_key` | `str` | `"policy"` | TensorDict key for actor observations |
| `clip_actions` | `float \| None` | `None` | Forwarded to `RslRlVecEnvWrapper` |
| `env_name` | `str` | `"Unknown"` | Human-readable task identifier written to the run record |

### Key methods

```python
env.attach(runner.alg.actor)           # Pre-register actor (avoids passing it each time)
env.on_episode_end(callback)           # Register episode-end callback (receives episode index)

with env.collecting(actor) as run_id:  # Passive collection context manager
    runner.learn(num_learning_iterations=500)

result = env.collect(steps=2000)       # Standalone rollout collection

env.upload(run_id=..., tags={...})     # Upload DB to Interlatent platform manually
```

## Notes

- Only env index 0 is recorded. For multi-env collection, run separate processes with separate DB files.
- Isaac Lab's standard obs group key is `"policy"`. If you use asymmetric actor-critic (separate `"actor"` and `"critic"` groups), set `actor_obs_key="actor"`.
- Frame capture requires `render_mode="rgb_array"` on the base env before wrapping.
- The actor model can be pre-registered with `attach()` so `collect()` / `collecting()` need no explicit `actor_model=` argument.
- `DirectRLEnv` is fully supported but obs labels will use generic `obs_N` names since there is no `ObservationManager` to introspect.
