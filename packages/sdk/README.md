# interlatent (Python SDK)

The robot-side half of [Interlatent](https://github.com/interlatent/interlatent): run VLA
policies on real hardware against a self-hosted or hosted inference server, teleoperate,
and collect LeRobot datasets.

What's in this package:

1. **DRTC inference client** (`interlatent.inference`) — `connect_drtc()` opens a real-time
   action-chunking session against any DRTC server: your own
   [`interlatent-server`](../server) box (no account needed) or Interlatent Cloud
   (`api_key=`).
2. **Robot node daemon** (`interlatent-node`) — a long-running daemon for always-on robots,
   with camera capture and DAgger keyboard takeover (uses Interlatent Cloud for session
   assignment).
3. **Collection** — `watch()` / `tick()` / `collect()` record per-step observations,
   actions, rewards, and metrics into a local SQLite staging cache; build a local LeRobot
   v3.0 dataset from it, or `upload()` it to a hosted environment.

For inference quickstarts see the [repo README](../../README.md) and
[examples](../../examples/). The rest of this document covers the collection/upload API.

> **Note (hosted uploads):** a collection session binds to a backend **environment**
> (env-as-collection). There is no `model_id` — policy attribution lives on the
> environment. The environment must already exist in the dashboard before you upload.
> Purely local collection needs no account.

## Install

```bash
pip install interlatent
```

Optional extras for environment integrations:

```bash
pip install 'interlatent[isaaclab]'   # Isaac Lab (requires isaaclab installed separately)
pip install 'interlatent[mjlab]'      # MuJoCo Lab (requires mjlab installed separately)
pip install 'interlatent[lerobot]'    # LeRobot (adds huggingface_hub for checkpoint naming)
```

**Requirements:** Python >= 3.11

## Quickstart

```python
from interlatent import Interlatent

client = Interlatent(api_key="ilat_...")

# 1. Hook a model and bind the session to a backend environment
watcher = client.watch(
    model,
    env,
    environment="my-policy",   # backend environment slug (must already exist)
    capture_frames=True,
)

# 2. Run your environment loop — obs and action are both required
obs, _ = env.reset()
for step in range(3000):
    action, _ = model.predict(obs, deterministic=True)
    next_obs, reward, done, truncated, info = env.step(action)
    client.tick(obs=obs, action=action, reward=reward, done=done, truncated=truncated, info=info)
    obs = next_obs
    if done or truncated:
        obs, _ = env.reset()

# 3. Upload + trigger server-side analysis
job = client.checkpoint()
# job = {"environment": "...", "job_id": "...", "status": "pending", "checkpoint_count": 1, "step_count": 3000}

# 4. Poll for completion
status = client.environments.processing_status(job["environment"])

client.close()
```

## Client Constructor

```python
client = Interlatent(
    api_key="ilat_...",       # API key (or set INTERLATENT_API_KEY env var)
    base_url=None,            # Override API base URL (default: https://interlatent.com, or INTERLATENT_BASE_URL env var)
    bypass_token=None,        # Vercel bypass token (or INTERLATENT_BYPASS_TOKEN env var)
    timeout=30.0,             # HTTP request timeout in seconds
    db_path=None,             # Custom path for the local SQLite staging cache
    fps=30,                   # Frame rate stamped into the LeRobot dataset at upload
)
```

All arguments are keyword-only. There is no `model_id` — bind to a backend environment via `watch(environment=...)`.

## Collection

### `watch()` — passive collection (you drive the loop)

Hook a model and start recording. You control the environment loop and call `tick()` after each step.

```python
watcher = client.watch(
    model,                          # PyTorch model or SB3 model
    env,                            # Gymnasium environment (optional; used for env-name and metric auto-detection)
    environment="my-policy",        # Required — backend environment slug (or id) to attach this collection to
    env_name=None,                  # Override the human-readable env name (auto-detected from env if omitted)
    task=None,                      # Task label stamped into the dataset (defaults to env name)
    metrics=None,                   # Custom metrics (auto-detected from env if omitted)
    context_fn=None,                # Callable returning extra per-step context dict
    total_steps=None,               # Expected total steps (for progress display)
    capture_frames=False,           # Capture rendered frames
    frame_every=1,                  # Capture a frame every N steps
    frame_quality=85,               # JPEG quality for saved frames
    frame_dir=None,                 # Custom directory for frame storage
    episode_id=None,                # Override the generated episode UUID
)
```

`environment=` is required — it routes uploads to the right dashboard env. The env must already exist, and its policy attribution (`layer`, `base_model`, `model_source`) is resolved/locked on first write.

Then drive your loop:

```python
obs, _ = env.reset()
for step in range(steps):
    action, _ = model.predict(obs, deterministic=True)
    next_obs, reward, done, truncated, info = env.step(action)
    client.tick(
        obs=obs,
        action=action,             # required — becomes the LeRobot `action` column
        reward=float(reward),
        done=done,
        truncated=truncated,
        info=info,
        frame=env.render(),        # optional — pass frames directly via tick()
    )
    obs = next_obs
    if done or truncated:
        obs, _ = env.reset()
```

`obs` and `action` are both required — they become the `observation.state` and `action` columns of the LeRobot dataset at upload time, and downstream Q(s, a) post-training cannot recover from null values.

### `collect()` — automatic collection (SDK drives the loop)

Runs the full environment loop for you. The SDK calls `model.predict()` / the model forward and `env.step()` internally.

```python
result = client.collect(
    model,
    env,
    steps=5000,
    task=None,
    metrics=None,
    context_fn=None,
    deterministic=True,
    capture_frames=True,
    frame_every=1,
    frame_quality=85,
)
# result = {"episode_id": "...", "steps": 5000, "env_name": "...", "start_time": ...}
```

`collect()` records into the same staging cache `watch()` uses; bind the environment with `watch(environment=...)` first (or set it before calling `upload()` / `checkpoint()`).

### Multicamera frame capture

Register camera names to capture from multiple viewpoints:

```python
client.register_cameras(["front", "side", "overhead"])

# Then pass a dict of camera images per tick:
client.tick(
    obs=obs, action=action, reward=reward, done=done, truncated=truncated,
    frame={"front": front_img, "side": side_img, "overhead": overhead_img},
)
```

## Upload and Processing

### `upload()` — build a LeRobot dataset and push it to the server

Builds a LeRobot v3.0 dataset from the staging cache, uploads it under the environment's `_inbox/<session_uuid>/` prefix via presigned URLs, registers each episode, and calls `upload-complete` so the server-side merge picks it up. On success the local staging cache and frame buffer are wiped.

```python
client.upload(
    tags={"experiment": "v2"},    # optional per-episode metadata
    label="",                      # optional label forwarded to episodes.create
    workers=8,                     # parallel upload threads
    reward_config=None,            # optional reward config dict
)
```

Uploads are session-scoped: each `upload()` / `checkpoint()` produces one inbox session that the backend merges into the env's single canonical dataset.

### `checkpoint()` — upload + trigger server-side analysis

Calls `upload()` internally, then triggers the analysis pipeline on the server:

```python
job = client.checkpoint(label="")
# job = {"environment": "...", "job_id": "...", "status": "pending", "checkpoint_count": 1, "step_count": ...}
```

The server-side pipeline runs: SAE training, latent statistics, optional VLM frame scoring, optional autolabeling (if `OPENAI_API_KEY` is set on the server), episode export, failure classification, and report generation.

### Poll for results

Status is environment-scoped:

```python
# Processing status for the whole environment
status = client.environments.processing_status(job["environment"])
```

Per-episode status/results are available on the episodes resource:

```python
# Block until an episode finishes processing
status = client.episodes.wait(episode_id, timeout=600, poll=5.0)

# Or poll manually
data = client.episodes.status(episode_id)

# Retrieve results
results = client.episodes.results(episode_id)
```

## Stable Baselines3 Integration

Use the SB3 callback to automatically checkpoint during training:

```python
client = Interlatent(api_key="...")

client.watch(model, env, environment="my-sb3-agent", capture_frames=True)

callback = client.sb3_callback(checkpoint_every=10_000)
model.learn(100_000, callback=callback)

client.close()
```

## Isaac Lab Integration

> ⚠️ The Isaac Lab / MuJoCo Lab collection-env wrappers are currently out of date with the public SDK surface and are scheduled for a rewrite (see `FUTURE.md`, item 9). Treat the examples below as indicative of the intended shape, not a verified contract — prefer the core `watch()` / `tick()` / `checkpoint()` API above.

The `IsaacSimCollectionEnv` wraps an Isaac Lab environment for collection. It supports both `ManagerBasedRLEnv` and `DirectRLEnv`.

### Mode 1 — standalone collection

The wrapper drives its own rollout loop:

```python
from interlatent import Interlatent
from interlatent.isaaclab.collection_env import IsaacSimCollectionEnv

client = Interlatent(api_key="...")
env = gym.make("Isaac-Velocity-Flat-Spot-v0", cfg=env_cfg)

col_env = IsaacSimCollectionEnv(
    env,
    interlatent_client=client,
    env_name="Isaac-Velocity-Flat-Spot-v0",
)

runner = OnPolicyRunner(col_env, asdict(agent_cfg), device=device)
runner.load(checkpoint_path, load_cfg={"actor": True}, strict=True)
col_env.attach(runner.alg.actor)

result = col_env.collect(steps=2000)
# result = {"episode_id": "...", "steps": 2000, "env_name": "..."}
```

### Mode 2 — passive collection (hooks into an existing training/eval loop)

The wrapper intercepts `env.step()` calls and automatically feeds data to the Interlatent client:

```python
col_env = IsaacSimCollectionEnv(env, interlatent_client=client, env_name="Spot-v0")
runner = OnPolicyRunner(col_env, asdict(agent_cfg), device=device)

with col_env.collecting(runner.alg.actor) as episode_id:
    runner.learn(num_learning_iterations=500)

print("episode:", episode_id)
```

Both modes automatically:
- Auto-detect observation and action labels from Isaac Lab managers
- Capture rendered frames
- Upload data on completion
- Pass reward configuration to the server

### Reward inspection

```python
# Snapshot current reward manager state
rewards = col_env.inspect_rewards(env_idx=0)

# JSON-serializable reward config
config = col_env.reward_config_json(env_idx=0)
```

## MuJoCo Lab Integration

> ⚠️ Same caveat as Isaac Lab above — this wrapper is scheduled for a rewrite (see `FUTURE.md`, item 9).

The `CollectionEnv` wrapper in `interlatent.mjlab.collection_env` has the same API as `IsaacSimCollectionEnv`, adapted for mjlab environments. Usage is identical — see the Isaac Lab section above.

```python
from interlatent.mjlab.collection_env import CollectionEnv

col_env = CollectionEnv(
    env,
    interlatent_client=client,
    env_name="my-mjlab-env",
    actor_obs_key="actor",  # default is "actor" for mjlab (vs "policy" for isaaclab)
)
```

## LeRobot Integration

Two CLI entry points are provided for instrumenting LeRobot policy servers with Interlatent activation capture.

### Async inference (gRPC policy server)

Drop-in replacement for lerobot's `PolicyServer`. Hooks into the policy after it loads and captures activations on every forward pass. Uploads all data on client disconnect or shutdown.

```bash
interlatent-rollout \
    --host=0.0.0.0 \
    --port=8080 \
    --fps=30 \
    --layer=auto \
    --api-key="ilat_..." \
    --env-name="my-robot-env" \
    --model-name="my-policy"    # optional — auto-derived from policy path
```

Then run lerobot's robot client as normal:

```bash
python -m lerobot.async_inference.robot_client \
    --robot.type=so100_follower \
    --server_address=HOST:8080
```

### Sync inference (local rollout)

Replaces lerobot's teleop loop — runs policy inference locally on the robot with Interlatent instrumentation:

```bash
interlatent-sync-rollout \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --robot.id=black \
    --pretrained_name_or_path=user/model \
    --policy_type=smolvla \
    --task="pick up the cube" \
    --fps=30 \
    --layer=auto \
    --api-key="ilat_..."
```

The environment and model must already exist in the Interlatent dashboard. The model name is derived from the policy path (HuggingFace repo `org/model-name` becomes `model-name`; local file uses the basename). Override with `--model-name`.

## VLA Layer Inspection

The `vla` module provides a utility for discovering hookable layers in vision-language-action models:

```python
from interlatent.vla.hook_vla_layers import inspect_policy

attn_layers, linear_layers = inspect_policy(
    policy,
    max_attn_layers=10,
    max_linear_layers=10,
    verbose=True,
)

# Each layer has: full_name, module, kind ("attention"/"linear"), metadata
for layer in attn_layers:
    print(f"{layer.full_name}: heads={layer.num_heads}, dim={layer.embed_dim}")
```

This is an inspection utility — it is not integrated into `watch()` and does not collect activations on its own.

## HTTP Resources

The client also exposes HTTP resource objects for direct API access:

```python
client = Interlatent(api_key="ilat_...")

# Environments
envs = client.environments.list()
env = client.environments.create(slug="ant-v5", display_name="Ant-v5")
status = client.environments.processing_status("ant-v5")

# Episodes
episode = client.episodes.retrieve("episode-id")
status = client.episodes.status("episode-id")
results = client.episodes.results("episode-id")
meta = client.episodes.meta("episode-id")
chunk = client.episodes.chunk("episode-id", 0)
```

| Resource | Methods |
|----------|---------|
| `client.environments` | `list()`, `get()`, `create()`, `episodes()`, `process()`, `processing_status()`, `cancel_processing()`, `analyze()` |
| `client.episodes` | `retrieve()`, `create()`, `upload_urls()`, `upload_complete()`, `gc_inbox()`, `status()`, `results()`, `wait()`, `meta()`, `chunk()` |

> The `Model`, `Run`, latents, checkpoint, and analysis-report resources were retired when the platform moved to env-as-collection. The `client.index` / `client.auth` resources and the per-frame `episodes.frame()` / `episodes.update()` methods were removed when their backend routes were retired (auth is Auth0 + API keys; media is served per-camera via the dashboard). Use the environment- and episode-scoped resources above.

## Running the Demo Script

The repository includes a full end-to-end demo at `scripts/demo_processing.py`:

```bash
# Install dependencies
pip install interlatent stable-baselines3 gymnasium box2d-py

# Run with the hosted API
python scripts/demo_processing.py --api-key "ilat_..."

# Customize training and collection
python scripts/demo_processing.py \
    --api-key "ilat_..." \
    --train-steps 50000 \
    --collect-steps 5000 \
    --sae-k 64

# Skip training and load a saved model
python scripts/demo_processing.py \
    --skip-train \
    --model-path models/lunarlander.zip \
    --api-key "ilat_..."
```

The demo trains a PPO agent on LunarLander-v3, collects data, uploads to the server, triggers the analysis pipeline, polls until completion, and prints the results including the dashboard URL.

## Environment Management

Create and configure environments programmatically:

```python
client.create_environment(
    env_id="my-robot-env",
    slug="my-robot-env",
    display_name="My Robot Environment",
    robot_type="so100",
    num_cameras=2,
    camera_names=["front", "wrist"],
    action_dim=7,
    observation_keys=["observation.state"],
    task_description="Pick and place task",
    preset=None,
    notes=None,
    environment_type="robotics",
)
```

## Context Manager

The client supports context manager usage for automatic cleanup:

```python
with Interlatent(api_key="...") as client:
    client.watch(model, env, environment="my-policy")
    # ... collect data ...
    client.checkpoint()
# client.close() called automatically
```
