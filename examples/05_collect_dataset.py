"""Collect a LeRobot v3.0 dataset locally — no account, no upload.

Drives a gym environment with a random policy, records every step
through the SDK's watch()/tick() staging path, then builds a real
LeRobot dataset on disk that loads with `lerobot.LeRobotDataset` (and
uploads to the Hugging Face Hub with lerobot's own tooling, if you
want).

Run:
    pip install 'interlatent[lerobot]' gymnasium
    python examples/05_collect_dataset.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import gymnasium as gym

from interlatent import Interlatent
from interlatent._dataset import LeRobotRebuilder

DB_PATH = "./interlatent_staging_demo.db"
DATASET_ROOT = Path("./datasets/cartpole-demo")
EPISODES = 5


def main() -> None:
    # Offline client: no api_key, nothing leaves this machine.
    client = Interlatent(db_path=DB_PATH)
    env = gym.make("CartPole-v1")

    # watch() binds the session and starts the local recorder. The
    # `environment` slug is just a label here — with an api_key it would
    # route uploads to that dashboard environment.
    watcher = client.watch(None, env, environment="cartpole-local")

    for ep in range(EPISODES):
        obs, _ = env.reset(seed=ep)
        done = truncated = False
        steps = 0
        while not (done or truncated):
            action = env.action_space.sample()
            next_obs, reward, done, truncated, info = env.step(action)
            client.tick(obs=obs, action=action, reward=reward,
                        done=done, truncated=truncated, info=info)
            obs = next_obs
            steps += 1
        print(f"episode {ep}: {steps} steps")

    # Flush the staging cache, then build the LeRobot dataset from it.
    watcher.stop()
    if DATASET_ROOT.exists():
        shutil.rmtree(DATASET_ROOT)  # LeRobotDataset.create requires a fresh dir
    rebuilder = LeRobotRebuilder(
        DATASET_ROOT,
        fps=30,
        task="keep the pole upright",
        env_slug="cartpole-local",
    )
    root, episode_uuids = rebuilder.build_from_staging(db_path=DB_PATH, media=None)

    print(f"\nLeRobot dataset written to {root}")
    print(f"episodes: {len(episode_uuids)}")
    print("\nLoad it like any LeRobot dataset:")
    print(f'  LeRobotDataset("interlatent/cartpole-local", root="{root}")')


if __name__ == "__main__":
    main()
