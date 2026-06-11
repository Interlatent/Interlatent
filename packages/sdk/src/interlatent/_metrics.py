"""Metric protocol and auto-detection for RL environments (pure Python)."""
from __future__ import annotations

from collections import deque
from typing import Callable, Optional, Protocol


class Metric(Protocol):
    """A metric produces one scalar per timestep and can reset at episode-end."""

    name: str

    def reset(self) -> None: ...
    def step(self, *, obs, reward, info, done=None, truncated=None) -> Optional[float]: ...


class LambdaMetric:
    """Wrap any callable into a Metric."""

    def __init__(self, name: str, fn: Callable[..., float | None]):
        self.name, self._fn = name, fn

    def reset(self) -> None:
        pass

    def step(self, *, obs, reward, info, done=None, truncated=None):
        return self._fn(obs=obs, reward=reward, info=info, done=done, truncated=truncated)


class EpisodeAccumulator:
    """Accumulates a per-step value over an episode."""

    def __init__(self, name: str, fn: Callable[..., float]):
        self.name, self._fn = name, fn
        self._acc = 0.0

    def reset(self) -> None:
        self._acc = 0.0

    def step(self, *, obs, reward, info, done=None, truncated=None):
        self._acc += self._fn(obs=obs, reward=reward, info=info, done=done, truncated=truncated)
        return self._acc


class StuckMetric:
    """Flags when a signal stays below a threshold for a rolling window."""

    def __init__(self, name: str, fn: Callable[..., float], window: int = 50, threshold: float = 0.05):
        self.name = name
        self._fn = fn
        self._window = window
        self._threshold = threshold
        self._buf: deque = deque(maxlen=window)

    def reset(self) -> None:
        self._buf.clear()

    def step(self, *, obs, reward, info, done=None, truncated=None):
        val = float(self._fn(obs=obs, reward=reward, info=info, done=done, truncated=truncated))
        self._buf.append(val)
        if len(self._buf) < self._window:
            return 0.0
        mean_abs = sum(abs(v) for v in self._buf) / len(self._buf)
        return 1.0 if mean_abs < self._threshold else 0.0


# ---------------------------------------------------------------------------
# Builder / factory helpers
# ---------------------------------------------------------------------------


def obs(name: str, index: int) -> LambdaMetric:
    """Extract a single observation dimension as a metric."""
    return LambdaMetric(name, lambda *, obs, _i=index, **_: float(obs[_i]))


def obs_range(start: int, end: int, prefix: str = "obs") -> list:
    """Batch-create observation metrics for indices ``[start, end)``."""
    return [obs(f"{prefix}_{i}", i) for i in range(start, end)]


def reward_total(name: str = "episode_reward") -> EpisodeAccumulator:
    """Accumulated episode reward."""
    return EpisodeAccumulator(name, lambda *, reward, **_: float(reward))


def reward_flag(name: str, *, above: float | None = None, below: float | None = None) -> LambdaMetric:
    def fn(*, reward, done=None, **_):
        if not done:
            return 0.0
        if above is not None and reward >= above:
            return 1.0
        if below is not None and reward < below:
            return 1.0
        return 0.0
    return LambdaMetric(name, fn)


def obs_flag(
    name: str,
    index: int,
    *,
    above: float | None = None,
    below: float | None = None,
    when_done: bool = False,
) -> LambdaMetric:
    def fn(*, obs, done=None, truncated=None, _i=index, **_):
        if when_done and not (done or truncated):
            return 0.0
        v = float(obs[_i])
        if above is not None and v >= above:
            return 1.0
        if below is not None and v < below:
            return 1.0
        return 0.0
    return LambdaMetric(name, fn)


def info_val(name: str, key: str | None = None, default: float = 0.0) -> LambdaMetric:
    k = key or name
    return LambdaMetric(
        name,
        lambda *, info=None, _k=k, _d=default, **_: float(info.get(_k, _d)) if info and isinstance(info, dict) else _d,
    )


def done_flag(name: str = "done") -> LambdaMetric:
    return LambdaMetric(name, lambda *, done=None, truncated=None, **_: 1.0 if (done or truncated) else 0.0)


def ant_fallen(obs, **_) -> float:
    try:
        return 1.0 if float(obs[0]) < 0.2 else 0.0
    except Exception:
        return 0.0


def ant_backward(*, info=None, **_) -> float:
    if info and isinstance(info, dict):
        try:
            return 1.0 if float(info.get("x_velocity", 0.0)) < -0.1 else 0.0
        except Exception:
            return 0.0
    return 0.0


def auto_metrics(env) -> list:
    """Return appropriate metrics based on the environment's spec ID."""
    env_id = extract_env_id(env)

    if "Ant" in env_id:
        return [
            LambdaMetric("fallen", ant_fallen),
            LambdaMetric("backward", ant_backward),
            StuckMetric(
                "stuck",
                lambda *, info=None, **_: float(
                    info.get("x_velocity", 0.0) if info and isinstance(info, dict) else 0.0
                ),
            ),
        ]

    if "LunarLander" in env_id:
        return [
            obs("x_position", 0),
            obs("y_position", 1),
            obs("x_velocity", 2),
            obs("y_velocity", 3),
            obs("angle", 4),
            obs("angular_velocity", 5),
            obs("left_leg_contact", 6),
            obs("right_leg_contact", 7),
            reward_total(),
            reward_flag("crashed", below=-50),
            reward_flag("success", above=100),
        ]

    if "CartPole" in env_id:
        return [
            obs("pole_angle", 2),
            obs("cart_position", 0),
            obs("cart_velocity", 1),
            obs("pole_angular_velocity", 3),
            reward_total(),
            done_flag("failed"),
        ]

    if "MountainCar" in env_id:
        return [
            obs("position", 0),
            obs("velocity", 1),
            reward_total(),
            obs_flag("reached_goal", 0, above=0.5, when_done=True),
        ]

    if "Acrobot" in env_id:
        return [
            obs("cos_theta1", 0),
            obs("sin_theta1", 1),
            obs("cos_theta2", 2),
            obs("sin_theta2", 3),
            obs("angular_vel_1", 4),
            obs("angular_vel_2", 5),
            reward_total(),
        ]

    if "HalfCheetah" in env_id:
        return [
            obs("z_position", 0),
            obs("torso_angle", 1),
            obs("back_thigh_angle", 2),
            obs("back_shin_angle", 3),
            obs("back_foot_angle", 4),
            obs("front_thigh_angle", 5),
            obs("front_shin_angle", 6),
            obs("front_foot_angle", 7),
            obs("x_velocity", 8),
            obs("z_velocity", 9),
            obs("back_thigh_vel", 10),
            obs("back_shin_vel", 11),
            obs("back_foot_vel", 12),
            obs("front_thigh_vel", 13),
            obs("front_shin_vel", 14),
            obs("front_foot_vel", 15),
            obs("torso_angular_vel", 16),
            reward_total(),
        ]

    # Generic fallback
    metrics: list = [
        reward_total(),
        reward_flag("low_reward", below=-1.0),
    ]
    obs_dim = _get_obs_dim(env)
    metrics.extend(obs_range(0, min(obs_dim, 8)))
    return metrics


def _get_obs_dim(env) -> int:
    try:
        obs_space = getattr(env, "observation_space", None)
        if obs_space is not None and hasattr(obs_space, "shape") and obs_space.shape:
            return int(obs_space.shape[0])
    except Exception:
        pass
    return 0


def extract_env_id(env) -> str:
    """Best-effort extraction of environment identifier string."""
    spec = getattr(env, "spec", None)
    if spec is not None:
        env_id = getattr(spec, "id", None)
        if env_id:
            return str(env_id)
    return type(env).__name__


__all__ = [
    "Metric", "LambdaMetric", "EpisodeAccumulator", "StuckMetric",
    "ant_fallen", "ant_backward", "auto_metrics", "extract_env_id",
    "obs", "obs_range", "reward_total", "reward_flag",
    "obs_flag", "info_val", "done_flag",
]
