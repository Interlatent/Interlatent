"""Headless step recorder for user-driven training/inference loops.

After the 2026-05 cleanup the watcher no longer hooks the policy — it
just records per-step ``observation``, ``action``, ``reward``, ``done``,
``truncated``, and any user-supplied context into the local SQLite
staging cache. The upload-time LeRobot rebuild reads from this cache.
"""
from __future__ import annotations

import uuid
from typing import Any, Callable, Dict, List

from ._db import CollectionDB
from ._schema import ActivationEvent, EpisodeInfo


# Sentinel "layer" written for every step. The row carries the full
# per-step context dict. The upload-time rebuild keys on this layer.
_CTX_LAYER = "_ctx"


def _flatten(value) -> List[float]:
    if value is None:
        return []
    try:
        import numpy as np
        arr = np.asarray(value).reshape(-1).astype(float)
        return arr.tolist()
    except Exception:
        if hasattr(value, "__iter__"):
            try:
                return [float(v) for v in value]
            except Exception:
                return []
        try:
            return [float(value)]
        except Exception:
            return []


class Watcher:
    """Records per-step context into a ``CollectionDB``."""

    def __init__(
        self,
        model=None,
        *,
        env_name: str,
        db: CollectionDB,
        metrics: List | None = None,
        context_fn: Callable[..., Dict[str, Any]] | None = None,
        total_steps: int | None = None,
        run_id: str | None = None,
        episode_id: str | None = None,
    ) -> None:
        self._env_name = env_name
        self._db = db
        self._metrics = {m.name: m for m in (metrics or [])}
        self._context_fn = context_fn
        self._total_steps = total_steps

        self._run_id = run_id or str(uuid.uuid4())
        self._step = 0
        self._episode_id = episode_id or str(uuid.uuid4())
        self._all_episode_ids: List[str] = [self._episode_id]
        self._episode_step = 0
        self._episode_reward_acc = 0.0

        self._step_ctx: Dict[str, Any] = {}
        self._started = False

        self._episode_info = EpisodeInfo(
            episode_id=self._episode_id,
            env_name=self._env_name,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def start_time(self) -> str:
        return self._episode_info.start_time

    @property
    def episode_id(self) -> str:
        return self._episode_id

    @property
    def all_episode_ids(self) -> List[str]:
        return list(self._all_episode_ids)

    @property
    def step(self) -> int:
        return self._step

    @property
    def db(self) -> CollectionDB:
        return self._db

    def start(self) -> "Watcher":
        self._started = True
        return self

    def tick(
        self,
        *,
        obs,
        action,
        reward: float = 0.0,
        done: bool = False,
        truncated: bool = False,
        info: dict | None = None,
        control_source: str | None = None,
    ) -> None:
        """Record one environment step.

        ``obs`` is the state *before* ``action`` was taken (``s_t``).
        Both vectors are flattened to ``list[float]`` and become the
        canonical ``observation.state`` / ``action`` columns of the
        LeRobot dataset at upload time.
        """
        metric_vals: Dict[str, float | None] = {}
        for m in self._metrics.values():
            val = m.step(
                obs=obs, reward=reward, info=info,
                done=done, truncated=truncated,
            )
            metric_vals[m.name] = val

        self._episode_reward_acc += float(reward)

        ctx: Dict[str, Any] = {
            "env_id": self._env_name,
            "episode_id": self._episode_id,
            "t": self._episode_step,
            "step": self._step,
            "reward": float(reward),
            "done": bool(done),
            "truncated": bool(truncated),
            "metrics": metric_vals,
            "observation": _flatten(obs),
            "action": _flatten(action),
        }
        if control_source is not None:
            ctx["control_source"] = str(control_source)

        if self._context_fn is not None:
            extra = self._context_fn(
                obs=obs, reward=reward, done=done,
                truncated=truncated, info=info,
            )
            if extra:
                ctx.update(extra)

        self._step_ctx.clear()
        self._step_ctx.update(ctx)

        self._db.write_event(ActivationEvent(
            episode_id=self._episode_id,
            step=self._step,
            layer=_CTX_LAYER,
            channel=0,
            tensor=[],
            context=ctx,
        ))

        self._step += 1
        self._episode_step += 1

        if done or truncated:
            self.reset_episode()

    def reset_episode(self) -> None:
        for m in self._metrics.values():
            m.reset()
        self._episode_id = str(uuid.uuid4())
        self._all_episode_ids.append(self._episode_id)
        self._episode_step = 0
        self._episode_reward_acc = 0.0

    def reset_session(self) -> None:
        """Start a fresh upload session after a successful upload."""
        self.reset_episode()
        self._all_episode_ids = [self._episode_id]
        self._episode_info = EpisodeInfo(
            episode_id=self._episode_id,
            env_name=self._env_name,
        )

    def stop(self) -> None:
        if not self._started:
            return
        self._db.flush()
        self._started = False

    def close(self) -> None:
        self.stop()
