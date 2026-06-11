"""
CollectionEnv: a RslRlVecEnvWrapper subclass that hooks Interlatent trajectory
and activation collection into a mjlab environment.

Two usage modes:

    Mode 1 – standalone collection (drives its own rollout loop):

        from interlatent import Interlatent

        client = Interlatent(api_key="...")
        env = ManagerBasedRlEnv(cfg=env_cfg)
        col_env = CollectionEnv(env, interlatent_client=client)
        runner = MjlabOnPolicyRunner(col_env, asdict(agent_cfg), device=device)
        runner.load(checkpoint_path, load_cfg={"actor": True}, strict=True)
        client.watch(runner.alg.actor, env, environment="g1-velocity", layer="auto")

        result = col_env.collect(
            actor_model=runner.alg.actor,
            steps=2000,
        )

    Mode 2 – passive collection (hooks into an existing training/eval loop):

        client = Interlatent(api_key="...")
        env = ManagerBasedRlEnv(cfg=env_cfg)
        col_env = CollectionEnv(env, interlatent_client=client)
        runner = MjlabCollectingOnPolicyRunner(col_env, asdict(agent_cfg), device=device)
        client.watch(runner.alg.actor, env, environment="g1-velocity", layer="auto")

        with col_env.collecting(runner.alg.actor) as run_id:
            runner.learn(max_iterations=500)
        print("run:", run_id)

Notes
-----
- Both modes track env index 0 only. For multi-env collection run several
  independent processes with separate DB files and merge them.
- Mode 1 requires num_envs=1 (or at most env 0 is observed). Mode 2 works
  with any num_envs but still records env 0 for trajectory context.
- The actor_model can be pre-registered via attach() instead of passing it
  to collect() / collecting() every time.
- The Interlatent client manages local DB creation and platform upload.
  Pass db_path= to the Interlatent constructor to control the DB location.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional, Sequence

import inspect

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from interlatent._metrics import LambdaMetric
from interlatent._metrics import obs as _obs_metric

_LOG = logging.getLogger(__name__)


class CollectionEnv(RslRlVecEnvWrapper):
    """RslRlVecEnvWrapper that hooks Interlatent trajectory and activation collection.

    Drop-in replacement for RslRlVecEnvWrapper.  All runner / PPO machinery sees
    the standard VecEnv interface; collection is layered on top without changing
    the training contract.

    Parameters
    ----------
    env:
        The unwrapped ManagerBasedRlEnv.
    interlatent_client:
        An ``Interlatent`` SDK client instance.  The client owns DB creation,
        hook management, and platform upload.  Construct it with ``db_path=``
        to control where the local SQLite file is written.
    max_layers:
        Maximum number of actor layers to auto-detect for hooking.
    actor_obs_key:
        Key used to extract the actor observation from the TensorDict returned
        by the vec-env (default: ``"actor"``).
    clip_actions:
        Forwarded unchanged to RslRlVecEnvWrapper.
    env_name:
        Human-readable environment / task identifier written to the run record
        on the Interlatent platform (e.g. ``"Mjlab-Velocity-Flat-Unitree-G1"``).
    failure_rules:
        Optional dict mapping rule names to compilable boolean expressions,
        e.g. ``{"fell_over": "base_height < 0.3", "low_reward": "episode_reward < -10"}``.
        Compiled into a ``FailureTaxonomy`` and evaluated at collection time.
    success_rules:
        Optional dict mapping success rule names to compilable boolean expressions.
    """

    def __init__(
        self,
        env: ManagerBasedRlEnv,
        *,
        interlatent_client: Any,
        max_layers: int = 3,
        actor_obs_key: str = "actor",
        clip_actions: float | None = None,
        env_name: str = "Unknown",
        failure_rules: dict[str, str] | None = None,
        success_rules: dict[str, str] | None = None,
    ) -> None:
        super().__init__(env, clip_actions=clip_actions)

        self._il: Any = interlatent_client
        self._max_layers = max_layers
        self._actor_obs_key = actor_obs_key
        self._env_name = env_name

        # Failure / success rules (compiled into a FailureTaxonomy at watch time)
        self._failure_rules = failure_rules
        self._success_rules = success_rules

        self._obs_labels: list[LambdaMetric] | None = None
        self._action_labels: list[str] | None = None
        self._last_action_np: np.ndarray | None = None

        # Set by attach() or passed directly to collect() / collecting()
        self._actor_model: torch.nn.Module | None = None

        # ── Mode 2 state ────────────────────────────────────────────────
        self._collecting: bool = False
        self._active_run_id: str = ""
        self._global_step: int = 0
        self._episode_count: int = 0

        # ── Episode callbacks ────────────────────────────────────────────
        self._episode_end_callbacks: list[Callable[[int], None]] = []

    # ------------------------------------------------------------------
    # wandb.watch() equivalent
    # ------------------------------------------------------------------

    def attach(self, actor_model: torch.nn.Module) -> None:
        """Pre-register the actor model so collect() / collecting() need no argument.

        Equivalent to ``wandb.watch(model)`` – call once after runner is built.

        Example
        -------
        >>> col_env.attach(runner.alg.actor)
        >>> col_env.collect(steps=2000)
        """
        self._actor_model = actor_model

    def on_episode_end(self, callback: Callable[[int], None]) -> None:
        """Register a callback invoked at the end of each episode.

        The callback receives the zero-based index of the episode that just
        ended.  Fires regardless of whether passive collection is active.

        Example
        -------
        >>> env.on_episode_end(lambda ep: print(f"Episode {ep} done"))
        """
        self._episode_end_callbacks.append(callback)

    # ------------------------------------------------------------------
    # Reward inspection
    # ------------------------------------------------------------------

    def inspect_rewards(self, env_idx: int = 0) -> dict[str, dict[str, Any]]:
        """Return a snapshot of the current RewardManager state."""
        rm = self.unwrapped.reward_manager
        step_values = {name: vals[0] for name, vals in rm.get_active_iterable_terms(env_idx)}
        result = {}
        for name in rm.active_terms:
            cfg = rm.get_term_cfg(name)
            try:
                src = inspect.getsource(cfg.func)
            except (TypeError, OSError):
                src = None
            result[name] = {
                "weight": cfg.weight,
                "func": repr(cfg.func),
                "src": src,
                "last_step_value": step_values.get(name),
            }
        return result

    def reward_config_json(self, env_idx: int = 0) -> dict[str, Any]:
        """Serialize the current RewardManager state as a JSON-ready dict."""
        terms = self.inspect_rewards(env_idx=env_idx)
        return {
            "type": "reward_config",
            "env_name": self._env_name,
            "num_terms": len(terms),
            "terms": {
                name: {
                    "weight": info["weight"],
                    "func": info["func"],
                    "last_step_value": info["last_step_value"],
                    "src": str(info["src"]) if info["src"] is not None else None,
                }
                for name, info in terms.items()
            },
        }

    # ------------------------------------------------------------------
    # Mode 1 – standalone collection
    # ------------------------------------------------------------------

    def collect(
        self,
        *,
        steps: int,
        actor_model: torch.nn.Module | None = None,
        policy_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        context_fn: Optional[Callable[..., Dict[str, Any]]] = None,
        tags: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict:
        """Drive a standalone rollout loop, collecting trajectories and activations.

        Hooks the actor via ``client.watch()``, drives the environment loop,
        and calls ``client.upload()`` at the end to push raw data to the
        server.

        Parameters
        ----------
        steps:
            Total environment steps to collect.
        actor_model:
            Actor nn.Module to hook.  Falls back to the model set by attach().
        policy_fn:
            ``obs_np -> action_np`` callable.  If *None*, a default wrapper
            around *actor_model* is built automatically.
        context_fn:
            Optional ``(**step_kwargs) -> dict`` forwarded to ``watch()`` as
            ``context_fn=`` for extra per-step context written alongside
            activations.
        tags:
            Arbitrary key-value metadata passed to ``upload()``.
        **kwargs:
            Accepted for API compatibility; not forwarded.

        Returns
        -------
        dict
            Summary with ``episode_id``, ``steps``, ``env_name``.
        """
        model = self._resolve_model(actor_model)
        if policy_fn is None:
            policy_fn = _make_rsl_policy_fn(model, self._actor_obs_key, self.device)

        self._il.watch(
            model,
            env_name=self._env_name,
            context_fn=context_fn or self._make_obs_context_fn(),
            metrics=self._obs_labels,
            capture_frames=True,
            frame_quality=100,
        )

        self._episode_count = 0
        shim = _GymShim(self, actor_obs_key=self._actor_obs_key)
        obs, _ = shim.reset()
        for _ in range(steps):
            action = policy_fn(obs)
            self._last_action_np = action
            obs, reward, done, truncated, info = shim.step(action)
            self._il.tick(
                obs=obs,
                reward=float(reward),
                done=bool(done),
                truncated=bool(truncated),
                info=info,
                frame=self._render_frame(),
            )
            if done or truncated:
                self._episode_count += 1
                obs, _ = shim.reset()

        # Upload raw data to the server
        episode_id = self._il.episode_id or ""
        self._active_run_id = episode_id
        try:
            self._il.upload(
                tags={k: str(v) for k, v in (tags or {}).items()} or None,
                reward_config=self.reward_config_json(env_idx=0),
            )
        except Exception as exc:
            _LOG.warning("Interlatent upload failed (non-fatal): %s", exc)

        return {
            "episode_id": episode_id,
            "steps": steps,
            "env_name": self._env_name,
        }

    # ------------------------------------------------------------------
    # Mode 2 – passive collection (hooks into an external loop)
    # ------------------------------------------------------------------

    @contextmanager
    def collecting(
        self,
        actor_model: torch.nn.Module | None = None,
        *,
        tags: Dict[str, Any] | None = None,
    ):
        """Context manager that hooks collection into an existing loop.

        Hooks the actor via ``client.watch()``, then yields the episode UUID.
        Each ``env.step()`` call automatically triggers ``client.tick()`` to
        record trajectory context.  Calls ``client.upload()`` at context exit.

        Usage
        -----
        >>> with col_env.collecting(runner.alg.actor) as episode_id:
        ...     runner.learn(max_iterations=500)
        >>> print("episode:", episode_id)

        Parameters
        ----------
        actor_model:
            Actor nn.Module to hook.  Falls back to the model set by attach().
        tags:
            Arbitrary key-value metadata passed to ``upload()``.

        Yields
        ------
        str
            The episode UUID (useful for downstream pipeline calls).
        """
        model = self._resolve_model(actor_model)

        watcher = self._il.watch(
            model,
            env_name=self._env_name,
            context_fn=self._make_obs_context_fn(),
            capture_frames=True,
            frame_quality=100,
            metrics=self._obs_labels,
        )

        self._active_run_id = self._il.episode_id or ""
        self._collecting = True
        self._global_step = 0
        self._episode_count = 0

        try:
            yield self._active_run_id
        finally:
            self._collecting = False
            try:
                self._il.upload(
                    tags={k: str(v) for k, v in (tags or {}).items()} or None,
                    reward_config=self.reward_config_json(env_idx=0),
                )
            except Exception as exc:
                _LOG.warning("Interlatent upload failed (non-fatal): %s", exc)

    # ------------------------------------------------------------------
    # Overrides – intercept step for Mode 2 context tracking
    # ------------------------------------------------------------------

    def step(self, actions: torch.Tensor) -> tuple:
        obs_td, rew, dones, extras = super().step(actions)

        if self._collecting:
            time_outs = extras.get(
                "time_outs", torch.zeros(self.num_envs, device=self.device)
            )
            obs = obs_td[self._actor_obs_key][0].cpu().numpy()
            done = bool(dones[0].item())
            truncated = bool(time_outs[0].item())
            self._last_action_np = actions[0].cpu().numpy()

            frame = self._render_frame()

            self._il.tick(
                obs=obs,
                reward=float(rew[0].item()),
                done=done,
                truncated=truncated,
                frame=frame,
            )

            self._global_step += 1
            if done or truncated:
                for cb in self._episode_end_callbacks:
                    cb(self._episode_count)
                self._episode_count += 1

        return obs_td, rew, dones, extras

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_obs_labels(self) -> list[LambdaMetric]:
        """Build a flat list of LambdaMetric objects from the observation manager.

        For each term in the actor obs group, generates one metric per scalar:
        e.g. a ``base_lin_vel`` term of shape (3,) becomes metrics named
        ``["base_lin_vel_0", "base_lin_vel_1", "base_lin_vel_2"]`` each
        extracting the corresponding index from the obs array.
        """
        if self._obs_labels is not None:
            return self._obs_labels
        obs_manager = self.unwrapped.observation_manager
        group = self._actor_obs_key
        if group not in obs_manager.active_terms:
            self._obs_labels = []
            return self._obs_labels
        term_names = obs_manager.active_terms[group]
        term_dims = obs_manager.group_obs_term_dim.get(group, [])
        metrics: list[LambdaMetric] = []
        flat_idx = 0
        for name, dim in zip(term_names, term_dims, strict=False):
            n = int(np.prod(dim))
            if n == 1:
                metrics.append(_obs_metric(name, flat_idx))
            else:
                for i in range(n):
                    metrics.append(_obs_metric(f"{name}_{i}", flat_idx + i))
            flat_idx += n
        self._obs_labels = metrics
        return metrics

    def _build_action_labels(self) -> list[str]:
        """Build a flat list of action labels from the action manager."""
        if self._action_labels is not None:
            return self._action_labels
        action_manager = self.unwrapped.action_manager
        labels: list[str] = []
        for term_name, term in action_manager._terms.items():
            target_names = getattr(term, "target_names", None)
            if target_names:
                for tgt in target_names:
                    labels.append(f"{term_name}/{tgt}")
            else:
                for i in range(term.action_dim):
                    labels.append(f"{term_name}_{i}")
        self._action_labels = labels
        return labels

    def _make_obs_context_fn(
        self,
    ) -> Callable[..., Dict[str, Any]] | None:
        """Return a context_fn that maps obs and action vectors to named fields."""
        obs_labels = self._build_obs_labels()
        act_labels = self._build_action_labels()
        if not obs_labels and not act_labels:
            return None

        def _fn(*, obs: np.ndarray | None = None, **_kwargs: Any) -> Dict[str, Any]:
            result: Dict[str, Any] = {}
            if obs is not None and obs_labels:
                result["observations"] = {
                    m.name: float(obs[i])
                    for i, m in enumerate(obs_labels)
                    if i < len(obs)
                }
            action_np = self._last_action_np
            if action_np is not None and act_labels:
                result["actions"] = {
                    label: float(action_np[i])
                    for i, label in enumerate(act_labels)
                    if i < len(action_np)
                }
            return result

        return _fn

    def _resolve_model(self, actor_model: torch.nn.Module | None) -> torch.nn.Module:
        model = actor_model or self._actor_model
        if model is None:
            raise RuntimeError(
                "No actor model available.  Either pass actor_model= or call "
                "col_env.attach(runner.alg.actor) after building the runner."
            )
        return model

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    def _render_frame(self) -> Any:
        """Call the underlying env's ``render()`` and log its output once.

        Catches the silent-None failure mode where the mjlab env was built
        without an offscreen render config: ``render()`` returns ``None``,
        ``_client.tick()`` drops the frame on its ``is not None`` guard, and
        the run completes successfully with zero frames buffered.

        Fires exactly one print per ``CollectionEnv`` instance, so it is safe
        to call on every step. Uses explicit ``is None`` checks rather than
        ``not frame`` because numpy arrays with more than one element raise
        ``ValueError`` on boolean evaluation.
        """
        frame = self.unwrapped.render()
        if not getattr(self, "_rendered_once", False):
            shape = getattr(frame, "shape", None)
            dtype = getattr(frame, "dtype", None)
            print(
                f"[mjlab:debug] first render() -> type={type(frame).__name__} "
                f"shape={shape} dtype={dtype}"
            )
            if frame is None:
                print(
                    "[mjlab:debug] WARNING: env.render() returned None. "
                    "Frames will NOT be captured or uploaded. "
                    "Check that the mjlab env was built with a render config / "
                    "camera, and that MUJOCO_GL is set (e.g. MUJOCO_GL=egl)."
                )
            self._rendered_once = True
        return frame


# ---------------------------------------------------------------------------
# Internal gym shim (Mode 1 helper)
# ---------------------------------------------------------------------------


class _GymShim:
    """Adapt a batched CollectionEnv to the single-env gym interface used in
    Mode 1 (standalone collection).

    Squeezes the batch dimension (env index 0) and converts tensors to numpy.
    """

    def __init__(self, vec_env: CollectionEnv, *, actor_obs_key: str = "actor") -> None:
        self._env = vec_env
        self._actor_obs_key = actor_obs_key
        # Expose action_space so callers can inspect it if needed.
        self.action_space = vec_env.unwrapped.action_space
        self.spec = None

    def _obs_to_array(self, obs_td) -> np.ndarray:
        return obs_td[self._actor_obs_key].squeeze(0).cpu().numpy()

    def reset(self) -> tuple[np.ndarray, dict]:
        obs_td, extras = self._env.reset()
        return self._obs_to_array(obs_td), extras

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action_t = (
            torch.as_tensor(action, dtype=torch.float32)
            .unsqueeze(0)
            .to(self._env.device)
        )
        obs_td, rew, dones, extras = self._env.step(action_t)
        obs = self._obs_to_array(obs_td)
        reward = float(rew[0].item())
        done = bool(dones[0].item())
        truncated = bool(
            extras.get("time_outs", torch.zeros(1))[0].item()
        )
        return obs, reward, done, truncated, extras

    def render(self) -> None:
        return None

    def close(self) -> None:
        self._env.close()


# ---------------------------------------------------------------------------
# Policy factory (Mode 1 default)
# ---------------------------------------------------------------------------


def _make_rsl_policy_fn(
    actor_model: torch.nn.Module,
    actor_obs_key: str,
    device: torch.device,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a numpy-obs -> numpy-action policy function from an RSL-RL actor.

    Passes the observation tensor directly to the actor (MLPModel), which applies
    its own obs normalizer internally.  A TensorDict must NOT be used here because
    the RSL-RL MLPModel expects a plain tensor, not a TensorDict.
    """

    def policy_fn(obs_np: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action_t = actor_model(obs_t)
        return action_t.squeeze(0).cpu().numpy()

    return policy_fn
