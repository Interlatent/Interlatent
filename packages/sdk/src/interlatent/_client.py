from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict

if TYPE_CHECKING:
    from ._media import MediaBuffer

from ._exceptions import APIError
from ._http import HTTPClient
from ._resources import (
    EnvironmentsResource,
    EpisodesResource,
)


def _get_sdk_version() -> str:
    try:
        from importlib import metadata
        return metadata.version("interlatent")
    except Exception:
        return "0.0.dev"


def _detect_framework(model) -> str | None:
    if model is None:
        return None
    cls_name = type(model).__module__
    if "stable_baselines3" in cls_name:
        return "sb3"
    try:
        import torch
        if isinstance(model, torch.nn.Module):
            return "pytorch"
    except ImportError:
        pass
    return None


class Interlatent:
    """SDK client for the hosted Interlatent API with local collection support.

    Example (HTTP-only):
        client = Interlatent(api_key="...")
        eps = client.environments.episodes("ant-v5")

    Example (collection + upload):
        client = Interlatent(api_key="...")
        client.watch(model, env, environment="ant-v5", layer="auto")
        model.learn(100_000, callback=client.sb3_callback(checkpoint_every=10_000))
        client.checkpoint()
        client.episodes.wait(client.episode_id)
        client.close()
    """

    _BASE_URL = "https://interlatent.com"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
        db_path: str | None = None,
        fps: int = 30,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url or os.environ.get("INTERLATENT_API_BASE") or self._BASE_URL
        self._http = HTTPClient(
            base_url=self._base_url,
            api_key=api_key,
            timeout=timeout,
        )
        self._fps = int(fps)

        # Resource groups
        self.environments = EnvironmentsResource(self._http)
        self.episodes = EpisodesResource(self._http)

        # Collection state (lazy-init). The local SQLite at ``db_path`` is a
        # transient staging cache only — it is never uploaded directly. At
        # ``upload()`` we read it once to build a LeRobot dataset, then wipe.
        self._db_path_override = db_path
        self._db = None
        self._watcher = None
        self._metrics: list | None = None
        self._env_name: str = "Unknown"
        self._env_slug: str | None = None  # backend env slug for routing
        self._env_config: dict | None = None  # cached backend env config
        self._task: str | None = None
        self._model_ref = None
        self._checkpoint_count = 0

        # Media / frame state (lazy-init)
        self._media: "MediaBuffer | None" = None
        self._frame_every: int = 1
        self._frame_quality: int = 85
        self._camera_names: list[str] | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def db_path(self) -> str | None:
        if self._db is not None:
            return str(self._db._store._path)
        return self._db_path_override

    @property
    def episode_id(self) -> str | None:
        if self._watcher is not None:
            return self._watcher.episode_id
        return None

    @property
    def step_count(self) -> int:
        if self._watcher is not None:
            return self._watcher.step
        return 0

    @property
    def environment(self) -> str | None:
        """Backend environment slug this client is bound to."""
        return self._env_slug

    # ------------------------------------------------------------------
    # Backend config helpers
    # ------------------------------------------------------------------

    def _fetch_env_config(self) -> dict | None:
        """Fetch and cache the environment config from the backend."""
        if self._env_config is not None:
            return self._env_config
        if self._env_slug is None:
            return None
        try:
            self._env_config = self.environments.get(self._env_slug)
            return self._env_config
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Environment Management
    # ------------------------------------------------------------------

    def create_environment(
        self,
        *,
        env_id: str | None = None,
        slug: str | None = None,
        display_name: str | None = None,
        robot_type: str | None = None,
        num_cameras: int | None = None,
        camera_names: list[str] | None = None,
        action_dim: int | None = None,
        observation_keys: list[str] | None = None,
        task_description: str | None = None,
        preset: str | None = None,
        notes: str | None = None,
        environment_type: str | None = None,
    ) -> dict:
        """Create a new environment on the Interlatent platform.

        Uses ``self._env_name`` if set, otherwise falls back to the
        provided ``env_id``.  Raises ``ValueError`` if neither is available.

        Returns the created environment config dict.
        """
        resolved = self._env_name if self._env_name and self._env_name != "Unknown" else None
        resolved = resolved or env_id
        if resolved is None:
            raise ValueError(
                "env_id is required. Either pass env_id= or set an environment "
                "name via watch() before calling create_environment()."
            )

        resolved_slug = slug or resolved.lower().replace(" ", "-")
        resolved_display = display_name or resolved

        return self.environments.create(
            slug=resolved_slug,
            display_name=resolved_display,
            robot_type=robot_type,
            num_cameras=num_cameras,
            camera_names=camera_names,
            action_dim=action_dim,
            observation_keys=observation_keys,
            task_description=task_description,
            preset=preset,
            notes=notes,
            environment_type=environment_type,
        )

    # ------------------------------------------------------------------
    # Camera registration
    # ------------------------------------------------------------------

    def register_cameras(self, camera_names: list[str]) -> None:
        """Register camera names for multicamera frame capture.

        When cameras are registered, ``tick(frame={cam: image})`` stores
        each image with the camera name embedded in the filename
        (``frame_{cam}_{step}.jpg``).
        """
        self._camera_names = list(camera_names)

    # ------------------------------------------------------------------
    # Collection API
    # ------------------------------------------------------------------

    def watch(
        self,
        model=None,
        env=None,
        *,
        environment: str | None = None,
        env_name: str | None = None,
        task: str | None = None,
        metrics: list | None = None,
        context_fn: Callable[..., Dict[str, Any]] | None = None,
        total_steps: int | None = None,
        capture_frames: bool = False,
        frame_every: int = 1,
        frame_quality: int = 85,
        frame_dir: str | None = None,
        episode_id: str | None = None,
    ):
        """Hook a model and start capturing activations.

        Args:
            model: The policy / nn.Module to instrument.
            env: Optional gym-style environment (used for auto metrics and
                taxonomy resolution; the runtime env, not the backend
                environment).
            environment: Backend environment slug (or id) to attach this
                collection to. Required. The environment must already
                exist on the dashboard.
            env_name: Optional override for the human-readable env name
                stamped into the staging records. Defaults to the slug
                inferred from ``env``.

        The watcher records observations / actions / rewards / metrics
        into the local staging cache. Activation hooks were removed in
        the 2026-05 cleanup.

        Returns the Watcher instance (already started).
        """
        from ._metrics import extract_env_id, auto_metrics
        from ._watcher import Watcher

        if environment is not None:
            self._env_slug = environment.lower().replace(" ", "-")
        elif self._env_slug is None:
            raise ValueError(
                "environment= is required for watch(). Pass the backend "
                "environment slug (e.g. environment=\"spot-velocity\") so "
                "uploads can route to the right dashboard env."
            )

        if env_name is not None:
            self._env_name = env_name
        elif env is not None:
            self._env_name = extract_env_id(env)
        else:
            self._env_name = self._env_slug or "Unknown"

        self._task = task or self._env_name or "rollout"

        if metrics is None and env is not None:
            try:
                metrics = auto_metrics(env)
            except Exception:
                metrics = None
        self._metrics = metrics

        self._model_ref = model
        self._ensure_db()

        if frame_dir is not None:
            from ._media import MediaBuffer
            self._media = MediaBuffer(
                tempfile.mkdtemp(prefix="interlatent_media_"),
                frames_dir=frame_dir,
            )
            self._frame_every = frame_every
            self._frame_quality = frame_quality
        elif capture_frames:
            from ._media import MediaBuffer
            self._media = MediaBuffer(tempfile.mkdtemp(prefix="interlatent_media_"))
            self._frame_every = frame_every
            self._frame_quality = frame_quality

        self._watcher = Watcher(
            model,
            env_name=self._env_name,
            db=self._db,
            metrics=metrics,
            context_fn=context_fn,
            total_steps=total_steps,
            episode_id=episode_id,
        )
        self._watcher.start()
        return self._watcher

    def collect(
        self,
        model,
        env,
        *,
        steps: int = 5000,
        task: str | None = None,
        metrics: list | None = None,
        context_fn: Callable[..., Dict[str, Any]] | None = None,
        deterministic: bool = True,
        capture_frames: bool = False,
        frame_every: int = 1,
        frame_quality: int = 85,
    ) -> dict:
        """Drive ``model`` through ``env`` for ``steps`` and record each step.

        Returns a summary dict with episode_id and step count.
        """
        from ._metrics import auto_metrics, extract_env_id

        if self._watcher is not None:
            self._watcher.stop()

        self._env_name = extract_env_id(env)
        self._task = task or self._task or self._env_name or "rollout"

        if metrics is None:
            metrics = auto_metrics(env)
        self._metrics = metrics

        self._model_ref = model
        self._ensure_db()

        # Initialise frame capture
        if capture_frames:
            from ._media import MediaBuffer
            self._media = MediaBuffer(tempfile.mkdtemp(prefix="interlatent_media_"))
            self._frame_every = frame_every
            self._frame_quality = frame_quality

        # Create a watcher for the collection run
        from ._watcher import Watcher

        watcher = Watcher(
            model,
            env_name=self._env_name,
            db=self._db,
            metrics=metrics,
            context_fn=context_fn,
        )
        watcher.start()

        # Drive the env loop
        obs, _ = env.reset()
        for step_i in range(steps):
            # SB3 predict or torch forward
            if hasattr(model, "predict"):
                action, _ = model.predict(obs, deterministic=deterministic)
            else:
                import torch
                with torch.no_grad():
                    action = model(torch.as_tensor(obs).float().unsqueeze(0))
                    if hasattr(action, "numpy"):
                        action = action.squeeze(0).numpy()

            next_obs, reward, done, truncated, info = env.step(action)

            # Auto-capture frame
            if (
                self._media is not None
                and self._frame_every > 0
                and step_i % self._frame_every == 0
            ):
                try:
                    frame = env.render()
                    if frame is not None:
                        self._media.add_frame(
                            step_i, frame,
                            episode_id=watcher.episode_id,
                            quality=self._frame_quality,
                        )
                except Exception as exc:
                    if step_i == 0:
                        import logging
                        logging.getLogger(__name__).warning(
                            "Frame capture failed: %s. "
                            "Ensure env was created with render_mode='rgb_array' "
                            "and rendering dependencies are installed.",
                            exc,
                        )

            watcher.tick(
                obs=obs, action=action, reward=float(reward),
                done=bool(done), truncated=bool(truncated), info=info,
            )

            obs = next_obs

            if done or truncated:
                obs, _ = env.reset()

        watcher.stop()

        return {
            "episode_id": watcher.episode_id,
            "steps": steps,
            "env_name": self._env_name,
            "start_time": watcher.start_time,
        }

    def tick(
        self,
        *,
        obs,
        action,
        reward: float = 0.0,
        done: bool = False,
        truncated: bool = False,
        info: dict | None = None,
        frame=None,
    ) -> None:
        """Record one environment step.

        ``obs`` and ``action`` are required — they become the
        ``observation.state`` and ``action`` columns of the LeRobot
        dataset at upload time and downstream Q(s, a) post-training
        cannot recover from null values.

        Args:
            obs: Environment observation. Numpy array / list / scalar.
                Flattened to a 1-D float32 vector at upload time.
            action: Action commanded for this step. Same shape conventions
                as ``obs``.
            reward: Scalar step reward.
            done, truncated: Gymnasium episode-end flags.
            info: Optional info dict from the env.
            frame: Optional rendered frame. Can be:
                - A single image (numpy array, PIL Image, or path)
                - A dict mapping camera names to images for multicamera
                Saved to the media buffer if one is active.
        """
        if self._watcher is None:
            raise RuntimeError("No active watcher. Call watch() first.")
        if frame is not None and self._media is not None:
            step = self._watcher.step
            episode_id = self._watcher.episode_id
            if isinstance(frame, dict):
                for cam_name, cam_image in frame.items():
                    self._media.add_frame(
                        step, cam_image,
                        episode_id=episode_id,
                        camera_name=cam_name,
                        quality=self._frame_quality,
                    )
            else:
                self._media.add_frame(
                    step, frame,
                    episode_id=episode_id,
                    quality=self._frame_quality,
                )
        self._watcher.tick(
            obs=obs, action=action, reward=reward, done=done,
            truncated=truncated, info=info,
        )

    def add_frame(
        self, step: int, image, *,
        episode_id: str | None = None,
        camera_name: str | None = None,
        quality: int = 85,
    ) -> None:
        """Manually add a frame to the media buffer.

        Lazy-initialises the MediaBuffer if it doesn't exist yet.

        Args:
            camera_name: Optional camera identifier for multicamera setups.
        """
        if self._media is None:
            from ._media import MediaBuffer
            self._media = MediaBuffer(tempfile.mkdtemp(prefix="interlatent_media_"))
        if episode_id is None and self._watcher is not None:
            episode_id = self._watcher.episode_id
        self._media.add_frame(
            step, image,
            episode_id=episode_id,
            camera_name=camera_name,
            quality=quality,
        )

    def checkpoint(
        self,
        *,
        label: str = "",
    ) -> dict:
        """Upload the current staging cache and trigger server-side analysis.

        Returns a summary dict with ``environment`` / ``job_id`` / ``status``.
        """
        if self._env_slug is None:
            raise RuntimeError(
                "environment is required for processing. Bind one via "
                "watch(environment=...) before calling checkpoint()."
            )

        step_count = self.step_count
        self._checkpoint_count += 1

        # Upload the staged data as a fresh inbox session.
        self.upload(label=label)

        job_data = self.environments.process(self._env_slug)

        return {
            "environment": self._env_slug,
            "job_id": job_data.get("job_id", ""),
            "status": job_data.get("status", "pending"),
            "checkpoint_count": self._checkpoint_count,
            "step_count": step_count,
        }

    def upload(
        self,
        *,
        tags: dict[str, str] | None = None,
        label: str = "",
        workers: int = 8,
        reward_config: dict[str, Any] | None = None,
    ) -> None:
        """Build a LeRobot dataset from the staging cache and upload it.

        End-to-end:

        1. Drain hook buffers + flush the SQLite staging cache so all
           in-flight inference work has landed on disk.
        2. Build a complete LeRobot v3.0 dataset on disk (in a tempdir)
           via ``LeRobotRebuilder``.
        3. Walk the dataset directory, request presigned PUT URLs for each
           file, upload them under the environment's ``_inbox/<session_uuid>/``
           prefix on S3.
        4. Register each episode UUID with the backend and call
           ``upload-complete`` so the server-side merge picks it up.
        5. On confirmed success: delete the local SQLite cache, the
           LeRobot tempdir, and the frame staging buffer.
        6. On failure: leave local state intact so the user can retry,
           but best-effort GC the partial inbox prefix on S3.

        ``label`` and ``tags`` are forwarded to ``episodes.create`` for
        per-episode metadata. ``reward_config`` is forwarded through the
        same path. Uploads are session-scoped.
        """
        import logging
        import requests
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from ._dataset import LeRobotRebuilder

        log = logging.getLogger(__name__)

        if self._env_slug is None:
            raise RuntimeError(
                "environment is required for upload. Bind one via "
                "watch(environment=...) before calling upload()."
            )

        # 1. Flush SQLite so the rebuild sees a consistent snapshot.
        if self._db is not None:
            self._db.flush()

        if self._db is None or self.db_path is None or not Path(self.db_path).exists():
            log.warning("upload() called with no staging cache — nothing to do")
            return

        db_path = self.db_path

        # 2. Build the LeRobot dataset locally. ``LeRobotDataset.create``
        #    requires its root to NOT exist (it does an mkdir(exist_ok=False)
        #    internally), so we let mkdtemp create a unique *parent* and
        #    nest the dataset one level inside.
        session_uuid = uuid.uuid4().hex
        parent_dir = Path(
            tempfile.mkdtemp(prefix=f"interlatent_dataset_{session_uuid}_")
        )
        dataset_root = parent_dir / "dataset"
        rebuilder = LeRobotRebuilder(
            root=dataset_root,
            fps=self._fps,
            task=self._task or self._env_name or "rollout",
            env_slug=self._env_slug or self._env_name,
        )

        try:
            _root, episode_uuids = rebuilder.build_from_staging(
                db_path=db_path,
                media=self._media,
            )
        except Exception as e:
            print("[ERROR] Failed to build LeRobot dataset from staging cache:", e)
            rebuilder.cleanup()
            raise

        if not episode_uuids:
            log.warning("upload() found no episodes in the staging cache — nothing to do")
            rebuilder.cleanup()
            return

        # 3. Register every episode with the backend.
        common_create_kwargs: dict[str, Any] = dict(
            environment=self._env_slug or self._env_name,
            collect_steps=None,
            tags=tags or {},
            sdk_version=_get_sdk_version(),
            model_framework=_detect_framework(self._model_ref),
            reward_config=reward_config,
        )
        for eid in episode_uuids:
            try:
                self.episodes.create(episode_id=eid, **common_create_kwargs)
            except APIError as exc:
                # 409 = already registered (e.g. user retrying after a partial
                # failure). Tolerate; continue with upload.
                if exc.status_code != 409:
                    raise

        # 4. Walk the dataset directory and build the upload manifest.
        manifest: list[dict[str, Any]] = []
        for f in sorted(dataset_root.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(dataset_root).as_posix()
            key = f"_inbox/{session_uuid}/{rel}"
            manifest.append({
                "key": key,
                "local": str(f),
                "size": f.stat().st_size,
            })

        if not manifest:
            log.warning("upload(): LeRobot dataset built but contains no files; aborting")
            rebuilder.cleanup()
            return

        anchor_eid = episode_uuids[0]

        def _put_file(local_path: str, put_url: str) -> None:
            with open(local_path, "rb") as fh:
                resp = requests.put(put_url, data=fh, timeout=300)
                if not resp.ok:
                    raise RuntimeError(
                        f"PUT failed for {local_path}: {resp.status_code} {resp.text[:200]}"
                    )

        def _upload_batch(files: list[dict[str, Any]], pool: ThreadPoolExecutor) -> None:
            key_to_local = {f["key"]: f["local"] for f in files}
            batch_size = 100
            for i in range(0, len(files), batch_size):
                batch = files[i : i + batch_size]
                url_resp = self.episodes.upload_urls(
                    anchor_eid,
                    files=[{"key": f["key"], "size": f["size"]} for f in batch],
                )
                presigned = url_resp.get("presigned_urls", {})
                futures = []
                for key, put_url in presigned.items():
                    local = key_to_local.get(key)
                    if local:
                        futures.append(pool.submit(_put_file, local, put_url))
                for fut in as_completed(futures):
                    fut.result()

        upload_succeeded = False
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                _upload_batch(manifest, pool)

            # 5. Notify upload-complete per episode (server enqueues the merge).
            for eid in episode_uuids:
                self.episodes.upload_complete(eid)

            upload_succeeded = True
        finally:
            if upload_succeeded:
                self._cleanup_after_upload(rebuilder)
            else:
                self._gc_partial_inbox(anchor_eid, session_uuid, manifest)
                rebuilder.cleanup()

    def _cleanup_after_upload(self, rebuilder) -> None:
        """Wipe local staging on a successful upload (Q11)."""
        # Drop the LeRobot tempdir.
        rebuilder.cleanup()

        # Wipe the SQLite staging cache.
        db_path_str = self.db_path
        if self._db is not None:
            self._db.close()
            self._db = None
        if db_path_str:
            try:
                Path(db_path_str).unlink(missing_ok=True)
            except OSError:
                pass
        self._db_path_override = None

        # Wipe the JPEG frame buffer.
        if self._media is not None:
            self._media.cleanup()
            self._media = None

        # Reset watcher session state so subsequent ticks belong to a
        # fresh inbox session.
        if self._watcher is not None and hasattr(self._watcher, "reset_session"):
            self._watcher.reset_session()

    def _gc_partial_inbox(
        self,
        anchor_eid: str,
        session_uuid: str,
        manifest: list[dict[str, Any]],
    ) -> None:
        """Best-effort cleanup of an inbox prefix after a failed upload (Q11).

        Uses the same presigned-URL path as the upload (DELETE) — if the
        server doesn't expose a presigned-DELETE we just log and move on.
        Local state is preserved either way so the user can retry.
        """
        import logging
        log = logging.getLogger(__name__)
        try:
            self.episodes.gc_inbox(anchor_eid, session_uuid=session_uuid)
        except Exception:
            log.warning(
                "Failed to GC inbox prefix _inbox/%s/ — orphaned files may "
                "remain in S3. Local staging preserved for retry.",
                session_uuid,
                exc_info=True,
            )

    def sb3_callback(self, *, checkpoint_every: int = 10_000):
        """Return an SB3 BaseCallback that auto-feeds the watcher.

        Args:
            checkpoint_every: Run checkpoint() every N training steps.
                Set to 0 to disable auto-checkpointing.
        """
        client = self

        class _SB3Callback:
            """Minimal SB3-compatible callback for activation collection."""

            def __init__(self):
                self.n_calls = 0
                self.num_timesteps = 0
                # These are set by SB3 during init_callback
                self.model = None
                self.training_env = None
                self.logger = None

            def init_callback(self, model) -> None:
                self.model = model

            def _on_step(self) -> bool:
                self.n_calls += 1
                self.num_timesteps += 1

                # Get step data from SB3's locals
                if client._watcher is not None:
                    # SB3 provides these via the training env
                    try:
                        from stable_baselines3.common.vec_env import VecEnv  # noqa: F401  (availability probe)
                        env = getattr(self.model, "env", None)
                        if env is not None:
                            # VecEnv: get info from buffer
                            obs = getattr(self, "_last_obs", None)
                            if obs is None and hasattr(self.model, "_last_obs"):
                                obs = self.model._last_obs
                                if obs is not None and hasattr(obs, "__len__") and len(obs) > 0:
                                    obs = obs[0]
                    except ImportError:
                        pass

                if (
                    checkpoint_every > 0
                    and self.n_calls % checkpoint_every == 0
                    and client._watcher is not None
                ):
                    try:
                        client.checkpoint()
                    except Exception:
                        pass

                return True

            # SB3 BaseCallback protocol
            def on_step(self) -> bool:
                return self._on_step()

            def on_training_start(self, locals_dict, globals_dict) -> None:
                pass

            def on_training_end(self) -> None:
                pass

            def on_rollout_start(self) -> None:
                pass

            def on_rollout_end(self) -> None:
                pass

        return _SB3Callback()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._watcher is not None:
            self._watcher.close()
            self._watcher = None
        if self._db is not None:
            self._db.close()
            self._db = None
        if self._media is not None:
            self._media.cleanup()
            self._media = None
        self._http.close()

    def __enter__(self) -> "Interlatent":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_db(self) -> None:
        """Lazily create the CollectionDB on first use."""
        if self._db is not None:
            return
        from ._db import CollectionDB

        path = self._db_path_override
        if path is None:
            path = f"interlatent_{uuid.uuid4().hex[:8]}.db"
            self._db_path_override = path
        self._db = CollectionDB(path)
