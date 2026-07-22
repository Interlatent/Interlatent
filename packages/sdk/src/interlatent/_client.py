from __future__ import annotations

import os
import warnings
from typing import NoReturn


from ._http import HTTPClient
from ._resources import (
    EnvironmentsResource,
    EpisodesResource,
)

# Collection is streaming-first and server-side (ADR 0022): a robot node
# streams JPEG RecordTicks to a hosted recorder (the DRTC GPU box, or the
# teleop recorder pod), which builds the LeRobot dataset and uploads it
# through the inbox→merge path. The old client-side path — stage steps +
# JPEGs locally, build a LeRobot dataset on-device, upload() it — was
# product-deprecated and has been removed; the verbs below raise with a
# pointer instead of silently half-working.
_REMOVED_COLLECTION_MSG = (
    "Interlatent.{name}() has been removed: client-side/local collection is "
    "deprecated (see the SDK docs and ADR 0022 'collection is streaming-"
    "first'). Record episodes through a hosted session instead — run the "
    "node daemon (`interlatent node run`) against an inference session or "
    "teleop recording; the server builds and uploads the dataset."
)


def _get_sdk_version() -> str:
    try:
        from importlib import metadata
        return metadata.version("interlatent")
    except Exception:
        return "0.0.dev"


class Interlatent:
    """SDK client for the hosted Interlatent API.

    Example:
        client = Interlatent(api_key="...")
        eps = client.environments.episodes("ant-v5")

    Episode *collection* is not done through this class anymore — a robot
    node streams ticks to a hosted recorder (see the node daemon /
    ``connect_drtc``); this client is the HTTP surface (environments,
    episodes, routing).
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
        if db_path is not None:
            warnings.warn(
                "Interlatent(db_path=...) is ignored: the local staging "
                "cache was removed with client-side collection (ADR 0022).",
                DeprecationWarning,
                stacklevel=2,
            )
        del db_path, fps  # accepted for signature compatibility only
        self._api_key = api_key
        self._base_url = base_url or os.environ.get("INTERLATENT_API_BASE") or self._BASE_URL
        self._http = HTTPClient(
            base_url=self._base_url,
            api_key=api_key,
            timeout=timeout,
        )

        # Resource groups
        self.environments = EnvironmentsResource(self._http)
        self.episodes = EpisodesResource(self._http)

        self._env_name: str = "Unknown"
        self._env_slug: str | None = None  # backend env slug for routing
        self._env_config: dict | None = None  # cached backend env config

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

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

        Returns the created environment config dict.
        """
        resolved = self._env_name if self._env_name and self._env_name != "Unknown" else None
        resolved = resolved or env_id
        if resolved is None:
            raise ValueError("env_id is required.")

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
    # Removed collection API (ADR 0022) — stubs for one release, so old
    # integrations fail loudly with a pointer instead of an AttributeError.
    # ------------------------------------------------------------------

    def _removed(self, name: str) -> NoReturn:
        raise RuntimeError(_REMOVED_COLLECTION_MSG.format(name=name))

    def watch(self, *args, **kwargs):
        self._removed("watch")

    def collect(self, *args, **kwargs):
        self._removed("collect")

    def tick(self, *args, **kwargs):
        self._removed("tick")

    def add_frame(self, *args, **kwargs):
        self._removed("add_frame")

    def checkpoint(self, *args, **kwargs):
        self._removed("checkpoint")

    def upload(self, *args, **kwargs):
        self._removed("upload")

    def register_cameras(self, *args, **kwargs):
        self._removed("register_cameras")

    def sb3_callback(self, *args, **kwargs):
        self._removed("sb3_callback")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "Interlatent":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
