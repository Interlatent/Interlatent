from __future__ import annotations

import time
from typing import Any

from ._http import HTTPClient


class EnvironmentsResource:
    """Resource for environment CRUD and env-scoped processing/analysis.

    After the model_id retirement, every operation that used to be keyed
    by model_id is now keyed by env_id. The model concept is fully gone
    from the public SDK surface.
    """

    def __init__(self, http: HTTPClient) -> None:
        self._http = http

    def list(self) -> list[dict[str, Any]]:
        return self._http.request("GET", "/api/v1/environments")

    def get(self, env_id: str) -> dict[str, Any]:
        """Fetch an environment's config from the backend.

        ``env_id`` accepts either the UUID id or the user-scoped slug;
        the backend resolves both.
        """
        return self._http.request("GET", f"/api/v1/environments/{env_id}/config")

    def create(
        self,
        *,
        slug: str,
        display_name: str,
        robot_type: str | None = None,
        num_cameras: int | None = None,
        camera_names: list[str] | None = None,
        action_dim: int | None = None,
        observation_keys: list[str] | None = None,
        task_description: str | None = None,
        preset: str | None = None,
        notes: str | None = None,
        environment_type: str | None = None,
        failure_cases: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "slug": slug,
            "display_name": display_name,
        }
        if robot_type is not None:
            body["robot_type"] = robot_type
        if num_cameras is not None:
            body["num_cameras"] = num_cameras
        if camera_names is not None:
            body["camera_names"] = camera_names
        if action_dim is not None:
            body["action_dim"] = action_dim
        if observation_keys is not None:
            body["observation_keys"] = observation_keys
        if task_description is not None:
            body["task_description"] = task_description
        if preset is not None:
            body["preset"] = preset
        if notes is not None:
            body["notes"] = notes
        if environment_type is not None:
            body["environment_type"] = environment_type
        if failure_cases is not None:
            body["failure_cases"] = failure_cases
        return self._http.request("POST", "/api/v1/environments", json_body=body)

    def episodes(self, env_id: str, *, limit: int = 20, offset: int = 0) -> dict[str, Any]:
        return self._http.request(
            "GET",
            f"/api/v1/environments/{env_id}/episodes",
            params={"limit": limit, "offset": offset},
        )

    # ------------------------------------------------------------------
    # Processing / analysis (env-scoped after model_id retirement)
    # ------------------------------------------------------------------

    def process(self, env_id: str) -> dict[str, Any]:
        """Trigger server-side processing for all episodes under an env."""
        return self._http.request(
            "POST",
            f"/api/v1/environments/{env_id}/process",
            json_body={},
            timeout=120,
        )

    def processing_status(self, env_id: str) -> dict[str, Any]:
        return self._http.request(
            "GET", f"/api/v1/environments/{env_id}/processing-status"
        )

    def cancel_processing(self, env_id: str) -> dict[str, Any]:
        return self._http.request(
            "POST", f"/api/v1/environments/{env_id}/cancel-processing"
        )

    def analyze(self, env_id: str, **body: Any) -> dict[str, Any]:
        return self._http.request(
            "POST",
            f"/api/v1/environments/{env_id}/analyze",
            json_body=body or None,
        )


class EpisodesResource:
    """Resource for episode CRUD, upload, and processing."""

    def __init__(self, http: HTTPClient) -> None:
        self._http = http

    def retrieve(self, episode_id: str) -> dict[str, Any]:
        return self._http.request("GET", f"/api/v1/episodes/{episode_id}")

    def create(
        self,
        *,
        episode_id: str,
        environment: str,
        base_model: str | None = None,
        model_source: str | None = None,
        created_at: str | None = None,
        policy_path: str | None = None,
        collect_steps: int | None = None,
        pipeline_status: dict[str, bool] | None = None,
        tags: dict[str, str] | None = None,
        sdk_version: str | None = None,
        model_framework: str | None = None,
        metric_definitions: list[str] | None = None,
        reward_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create / register an episode.

        ``environment`` is the env slug (or id). ``base_model`` /
        ``model_source`` are optional policy descriptors the backend
        reconciles under one-policy-per-env (first writer sets;
        disagreement → 409).
        """
        body: dict[str, Any] = {
            "episode_id": episode_id,
            "environment": environment,
            "created_at": created_at,
            "policy_path": policy_path,
            "collect_steps": collect_steps,
            "pipeline_status": pipeline_status or {},
            "tags": tags or {},
            "reward_config": reward_config,
        }
        if base_model is not None:
            body["base_model"] = base_model
        if model_source is not None:
            body["model_source"] = model_source
        if sdk_version is not None:
            body["sdk_version"] = sdk_version
        if model_framework is not None:
            body["model_framework"] = model_framework
        if metric_definitions is not None:
            body["metric_definitions"] = metric_definitions
        return self._http.request("POST", "/api/v1/episodes", json_body=body)

    def upload_urls(self, episode_id: str, *, files: list[dict[str, Any]]) -> dict[str, Any]:
        return self._http.request(
            "POST",
            f"/api/v1/episodes/{episode_id}/upload-urls",
            json_body={"files": files},
        )

    def upload_complete(self, episode_id: str, *, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._http.request(
            "POST",
            f"/api/v1/episodes/{episode_id}/upload-complete",
            json_body={"manifest": manifest},
        )

    def gc_inbox(self, episode_id: str, *, session_uuid: str) -> dict[str, Any]:
        """Best-effort cleanup of an inbox prefix after a failed upload.

        Anchors on ``episode_id`` (any episode in the session works) and
        asks the backend to delete every object under
        ``environments/{env_id}/_inbox/{session_uuid}/`` for that env.
        The SDK calls this in ``upload()``'s failure path; local staging
        is preserved on the client side so the user can retry.
        """
        return self._http.request(
            "POST",
            f"/api/v1/episodes/{episode_id}/inbox-gc",
            json_body={"session_uuid": session_uuid},
        )

    def status(self, episode_id: str) -> dict[str, Any]:
        """Get episode status including latest processing job."""
        return self._http.request("GET", f"/api/v1/episodes/{episode_id}/status")

    def results(self, episode_id: str) -> dict[str, Any]:
        """Get processing results (report, export URLs, etc.)."""
        return self._http.request("GET", f"/api/v1/episodes/{episode_id}/results")

    def wait(
        self,
        episode_id: str,
        *,
        timeout: float = 300.0,
        poll: float = 5.0,
    ) -> dict[str, Any]:
        """Poll until server-side processing is completed or failed."""
        deadline = time.monotonic() + timeout
        while True:
            data = self.status(episode_id)
            processing = data.get("processing")
            if processing:
                st = processing.get("status", "")
                if st in ("completed", "failed"):
                    return data
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Processing for episode {episode_id} did not complete "
                    f"within {timeout}s (last status: {data})"
                )
            time.sleep(poll)

    # -- Episode data (frames, chunks, meta) --

    def meta(self, episode_id: str) -> dict[str, Any]:
        return self._http.request("GET", f"/api/v1/episodes/{episode_id}/meta")

    def chunk(self, episode_id: str, chunk_index: int) -> dict[str, Any]:
        return self._http.request("GET", f"/api/v1/episodes/{episode_id}/chunks/{chunk_index}")


# Backward compat alias — old code used RunsResource
RunsResource = EpisodesResource
