from __future__ import annotations

from typing import Any

from ._http import HTTPClient


class EnvironmentsResource:
    """Resource for environment CRUD against a coordinator.

    Environments are keyed by ``env_id`` — either the UUID id or the
    user-scoped slug; the coordinator resolves both.
    """

    def __init__(self, http: HTTPClient) -> None:
        self._http = http

    def list(self) -> list[dict[str, Any]]:
        return self._http.request("GET", "/api/v1/environments")

    def get(self, env_id: str) -> dict[str, Any]:
        """Fetch an environment's config from the coordinator.

        ``env_id`` accepts either the UUID id or the user-scoped slug;
        the coordinator resolves both.
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
