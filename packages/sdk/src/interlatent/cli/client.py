"""Tiny stdlib HTTP client for the coordinator's ``/admin/*`` API."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Optional


class CoordinatorError(RuntimeError):
    pass


class CoordinatorClient:
    def __init__(self, base_url: str) -> None:
        self.base = base_url.rstrip("/")

    def _req(self, method: str, path: str, body: Optional[dict] = None, *, timeout: float = 30.0) -> Any:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            self.base + path, data=data, method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = json.loads(e.read() or b"{}").get("error", "")
            except Exception:
                pass
            raise CoordinatorError(detail or f"HTTP {e.code} {e.reason}")
        except urllib.error.URLError as e:
            raise CoordinatorError(f"cannot reach coordinator at {self.base} ({e.reason})")

    # -- gpus --
    def list_gpus(self) -> list[dict]:
        return self._req("GET", "/admin/gpus")["gpus"]

    def add_gpu(self, name: str, url: str, method: str = "direct") -> dict:
        return self._req("POST", "/admin/gpus", {"name": name, "url": url, "method": method})

    def remove_gpu(self, name: str) -> None:
        self._req("DELETE", f"/admin/gpus/{name}")

    # -- nodes --
    def list_nodes(self) -> list[dict]:
        return self._req("GET", "/admin/nodes")["nodes"]

    def remove_node(self, node_id: str) -> None:
        self._req("DELETE", f"/admin/nodes/{node_id}")

    # -- sessions --
    def list_sessions(self) -> list[dict]:
        return self._req("GET", "/admin/sessions")["sessions"]

    def start_session(self, params: dict) -> dict:
        return self._req("POST", "/admin/sessions", params)

    def stop_session(self, session_id: str) -> None:
        self._req("DELETE", f"/admin/sessions/{session_id}")

    # -- destination --
    def get_destination(self) -> dict:
        return self._req("GET", "/admin/destination")["recording"]

    def set_destination(self, recording: dict) -> dict:
        return self._req("POST", "/admin/destination", {"recording": recording})

    def ping(self, timeout: float = 1.0) -> bool:
        try:
            self._req("GET", "/admin/gpus", timeout=timeout)
            return True
        except CoordinatorError:
            return False
