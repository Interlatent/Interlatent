from __future__ import annotations

import time
from typing import Any

from ._exceptions import APIError, AuthenticationError, NotFoundError


class HTTPClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        bypass_token: str | None = None,
        timeout: float = 30.0,
        session=None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key or ""
        self._bypass_token = bypass_token or ""
        self._timeout = timeout
        if session is None:
            try:
                import requests
            except ImportError as e:
                raise ImportError(
                    "The public Interlatent SDK requires 'requests'. Install with: pip install interlatent"
                ) from e
            session = requests.Session()
        self._session = session

    @property
    def base_url(self) -> str:
        return self._base_url

    def close(self) -> None:
        self._session.close()

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: Any | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> Any:
        url = f"{self._base_url}{path}"
        req_headers: dict[str, str] = {"Accept": "application/json"}
        if self._api_key:
            req_headers["x-api-key"] = self._api_key
        if self._bypass_token:
            req_headers["x-vercel-protection-bypass"] = self._bypass_token
        if headers:
            req_headers.update(headers)

        max_retries = 3
        for attempt in range(max_retries):
            resp = self._session.request(
                method=method.upper(),
                url=url,
                params=params,
                json=json_body,
                headers=req_headers,
                timeout=timeout or self._timeout,
            )
            if resp.status_code < 500 or attempt == max_retries - 1:
                return self._handle_response(resp)
            time.sleep(5)

    @staticmethod
    def _handle_response(resp) -> Any:
        content_type = resp.headers.get("content-type", "")
        body: Any
        if "application/json" in content_type:
            try:
                body = resp.json()
            except Exception:
                body = resp.text
        else:
            body = resp.content

        if 200 <= resp.status_code < 300:
            return body

        detail = None
        if isinstance(body, dict):
            detail = body.get("detail") or body.get("message")
        message = detail or f"HTTP {resp.status_code}"

        exc_cls = APIError
        if resp.status_code in (401, 403):
            exc_cls = AuthenticationError
        elif resp.status_code == 404:
            exc_cls = NotFoundError
        raise exc_cls(message, status_code=resp.status_code, body=body)
