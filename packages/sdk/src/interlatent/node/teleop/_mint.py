"""Shared teleop-token mint call.

Both the WS channel and the QUIC channel POST the same node-role token
endpoint; the response now also carries ``transport`` + ``webtransport_url``
so a factory can pick the path. Kept transport-free (httpx only) so importing
it never drags in aioquic.
"""
from __future__ import annotations

from typing import Optional


def mint_teleop_token(
    *,
    api_base: str,
    token_path: str,
    api_key: str,
    bypass_key: Optional[str] = None,
    role: str = "node",
    timeout: float = 10.0,
) -> dict:
    """POST the teleop-token endpoint and return the full response JSON.

    Raises on any non-2xx. The response includes ``token``, ``transport``, and
    ``webtransport_url`` (teleop runs over QUIC/WebTransport).
    """
    import httpx

    url = f"{api_base.rstrip('/')}{token_path}"
    headers = {"x-api-key": api_key}
    if (bypass_key or "").strip():
        # Protected preview deployments challenge un-bypassed requests.
        headers["x-vercel-protection-bypass"] = bypass_key.strip()
    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, params={"role": role}, headers=headers)
        r.raise_for_status()
        data = r.json()
    if not isinstance(data, dict):
        raise RuntimeError("teleop-token: non-object response")
    return data
