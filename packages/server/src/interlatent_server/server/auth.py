"""Interlatent API-key validation for the DRTC server.

Imported by the production launcher (:mod:`interlatent_server.serve_gpu`)
to guard the public-facing gRPC endpoint. On a self-hosted box the guard
is ON by default with the *owner-scoped* check
(:func:`build_box_key_validator`): a presented ``x-api-key`` passes only
if the backend confirms it belongs to this box's owner. The weaker
"any valid Interlatent key" check (:func:`build_api_key_validator`)
survives for tests and custom frontings.

Each RPC checks the ``x-api-key`` metadata, validates against the
Interlatent backend, and caches the result in-process.

The Interlatent backend treats API keys as ``X-Api-Key`` HTTP headers
(see ``site/app/deps.py``). The owner check probes
``GET /api/v1/compute/boxes/{box_id}/authz`` (200 iff the key may drive
this box); the any-key check probes ``/environments`` because it uses
``require_auth`` (accepts API keys). Bodies are discarded either way.
"""

from __future__ import annotations

import time
from typing import Callable

DEFAULT_API_BASE = "https://interlatent.com/api/v1"
DEFAULT_TTL_S = 60.0


def validate_api_key(token: str, *, api_base: str = DEFAULT_API_BASE) -> bool:
    """One-shot validation. Returns True iff the backend accepts the key."""
    if not token:
        return False
    import httpx

    base = api_base.rstrip("/")
    if not base.endswith("/api/v1"):
        base = f"{base}/api/v1"
    try:
        r = httpx.get(
            f"{base}/environments",
            headers={"X-Api-Key": token},
            timeout=5.0,
        )
        return r.status_code == 200
    except httpx.HTTPError:
        return False


def build_api_key_validator(
    api_base: str = DEFAULT_API_BASE,
    ttl_s: float = DEFAULT_TTL_S,
) -> Callable[[str], bool]:
    """Return a stateful `check(token)` with in-process LRU+TTL cache.

    Cache lives for the process lifetime — the DRTC server is a long-
    running asyncio process, so warm cache hits are the steady state.
    The 60-second TTL means an active client pays one backend
    roundtrip per minute regardless of inference rate.
    """
    cache: dict[str, tuple[float, bool]] = {}

    def check(token: str) -> bool:
        if not token:
            return False
        now = time.time()
        hit = cache.get(token)
        if hit and (now - hit[0]) < ttl_s:
            return hit[1]
        ok = validate_api_key(token, api_base=api_base)
        cache[token] = (now, ok)
        if len(cache) > 1024:
            cutoff = now - ttl_s
            for k, (t, _) in list(cache.items()):
                if t < cutoff:
                    cache.pop(k, None)
        return ok

    return check


def validate_key_for_box(
    token: str, *, box_id: str, api_base: str = DEFAULT_API_BASE
) -> bool:
    """One-shot owner check: True iff the backend confirms `token` may
    drive box `box_id` (i.e. it is the box owner's key, or the system
    key). 403/401/404 and network failures are all False."""
    if not token:
        return False
    import httpx

    base = api_base.rstrip("/")
    if not base.endswith("/api/v1"):
        base = f"{base}/api/v1"
    try:
        r = httpx.get(
            f"{base}/compute/boxes/{box_id}/authz",
            headers={"X-Api-Key": token},
            timeout=5.0,
        )
        return r.status_code == 200
    except httpx.HTTPError:
        return False


def build_box_key_validator(
    *,
    box_id: str,
    api_base: str = DEFAULT_API_BASE,
    ttl_s: float = DEFAULT_TTL_S,
) -> Callable[[str], bool]:
    """Owner-scoped `check(token)` with the same in-process LRU+TTL cache
    as :func:`build_api_key_validator`. The steady state is one backend
    roundtrip per presented key per minute."""
    cache: dict[str, tuple[float, bool]] = {}

    def check(token: str) -> bool:
        if not token:
            return False
        now = time.time()
        hit = cache.get(token)
        if hit and (now - hit[0]) < ttl_s:
            return hit[1]
        ok = validate_key_for_box(token, box_id=box_id, api_base=api_base)
        cache[token] = (now, ok)
        if len(cache) > 1024:
            cutoff = now - ttl_s
            for k, (t, _) in list(cache.items()):
                if t < cutoff:
                    cache.pop(k, None)
        return ok

    return check


def wrap_servicer_with_auth(servicer, *, check_token: Callable[[str], bool]):
    """Replace each RPC method on `servicer` with an auth-gated
    version. The first action of every RPC becomes a check on the
    `x-api-key` metadata; on failure the call is aborted with
    UNAUTHENTICATED. On success the original method runs."""
    import grpc

    # RecordTick/RecordTicks are unary RPCs like the rest — leaving them
    # unguarded would let anyone stream ticks into this box's recorder.
    rpc_names = (
        "OpenSession",
        "CloseSession",
        "Infer",
        "Stream",
        "RecordTick",
        "RecordTicks",
    )

    def _token_from(context) -> str:
        md = dict(context.invocation_metadata() or [])
        return md.get("x-api-key", "").strip()

    for name in rpc_names:
        original = getattr(servicer, name, None)
        if original is None:
            continue

        if name == "Stream":
            async def _guarded_stream(request_iterator, context, _orig=original):
                if not check_token(_token_from(context)):
                    await context.abort(
                        grpc.StatusCode.UNAUTHENTICATED,
                        "missing or invalid Interlatent API key",
                    )
                async for resp in _orig(request_iterator, context):
                    yield resp
            setattr(servicer, name, _guarded_stream)
        else:
            async def _guarded_unary(request, context, _orig=original):
                if not check_token(_token_from(context)):
                    await context.abort(
                        grpc.StatusCode.UNAUTHENTICATED,
                        "missing or invalid Interlatent API key",
                    )
                return await _orig(request, context)
            setattr(servicer, name, _guarded_unary)

    return servicer
