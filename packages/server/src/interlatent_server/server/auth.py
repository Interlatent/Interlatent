"""Interlatent API-key validation for the DRTC server.

Imported by the production launcher (:mod:`interlatent_server.serve_gpu`)
to guard the public-facing gRPC endpoint. The guard is ON by default with
the *owner-scoped* check (:func:`build_box_key_validator`): a presented
``x-api-key`` passes only if this box's coordinator confirms the key may
drive this box. The weaker "any valid Interlatent key" check
(:func:`build_api_key_validator`) survives for tests and custom
frontings.

Each RPC checks the ``x-api-key`` metadata, validates it against the
coordinator the box registered with, and caches the result in-process.

Keys travel as ``X-Api-Key`` HTTP headers. The owner check probes
``GET /api/v1/compute/boxes/{box_id}/authz`` (200 iff the key may drive
this box); the any-key check probes ``/environments``, which any
authenticated key can read. Bodies are discarded either way.
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

# Absolute, not relative, and deliberately so: `tests/test_auth_cli.py` loads
# this file by path (with no package) to avoid importing
# `interlatent_server.server`, whose __init__ registers the policy backends and
# drags in torch. A relative import breaks that; this one only executes the
# light top-level __init__ plus a stdlib-only module.
from interlatent_server.coordinator import api_v1

DEFAULT_TTL_S = 60.0

_MAX_CACHE_ENTRIES = 1024

_auth_executor: Optional[ThreadPoolExecutor] = None
_auth_executor_lock = threading.Lock()


def _auth_probe_executor() -> ThreadPoolExecutor:
    """The pool the blocking coordinator probe runs on.

    Deliberately *not* the loop's default executor. On a GPU box that
    pool is the recording executor, pinned to the reserved cores (see
    :mod:`interlatent_server.serve_gpu`), so a 5-second auth probe parked
    there would sit on a worker the recorder needs for disk writes. Two
    workers is enough because concurrent misses on the same token are
    deduped — the ceiling is *distinct cold keys*, not RPC rate.
    """
    global _auth_executor
    with _auth_executor_lock:
        if _auth_executor is None:
            _auth_executor = ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="il-auth"
            )
        return _auth_executor


def validate_api_key(token: str, *, api_base: str) -> bool:
    """One-shot validation. True iff the coordinator accepts the key."""
    if not token:
        return False
    import httpx

    base = api_v1(api_base)
    try:
        r = httpx.get(
            f"{base}/environments",
            headers={"X-Api-Key": token},
            timeout=5.0,
        )
        return r.status_code == 200
    except httpx.HTTPError:
        return False


class CachedValidator:
    """A stateful `check(token)` with an in-process LRU+TTL cache.

    Cache lives for the process lifetime — the DRTC server is a long-
    running asyncio process, so warm cache hits are the steady state.
    The 60-second TTL means an active client pays one coordinator
    roundtrip per minute regardless of inference rate.

    Callable, so it satisfies the historical ``Callable[[str], bool]``
    contract for sync callers. Async callers must use
    :meth:`check_async`: the underlying probe is a blocking
    ``httpx.get(timeout=5.0)``, and awaiting it on the event loop would
    stall every concurrent RPC — including 30 Hz RecordTick ingest — for
    up to five seconds on each cold key. ``check_async`` serves warm
    hits inline (a dict lookup, no thread hop, no yield) and offloads
    only the miss.
    """

    def __init__(self, probe: Callable[[str], bool], ttl_s: float) -> None:
        self._probe = probe
        self._ttl_s = ttl_s
        self._cache: dict[str, tuple[float, bool]] = {}
        # token -> the single in-flight probe for it. Without this, a
        # cold key arriving at 30 Hz would fan out one blocking request
        # per tick against a 2-worker pool.
        self._inflight: dict[str, asyncio.Future] = {}

    def _peek(self, token: str, now: float) -> Optional[bool]:
        hit = self._cache.get(token)
        if hit is not None and (now - hit[0]) < self._ttl_s:
            return hit[1]
        return None

    def _store(self, token: str, now: float, ok: bool) -> None:
        self._cache[token] = (now, ok)
        if len(self._cache) > _MAX_CACHE_ENTRIES:
            cutoff = now - self._ttl_s
            for k, (t, _) in list(self._cache.items()):
                if t < cutoff:
                    self._cache.pop(k, None)

    def __call__(self, token: str) -> bool:
        """Blocking check. For sync callers only — see :meth:`check_async`."""
        if not token:
            return False
        now = time.time()
        cached = self._peek(token, now)
        if cached is not None:
            return cached
        ok = self._probe(token)
        self._store(token, time.time(), ok)
        return ok

    async def check_async(self, token: str) -> bool:
        """Non-blocking check for the asyncio server."""
        if not token:
            return False
        now = time.time()
        cached = self._peek(token, now)
        if cached is not None:
            return cached

        pending = self._inflight.get(token)
        if pending is not None:
            # Someone else is already probing this key. Shield: our
            # cancellation must not cancel the probe they're awaiting.
            return await asyncio.shield(pending)

        loop = asyncio.get_running_loop()
        pending = loop.run_in_executor(_auth_probe_executor(), self._probe, token)
        self._inflight[token] = pending
        try:
            ok = await pending
        finally:
            self._inflight.pop(token, None)
        self._store(token, time.time(), ok)
        return ok


def build_api_key_validator(
    api_base: str,
    ttl_s: float = DEFAULT_TTL_S,
) -> CachedValidator:
    """Cached "is this a valid Interlatent key?" check."""
    return CachedValidator(
        lambda token: validate_api_key(token, api_base=api_base), ttl_s
    )


def validate_key_for_box(
    token: str, *, box_id: str, api_base: str
) -> bool:
    """One-shot owner check: True iff the coordinator confirms `token`
    may drive box `box_id`. 403/401/404 and network failures are all
    False."""
    if not token:
        return False
    import httpx

    base = api_v1(api_base)
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
    api_base: str,
    ttl_s: float = DEFAULT_TTL_S,
) -> CachedValidator:
    """Owner-scoped `check(token)`, same cache as
    :func:`build_api_key_validator`. The steady state is one coordinator
    roundtrip per presented key per minute."""
    return CachedValidator(
        lambda token: validate_key_for_box(token, box_id=box_id, api_base=api_base),
        ttl_s,
    )


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

    # A CachedValidator knows how to keep its blocking probe off the
    # loop. Anything else is assumed to block too — a plain callable
    # here is almost always a coordinator roundtrip — so it gets the same
    # treatment rather than being trusted to be cheap.
    _check_async = getattr(check_token, "check_async", None)
    if _check_async is None:

        async def _check_async(token: str, _fn=check_token) -> bool:
            if not token:
                return False
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(_auth_probe_executor(), _fn, token)

    for name in rpc_names:
        original = getattr(servicer, name, None)
        if original is None:
            continue

        # grpc.aio's context.abort() raises, so the returns below are
        # unreachable in production. They are here so the contract does
        # not rest on that: a non-raising context (a test double, a
        # future grpc change) must still not reach the real handler.
        if name == "Stream":
            async def _guarded_stream(request_iterator, context, _orig=original):
                if not await _check_async(_token_from(context)):
                    await context.abort(
                        grpc.StatusCode.UNAUTHENTICATED,
                        "missing or invalid Interlatent API key",
                    )
                    return
                async for resp in _orig(request_iterator, context):
                    yield resp
            setattr(servicer, name, _guarded_stream)
        else:
            async def _guarded_unary(request, context, _orig=original):
                if not await _check_async(_token_from(context)):
                    await context.abort(
                        grpc.StatusCode.UNAUTHENTICATED,
                        "missing or invalid Interlatent API key",
                    )
                    return None
                return await _orig(request, context)
            setattr(servicer, name, _guarded_unary)

    return servicer
