"""Coordinator address resolution for the server dist.

A deliberate twin of ``interlatent._coordinator``. ADR 0023 makes
``interlatent-server`` installable on its own — a GPU box does not have the SDK
— so it cannot import that module, and a shared dist just to hold thirty lines
would undo the split. The two must stay in agreement; the pair is small enough
that agreement is inspectable, and ``tests/test_coordinator_resolution.py``
asserts they behave identically.

One convention: a coordinator address is a **bare origin**. Callers append
``/api/v1/...``. See ``docs/coordinator-protocol.md``.
"""

from __future__ import annotations

import os
import warnings

__all__ = [
    "CoordinatorNotConfigured",
    "ENV_VAR",
    "LEGACY_ENV_VAR",
    "HOSTED_COORDINATOR",
    "API_PREFIX",
    "normalize",
    "resolve",
    "api_v1",
]

ENV_VAR = "INTERLATENT_COORDINATOR"
LEGACY_ENV_VAR = "INTERLATENT_API_BASE"
#: Where the hosted dashboard serves the protocol. Named for humans and
#: suggested in errors; never resolved to implicitly.
HOSTED_COORDINATOR = "https://interlatent.com"
API_PREFIX = "/api/v1"


class CoordinatorNotConfigured(RuntimeError):
    """No coordinator address could be resolved."""


_REMEDY = (
    "interlatent-serve needs a coordinator to register with. Pass "
    "--coordinator <url>, or set INTERLATENT_COORDINATOR. Run one with "
    "`interlatent up` on your control-plane host, or point at a hosted "
    "dashboard."
)


def normalize(url: str) -> str:
    """Canonicalise to a bare origin: no trailing slash, no ``/api/v1``.

    Both conventions existed in this repo and were reconciled at runtime in
    three places. This is the one place now.
    """
    base = url.strip().rstrip("/")
    if base.endswith(API_PREFIX):
        base = base[: -len(API_PREFIX)]
    return base.rstrip("/")


def api_v1(url: str) -> str:
    """``https://host`` -> ``https://host/api/v1``, idempotent."""
    return f"{normalize(url)}{API_PREFIX}"


def resolve(explicit: str | None = None) -> str:
    """Resolve the coordinator address, or raise.

    Precedence: ``explicit`` -> ``INTERLATENT_COORDINATOR`` ->
    ``INTERLATENT_API_BASE`` (deprecated) -> raise.
    """
    if explicit and explicit.strip():
        return normalize(explicit)

    from_env = os.environ.get(ENV_VAR, "").strip()
    if from_env:
        return normalize(from_env)

    legacy_env = os.environ.get(LEGACY_ENV_VAR, "").strip()
    if legacy_env:
        warnings.warn(
            f"{LEGACY_ENV_VAR} is deprecated; use {ENV_VAR}.",
            DeprecationWarning,
            stacklevel=2,
        )
        return normalize(legacy_env)

    raise CoordinatorNotConfigured(_REMEDY)
