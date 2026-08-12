"""One place that answers "which coordinator?".

Before this module the answer was hardcoded in eight files, in two mutually
incompatible spellings — some stored a bare origin, some stored one with
``/api/v1`` glued on — and three separate runtime fixups existed to paper over
the difference. One convention now (**bare origin**), resolved here, appended to
by callers.

A coordinator address is **required**. There is no such thing as "the" control
plane, so defaulting to one is how a fleet ends up quietly phoning home, and it
is what made "one code path" untestable: with a default, nothing ever proves
the SDK works against a coordinator you chose. :func:`resolve` raises
:class:`CoordinatorNotConfigured` with a remediation sentence naming the
caller's own flag.

See ``docs/coordinator-protocol.md`` and
``docs/adr/0038-coordinator-protocol-one-control-plane.md``.
"""

from __future__ import annotations

import os
import warnings

__all__ = [
    "CoordinatorNotConfigured",
    "ENV_VAR",
    "LEGACY_ENV_VAR",
    "normalize",
    "resolve",
]

#: The env var callers should set.
ENV_VAR = "INTERLATENT_COORDINATOR"

#: What it used to be called. Read for one minor, with a warning.
LEGACY_ENV_VAR = "INTERLATENT_API_BASE"


class CoordinatorNotConfigured(RuntimeError):
    """No coordinator address could be resolved.

    Carries a remediation sentence naming the *caller's own* flag, because
    "set INTERLATENT_COORDINATOR" is unhelpful when the user is running
    ``interlatent-serve`` and the flag is ``--coordinator``.
    """


# How each entry point tells the user to fix it. Keyed by the ``purpose``
# argument so the message names the flag actually in front of them.
_REMEDIES = {
    "node": (
        "interlatent-node needs a coordinator. Pass --coordinator <url> to "
        "`interlatent-node pair`, or set {env}."
    ),
    "serve": (
        "interlatent-serve needs a coordinator to register with. Pass "
        "--coordinator <url>, or set {env}. Run one with `interlatent up` on "
        "your control-plane host."
    ),
    "cli": (
        "This command needs a coordinator. Pass --coordinator <url>, set "
        "{env}, or run `interlatent up` to start one locally."
    ),
    "connect": (
        "connect_drtc() needs a coordinator to resolve your account and GPU "
        "pod. Pass coordinator=<url>, or set {env}."
    ),
    "preflight": (
        "interlatent-preflight needs a coordinator. Pass --coordinator <url>, "
        "or set {env}. (Use --server to dial a GPU box directly instead.)"
    ),
    "client": (
        "Interlatent() needs a coordinator. Pass base_url=<url>, or set {env}."
    ),
}


def normalize(url: str) -> str:
    """Canonicalise an address to a **bare origin**.

    Strips trailing slashes and a trailing ``/api/v1``, so a value written
    under either of the old conventions resolves to the same thing. Callers
    append ``/api/v1/...`` themselves — see
    ``interlatent.coordinator.protocol.API_PREFIX``.
    """
    base = url.strip().rstrip("/")
    if base.endswith("/api/v1"):
        base = base[: -len("/api/v1")]
    return base.rstrip("/")


def resolve(
    explicit: str | None = None,
    *,
    config: str | None = None,
    purpose: str = "cli",
) -> str:
    """Resolve the coordinator address, or raise.

    Precedence: ``explicit`` (a flag or kwarg) → :data:`ENV_VAR` →
    :data:`LEGACY_ENV_VAR` → ``config`` (a stored value, e.g. ``node.toml``) →
    raise.

    ``purpose`` selects the remediation sentence in the error; see
    :data:`_REMEDIES` for the accepted values.
    """
    if explicit and explicit.strip():
        return normalize(explicit)

    from_env = os.environ.get(ENV_VAR, "").strip()
    if from_env:
        return normalize(from_env)

    legacy_env = os.environ.get(LEGACY_ENV_VAR, "").strip()
    if legacy_env:
        warnings.warn(
            f"{LEGACY_ENV_VAR} is deprecated; use {ENV_VAR}. "
            "The two mean the same thing and the old name will stop being "
            "read in the next major.",
            DeprecationWarning,
            stacklevel=2,
        )
        return normalize(legacy_env)

    if config and config.strip():
        return normalize(config)

    raise CoordinatorNotConfigured(_remedy(purpose))


def _remedy(purpose: str) -> str:
    template = _REMEDIES.get(purpose, _REMEDIES["cli"])
    return template.format(env=ENV_VAR)
