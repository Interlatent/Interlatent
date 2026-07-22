"""Shared env-var parsing for node knobs.

One clamp-parse surface for the whole node, replacing the copies that used to
live in each leaf (`_quic_proc._env_int`, `channel._preview_period_s`,
`control._preview_max_dim`/`_preview_jpeg_quality`). Every knob read through
these helpers is recorded in a process-local registry, so a process can log all
of its non-default overrides in one place (:func:`overrides`) instead of a
per-knob ``if`` — see ``_quic_proc.main``.

Semantics match the old idiom exactly: an unset/empty value → the default;
unparseable → the default (never raises); the result is clamped to ``[lo, hi]``.
"""
from __future__ import annotations

import os
from typing import Dict, Tuple

# name -> (parsed value, default) for every knob read through this module.
_REGISTRY: "Dict[str, Tuple[object, object]]" = {}


def _record(name: str, value: object, default: object) -> None:
    _REGISTRY[name] = (value, default)


def env_int(name: str, default: int, lo: int, hi: int) -> int:
    """Parse ``name`` as an int, falling back to ``default``, clamped to
    ``[lo, hi]``."""
    try:
        value = int(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        value = default
    value = max(lo, min(hi, value))
    _record(name, value, default)
    return value


def env_float(name: str, default: float, lo: float, hi: float) -> float:
    """Parse ``name`` as a float, falling back to ``default``, clamped to
    ``[lo, hi]``."""
    try:
        value = float(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        value = default
    value = max(lo, min(hi, value))
    _record(name, value, default)
    return value


def env_bool(name: str, default: bool) -> bool:
    """Parse ``name`` as a bool with the node's ``!= "0"`` convention: unset
    or empty → ``default``; ``"0"`` → False; anything else → True."""
    raw = os.environ.get(name)
    value = default if raw is None or raw == "" else raw != "0"
    _record(name, value, default)
    return value


def overrides() -> "Dict[str, object]":
    """Every knob whose parsed value differs from its default, ``{name: value}``.
    Empty when the process runs on stock defaults."""
    return {name: value for name, (value, default) in _REGISTRY.items()
            if value != default}
