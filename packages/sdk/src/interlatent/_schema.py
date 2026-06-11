"""Lightweight data models for SDK collection (no pydantic dependency)."""
from __future__ import annotations

import datetime as _dt
import uuid as _uuid_mod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _now() -> str:
    """Return current UTC time in ISO-8601 with trailing Z."""
    return _dt.datetime.utcnow().isoformat(timespec="milliseconds") + "Z"


def _uuid() -> str:
    return _uuid_mod.uuid4().hex


@dataclass
class ActivationEvent:
    """Flattened activation tensor captured at a single forward step."""

    episode_id: str
    step: int
    layer: str
    channel: int = 0
    tensor: List[float] = field(default_factory=list)
    value_sum: Optional[float] = None
    value_sq_sum: Optional[float] = None
    timestamp: str = field(default_factory=_now)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeInfo:
    """Metadata about a collection episode."""

    episode_id: str = field(default_factory=_uuid)
    env_name: str = ""
    start_time: str = field(default_factory=_now)
    tags: Dict[str, Any] = field(default_factory=dict)
