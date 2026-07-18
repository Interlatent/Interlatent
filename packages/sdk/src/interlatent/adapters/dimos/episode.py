"""Episode markers: the correlation bridge between the two recording worlds.

The interlatent node records the episode of record (observations + actions +
``control_source``); a dimos-side memory2 ``Recorder`` may record low-level
streams (lidar/odom/tf) locally. The ONLY thing they share is this marker,
published on the dimos bus (pickled transport) at episode start/stop so the
dimos-side recording can be segmented per episode and joined by episode id +
same-host timestamps. Deliberately NOT dimos's ``EpisodeStatus`` — that type is
the control signal of dimos's own recording state machine, and publishing it
would make the node an actor in that machine (ADR 0018: no recorder-to-recorder
entanglement; markers are correlation metadata).

Plain dataclass on purpose: it pickles with only the class importable (any
dimos-side consumer has the ``interlatent`` package installed by definition)
and adds no pydantic dependency to the base SDK.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Literal

_logger = logging.getLogger(__name__)

MARKER_SCHEMA = 1


@dataclass(frozen=True)
class EpisodeMarker:
    episode_id: str
    event: Literal["start", "stop"]
    robot_kind: str
    ts: float = field(default_factory=time.time)  # dimos Timestamped convention
    source: str = "interlatent"
    schema: int = MARKER_SCHEMA


def publish_marker(bus, episode_id: str, event: str, robot_kind: str) -> None:
    """Best-effort marker publish — never raises into the control loop."""
    try:
        bus.publish_episode_marker(
            EpisodeMarker(episode_id=episode_id, event=event, robot_kind=robot_kind)  # type: ignore[arg-type]
        )
    except Exception:  # noqa: BLE001
        _logger.warning("episode %s marker publish failed", event, exc_info=True)


__all__ = ["EpisodeMarker", "publish_marker", "MARKER_SCHEMA"]
