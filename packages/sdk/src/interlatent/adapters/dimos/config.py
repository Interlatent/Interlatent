"""Build the dimos adapter config from the node's passthrough dicts.

The node hands robot construction a flat ``extra`` dict (``--robot-arg key=value``)
and a ``cameras`` dict (``--camera name=topic``). The dimos adapter binds to a
RUNNING dimos stack as an external bus peer (LCM/Zenoh) — cameras here are bus
*topics* carrying ``dimos.msgs.sensor_msgs.Image``, not vendor camera devices.

Nothing here imports ``dimos`` — that is resolved lazily inside the adapter — so
importing this module never requires the ``[dimos]`` extra.

There is deliberately NO ``verify=false`` escape hatch: connect-time
declare-then-verify is fail-closed by design (ADR 0018). Tests inject fakes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .kinds import DimosKind, get_kind

_logger = logging.getLogger(__name__)

_VALID_TRANSPORTS = ("lcm", "zenoh")

DEFAULT_JOINT_STATE_TOPIC = "/coordinator_joint_state"
DEFAULT_JOINT_COMMAND_TOPIC = "/joint_command"
DEFAULT_EPISODE_TOPIC = "/interlatent/episode"


@dataclass
class DimosAdapterConfig:
    """Everything the native dimos inference loop needs to bind and drive."""

    kind: DimosKind
    # None = follow dimos's own resolution (DIMOS_TRANSPORT / .env). Both sides
    # MUST agree or they silently do not see each other; verify.py's ping failure
    # message names this hazard.
    transport: str | None = None
    joint_state_topic: str = DEFAULT_JOINT_STATE_TOPIC
    joint_command_topic: str = DEFAULT_JOINT_COMMAND_TOPIC
    episode_topic: str = DEFAULT_EPISODE_TOPIC
    # --camera name=topic; name must match the policy's training camera keys.
    cameras: dict[str, str] = field(default_factory=dict)
    # Joint-state freshness gate (the loop holds when stale — nori precedent).
    staleness_ms: float = 200.0
    # Per-camera freshness: a stale camera serves the last frame + a one-shot
    # warning (frozen image, never a dead session).
    camera_staleness_ms: float = 500.0
    # connect() blocks until each camera topic delivered one frame.
    camera_warmup_s: float = 10.0
    # Execution-safety per-tick clamp on arm joints, radians. This is the ONLY
    # clamp in the whole path — dimos applies no limits to streamed joint_command.
    # The gripper is exempt (commanded across its whole range in one step by
    # design) and rides the same joint_command message as the arm joints.
    max_step_rad: float = 0.05
    # Budget for CoordinatorRPC ping + first-JointState wait at connect().
    connect_timeout_s: float = 10.0

    def __post_init__(self) -> None:
        if self.transport is not None and self.transport not in _VALID_TRANSPORTS:
            raise ValueError(
                f"transport must be one of {_VALID_TRANSPORTS}, got {self.transport!r}"
            )
        if self.max_step_rad <= 0:
            raise ValueError(f"max_step_rad must be > 0, got {self.max_step_rad}")


def build_adapter_config(
    extra: dict[str, str] | None, cameras: dict[str, str] | None
) -> DimosAdapterConfig:
    """Build a :class:`DimosAdapterConfig` from ``--robot-arg`` / ``--camera``.

    Recognized ``--robot-arg`` keys: ``kind`` (REQUIRED — the declared
    embodiment, e.g. ``xarm7``), ``transport`` (``lcm|zenoh``),
    ``joint_state_topic`` / ``joint_command_topic`` / ``episode_topic``,
    ``staleness_ms`` / ``camera_staleness_ms`` / ``camera_warmup_s``,
    ``max_step_rad``, ``connect_timeout_s``. Unrecognized keys warn + are
    ignored.

    ``--camera <name>=<topic>`` maps an observation key to a dimos bus topic
    (e.g. ``wrist=/camera/wrist/color``). Names must match the policy's training
    camera keys. Cameras are optional (a manual ``interlatent-act`` joint move
    needs none).
    """
    extra = dict(extra or {})
    cameras = dict(cameras or {})

    kind_name = extra.pop("kind", None)
    if not kind_name:
        raise ValueError(
            "the dimos adapter requires --robot-arg kind=<kind> declaring the "
            "embodiment the running dimos stack drives (it is verified against "
            "the live stack at connect). Example: --robot-arg kind=xarm7"
        )
    kind = get_kind(kind_name)

    transport = extra.pop("transport", None)
    if transport is not None:
        transport = str(transport).strip().lower()

    cfg = DimosAdapterConfig(
        kind=kind,
        transport=transport,
        joint_state_topic=str(extra.pop("joint_state_topic", DEFAULT_JOINT_STATE_TOPIC)),
        joint_command_topic=str(
            extra.pop("joint_command_topic", DEFAULT_JOINT_COMMAND_TOPIC)
        ),
        episode_topic=str(extra.pop("episode_topic", DEFAULT_EPISODE_TOPIC)),
        cameras={str(k): str(v) for k, v in cameras.items()},
        staleness_ms=float(extra.pop("staleness_ms", "200")),
        camera_staleness_ms=float(extra.pop("camera_staleness_ms", "500")),
        camera_warmup_s=float(extra.pop("camera_warmup_s", "10")),
        max_step_rad=float(extra.pop("max_step_rad", "0.05")),
        connect_timeout_s=float(extra.pop("connect_timeout_s", "10")),
    )

    if extra:
        _logger.warning(
            "Ignoring unrecognized --robot-arg key(s) for dimos: %s",
            ", ".join(sorted(extra)),
        )
    return cfg


__all__ = ["DimosAdapterConfig", "build_adapter_config"]
