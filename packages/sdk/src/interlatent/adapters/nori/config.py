"""Build the Nori adapter config from the node's passthrough dicts.

The node hands robot construction a flat ``extra`` dict (``--robot-arg key=value``)
and a ``cameras`` dict (``--camera name=value``). Nori is driven through its on-Pi
daemon over TCP NDJSON (Nori-Protocol v1), so there are no motor-bus or camera-SDK
settings here — cameras arrive on the daemon's companion ZeroMQ MJPEG channel and
``--camera <obs_key>=<daemon_camera_name>`` only maps daemon camera names onto the
policy's observation keys.

Nothing here imports ``zmq``/``cv2`` — those are resolved lazily inside the
adapter — so importing this module never requires the ``[nori]`` extra.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

_logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7777
# The daemon's own token file; the node runs on the same Pi (LAN/on-Pi only in
# v1), so reading it directly is the same trust domain as the daemon.
DEFAULT_TOKEN_PATH = "/etc/nori/agent.token"
DEFAULT_CAM_BASE_PORT = 5555


@dataclass
class NoriAdapterConfig:
    """Everything the native Nori loop needs to reach and drive the daemon."""

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    token: Optional[str] = None  # explicit override beats token_path
    token_path: str = DEFAULT_TOKEN_PATH
    bus_choice: str = "3"
    # Execution-safety per-send clamp on arm joints, daemon-normalized units
    # ([-100,100] scale; grippers exempt). float("inf") disables.
    max_step: float = 3.0
    # Keep-alive pump rate. The daemon watchdog counts control-frame arrival;
    # ~50 Hz matches the native teleop cadence (see ADR 0015).
    pump_hz: float = 50.0
    cam_host: Optional[str] = None  # None -> same as host
    cam_base_port: int = DEFAULT_CAM_BASE_PORT
    # obs_key -> daemon camera name; {} = subscribe every descriptor camera
    # under its native name.
    cameras: dict[str, str] = field(default_factory=dict)
    connect_timeout_s: float = 5.0
    # connect() blocks until every configured camera has delivered one frame,
    # up to this long — the Pi's capture bridge publishes lazily, so the first
    # frames can trail the session by a few seconds. 0 disables the wait.
    camera_warmup_s: float = 10.0
    reconnect_backoff_s: float = 0.5  # doubles up to max_backoff_s
    max_backoff_s: float = 5.0
    # TCP down longer than this => the session is dead (loop ends the episode).
    reconnect_window_s: float = 10.0
    # Telemetry older than this => observation is not fresh (loop holds).
    staleness_ms: float = 250.0

    @property
    def camera_host(self) -> str:
        return self.cam_host or self.host


def resolve_token(cfg: NoriAdapterConfig) -> Optional[str]:
    """Explicit token > token file > None (dev daemon without a token file)."""
    if cfg.token:
        return cfg.token
    try:
        if os.path.exists(cfg.token_path):
            content = open(cfg.token_path, encoding="utf-8").read().strip()
            return content or None
    except OSError:
        _logger.warning("Could not read Nori token file %s", cfg.token_path)
    return None


def build_adapter_config(
    extra: dict[str, str] | None, cameras: dict[str, str] | None
) -> NoriAdapterConfig:
    """Build a :class:`NoriAdapterConfig` from ``--robot-arg`` / ``--camera``.

    Recognized ``--robot-arg`` keys: ``host``, ``port``, ``token``,
    ``token_path``, ``bus_choice``, ``max_step``, ``pump_hz``, ``cam_host``,
    ``cam_base_port``, ``connect_timeout_s``, ``reconnect_window_s``,
    ``staleness_ms``. Unrecognized keys warn + are ignored.

    ``--camera <obs_key>=<daemon_camera_name>`` maps a daemon camera (as named
    in ``ack.descriptor.cameras``) onto a policy observation key. With no
    ``--camera`` flags, every descriptor camera is subscribed under its native
    name.
    """
    extra = dict(extra or {})

    cfg = NoriAdapterConfig(
        host=str(extra.pop("host", DEFAULT_HOST)),
        port=int(extra.pop("port", DEFAULT_PORT)),
        token=(extra.pop("token", None) or None),
        token_path=str(extra.pop("token_path", DEFAULT_TOKEN_PATH)),
        bus_choice=str(extra.pop("bus_choice", "3")),
        max_step=float(extra.pop("max_step", "3.0")),
        pump_hz=float(extra.pop("pump_hz", "50.0")),
        cam_host=(extra.pop("cam_host", None) or None),
        cam_base_port=int(extra.pop("cam_base_port", DEFAULT_CAM_BASE_PORT)),
        cameras=dict(cameras or {}),
        connect_timeout_s=float(extra.pop("connect_timeout_s", "5.0")),
        camera_warmup_s=float(extra.pop("camera_warmup_s", "10.0")),
        reconnect_window_s=float(extra.pop("reconnect_window_s", "10.0")),
        staleness_ms=float(extra.pop("staleness_ms", "250.0")),
    )

    if extra:
        _logger.warning(
            "Ignoring unrecognized --robot-arg key(s) for Nori: %s",
            ", ".join(sorted(extra)),
        )
    return cfg


__all__ = [
    "NoriAdapterConfig",
    "build_adapter_config",
    "resolve_token",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "DEFAULT_TOKEN_PATH",
    "DEFAULT_CAM_BASE_PORT",
]
