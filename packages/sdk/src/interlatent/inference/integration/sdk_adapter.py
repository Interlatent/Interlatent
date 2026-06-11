"""Adapter that lets `Interlatent.watch()` / `tick()` route through DRTC.

Today's `watch()` attaches local PyTorch hooks to a model the user
already holds. Under DRTC the policy lives on Modal, so there is no
local model to hook. This adapter swaps the local-inference part of
the SDK's collection loop with a DRTC client call while preserving
the existing public API.

Activation capture is intentionally NOT wired in v1 — see parent
package docstring. The hook seam exists on the server.

Wiring (planned for v1):

    client = Interlatent(api_key=...)
    client.watch_remote(
        env,
        environment="smolvla-x",
        server_address="https://...modal.run",
        chunk_size=32,
    )
    while running:
        obs = capture()
        action = client.tick_remote(obs)
        env.step(action)

The full integration into the existing watch()/tick() methods is
deferred to the SDK refactor; for now this module exposes a
self-contained `RemoteRollout` wrapper that callers can use directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..client import DRTCClient, DRTCConfig

log = logging.getLogger(__name__)


@dataclass
class RemoteRolloutConfig:
    server_address: str
    environment: str
    chunk_size: int = 50
    min_execution_horizon: int = 12
    control_period_s: float = 1 / 30
    use_grpc_web: bool = False


class RemoteRollout:
    """Minimal user-facing wrapper.

    Owns a DRTCClient, exposes (open/step/close) so callers can drop
    it into any env loop without thinking about the underlying
    threading model.
    """

    def __init__(self, cfg: RemoteRolloutConfig) -> None:
        # The DRTC wire protocol still spells the session identifier
        # ``model_id`` (out of scope for the SDK retirement). We pass
        # the backend env slug through it.
        self._client = DRTCClient(
            DRTCConfig(
                server_address=cfg.server_address,
                model_id=cfg.environment,
                chunk_size=cfg.chunk_size,
                min_execution_horizon=cfg.min_execution_horizon,
                control_period_s=cfg.control_period_s,
                use_grpc_web=cfg.use_grpc_web,
            )
        )

    def open(self) -> None:
        self._client.open()

    def step(self, observation: np.ndarray) -> Optional[np.ndarray]:
        if observation.dtype != np.float32:
            observation = observation.astype(np.float32)
        return self._client.step(observation.tobytes())

    def close(self) -> None:
        self._client.close()

    @property
    def queue_depth(self) -> int:
        return self._client.queue_depth

    @property
    def estimated_latency_s(self) -> float:
        return self._client.estimated_latency_s
