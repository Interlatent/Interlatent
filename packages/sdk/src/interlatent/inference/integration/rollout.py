"""Backend for the `interlatent-rollout` console script (DRTC path).

After this refactor, `interlatent-rollout` no longer runs a local
PolicyServer. It launches a DRTC client that:
    - opens a session against the Modal-hosted DRTC server
    - drives a local control loop (or hands the rollout object back
      to a host environment / lerobot's robot client)

The old PolicyServer-based behavior in
`lerobot/async_inference/async_rollout.py` is kept for backwards
compatibility but is no longer the default; that module will be
converted into a thin pass-through to this one in a follow-up.

This file is intentionally thin — all the interesting code is in
`inference/client/` and `inference/integration/sdk_adapter.py`.
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np

from .sdk_adapter import RemoteRollout, RemoteRolloutConfig

log = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="interlatent-rollout")
    p.add_argument("--server", required=True,
                   help="DRTC server address (host:port or https://...modal.run)")
    p.add_argument("--environment", required=True,
                   help="Interlatent environment slug registered in the dashboard")
    p.add_argument("--chunk-size", type=int, default=32)
    p.add_argument("--min-horizon", type=int, default=8)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--obs-dim", type=int, default=32,
                   help="Synthetic observation dim for the smoke loop")
    p.add_argument("--steps", type=int, default=300,
                   help="How many control steps to run before exiting")
    p.add_argument("--grpc-web", action="store_true",
                   help="Use gRPC-Web transport (needed for Modal asgi)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO)
    cfg = RemoteRolloutConfig(
        server_address=args.server,
        environment=args.environment,
        chunk_size=args.chunk_size,
        min_execution_horizon=args.min_horizon,
        control_period_s=1.0 / args.fps,
        use_grpc_web=args.grpc_web,
    )
    rollout = RemoteRollout(cfg)
    rollout.open()
    try:
        period = 1.0 / args.fps
        for i in range(args.steps):
            t0 = time.monotonic()
            obs = np.random.randn(args.obs_dim).astype(np.float32)
            action = rollout.step(obs)
            if action is None:
                log.info("step=%d action=None (warming up)", i)
            elif i % 30 == 0:
                log.info(
                    "step=%d action_norm=%.3f queue=%d est_latency_ms=%.1f",
                    i, float(np.linalg.norm(action)),
                    rollout.queue_depth,
                    rollout.estimated_latency_s * 1000,
                )
            dt = time.monotonic() - t0
            if dt < period:
                time.sleep(period - dt)
    finally:
        rollout.close()


if __name__ == "__main__":
    main()
