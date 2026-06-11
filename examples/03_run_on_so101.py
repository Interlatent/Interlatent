"""Drive an SO-101-shaped robot against your self-hosted inference server.

Without hardware this still runs: it introspects the policy's expected
observation schema locally (camera keys + state shape, config json only —
no weights downloaded) and synthesizes matching observations, so you can
validate your server + network path before touching a robot. When you
wire real hardware, replace `synth_observation()`'s per-key logic with
camera capture + joint reads — keys and shapes stay identical.

Run (after `interlatent-serve --policy lerobot/smolvla_base` on your GPU box):

    pip install interlatent lerobot
    python examples/03_run_on_so101.py --server gpu-box:50051 \\
        --task "pick up the red cube"

For a hands-off daemon on the robot (auto camera capture, DAgger teleop
takeover, dashboard-assigned sessions) see `interlatent-node` — that path
uses Interlatent Cloud for session management.
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--server", default="localhost:50051",
                   help="your interlatent-serve address (host:port)")
    p.add_argument("--policy-uri", default="lerobot/smolvla_base")
    p.add_argument("--task", default="pick up the red cube")
    p.add_argument("--fps", type=float, default=10.0,
                   help="control rate; VLA chunks are long, 10 Hz is plenty")
    p.add_argument("--steps", type=int, default=200)
    return p.parse_args()


def introspect(policy_uri: str) -> tuple[dict[str, tuple[int, ...]], int, int]:
    """Discover the policy's input feature shapes from its config json."""
    try:
        from lerobot.policies import factory as _f  # noqa: F401  (registers choices)
        from lerobot.configs.policies import PreTrainedConfig
    except ImportError:
        sys.exit("lerobot is required to introspect the policy: pip install lerobot")

    cfg = PreTrainedConfig.from_pretrained(policy_uri)
    shapes = {k: tuple(f.shape) if hasattr(f, "shape") else ()
              for k, f in cfg.input_features.items()}
    chunk = getattr(cfg, "chunk_size", None) or getattr(cfg, "n_action_steps", None) or 32
    action_feat = cfg.output_features.get("action")
    action_dim = action_feat.shape[0] if action_feat else 6
    return shapes, int(chunk), int(action_dim)


def synth_observation(shapes: dict[str, tuple[int, ...]], task: str,
                      rng: np.random.Generator) -> bytes:
    """One frame's observation as an npz blob.

    >>> REPLACE THIS with real reads when wiring hardware:
        - "observation.images.*" -> your camera frames, uint8 HWC
        - "observation.state"    -> joint positions, float32
    """
    obs: dict[str, Any] = {}
    for key, shape in shapes.items():
        if not shape:
            continue
        if "image" in key:
            c, h, w = shape if shape[0] in (1, 3) else (3, 224, 224)
            obs[key] = rng.integers(0, 256, size=(h, w, c), dtype=np.uint8)
        else:
            obs[key] = (rng.standard_normal(shape) * 0.1).astype(np.float32)
    obs["task"] = np.array(task)
    buf = io.BytesIO()
    np.savez(buf, **obs)
    return buf.getvalue()


def apply_action(action: np.ndarray) -> None:
    """>>> REPLACE THIS with your motor write (e.g. lerobot robot.send_action)."""


def main() -> None:
    args = parse_args()

    print(f"introspecting {args.policy_uri} ...")
    shapes, chunk_size, action_dim = introspect(args.policy_uri)
    for k, v in shapes.items():
        print(f"  {k}: {v}")

    from interlatent.inference.integration import connect_drtc

    print(f"connecting to {args.server} ...")
    print("  (first session on a fresh server loads the policy — can take a "
          "while unless the server was started with --policy)")
    client = connect_drtc(
        environment="so101-selfhost-demo",
        policy_uri=args.policy_uri,
        server_address=args.server,    # self-hosted: no api_key needed
        chunk_size=chunk_size,
        action_dim=action_dim,
        task=args.task,
        fps=args.fps,
    )
    print(f"session={client.session_id}")

    rng = np.random.default_rng(0)
    period = 1.0 / args.fps
    received = 0
    try:
        for i in range(args.steps):
            t0 = time.monotonic()
            action = client.step(synth_observation(shapes, args.task, rng), codec="npz")
            if action is not None:
                received += 1
                apply_action(action)
            if i % 20 == 0:
                print(f"  step={i:3d} queue={client.queue_depth:3d} "
                      f"latency_ms={client.estimated_latency_s * 1000:6.1f}")
            dt = time.monotonic() - t0
            if dt < period:
                time.sleep(period - dt)
    finally:
        client.close()

    print(f"\nactions received: {received}/{args.steps}")
    if received == 0:
        sys.exit("FAIL: no actions — check the server log and network path")
    print("OK")


if __name__ == "__main__":
    main()
