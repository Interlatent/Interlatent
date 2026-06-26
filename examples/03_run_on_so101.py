"""Drive an SO-101-shaped robot against a cloud GPU pod.

Inference runs on a managed pod provisioned by the Interlatent dashboard;
you authenticate with an API key (`ilat_...`). Without hardware this still
runs: it introspects the policy's expected observation schema locally
(camera keys + state shape, config json only — no weights downloaded) and
synthesizes matching observations, so you can validate your account +
network path before touching a robot. When you wire real hardware, replace
`synth_observation()`'s per-key logic with camera capture + joint reads —
keys and shapes stay identical.

Run:

    pip install interlatent lerobot
    export INTERLATENT_API_KEY=ilat_...
    python examples/03_run_on_so101.py --task "pick up the red cube"

For a hands-off daemon on the robot (auto camera capture, dashboard-assigned
sessions) see `interlatent-node`.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--api-key", default=os.environ.get("INTERLATENT_API_KEY"),
                   help="Interlatent API key (ilat_...); or set INTERLATENT_API_KEY")
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


def apply_action(action: np.ndarray, robot: Any = None) -> None:
    """Write one streamed action vector to the motors (engine path).

    This synthetic demo runs without hardware, so ``robot`` is None and this is a
    no-op. With a real robot, pass a connected robot: the flat action vector is
    zipped onto the robot's ordered joints and written fire-and-forget — one
    waypoint per control tick (this is the engine seam, not the manual ``action()``
    call; see ``examples/04_manual_action.py`` for that)::

        from interlatent.adapters.lerobot.robot import LeRobotAdapter
        robot = LeRobotAdapter("so101", port="/dev/ttyACM0")
        robot.connect()
        ...
        apply_action(action, robot)
    """
    if robot is None:
        return
    vec = np.asarray(action, dtype=np.float32).reshape(-1)
    robot.send_action(
        {f: float(vec[i]) for i, f in enumerate(robot.action_features)}
    )


def main() -> None:
    args = parse_args()

    print(f"introspecting {args.policy_uri} ...")
    shapes, chunk_size, action_dim = introspect(args.policy_uri)
    for k, v in shapes.items():
        print(f"  {k}: {v}")

    from interlatent.inference.integration import connect_drtc

    if not args.api_key:
        sys.exit("set INTERLATENT_API_KEY or pass --api-key ilat_...")

    print("connecting to Interlatent ...")
    print("  (the dashboard provisions a GPU pod for the session — the first "
          "action chunk can take a second or two to arrive)")
    client = connect_drtc(
        environment="so101-cloud-demo",
        policy_uri=args.policy_uri,
        api_key=args.api_key,          # resolves your account + attached GPU pod
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
        sys.exit("FAIL: no actions — check your API key, the session, and the network path")
    print("OK")


if __name__ == "__main__":
    main()
