"""Full client <-> server loop on one machine. No robot, no GPU, no account.

Launches `interlatent-serve` with the built-in `echo` test backend, then
drives it with the DRTC client exactly the way a robot would: stream
observations up, receive action chunks back, pull one action per control
tick. This exercises the entire real-time chunking path (sender/receiver
threads, chunk merging, latency estimation) end to end.

Run:
    pip install interlatent interlatent-server
    python examples/01_loopback_no_hardware.py
"""

from __future__ import annotations

import argparse
import io
import socket
import subprocess
import sys
import time

import numpy as np

from interlatent.inference.integration import connect_drtc

ACTION_DIM = 6
CHUNK_SIZE = 32


def wait_for_port(host: str, port: int, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError:
            time.sleep(0.2)
    raise TimeoutError(f"server did not come up on {host}:{port}")


def observation_npz(rng: np.random.Generator) -> bytes:
    """One control tick's observation, packed the same way a robot packs it.

    On real hardware these are joint readings + camera JPEGs; the echo
    backend only needs *something* to respond to.
    """
    buf = io.BytesIO()
    np.savez(buf, **{
        "observation.state": (rng.standard_normal(ACTION_DIM) * 0.1).astype(np.float32),
    })
    return buf.getvalue()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=50123)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--fps", type=float, default=30.0)
    args = p.parse_args()

    print(f"starting interlatent-serve on 127.0.0.1:{args.port} ...")
    server = subprocess.Popen(
        [sys.executable, "-m", "interlatent_server.server.app",
         "--host", "127.0.0.1", "--port", str(args.port)],
    )
    try:
        wait_for_port("127.0.0.1", args.port)

        # No api_key: self-hosted servers don't need an account.
        client = connect_drtc(
            environment="loopback-demo",
            policy_backend="echo",        # built-in test policy (sinusoid)
            server_address=f"127.0.0.1:{args.port}",
            chunk_size=CHUNK_SIZE,
            action_dim=ACTION_DIM,
            min_execution_horizon=8,
            cooldown_steps=8,
            fps=args.fps,
        )
        print(f"session open: {client.session_id}")

        rng = np.random.default_rng(0)
        period = 1.0 / args.fps
        actions = 0
        try:
            for i in range(args.steps):
                t0 = time.monotonic()
                a = client.step(observation_npz(rng), codec="npz")
                if a is not None:
                    actions += 1
                if i % 10 == 0:
                    print(
                        f"  step={i:3d} queue={client.queue_depth:3d} "
                        f"latency_ms={client.estimated_latency_s * 1000:6.1f} "
                        f"action={'-' if a is None else np.array2string(a[:3], precision=2)}"
                    )
                dt = time.monotonic() - t0
                if dt < period:
                    time.sleep(period - dt)
        finally:
            client.close()

        print(f"\ndone: {actions}/{args.steps} ticks had a scheduled action")
        if actions == 0:
            sys.exit("FAIL: no actions received")
        print("OK — you just ran the same loop a real robot runs.")
        print("Next: serve a real policy (examples/02_serve_policy.md).")
    finally:
        server.terminate()
        server.wait(timeout=10)


if __name__ == "__main__":
    main()
