"""The cloud upgrade: same code, one different argument.

Everything else in examples/ runs against infrastructure you operate.
This example is the bridge to Interlatent Cloud — drop the
`server_address` and pass an `api_key`, and the same DRTC loop runs on
managed warm GPUs with server-side episode recording into your hosted
datasets:

    -    server_address="gpu-box:50051",
    +    api_key=os.environ["INTERLATENT_API_KEY"],

What that buys (see README "OSS vs Cloud" for the honest table):
  - managed warm GPUs — no box to rent, no torch.compile cold starts
  - server-side recording into hosted, versioned LeRobot datasets
  - the dashboard: episode viewer, policy analysis, reward labeling
    (Robometer)

Get a key at https://interlatent.com (the `environment` slug below must
exist in your dashboard).

Run:
    INTERLATENT_API_KEY=ilat_... python examples/06_connect_hosted.py
"""

from __future__ import annotations

import io
import os
import sys
import time

import numpy as np

from interlatent.inference.integration import connect_drtc


def main() -> None:
    api_key = os.environ.get("INTERLATENT_API_KEY")
    if not api_key:
        sys.exit("set INTERLATENT_API_KEY (get one at https://interlatent.com)")

    client = connect_drtc(
        api_key=api_key,                  # <- the one-line upgrade
        environment="my-arm",             # dashboard environment slug
        policy_uri="lerobot/smolvla_base",
        task="pick up the red cube",
        fps=10,
        record=True,                      # cloud records the episode server-side
    )
    print(f"session={client.session_id} (hosted)")

    rng = np.random.default_rng(0)
    try:
        for i in range(100):
            buf = io.BytesIO()
            np.savez(buf, **{
                # Replace with your real camera frames + joint reads —
                # identical to the self-hosted loop in 03_run_on_so101.py.
                "observation.state": (rng.standard_normal(6) * 0.1).astype(np.float32),
                "task": np.array("pick up the red cube"),
            })
            action = client.step(buf.getvalue(), codec="npz")
            if i % 10 == 0:
                print(f"  step={i:3d} queue={client.queue_depth:3d} "
                      f"latency_ms={client.estimated_latency_s * 1000:6.1f} "
                      f"action={'-' if action is None else 'ok'}")
            time.sleep(0.1)
    finally:
        client.close()  # finalizes the hosted recording

    print("done — the episode is in your dashboard")


if __name__ == "__main__":
    main()
