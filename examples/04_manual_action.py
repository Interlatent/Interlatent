"""Drive an SO-101 by hand with the manual action interface — no policy, no cloud.

The same robot adapter the DRTC engine path uses also exposes a manual,
programmatic ``action()`` call: name the joints, give absolute targets, and the call
**blocks until the arm settles** (raising on timeout). Targets are joint angles in
the robot's own frame (degrees for SO-101) — joint-space only, no IK.

Motion is gated by the same client-side safety model as teleop: each step is
velocity/workspace/deadman-clamped by the SafetyGate (which needs a RobotProfile for
the kind — so101 has one) and then delta-clamped inside ``send_action``.

Requires hardware:

    pip install 'interlatent[lerobot]'
    python examples/04_manual_action.py --port /dev/ttyACM0
"""
from __future__ import annotations

import argparse

from interlatent.adapters.lerobot.robot import LeRobotAdapter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--robot", default="so101", help="robot kind (needs a RobotProfile)")
    p.add_argument("--port", required=True, help="serial port, e.g. /dev/ttyACM0")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    adapter = LeRobotAdapter(args.robot, port=args.port)
    adapter.connect()
    print(f"connected {args.robot!r}; joints: {adapter.action_features}")

    try:
        # Sequential, blocking moves — each returns once the arm has settled.
        print("centering the arm ...")
        adapter.action(
            shoulder_pan=0.0, shoulder_lift=0.0, elbow_flex=0.0,
            wrist_flex=0.0, wrist_roll=0.0, gripper=50.0,
            timeout=8.0,
        )

        print("nudging just one joint (others held at their present position) ...")
        adapter.action(shoulder_pan=30.0, hold_missing=True, timeout=8.0)

        print("opening the gripper ...")
        adapter.action(gripper=100.0, hold_missing=True, timeout=8.0)

        print("done.")
    finally:
        adapter.disconnect()


if __name__ == "__main__":
    main()
