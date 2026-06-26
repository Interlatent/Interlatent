"""Drive a robot by hand with the manual action interface — no policy, no cloud.

The same robot robot the DRTC engine path uses also exposes a manual,
programmatic ``action()`` call: name the joints, give absolute targets, and the call
**blocks until the arm settles** (raising on timeout). Targets are joint angles in
the robot's own frame (degrees for SO-101 / Koch) — joint-space only, no IK. So
``action(shoulder_pan=30, ...)`` means "drive that joint to 30 degrees", not a
Cartesian point.

Motion is gated by the same client-side safety model as teleop: each step is
velocity/workspace/deadman-clamped by the SafetyGate (which needs a RobotProfile for
the kind — so101 and koch ship with one) and then delta-clamped inside
``send_action``. A robot kind with no profile refuses to move.

Requires hardware:

    pip install 'interlatent[lerobot]'
    python examples/04_manual_action.py --port /dev/ttyACM0
    python examples/04_manual_action.py --robot koch --port /dev/ttyACM0
"""
from __future__ import annotations

import argparse

from interlatent.adapters.lerobot.robot import LeRobotAdapter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--robot", default="so101", help="robot kind (needs a RobotProfile)")
    p.add_argument("--port", required=True, help="serial port, e.g. /dev/ttyACM0")
    return p.parse_args()


def show_pose(robot: LeRobotAdapter, label: str) -> None:
    """Read the live joint positions and print them by name."""
    obs = robot.get_observation()
    pose = ", ".join(
        f"{f.removesuffix('.pos')}={obs[f]:.1f}"
        for f in robot.action_features
        if f in obs
    )
    print(f"  [{label}] {pose}")


def main() -> None:
    args = parse_args()

    robot = LeRobotAdapter(args.robot, port=args.port)
    robot.connect()
    # `action_features` is the ordered list of joints; drop the ".pos" suffix to
    # get the names you pass to action(). joint_specs declares each joint's mode.
    print(f"connected {args.robot!r}")
    print(f"  joints: {[f.removesuffix('.pos') for f in robot.action_features]}")
    show_pose(robot, "start")

    try:
        # 1) Full pose: name every joint with an absolute target. Blocks until
        #    the arm settles within each joint's tolerance, then returns.
        print("\n1) centering the arm (all joints named) ...")
        robot.action(
            shoulder_pan=0.0, shoulder_lift=0.0, elbow_flex=0.0,
            wrist_flex=0.0, wrist_roll=0.0, gripper=50.0,
            timeout=8.0,
        )
        show_pose(robot, "centered")

        # 2) Partial move with hold_missing=True: name only the joints you want to
        #    move; every joint you omit is held at its *measured present position*
        #    (and the held joints are logged). Without the flag, omitting a joint
        #    is an error — so a typo or embodiment mismatch can't silently move
        #    the wrong joint.
        print("\n2) sweeping the base left and right (other joints held) ...")
        for pan in (30.0, -30.0, 0.0):
            robot.action(shoulder_pan=pan, hold_missing=True, timeout=8.0)
            show_pose(robot, f"pan={pan:+.0f}")

        # 3) Gripper: a gripper is not a position joint, so the call does NOT wait
        #    for it to reach its target (a gripper closing on an object never
        #    does). It settles as soon as it is commanded.
        print("\n3) open then close the gripper ...")
        robot.action(gripper=100.0, hold_missing=True, timeout=8.0)  # open
        robot.action(gripper=0.0, hold_missing=True, timeout=8.0)    # close

        # 4) The contract guards — these raise *before* any motion. Uncomment to
        #    see them; nothing moves.
        #
        #   robot.action(elbow=10.0, hold_missing=True)        # ValueError: unknown joint
        #   robot.action(shoulder_pan=10.0)                    # ValueError: missing joint
        #   robot.action(shoulder_pan=999.0, hold_missing=True)  # ValueError: outside limit

        print("\ndone.")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
