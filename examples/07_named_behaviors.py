"""Run named, deterministic behaviors — no cloud, no GPU, no policy, no API key.

Behaviors are named moves and trajectories (``home``, ``hello``, your own from TOML)
that run entirely on the robot side through the ordinary adapter action path. This is
the manual counterpart to the cloud policy path in ``03_run_on_so101.py``.

Like that example, this runs **without hardware**: if no ``--port`` is given it drives
an in-memory fake SO-101 adapter and prints the sampled action stream the executor
would send to the motors. Wire a real arm with ``--port /dev/ttyACM0`` and the exact
same code drives it.

Run:

    python examples/07_named_behaviors.py                     # fake adapter, prints stream
    python examples/07_named_behaviors.py --port /dev/ttyACM0 # real SO-101 ('interlatent[lerobot]')
"""
from __future__ import annotations

import argparse

import numpy as np

import interlatent as il


# --------------------------------------------------------------------------
# A hardware-free SO-101 adapter (mirrors 03_run_on_so101.py's synth path).
# Replace this with a real arm by passing --port; nothing else changes.
# --------------------------------------------------------------------------
_FEATURES = [
    "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
    "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
]


def _make_fake_adapter():
    from interlatent.adapters.base import JointSpec, ManualActionInterface

    class FakeSO101(ManualActionInterface):
        """In-memory arm: tracks commands exactly and prints every Nth action."""

        robot_kind = "so101"

        def __init__(self) -> None:
            self._pos = np.zeros(len(_FEATURES), dtype=np.float32)
            self._ticks = 0

        @property
        def action_features(self):
            return list(_FEATURES)

        @property
        def joint_specs(self):
            return [
                JointSpec(
                    name=f.rsplit(".", 1)[0],
                    control_mode="gripper" if "gripper" in f else "position",
                    settle_tolerance=2.0,
                )
                for f in _FEATURES
            ]

        def connect(self):
            print("[fake SO-101] connected (no hardware) — printing the action stream\n")

        def disconnect(self):
            print(f"\n[fake SO-101] disconnected after {self._ticks} commands")

        def get_observation(self):
            return {f: float(self._pos[i]) for i, f in enumerate(_FEATURES)}

        def send_action(self, action):
            vec = np.array([action[f] for f in _FEATURES], dtype=np.float32)
            if self._ticks % 15 == 0:  # ~ every 0.5 s at 30 Hz
                pretty = "  ".join(f"{f.rsplit('.', 1)[0]}={vec[i]:+6.1f}" for i, f in enumerate(_FEATURES))
                print(f"  tick {self._ticks:3d} | {pretty}")
            self._pos = vec
            self._ticks += 1
            return action

    return FakeSO101()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--robot", default="so101", help="robot kind (default: so101)")
    p.add_argument("--port", default=None, help="serial port, e.g. /dev/ttyACM0 (omit → fake)")
    return p.parse_args()


def _open_robot(args) -> il.Robot:
    """A real Robot when --port is given; otherwise one wrapping the fake adapter."""
    if args.port:
        return il.Robot(args.robot, port=args.port)
    # No hardware: hand the Robot facade a pre-built fake adapter. `connect=False`
    # skips resolving a real adapter; we connect the fake and build the executor here.
    from interlatent.behaviors.executor import TrajectoryExecutor
    from interlatent.node.teleop.robot_profile import get_profile

    robot = il.Robot(args.robot, connect=False, realtime=False)
    robot._adapter = _make_fake_adapter()  # noqa: SLF001 — example wiring
    robot._adapter.connect()
    robot._executor = TrajectoryExecutor(
        robot._adapter, get_profile(robot.robot_kind), control_hz=30.0, realtime=False
    )
    return robot


def main() -> None:
    args = parse_args()
    robot = _open_robot(args)
    try:
        print(f"available behaviors: {robot.behaviors()}\n")

        print("act('home') — move to the profile rest pose:")
        result = robot.act("home")
        print(f"  -> {'reached' if result.reached else 'aborted'} in {result.elapsed:.2f}s\n")

        print("act('hello') — play the built-in wave:")
        result = robot.act("hello")
        print(f"  -> {'reached' if result.reached else 'aborted'} in {result.elapsed:.2f}s\n")

        print("act('home', speed=0.5) — same behavior, time-scaled gentler:")
        robot.act("home", speed=0.5)

        print("\nmove(wrist_roll=30, duration=0.5) — an ad-hoc single-joint move:")
        robot.move(wrist_roll=30.0, duration=0.5)

        print(f"\nfinal pose: { {k: round(v, 1) for k, v in robot.pose().items()} }")
    finally:
        robot.close()
    print("\nOK")


if __name__ == "__main__":
    main()
