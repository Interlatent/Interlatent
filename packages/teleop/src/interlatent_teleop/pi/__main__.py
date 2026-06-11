"""CLI: `interlatent-teleop-pi --driver so101 --port /dev/ttyACM0`."""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time

import numpy as np

from ..common.config import SO101_JOINT_NAMES
from .server import serve
from .so101_driver import build_driver


# Sensible "teleop ready" pose for the SO-101 follower in motor degrees.
# Upper arm slightly back-tilted, forearm forward and slightly down,
# gripper open enough to grab a small object. Tweak via --home-pose if
# this doesn't match your particular assembly.
DEFAULT_HOME_POSE = (0.0, 30.0, -60.0, 30.0, 0.0, 50.0)


def _parse_home_pose(s: str) -> tuple[float, ...]:
    vals = [float(v.strip()) for v in s.split(",")]
    if len(vals) != len(SO101_JOINT_NAMES):
        raise argparse.ArgumentTypeError(
            f"--home-pose expects {len(SO101_JOINT_NAMES)} comma-separated values "
            f"(one per joint: {','.join(SO101_JOINT_NAMES)}); got {len(vals)}"
        )
    return tuple(vals)


def _go_to_home(driver, home_joints: np.ndarray, ramp_s: float = 1.5,
                control_hz: int = 50) -> None:
    """Smoothly ramp the arm from its current pose to `home_joints`, then hold briefly.

    Used at startup so the IK home reference matches a known, sensible
    physical pose instead of whatever the arm happened to be left at.
    """
    start = driver.read_joints().astype(np.float32)
    target = home_joints.astype(np.float32)
    n_ramp = max(1, int(ramp_s * control_hz))
    dt = 1.0 / control_hz
    for i in range(n_ramp):
        alpha = (i + 1) / n_ramp
        driver.write_joints(start * (1.0 - alpha) + target * alpha)
        time.sleep(dt)
    # Hold for 0.5s so the motors settle and read_joints reports the
    # arrived-at pose, not the in-flight commanded pose.
    for _ in range(int(0.5 * control_hz)):
        driver.write_joints(target)
        time.sleep(dt)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="interlatent-teleop-pi")
    parser.add_argument("--driver", choices=("so101", "mock"), default="so101",
                        help="hardware driver; 'mock' for dev without an arm")
    parser.add_argument("--port", default="/dev/ttyACM0",
                        help="serial device for the so101 driver")
    parser.add_argument("--robot-id", default="interlatent_so101_001",
                        help="lerobot calibration id; must match an existing calibration "
                             "(e.g. the one created by `interlatent-node`)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--grpc-port", type=int, default=50061)
    parser.add_argument("--control-hz", type=int, default=50)
    parser.add_argument("--session-token", default=os.environ.get("INTERLATENT_TELEOP_TOKEN"),
                        help="if set, producers must present this token on OpenTeleop")
    parser.add_argument("--home-pose", type=_parse_home_pose, default=DEFAULT_HOME_POSE,
                        help="comma-separated motor degrees, one per joint "
                             f"({','.join(SO101_JOINT_NAMES)}). The arm ramps here on "
                             f"startup and that pose becomes the IK reference. "
                             f"Default: {','.join(str(v) for v in DEFAULT_HOME_POSE)}. "
                             f"Pass an empty string to skip and use whatever pose the arm "
                             f"is already in.")
    parser.add_argument("--no-home", action="store_true",
                        help="skip the move-to-home ramp; use the arm's current pose as is")
    parser.add_argument("--pan-p-gain", type=int, default=32,
                        help="Feetech P_Coefficient for shoulder_pan. LeRobot default "
                             "is 16; that's fine for a static arm but causes visible "
                             "stick-slip jerkiness when teleop feeds a smoothly ramped "
                             "trajectory. Counterintuitively, pan needs MORE P than "
                             "the gravity-loaded joints (lift/elbow), not less: pan "
                             "carries the whole arm's inertia but has no gravity load, "
                             "so at small errors the motor sees zero torque demand, "
                             "static friction wins, the motor sits, error builds, the "
                             "motor unsticks and lurches, rings, then sticks again. "
                             "Lift gets a free torque bias from gravity that keeps it "
                             "out of the stick regime. 32 is enough P to keep pan "
                             "moving smoothly; bump to 40–48 if it's still jerky, drop "
                             "back to 24 if it twitches. Set 0 to leave lerobot's default.")
    parser.add_argument("--lift-p-gain", type=int, default=16,
                        help="Feetech P_Coefficient for shoulder_lift. LeRobot default "
                             "is 16 (too soft — arm sags); Feetech default 32 holds "
                             "but rings around each commanded position step. 24 is a "
                             "compromise that holds against gravity without visible "
                             "shake. Bump to 32–48 if the arm sags; drop toward 16 "
                             "if you still see shaking. Set 0 to leave lerobot's default.")
    parser.add_argument("--elbow-p-gain", type=int, default=24,
                        help="Feetech P_Coefficient for elbow_flex. Same notes as "
                             "--lift-p-gain.")
    parser.add_argument("--wflex-p-gain", type=int, default=16,
                        help="Feetech P_Coefficient for wrist_flex. Wrist is much less "
                             "gravity-loaded, lerobot's default is usually fine.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    driver = build_driver(args.driver, port=args.port, robot_id=args.robot_id)
    driver.connect()

    # Override lerobot's conservative P-gains on the gravity-loaded
    # joints. LeRobot ships with P_Coefficient=16 to keep the arm
    # quiet; that's too soft to actually lift the arm against gravity.
    if args.driver == "so101" and hasattr(driver, "set_motor_p_gains"):
        gains = {}
        if args.pan_p_gain > 0:
            gains["shoulder_pan"] = args.pan_p_gain
        if args.lift_p_gain > 0:
            gains["shoulder_lift"] = args.lift_p_gain
        if args.elbow_p_gain > 0:
            gains["elbow_flex"] = args.elbow_p_gain
        if args.wflex_p_gain > 0:
            gains["wrist_flex"] = args.wflex_p_gain
        if gains:
            try:
                driver.set_motor_p_gains(gains)
            except Exception:
                logging.exception("failed to set motor P gains; continuing with defaults")

    if not args.no_home:
        home = np.array(args.home_pose, dtype=np.float32)
        logging.info("ramping to home pose: %s", home.tolist())
        try:
            _go_to_home(driver, home, control_hz=args.control_hz)
        except Exception:  # noqa: BLE001
            logging.exception("home ramp failed; continuing from current pose")

    server = serve(
        driver=driver,
        host=args.host,
        port=args.grpc_port,
        control_hz=args.control_hz,
        session_token=args.session_token,
        connect_driver=False,
    )

    def _shutdown(signum, frame):  # noqa: ARG001
        logging.info("shutting down on signal %s", signum)
        server.stop(grace=1.0)
        try:
            driver.disconnect()
        except Exception:  # noqa: BLE001
            logging.exception("driver disconnect raised")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    server.wait_for_termination()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
