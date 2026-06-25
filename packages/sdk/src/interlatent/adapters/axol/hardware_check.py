"""On-hardware bring-up check for the native Axol adapter.

Runs **onboard the Jetson** to prove the adapter can read live observations and
drive the arms + grippers through the real ``send_action`` → ``motion_control``
path. This is NOT a mocked unit test — it needs the actual hardware:

  - CAN up (`axol can.setup`),
  - the GMSL-attached ZED cameras connected (opened directly by serial),
  - `almond_axol`, `pyzed`, and `cv2` installed (`pip install 'interlatent[axol]'`
    + the ZED SDK / `axol zed.install`).

**Safe by default.** Every motion is operator-gated and bounded: a small nudge
(`--nudge-rad`, default 0.05 rad, well under `max_step_rad`) on a single joint of
each arm, verified by reading it back, then a return-to-start, then a gripper
open/close that is restored. Pass `--no-move` to check observations only, or
`--yes` to skip the confirmation prompt.

Usage::

    python -m interlatent.adapters.axol.hardware_check \\
        --camera overhead=41234567 --camera left_arm=41234568 \\
        --camera right_arm=41234569 \\
        [--left-stiffness 0.5 --right-stiffness 0.5] \\
        [--nudge-rad 0.05 --nudge-joint wrist_1] [--yes] [--no-move]

Exit code is 0 only if every executed check passes. This is the conformance
template each future native robot adapter copies for its own bring-up check.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Any

import numpy as np

_logger = logging.getLogger(__name__)


class _Results:
    """Tiny PASS/FAIL/SKIP accumulator with live console output."""

    def __init__(self) -> None:
        self.failed = 0
        self.passed = 0
        self.skipped = 0

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        tag = "PASS" if ok else "FAIL"
        if ok:
            self.passed += 1
        else:
            self.failed += 1
        print(f"  [{tag}] {name}" + (f" — {detail}" if detail else ""), flush=True)
        return ok

    def skip(self, name: str, detail: str = "") -> None:
        self.skipped += 1
        print(f"  [SKIP] {name}" + (f" — {detail}" if detail else ""), flush=True)

    def summary(self) -> int:
        print(
            f"\n{self.passed} passed, {self.failed} failed, {self.skipped} skipped",
            flush=True,
        )
        return 1 if self.failed else 0


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m interlatent.adapters.axol.hardware_check",
        description="On-hardware bring-up check for the native Axol adapter.",
    )
    p.add_argument(
        "--camera", action="append", default=[], metavar="NAME=SERIAL",
        help="ZED camera as name=serial (repeatable). Names must match the policy's "
        "training camera keys, e.g. --camera overhead=41234567.",
    )
    p.add_argument("--left-channel", default=None)
    p.add_argument("--right-channel", default=None)
    p.add_argument("--left-stiffness", default=None,
                   help="Match data-collection. Scalar or comma-separated 7-vector.")
    p.add_argument("--right-stiffness", default=None)
    p.add_argument("--telemetry-hz", default=None)
    p.add_argument("--max-step-rad", default=None)
    p.add_argument("--nudge-rad", type=float, default=0.05,
                   help="Bounded per-joint nudge magnitude (rad). Must be < max_step_rad.")
    p.add_argument("--nudge-joint", default="wrist_1",
                   help="Joint name to nudge on each arm (default wrist_1, low-risk).")
    p.add_argument("--settle-s", type=float, default=1.0,
                   help="Seconds to wait for the arm to track a commanded target.")
    p.add_argument("--track-frac", type=float, default=0.3,
                   help="Min fraction of the commanded nudge the joint must actually "
                   "move (in the right direction) to pass.")
    p.add_argument("--yes", action="store_true", help="Skip the motion confirmation prompt.")
    p.add_argument("--no-move", action="store_true",
                   help="Observation checks only; do not command any motion.")
    return p.parse_args(argv)


def _build_extra(args: argparse.Namespace) -> dict[str, str]:
    """Map CLI flags onto the adapter's --robot-arg keys.

    gripper_mode is forced to "continuous" so an arm nudge never snaps the
    gripper; the gripper phase commands explicit 0/1 values itself.
    """
    extra: dict[str, str] = {"gripper_mode": "continuous"}
    for src, key in (
        (args.left_channel, "left_channel"),
        (args.right_channel, "right_channel"),
        (args.left_stiffness, "left_stiffness"),
        (args.right_stiffness, "right_stiffness"),
        (args.telemetry_hz, "telemetry_hz"),
        (args.max_step_rad, "max_step_rad"),
    ):
        if src is not None:
            extra[key] = str(src)
    return extra


def _cameras(args: argparse.Namespace) -> dict[str, str]:
    cams: dict[str, str] = {}
    for spec in args.camera:
        if "=" not in spec:
            raise SystemExit(f"--camera expects NAME=SERIAL, got {spec!r}")
        name, serial = spec.split("=", 1)
        cams[name.strip()] = serial.strip()
    if not cams:
        raise SystemExit("at least one --camera NAME=SERIAL is required")
    return cams


def _joint_action(obs: dict[str, Any], action_keys: list[str]) -> dict[str, float]:
    """The 16 joint values out of an observation (drops camera frames)."""
    return {k: float(obs[k]) for k in action_keys}


def _check_observation(robot: Any, action_keys: list[str], res: _Results) -> dict:
    """Read one observation and validate joints + live camera frames."""
    obs = robot.get_observation()

    have_all = all(k in obs for k in action_keys)
    res.check("observation has all 16 joint keys", have_all)
    vals = [obs.get(k) for k in action_keys if k in obs]
    finite = bool(vals) and all(
        isinstance(v, float) and np.isfinite(v) for v in vals
    )
    res.check("joint values are finite floats", finite,
              f"{len(vals)} values" if vals else "no values")

    cam_keys = [
        k for k, v in obs.items()
        if isinstance(v, np.ndarray) and v.dtype == np.uint8 and v.ndim == 3
    ]
    res.check("at least one camera frame present", bool(cam_keys),
              f"cameras={cam_keys}")
    for k in cam_keys:
        f = obs[k]
        res.check(
            f"camera {k!r} frame is uint8 HxWx3",
            f.ndim == 3 and f.shape[2] == 3 and f.shape[0] > 0 and f.shape[1] > 0,
            f"shape={f.shape}",
        )

    # Liveness: the receiver's receive-timestamp should advance between reads.
    views = getattr(robot, "_cam_views", {})
    for k, view in views.items():
        try:
            _f0, _c0, r0 = view.read_latest_with_ts()
            time.sleep(0.15)
            _f1, _c1, r1 = view.read_latest_with_ts()
            res.check(f"camera {k!r} stream is live (ts advancing)", r1 > r0,
                      f"Δrecv={1e3 * (r1 - r0):.1f}ms")
        except Exception as exc:  # noqa: BLE001
            res.check(f"camera {k!r} stream is live (ts advancing)", False, repr(exc))
    return obs


def _check_motion(
    robot: Any, args: argparse.Namespace, action_keys: list[str], res: _Results
) -> None:
    """Operator-gated: nudge one joint per arm, verify tracking, return to start."""
    nudge_keys = [f"left_{args.nudge_joint}.pos", f"right_{args.nudge_joint}.pos"]
    missing = [k for k in nudge_keys if k not in action_keys]
    if missing:
        res.check("nudge joint is valid", False, f"unknown key(s): {missing}")
        return

    start = _joint_action(robot.get_observation(), action_keys)
    target = dict(start)
    for k in nudge_keys:
        target[k] = start[k] + args.nudge_rad

    print(
        f"\n  About to nudge {nudge_keys} by +{args.nudge_rad} rad "
        f"(then return to start). Ensure the workspace is clear.",
        flush=True,
    )
    if not args.yes:
        try:
            ans = input("  Proceed with motion? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = "n"
        if ans != "y":
            res.skip("arm nudge + tracking", "declined at prompt")
            res.skip("gripper open/close", "declined at prompt")
            return

    # --- arm nudge + tracking ---
    robot.send_action(target)
    time.sleep(args.settle_s)
    after = _joint_action(robot.get_observation(), action_keys)
    for k in nudge_keys:
        moved = after[k] - start[k]
        ok = moved >= args.track_frac * args.nudge_rad  # right direction + magnitude
        res.check(f"{k} tracked the nudge", ok,
                  f"commanded +{args.nudge_rad:.3f}, moved {moved:+.3f} rad")
    robot.send_action(start)  # return to start
    time.sleep(args.settle_s)
    back = _joint_action(robot.get_observation(), action_keys)
    for k in nudge_keys:
        res.check(f"{k} returned to start", abs(back[k] - start[k]) <= args.nudge_rad,
                  f"residual {back[k] - start[k]:+.3f} rad")

    # --- gripper open/close (restored after) ---
    grip_keys = ["left_gripper.pos", "right_gripper.pos"]
    if all(k in action_keys for k in grip_keys):
        orig = _joint_action(robot.get_observation(), action_keys)
        for label, val in (("open", 1.0), ("close", 0.0)):
            cmd = dict(orig)
            for k in grip_keys:
                cmd[k] = val
            robot.send_action(cmd)
            time.sleep(args.settle_s)
            obs = _joint_action(robot.get_observation(), action_keys)
            moved_ok = all(abs(obs[k] - val) < abs(orig[k] - val) + 1e-6 for k in grip_keys)
            res.check(f"grippers moved toward {label} ({val})", moved_ok,
                      f"now={[round(obs[k], 2) for k in grip_keys]}")
        robot.send_action(orig)  # restore
        time.sleep(args.settle_s)
    else:
        res.skip("gripper open/close", "no gripper keys")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(level=logging.INFO)

    if args.nudge_rad <= 0:
        raise SystemExit("--nudge-rad must be > 0")

    # Heavy / hardware imports are deferred so --help works off-robot.
    from .config import build_adapter_config
    from .robot import AxolNativeRobot

    cfg = build_adapter_config(_build_extra(args), _cameras(args))
    max_step = float(getattr(cfg.axol_config, "max_step_rad", 0.5))
    if args.nudge_rad >= max_step:
        raise SystemExit(
            f"--nudge-rad ({args.nudge_rad}) must be < max_step_rad ({max_step}); "
            "a larger step would be dropped by motion_control."
        )

    res = _Results()
    robot = AxolNativeRobot(cfg)
    print("Connecting to Axol (CAN + telemetry + ZED cameras)...", flush=True)
    try:
        robot.connect()
    except ConnectionError as exc:
        # Most bring-up failures here are a wrong/disconnected serial; show the
        # serials the SDK can actually see so the operator can fix --camera.
        from almond_axol.lerobot.camera import ZedCamera

        try:
            found = ZedCamera.find_cameras()
        except Exception:  # noqa: BLE001
            found = []
        serials = ", ".join(str(c["serial"]) for c in found) or "none detected"
        raise SystemExit(
            f"Failed to open Axol cameras: {exc}\n"
            f"Connected ZED serials: {serials}. Check the --camera <name>=<serial> "
            "values (and that zed_x_daemon is up)."
        ) from exc
    try:
        action_keys = robot.action_features
        print("\n== Observation check ==", flush=True)
        _check_observation(robot, action_keys, res)

        print("\n== Action check ==", flush=True)
        if args.no_move:
            res.skip("arm nudge + tracking", "--no-move")
            res.skip("gripper open/close", "--no-move")
        else:
            _check_motion(robot, args, action_keys, res)
    finally:
        print("\nDisconnecting...", flush=True)
        try:
            robot.disconnect()
        except Exception:  # noqa: BLE001
            _logger.warning("disconnect failed", exc_info=True)

    return res.summary()


if __name__ == "__main__":
    raise SystemExit(main())
