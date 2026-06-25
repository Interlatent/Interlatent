"""``interlatent-act`` — a one-shot manual action from the command line.

A thin CLI over the manual :meth:`~interlatent.adapters.base.ManualActionInterface.action`
seam: connect a robot, drive the named joints to absolute targets (blocking until the
arm settles), then disconnect. Joint-space only — ``shoulder_pan=30`` is a joint angle
in the robot's own frame (degrees for SO-101 / Koch), not a Cartesian point.

    interlatent-act --robot so101 --port /dev/ttyACM0 shoulder_pan=30 gripper=80 --hold-missing
    interlatent-act --robot so101 --port /dev/ttyACM0 --show   # just print the live pose

This is the same safety-gated path the example script uses (SafetyGate + delta clamp,
RobotProfile required) — it refuses a robot kind with no profile rather than moving it
unguarded. It is NOT the engine/policy path; for a cloud policy use ``interlatent-node
run``.

Kept import-light (argparse + the Pi-importable adapter base); lerobot is imported
lazily inside ``adapter.connect()``.
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional


def _joint_kv(s: str) -> tuple[str, float]:
    """Parse a ``name=value`` joint target; value must be numeric."""
    if "=" not in s:
        raise argparse.ArgumentTypeError(
            f"expected joint as name=value (e.g. shoulder_pan=30), got: {s!r}"
        )
    name, _, raw = s.partition("=")
    name = name.strip()
    try:
        value = float(raw.strip())
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"joint {name!r} target must be a number, got: {raw.strip()!r}"
        )
    return name, value


def _extra_kv(s: str) -> tuple[str, str]:
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"--robot-arg expects key=value, got: {s!r}")
    k, _, v = s.partition("=")
    return k.strip(), v.strip()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="interlatent-act",
        description="Drive a robot to absolute joint targets once, then exit. "
        "Joint-space only (no IK): each target is a joint angle in the robot's "
        "own frame.",
    )
    p.add_argument(
        "joints",
        nargs="*",
        type=_joint_kv,
        metavar="name=value",
        help="Joint targets, e.g. shoulder_pan=30 gripper=80. Use --show to just "
        "read the current pose without moving.",
    )
    p.add_argument(
        "--robot",
        default="so101",
        help="Robot kind (must have a RobotProfile). Default: so101.",
    )
    p.add_argument(
        "--port",
        required=True,
        help="Serial port for the robot, e.g. /dev/ttyACM0.",
    )
    p.add_argument(
        "--robot-arg",
        type=_extra_kv,
        action="append",
        metavar="key=value",
        help="Extra key=value passed to the LeRobot robot config (repeatable).",
    )
    p.add_argument(
        "--hold-missing",
        action="store_true",
        help="Hold any joint you don't name at its measured present position. "
        "Without this, every joint must be named or the call errors.",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for the arm to settle before failing (default: 10).",
    )
    p.add_argument(
        "--rate-hz",
        type=float,
        default=30.0,
        help="Control rate for the settle loop in Hz (default: 30).",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Print the current joint pose and exit without moving.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _print_pose(adapter, label: str) -> None:
    obs = adapter.get_observation()
    pairs = ", ".join(
        f"{_strip(f)}={obs[f]:.1f}" for f in adapter.action_features if f in obs
    )
    print(f"[{label}] {pairs}")


def _strip(feature: str) -> str:
    return feature[:-4] if feature.endswith(".pos") else feature


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not args.joints and not args.show:
        print(
            "error: no joint targets given. Pass name=value pairs (e.g. "
            "shoulder_pan=30), or --show to read the current pose.",
            file=sys.stderr,
        )
        return 2

    # Imported here so --help / arg errors don't pay the adapter import cost.
    from ..adapters.lerobot.robot import LeRobotAdapter

    extra = dict(args.robot_arg or [])
    adapter = LeRobotAdapter(args.robot, port=args.port, extra=extra)

    try:
        adapter.connect()
    except Exception as exc:  # noqa: BLE001 - surface a clean message, not a traceback
        print(f"error: could not connect to {args.robot!r} on {args.port}: {exc}",
              file=sys.stderr)
        return 1

    try:
        _print_pose(adapter, "current")
        if args.show:
            return 0

        targets = dict(args.joints)
        print(
            "moving: "
            + ", ".join(f"{n}={v:g}" for n, v in targets.items())
            + (" (holding the rest)" if args.hold_missing else "")
        )
        adapter.action(
            hold_missing=args.hold_missing,
            timeout=args.timeout,
            rate_hz=args.rate_hz,
            **targets,
        )
        _print_pose(adapter, "settled")
        return 0
    except (ValueError, RuntimeError) as exc:
        # Contract violations (unknown/missing joint, out-of-range, no profile):
        # nothing moved.
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except TimeoutError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            adapter.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
