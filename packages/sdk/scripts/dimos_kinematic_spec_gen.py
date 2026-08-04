#!/usr/bin/env python3
"""Dev-time helper: print a URDF-derived joint chain for interlatent_robots'
``kinematic_spec.json`` (the browser-side VR IK descriptor,
see interlatent.robots.load_kinematic_spec).

dimos.robot.model_parser (used by dimos_profile_gen.py for joint LIMITS) does
not parse a joint's ``<origin>``/``<axis>`` -- the geometric transform data
the browser's IK solver needs -- so this script reads the URDF directly
(stdlib xml.etree, no dimos import at all) using the same "we already know
where dimos's own URDF for this kind lives" fact dimos_profile_gen.py
established. Requires the ``[dimos]`` extra only insofar as you need a URDF
path to point it at; the parsing itself has no dimos dependency.

Usage:
    python packages/sdk/scripts/dimos_kinematic_spec_gen.py \\
        --urdf /path/to/A1Z_G1Z.urdf \\
        --joint arm_joint1 --joint arm_joint2 --joint arm_joint3 \\
        --joint arm_joint4 --joint arm_joint5 --joint arm_joint6 \\
        --max-dq 0.05

Prints a JSON array of per-joint kinematic_spec entries (name, type, axis,
origin_pos, origin_quat_xyzw, limit, action_index, affine, q_rest, max_dq) --
paste into the "joints" list of a kinematic_spec.json chain. ``q_rest``
defaults to 0.0 for every joint here; override by hand to match the kind's
actual RobotProfile rest_pose if it differs.

This script does NOT attempt ik_config.json's solver-tuning fields (damping,
reach limits, anchor/tool offsets, webxr_to_base_R) -- those describe IK
solver convergence behavior and how the robot is physically mounted relative
to the VR user, neither of which is recoverable from a URDF. They need real
values from an existing working config (as a starting point) and empirical
verification against actual VR hardware, not generation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET


def _floats(text: str) -> tuple[float, ...]:
    return tuple(float(x) for x in text.split())


def _rpy_to_quat_xyzw(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """URDF's <origin rpy="..."> is a fixed-axis (extrinsic) XYZ rotation:
    R = Rz(yaw) @ Ry(pitch) @ Rx(roll). Standard closed-form conversion to a
    quaternion (matches ROS's quaternion_from_euler(..., axes='sxyz'))."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return (qx, qy, qz, qw)


def _find_joint(root: ET.Element, name: str) -> ET.Element:
    for joint in root.findall("joint"):
        if joint.get("name") == name:
            return joint
    raise SystemExit(f"joint {name!r} not found in URDF")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--joint", action="append", dest="joints", required=True,
                         help="Joint name, repeatable, in kinematic_spec order")
    parser.add_argument("--max-dq", type=float, default=0.05,
                         help="Per-joint max_dq (default matches the adapter's own "
                              "max_step_rad safety clamp, 0.05)")
    args = parser.parse_args(argv)

    tree = ET.parse(args.urdf)
    root = tree.getroot()

    entries = []
    for i, name in enumerate(args.joints):
        joint = _find_joint(root, name)
        joint_type = joint.get("type")
        origin = joint.find("origin")
        axis_el = joint.find("axis")
        limit_el = joint.find("limit")
        if origin is None or axis_el is None or limit_el is None:
            print(f"joint {name!r} missing <origin>/<axis>/<limit>", file=sys.stderr)
            return 1
        xyz = _floats(origin.get("xyz", "0 0 0"))
        rpy = _floats(origin.get("rpy", "0 0 0"))
        axis = _floats(axis_el.get("xyz", "0 0 0"))
        lower = float(limit_el.get("lower", "0"))
        upper = float(limit_el.get("upper", "0"))
        entries.append({
            "name": name,
            "type": "hinge" if joint_type == "revolute" else joint_type,
            "axis": list(axis),
            "origin_pos": list(xyz),
            "origin_quat_xyzw": list(_rpy_to_quat_xyzw(*rpy)),
            "limit": [lower, upper],
            "action_index": i,
            "affine": {"scale": 1.0, "offset": 0.0},
            "q_rest": 0.0,
            "max_dq": args.max_dq,
        })

    print(json.dumps(entries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
