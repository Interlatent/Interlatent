"""Kinematic-spec exporter — digest a robot bundle into a compact JSON the
**browser** IK consumes, so no URDF, meshes, or WASM MuJoCo ships to the client.

The in-browser solver needs exactly two things MuJoCo provides today: forward
kinematics and the site Jacobian, both over a fixed serial chain. Rather than
ship the URDF + a WASM engine, we drive :class:`~kinematic_model.KinematicModel`
once, offline, and emit a serial-chain descriptor the browser can walk in ~50
lines of TypeScript:

  * per IK joint (in IK order): the fixed transform from the previous joint
    frame to this one at ``q=0`` (``origin_pos`` + ``origin_quat_xyzw``), the
    joint ``axis`` in its own frame, ``type`` (hinge|slide), joint ``limit``,
    and the units/layout seams the browser's in-browser solver applies
    (``action_index``, ``affine``, ``q_rest``, ``max_dq``).
  * the ``tool0`` offset from the last joint frame.
  * the generic weighted-DLS tunables (``damping``, ``w_rot``) resolved to the
    shared ``DAMPING_DEFAULTS`` the browser solver uses, plus the browser
    mapper hints (``webxr_to_base_R``, scales, reach limits) and the gripper
    block.

FK from this spec is a forward walk:

    T = I
    for joint i:   T = T · Trans(origin_pos_i) · Rot(origin_quat_i) · Jexp(axis_i, q_i)
    tool0_world = T · Trans(tool0.origin_pos) · Rot(tool0.origin_quat)

and the geometric Jacobian falls out of the same walk (revolute column
``z_i × (p_e − p_i)`` / ``z_i``; prismatic ``z_i`` / ``0``), where ``z_i`` and
``p_i`` are the world axis/anchor of joint ``i`` captured just before its own
rotation is applied. :func:`fk_from_spec` is the reference implementation of
that walk, used by the parity test and mirrored by the TypeScript port.

Quaternions are **xyzw** on the wire (matching ``quat.ts`` / the browser),
converted from MuJoCo's wxyz at this seam.

The spec is computed from the digested URDF + ``ik_config.json``; any URDF
works for the math, but the semantic bits (which body is ``tool0``, which
joint is the gripper, the unit affines) still come from ``ik_config`` — see
:mod:`ik_config`.

Run it on a MuJoCo box, from the repo root, after any URDF or ``ik_config``
change::

    python packaging/kinematic_spec.py packages/sdk/src/interlatent_robots/<kind>

then re-verify the result with ``python packaging/verify_urdf.py <kind-dir>``,
which FK-checks the written spec against the compiled model. MuJoCo is a
maintainer dependency only (``pip install mujoco numpy``): nothing in the SDK
imports this file, and the wheel ships the generated JSON, not the generator.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import mujoco
import numpy as np

from ik_config import (
    IK_CONFIG_FILENAME,
    IkConfig,
    load_ik_config,
    parse_ik_config_bundle,
    resolve_damping,
)
from kinematic_model import KinematicModel, build_kinematic_model

SPEC_VERSION = 1

# Curated alongside ik_config.json inside the bundle and committed: the node
# serves this file to the headset verbatim (node/teleop/quic_channel.py), so no
# runtime anywhere needs MuJoCo.
KINEMATIC_SPEC_FILENAME = "kinematic_spec.json"

# The damping keys the browser's generic weighted-DLS solver reads, in the wire
# shape ``kinematics.ts`` declares: these five under ``damping``, with ``w_rot``
# a sibling field rather than a member. Keep that split — it is the contract.
#
# Values resolve against ``DAMPING_DEFAULTS["so101_5dof"]`` for *every* bundle,
# whatever its solver_type: the browser has one generic solver, so a
# decoupled_6dof bundle's wrist knobs (lam_rot/lam0_rot/w0_rot) describe a
# sub-solve that does not exist there and are dropped on purpose.
_BROWSER_DAMPING_KEYS = ("lam_pos", "lam0", "w0", "mu", "rot_err_hold")


def _R_to_quat_xyzw(R: np.ndarray) -> list[float]:
    """3x3 rotation -> quaternion [x, y, z, w] (MuJoCo emits wxyz)."""
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R, dtype=float).ravel())
    return [float(q[1]), float(q[2]), float(q[3]), float(q[0])]


def _quat_xyzw_to_R(q: list[float] | np.ndarray) -> np.ndarray:
    """[x, y, z, w] -> 3x3 rotation (via MuJoCo, wxyz)."""
    q = np.asarray(q, dtype=float).reshape(4)
    out = np.zeros(9)
    mujoco.mju_quat2Mat(out, np.array([q[3], q[0], q[1], q[2]]))
    return out.reshape(3, 3)


def _joint_world_frames(
    kin: KinematicModel, cfg: IkConfig
) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray, str]], np.ndarray, np.ndarray]:
    """At the zero pose, read each IK joint's world frame (anchor + orientation),
    local axis, and type, plus the tool0 world pose.

    A revolute/prismatic joint's frame is the joint anchor with the parent
    body's orientation; the axis is defined in that body frame, so it is the
    joint's local axis directly.
    """
    m, d = kin.model, kin.data
    kin.set_qpos(np.zeros(cfg.n_ik_joints))

    frames: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]] = []
    for name in cfg.urdf_joint_names:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        bid = int(m.jnt_bodyid[jid])
        R_body = d.xmat[bid].reshape(3, 3).copy()
        p_body = d.xpos[bid].copy()
        anchor = p_body + R_body @ np.asarray(m.jnt_pos[jid], dtype=float)
        axis_local = np.asarray(m.jnt_axis[jid], dtype=float).copy()
        jtype = (
            "slide"
            if int(m.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_SLIDE)
            else "hinge"
        )
        frames.append((anchor, R_body, axis_local, jtype))

    p_tool = kin.site_pos(kin.tool0_id)
    R_tool = kin.site_R(kin.tool0_id)
    return frames, p_tool, R_tool


def export_kinematic_spec(
    bundle_dir: str | Path, cfg: Optional[IkConfig] = None
) -> dict:
    """Build the browser-facing kinematic spec from a downloaded bundle dir."""
    cfg = cfg or load_ik_config(bundle_dir)
    kin = build_kinematic_model(bundle_dir, cfg)
    frames, p_tool, R_tool = _joint_world_frames(kin, cfg)

    joints: list[dict] = []
    prev_p = np.zeros(3)
    prev_R = np.eye(3)
    for k, (p, R, axis, jtype) in enumerate(frames):
        rel_R = prev_R.T @ R
        rel_p = prev_R.T @ (p - prev_p)
        aff = cfg.affine(k)
        joints.append({
            "name": cfg.urdf_joint_names[k],
            "type": jtype,
            "axis": [float(x) for x in axis],
            "origin_pos": [float(x) for x in rel_p],
            "origin_quat_xyzw": _R_to_quat_xyzw(rel_R),
            "limit": [
                float(kin.joint_limits[k, 0]),
                float(kin.joint_limits[k, 1]),
            ],
            "action_index": int(cfg.ik_joint_action_indices[k]),
            "affine": {"scale": float(aff.scale), "offset": float(aff.offset)},
            "q_rest": float(cfg.q_rest[k]) if cfg.q_rest else 0.0,
            "max_dq": (
                float(cfg.max_dq_per_joint[k])
                if cfg.max_dq_per_joint is not None else None
            ),
        })
        prev_p, prev_R = p, R

    tool_rel_R = prev_R.T @ R_tool
    tool_rel_p = prev_R.T @ (p_tool - prev_p)

    resolved_damping = resolve_damping("so101_5dof", cfg.damping)
    damping = {k: float(resolved_damping[k]) for k in _BROWSER_DAMPING_KEYS}

    gripper = None
    if cfg.gripper_action_index is not None:
        gripper = {
            "action_index": int(cfg.gripper_action_index),
            "range": [float(cfg.gripper_range[0]), float(cfg.gripper_range[1])],
        }

    return {
        "version": SPEC_VERSION,
        "solver_type": cfg.solver_type,
        "n_ik_joints": cfg.n_ik_joints,
        "joints": joints,
        "tool0": {
            "origin_pos": [float(x) for x in tool_rel_p],
            "origin_quat_xyzw": _R_to_quat_xyzw(tool_rel_R),
        },
        "gripper": gripper,
        "damping": damping,
        "w_rot": float(resolved_damping["w_rot"]),
        "webxr_to_base_R": [list(row) for row in cfg.webxr_to_base_R],
        "scale_translation": float(cfg.scale_translation),
        "scale_rotation": float(cfg.scale_rotation),
        "pos_reach_limit": float(cfg.pos_reach_limit),
        "rot_reach_limit": float(cfg.rot_reach_limit),
    }


def export_kinematic_spec_bundle(bundle_dir: str | Path) -> dict:
    """Bundle-aware spec export: flat or ``chains``-wrapped.

    Mirrors :func:`~ik_config.parse_ik_config_bundle`: a flat ``ik_config``
    exports the flat spec exactly as :func:`export_kinematic_spec` always
    has; a wrapped one exports a full spec per arm under the same shape —
    ``{"version": ..., "chains": {"right": <spec>}}`` — each chain digested
    independently against its own config (own ``urdf``, ``ee_body``,
    ``ik_joint_action_indices``, ...). The browser's ``QuicPoseSocket``
    detects the ``chains`` key and runs one solver per populated side,
    composing them into the one flat action vector."""
    import json

    with open(Path(bundle_dir) / IK_CONFIG_FILENAME, "r", encoding="utf-8") as f:
        cfg = parse_ik_config_bundle(json.load(f))
    if isinstance(cfg, dict):
        return {
            "version": SPEC_VERSION,
            "chains": {
                side: export_kinematic_spec(bundle_dir, side_cfg)
                for side, side_cfg in cfg.items()
            },
        }
    return export_kinematic_spec(bundle_dir, cfg)


def fk_from_spec(spec: dict, q_ik: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reference FK from a kinematic spec (the walk the browser mirrors).

    ``q_ik`` is the IK-joint vector in radians (hinge) / meters (slide), in IK
    order. Returns ``(tool0_pos_world, tool0_quat_xyzw)``. Kept here as the
    single source of truth for the TS port and the parity test.
    """
    q_ik = np.asarray(q_ik, dtype=float).reshape(-1)
    p = np.zeros(3)
    R = np.eye(3)
    for i, j in enumerate(spec["joints"]):
        off_R = _quat_xyzw_to_R(j["origin_quat_xyzw"])
        off_p = np.asarray(j["origin_pos"], dtype=float)
        # frame_i (before this joint's own motion) = frame_{i-1} · offset
        p = p + R @ off_p
        R = R @ off_R
        axis = np.asarray(j["axis"], dtype=float)
        if j["type"] == "slide":
            p = p + R @ (axis * q_ik[i])
        else:  # hinge
            R = R @ _axis_angle_to_R(axis, float(q_ik[i]))
    tool = spec["tool0"]
    off_R = _quat_xyzw_to_R(tool["origin_quat_xyzw"])
    off_p = np.asarray(tool["origin_pos"], dtype=float)
    p_tool = p + R @ off_p
    R_tool = R @ off_R
    return p_tool, np.asarray(_R_to_quat_xyzw(R_tool), dtype=float)


def _axis_angle_to_R(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation about a (local) axis by ``angle`` radians."""
    a = np.asarray(axis, dtype=float)
    n = np.linalg.norm(a)
    if n < 1e-12:
        return np.eye(3)
    a = a / n
    x, y, z = a
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def write_kinematic_spec(bundle_dir: str | Path) -> Path:
    """Curation step: compute the spec and write ``kinematic_spec.json`` into
    the bundle dir (run wherever MuJoCo is available; the result is reviewed
    and committed, never generated at install time). Returns the written
    path."""
    import json

    bundle_dir = Path(bundle_dir)
    spec = export_kinematic_spec_bundle(bundle_dir)
    out = bundle_dir / KINEMATIC_SPEC_FILENAME
    with open(out, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)
    return out


def _main(argv: list[str]) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(
        description="Curate kinematic_spec.json for a robot bundle (needs MuJoCo)."
    )
    ap.add_argument("bundle_dir", help="bundle dir with a URDF + ik_config.json")
    ap.add_argument("--stdout", action="store_true",
                    help="print the spec instead of writing kinematic_spec.json")
    args = ap.parse_args(argv)
    if args.stdout:
        print(json.dumps(export_kinematic_spec_bundle(args.bundle_dir), indent=2))
    else:
        print(f"wrote {write_kinematic_spec(args.bundle_dir)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    raise SystemExit(_main(sys.argv[1:]))


__all__ = [
    "export_kinematic_spec", "export_kinematic_spec_bundle",
    "write_kinematic_spec", "fk_from_spec",
    "SPEC_VERSION", "KINEMATIC_SPEC_FILENAME",
]
