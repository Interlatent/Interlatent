"""MuJoCo model construction behind the kinematic-spec exporter.

Generalization of vr-teleop-kit's ``ik/model.py``: the DK1-hardcoded body
names, tool0 offset, and anchor placement all come from :class:`IkConfig`
instead. Two named sites are added to the compiled spec:

  tool0   — the EE target site on ``cfg.ee_body`` (URDF importers collapse
            fixed-joint tool frames into their parent, so we re-attach it).
  anchor  — the wrist-invariant position-task anchor on ``cfg.anchor_body``
            (``decoupled_6dof`` only).

:class:`KinematicModel` wraps the compiled model and resolves the IK joints
by URDF joint *name* to qpos/dof addresses — no assumption that the IK
joints are the first N qpos entries (they aren't for SO101's gripper-bearing
URDF or a bimanual YAM).

Mesh paths: the shipped URDFs are kinematics-only, so normally no mesh is
referenced at all. One that does must reference meshes relative to the bundle
root (``package://`` URIs are rewritten at curation time — MjSpec cannot
resolve them); ``build_kinematic_model`` compiles with the URDF's own
directory as the mesh dir, which is the bundle root by construction.
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from ik_config import IkConfig

TOOL0_SITE = "tool0"
ANCHOR_SITE = "anchor"


def rpy_to_wxyz(rpy: np.ndarray) -> np.ndarray:
    r, p, y = rpy
    cr, sr = np.cos(r / 2), np.sin(r / 2)
    cp, sp = np.cos(p / 2), np.sin(p / 2)
    cy, sy = np.cos(y / 2), np.sin(y / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


class KinematicModel:
    """Compiled model + IK-joint indexing + site FK/Jacobian helpers.

    All public methods speak IK-joint-space vectors (length
    ``cfg.n_ik_joints``, radians, in ``cfg.urdf_joint_names`` order).
    """

    def __init__(self, urdf_path: str | Path, cfg: IkConfig) -> None:
        self.cfg = cfg
        urdf_path = Path(urdf_path)
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF not found at {urdf_path}")

        spec = mujoco.MjSpec.from_file(str(urdf_path))

        # mujoco renamed MjSpec.find_body -> MjSpec.body in 3.3.
        _find_body = getattr(spec, "body", None) or spec.find_body

        ee = _find_body(cfg.ee_body)
        if ee is None:
            raise RuntimeError(
                f"ik_config ee_body {cfg.ee_body!r} not found in URDF spec"
            )
        ee.add_site(
            name=TOOL0_SITE,
            pos=list(cfg.tool0_offset_xyz),
            quat=rpy_to_wxyz(np.asarray(cfg.tool0_offset_rpy)).tolist(),
        )
        if cfg.anchor_body is not None:
            anchor = _find_body(cfg.anchor_body)
            if anchor is None:
                raise RuntimeError(
                    f"ik_config anchor_body {cfg.anchor_body!r} not found in URDF spec"
                )
            anchor.add_site(
                name=ANCHOR_SITE,
                pos=list(cfg.anchor_offset_xyz or (0.0, 0.0, 0.0)),
            )

        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)

        # Resolve IK joints by URDF joint name → qpos / dof addresses.
        qpos_adr: list[int] = []
        dof_adr: list[int] = []
        limits: list[tuple[float, float]] = []
        for name in cfg.urdf_joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(
                    f"ik_config joint {name!r} not found in compiled model"
                )
            jtype = int(self.model.jnt_type[jid])
            if jtype not in (int(mujoco.mjtJoint.mjJNT_HINGE),
                             int(mujoco.mjtJoint.mjJNT_SLIDE)):
                raise RuntimeError(
                    f"ik_config joint {name!r} is not a scalar (hinge/slide) joint"
                )
            qpos_adr.append(int(self.model.jnt_qposadr[jid]))
            dof_adr.append(int(self.model.jnt_dofadr[jid]))
            if bool(self.model.jnt_limited[jid]):
                lo, hi = self.model.jnt_range[jid]
                limits.append((float(lo), float(hi)))
            else:
                limits.append((-np.inf, np.inf))

        self.qpos_adr = np.asarray(qpos_adr, dtype=int)
        self.dof_adr = np.asarray(dof_adr, dtype=int)
        # Joint limits straight from the compiled model (i.e. the URDF) —
        # nothing transcribed by hand.
        self.joint_limits = np.asarray(limits, dtype=float)

        self.tool0_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, TOOL0_SITE
        )
        self.anchor_id = (
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, ANCHOR_SITE)
            if cfg.anchor_body is not None else -1
        )
        if self.tool0_id < 0:
            raise RuntimeError("KinematicModel: tool0 site missing from model")

    # ---- FK ----------------------------------------------------------

    def set_qpos(self, q_ik: np.ndarray) -> None:
        """Run kinematics + comPos at the given IK-joint vector (radians).
        Non-IK joints are held at zero."""
        self.data.qpos[:] = 0.0
        self.data.qpos[self.qpos_adr] = np.asarray(q_ik, dtype=float)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)  # for site Jacobians

    def site_pos(self, site_id: int) -> np.ndarray:
        return self.data.site_xpos[site_id].copy()

    def site_R(self, site_id: int) -> np.ndarray:
        return self.data.site_xmat[site_id].reshape(3, 3).copy()

    def site_quat_wxyz(self, site_id: int) -> np.ndarray:
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[site_id])
        return quat

    def jac_site(self, site_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Positional + rotational site Jacobians restricted to the IK-joint
        dof columns: each is 3 x n_ik_joints."""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)
        return jacp[:, self.dof_adr], jacr[:, self.dof_adr]


def build_kinematic_model(bundle_dir: str | Path, cfg: IkConfig) -> KinematicModel:
    """Build the model from a downloaded bundle directory."""
    return KinematicModel(Path(bundle_dir) / cfg.urdf, cfg)


__all__ = [
    "KinematicModel", "build_kinematic_model", "rpy_to_wxyz",
    "TOOL0_SITE", "ANCHOR_SITE",
]
