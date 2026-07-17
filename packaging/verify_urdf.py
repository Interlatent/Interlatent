#!/usr/bin/env python3
"""Verify a robot's URDF compiles in MuJoCo and agrees with its kinematic_spec.

A maintainer/CI dev tool — NOT part of the installed SDK (the SDK ships no
MuJoCo; see ADR 0012). Run it on a MuJoCo box after editing a URDF/ik_config or
regenerating a spec, and especially after stripping meshes to a kinematics-only
URDF: it's the check that the model still builds with no mesh assets and that the
shipped `kinematic_spec.json` still matches the URDF.

Three levels, each gated on what's present in `robots/<kind>/`:

  1. COMPILE — load the URDF exactly as the engine does (`MjSpec.from_file` +
     `compile`) and report bodies / joints / DOF / mesh count. This is the
     "does it build without STLs" check. A missing referenced mesh fails here.
  2. IK-CONFIG — with `ik_config.json`: attach the `tool0` site on `ee_body`,
     resolve every `urdf_joint_names` entry to a scalar joint, read its limits.
     Mirrors `retarget/model.py::KinematicModel`, so a pass here means the
     engine's model builder will succeed too.
  3. SPEC PARITY — with `kinematic_spec.json`: forward-kinematics parity between
     the compiled MuJoCo model and the spec's serial-chain walk at N random
     joint configs. Catches URDF/ik_config/spec drift (the failure a spec that
     was hand-edited, or generated from a different URDF, would show).

Usage::

    python packaging/verify_urdf.py robots/nori
    python packaging/verify_urdf.py --all
    python packaging/verify_urdf.py path/to/robot.urdf --ik-config ic.json \
        --kinematic-spec spec.json --samples 512

Exit status is nonzero if any check fails, so it drops straight into CI.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ROBOTS_DIR = REPO_ROOT / "robots"

DEFAULT_SAMPLES = 256
DEFAULT_POS_TOL = 1e-4   # metres
DEFAULT_ROT_TOL = 1e-3   # radians


def _fail(msg: str) -> None:
    print(f"    FAIL  {msg}")


def _ok(msg: str) -> None:
    print(f"    ok    {msg}")


# --- small SO(3)/SE(3) helpers (spec side; MuJoCo gives us matrices directly) --

def _quat_xyzw_to_R(q):
    import numpy as np
    x, y, z, w = (float(v) for v in q)
    n = (x * x + y * y + z * z + w * w) ** 0.5
    if n == 0:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _axis_angle_to_R(axis, theta):
    import numpy as np
    a = np.asarray(axis, dtype=float)
    n = np.linalg.norm(a)
    if n == 0:
        return np.eye(3)
    a = a / n
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def _T(R, p):
    import numpy as np
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = p
    return M


def _fk_from_spec(chain: dict, q):
    """Serial-chain FK from a kinematic_spec chain — the exact walk the engine's
    exporter documents and the browser mirrors. Returns (pos[3], R[3x3])."""
    import numpy as np
    T = np.eye(4)
    for j, qi in zip(chain["joints"], q):
        T = T @ _T(_quat_xyzw_to_R(j["origin_quat_xyzw"]), j["origin_pos"])
        if j["type"] == "hinge":
            T = T @ _T(_axis_angle_to_R(j["axis"], qi), np.zeros(3))
        else:  # slide
            T = T @ _T(np.eye(3), np.asarray(j["axis"], float) * qi)
    tool0 = chain["tool0"]
    T = T @ _T(_quat_xyzw_to_R(tool0["origin_quat_xyzw"]), tool0["origin_pos"])
    return T[:3, 3].copy(), T[:3, :3].copy()


def _rot_angle(Ra, Rb) -> float:
    import numpy as np
    R = Ra.T @ Rb
    c = (np.trace(R) - 1.0) / 2.0
    return float(np.arccos(max(-1.0, min(1.0, c))))


# --- MuJoCo model (mirrors retarget/model.py::KinematicModel) ------------------

def _rpy_to_wxyz(rpy):
    import numpy as np
    r, p, y = rpy
    cr, sr = np.cos(r / 2), np.sin(r / 2)
    cp, sp = np.cos(p / 2), np.sin(p / 2)
    cy, sy = np.cos(y / 2), np.sin(y / 2)
    return [cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy]


class _Model:
    """Compiled model + IK-joint indexing + tool0 FK. `cfg` is a raw ik_config
    (flat) dict: ee_body, urdf_joint_names, tool0_offset_*, anchor_*."""

    def __init__(self, mujoco, urdf_path: Path, cfg: dict):
        self.mj = mujoco
        spec = mujoco.MjSpec.from_file(str(urdf_path))
        find_body = getattr(spec, "body", None) or spec.find_body

        ee = find_body(cfg["ee_body"])
        if ee is None:
            raise RuntimeError(f"ee_body {cfg['ee_body']!r} not in URDF")
        ee.add_site(
            name="tool0",
            pos=list(cfg.get("tool0_offset_xyz") or [0, 0, 0]),
            quat=_rpy_to_wxyz(cfg.get("tool0_offset_rpy") or [0, 0, 0]),
        )
        if cfg.get("anchor_body"):
            anchor = find_body(cfg["anchor_body"])
            if anchor is None:
                raise RuntimeError(f"anchor_body {cfg['anchor_body']!r} not in URDF")
            anchor.add_site(name="anchor",
                            pos=list(cfg.get("anchor_offset_xyz") or [0, 0, 0]))

        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)

        import numpy as np
        self.qpos_adr, limits = [], []
        for name in cfg["urdf_joint_names"]:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"joint {name!r} not in compiled model")
            jtype = int(self.model.jnt_type[jid])
            if jtype not in (int(mujoco.mjtJoint.mjJNT_HINGE),
                             int(mujoco.mjtJoint.mjJNT_SLIDE)):
                raise RuntimeError(f"joint {name!r} is not hinge/slide (scalar)")
            self.qpos_adr.append(int(self.model.jnt_qposadr[jid]))
            if bool(self.model.jnt_limited[jid]):
                lo, hi = self.model.jnt_range[jid]
                limits.append((float(lo), float(hi)))
            else:
                limits.append((-np.inf, np.inf))
        self.qpos_adr = np.asarray(self.qpos_adr, dtype=int)
        self.joint_limits = np.asarray(limits, dtype=float)
        self.tool0_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tool0")

    def fk_tool0(self, q_ik):
        import numpy as np
        self.data.qpos[:] = 0.0
        self.data.qpos[self.qpos_adr] = np.asarray(q_ik, float)
        self.mj.mj_kinematics(self.model, self.data)
        pos = self.data.site_xpos[self.tool0_id].copy()
        R = self.data.site_xmat[self.tool0_id].reshape(3, 3).copy()
        return pos, R


# --- per-robot verification ----------------------------------------------------

def _iter_chains(obj: dict):
    """Yield (label, section) for a flat or {"chains": {...}} ik_config/spec."""
    if isinstance(obj.get("chains"), dict):
        for side, sec in obj["chains"].items():
            yield side, sec
    else:
        yield "", obj


def _mesh_ref_count(urdf_path: Path) -> int:
    import xml.etree.ElementTree as ET
    return sum(1 for _ in ET.parse(urdf_path).getroot().iter("mesh"))


def verify(mujoco, urdf: Path, ik_cfg: dict | None, spec: dict | None,
           samples: int, pos_tol: float, rot_tol: float) -> bool:
    import numpy as np
    ok = True

    # 1. COMPILE (no ik_config needed) — the bare "builds without STLs" check.
    try:
        m0 = mujoco.MjSpec.from_file(str(urdf)).compile()
    except Exception as e:
        _fail(f"MuJoCo compile: {e}")
        return False
    n_mesh_file = _mesh_ref_count(urdf)
    _ok(f"compiles — bodies={m0.nbody} joints={m0.njnt} dof={m0.nv} "
        f"mesh-geoms={m0.nmesh} mesh-refs-in-urdf={n_mesh_file}"
        + (" (kinematics-only)" if n_mesh_file == 0 else ""))

    if ik_cfg is None:
        print("    (no ik_config.json — skipped ik-config + parity checks)")
        return ok

    spec_chains = dict(_iter_chains(spec)) if spec is not None else {}

    for label, cfg in _iter_chains(ik_cfg):
        tag = f"[{label}] " if label else ""
        # 2. IK-CONFIG — mirror KinematicModel construction.
        try:
            model = _Model(mujoco, urdf, cfg)
        except Exception as e:
            _fail(f"{tag}ik_config: {e}")
            ok = False
            continue
        n = len(cfg["urdf_joint_names"])
        _ok(f"{tag}ik_config resolves {n} IK joint(s) on ee_body "
            f"{cfg['ee_body']!r}")

        # 3. SPEC PARITY.
        if spec is None:
            continue
        sc = spec_chains.get(label) if label else spec
        if sc is None:
            _fail(f"{tag}kinematic_spec has no matching chain")
            ok = False
            continue
        if len(sc.get("joints", [])) != n:
            _fail(f"{tag}spec has {len(sc.get('joints', []))} joints, "
                  f"ik_config has {n}")
            ok = False
            continue

        rng = np.random.default_rng(0)
        lo = np.where(np.isfinite(model.joint_limits[:, 0]),
                      model.joint_limits[:, 0], -np.pi)
        hi = np.where(np.isfinite(model.joint_limits[:, 1]),
                      model.joint_limits[:, 1], np.pi)
        max_pos = max_rot = 0.0
        for _ in range(samples):
            q = lo + (hi - lo) * rng.random(n)
            p_mj, R_mj = model.fk_tool0(q)
            p_sp, R_sp = _fk_from_spec(sc, q)
            max_pos = max(max_pos, float(np.linalg.norm(p_mj - p_sp)))
            max_rot = max(max_rot, _rot_angle(R_mj, R_sp))
        if max_pos <= pos_tol and max_rot <= rot_tol:
            _ok(f"{tag}FK parity over {samples} configs: "
                f"max pos={max_pos:.2e} m, max rot={max_rot:.2e} rad")
        else:
            _fail(f"{tag}FK parity: max pos={max_pos:.2e} m "
                  f"(tol {pos_tol:.0e}), max rot={max_rot:.2e} rad "
                  f"(tol {rot_tol:.0e}) — URDF/spec disagree")
            ok = False
    return ok


def _load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _resolve_inputs(target: Path, args):
    """(urdf, ik_config_path, spec_path) from a dir or an explicit urdf path."""
    if target.is_dir():
        urdfs = sorted(target.glob("*.urdf"))
        if not urdfs:
            raise SystemExit(f"{target}: no .urdf found")
        if len(urdfs) > 1:
            raise SystemExit(f"{target}: {len(urdfs)} .urdf files, name one")
        urdf = urdfs[0]
        ic = Path(args.ik_config) if args.ik_config else target / "ik_config.json"
        sp = (Path(args.kinematic_spec) if args.kinematic_spec
              else target / "kinematic_spec.json")
    else:
        urdf = target
        ic = Path(args.ik_config) if args.ik_config else urdf.parent / "ik_config.json"
        sp = (Path(args.kinematic_spec) if args.kinematic_spec
              else urdf.parent / "kinematic_spec.json")
    return urdf, ic, sp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("target", nargs="?", help="a robots/<kind>/ dir or a .urdf path")
    ap.add_argument("--all", action="store_true", help="verify every robots/<kind>/")
    ap.add_argument("--ik-config", help="override ik_config.json path")
    ap.add_argument("--kinematic-spec", help="override kinematic_spec.json path")
    ap.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    ap.add_argument("--pos-tol", type=float, default=DEFAULT_POS_TOL)
    ap.add_argument("--rot-tol", type=float, default=DEFAULT_ROT_TOL)
    args = ap.parse_args()

    try:
        import mujoco  # noqa: F401
        import numpy  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            f"verify_urdf needs MuJoCo + numpy ({e}). Install: pip install mujoco numpy "
            "(this is a maintainer/CI tool; the SDK itself ships no MuJoCo)."
        )

    if args.all:
        targets = sorted(p for p in ROBOTS_DIR.iterdir()
                         if p.is_dir() and any(p.glob("*.urdf")))
        if not targets:
            raise SystemExit("no robots/<kind>/ dirs with a .urdf")
    elif args.target:
        targets = [Path(args.target)]
    else:
        ap.error("give a robots/<kind>/ dir, a .urdf path, or --all")

    all_ok = True
    for target in targets:
        urdf, ic_path, sp_path = _resolve_inputs(target, args)
        print(f"\n{target if target.is_dir() else urdf}:")
        ok = verify(
            mujoco, urdf, _load_json(ic_path), _load_json(sp_path),
            args.samples, args.pos_tol, args.rot_tol,
        )
        all_ok = all_ok and ok

    print("\n" + ("all URDFs verified" if all_ok else "VERIFICATION FAILED"))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
