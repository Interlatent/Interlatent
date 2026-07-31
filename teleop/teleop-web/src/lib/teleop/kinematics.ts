// Copied from Interlatent-Main site/src/lib/teleop/kinematics.ts @ f7e4bfb6 (2026-07-30). Upstream is the dashboard copy; sync fixes both ways.
/**
 * Browser-side forward kinematics + geometric Jacobian for a serial arm,
 * from the compact "kinematic spec" the backend exports
 * (engine `retarget/kinematic_spec.py`). This replaces the two things only
 * MuJoCo provided pod-side — FK (`mj_kinematics`) and the site Jacobian
 * (`mj_jacSite`) — so IK can run entirely in the browser with no WASM.
 *
 * The spec is a serial chain: each joint carries a fixed offset transform to
 * its frame at q=0 (`origin_pos` + `origin_quat_xyzw`), a local `axis`, and a
 * `type`. FK is the walk
 *
 *     for joint i:  frame = frame · offset_i · Jexp(axis_i, q_i)
 *     tool0 = frame · tool0_offset
 *
 * and the geometric Jacobian falls out of the same walk (revolute column
 * `z_i × (p_e − p_i)` / `z_i`; prismatic `z_i` / 0), where `z_i`/`p_i` are the
 * world axis/anchor of joint i captured just before its own rotation. This is
 * a 1:1 port of `fk_from_spec` / the analytic Jacobian in kinematic_spec.py,
 * verified against real MuJoCo at machine precision.
 *
 * Quaternions are xyzw throughout (matching quat.ts and the wire).
 */
import {
  Quat,
  Vec3,
  Mat3,
  quatMul,
  quatConj,
  quatToRotvec,
  rotateVec,
  rotvecToQuat,
} from './quat';

export interface SpecJoint {
  name: string;
  type: 'hinge' | 'slide';
  axis: Vec3;
  origin_pos: Vec3;
  origin_quat_xyzw: Quat;
  limit: [number, number];
  action_index: number;
  affine: { scale: number; offset: number };
  q_rest: number;
  max_dq: number | null;
}

export interface KinematicSpec {
  version: number;
  solver_type: string;
  n_ik_joints: number;
  joints: SpecJoint[];
  tool0: { origin_pos: Vec3; origin_quat_xyzw: Quat };
  gripper: { action_index: number; range: [number, number] } | null;
  damping: {
    lam_pos: number;
    lam0: number;
    w0: number;
    mu: number;
    rot_err_hold: number;
  };
  w_rot: number;
  webxr_to_base_R: Mat3;
  scale_translation: number;
  scale_rotation: number;
  pos_reach_limit: number;
  rot_reach_limit: number;
}

/** A bimanual bundle's spec: one complete `KinematicSpec` per arm, keyed
 *  under `chains` — the exact shape `export_kinematic_spec_bundle` emits
 *  (engine `retarget/kinematic_spec.py`) and the mirror of the bimanual
 *  `ik_config.json` / teleop-token `ik_hints` shape. */
export interface KinematicSpecChains {
  version?: number;
  chains: { left: KinematicSpec; right: KinematicSpec };
}

/** The spec the node serves over the relay uni stream (its installed
 *  `kinematic_spec.json`): flat for a single-arm rig, `chains`-wrapped for a
 *  bimanual one. */
export type KinematicSpecBundle = KinematicSpec | KinematicSpecChains;

export function isChainsSpec(s: KinematicSpecBundle): s is KinematicSpecChains {
  return (
    typeof s === 'object' && s !== null &&
    typeof (s as KinematicSpecChains).chains === 'object' &&
    (s as KinematicSpecChains).chains !== null
  );
}

export interface FkResult {
  /** tool0 world position (arm-base frame), meters. */
  pos: Vec3;
  /** tool0 world orientation, xyzw. */
  quat: Quat;
  /** per-joint world axis (z_i), before its own motion. */
  axes: Vec3[];
  /** per-joint world anchor (p_i). */
  anchors: Vec3[];
}

// --- tiny vec helpers (row-major, xyzw quats handled by quat.ts) ---
function add(a: Vec3, b: Vec3): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}
function sub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}
function scale(a: Vec3, s: number): Vec3 {
  return [a[0] * s, a[1] * s, a[2] * s];
}
export function cross(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}
function normalize3(v: Vec3): Vec3 {
  const n = Math.hypot(v[0], v[1], v[2]);
  return n < 1e-12 ? [0, 0, 0] : [v[0] / n, v[1] / n, v[2] / n];
}

/**
 * Forward kinematics + per-joint frames for the Jacobian.
 * `q` is the IK-joint vector in radians (hinge) / meters (slide), IK order.
 */
export function forwardKinematics(spec: KinematicSpec, q: number[]): FkResult {
  let p: Vec3 = [0, 0, 0];
  let ori: Quat = [0, 0, 0, 1];
  const axes: Vec3[] = [];
  const anchors: Vec3[] = [];

  for (let i = 0; i < spec.joints.length; i++) {
    const j = spec.joints[i];
    // Fixed offset into this joint's frame.
    p = add(p, rotateVec(j.origin_pos, ori));
    ori = quatMul(ori, j.origin_quat_xyzw);

    // World axis/anchor captured before this joint's own motion.
    const axisWorld = rotateVec(j.axis, ori);
    axes.push(axisWorld);
    anchors.push([p[0], p[1], p[2]]);

    if (j.type === 'slide') {
      p = add(p, scale(axisWorld, q[i]));
    } else {
      ori = quatMul(ori, rotvecToQuat(scale(normalize3(j.axis), q[i])));
    }
  }

  const pe = add(p, rotateVec(spec.tool0.origin_pos, ori));
  const qe = quatMul(ori, spec.tool0.origin_quat_xyzw);
  return { pos: pe, quat: qe, axes, anchors };
}

/**
 * Geometric Jacobian from an FK result. Returns `jacp` and `jacr` as 3×n
 * row-major matrices (each row length n = number of IK joints).
 */
export function siteJacobian(
  spec: KinematicSpec,
  fk: FkResult,
): { jacp: number[][]; jacr: number[][] } {
  const n = spec.joints.length;
  const jacp = [new Array(n).fill(0), new Array(n).fill(0), new Array(n).fill(0)];
  const jacr = [new Array(n).fill(0), new Array(n).fill(0), new Array(n).fill(0)];
  for (let i = 0; i < n; i++) {
    const z = fk.axes[i];
    if (spec.joints[i].type === 'slide') {
      jacp[0][i] = z[0];
      jacp[1][i] = z[1];
      jacp[2][i] = z[2];
    } else {
      const lever = sub(fk.pos, fk.anchors[i]);
      const c = cross(z, lever);
      jacp[0][i] = c[0];
      jacp[1][i] = c[1];
      jacp[2][i] = c[2];
      jacr[0][i] = z[0];
      jacr[1][i] = z[1];
      jacr[2][i] = z[2];
    }
  }
  return { jacp, jacr };
}

/** World-frame orientation error (rotation vector) from current→target, xyzw. */
export function orientationError(qCur: Quat, qTarget: Quat): Vec3 {
  // R_err = R_target · R_curᵀ  ⇔  q_target ⊗ q_cur⁻¹
  return quatToRotvec(quatMul(qTarget, quatConj(qCur)));
}
