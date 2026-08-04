// Copied from Interlatent-Main site/src/lib/teleop/dlsSolver.ts @ f7e4bfb6 (2026-07-30). Upstream is the dashboard copy; sync fixes both ways.
/**
 * Generic N-DOF weighted damped-least-squares IK, in the browser.
 *
 * One Newton step per call on the weighted stacked task
 *
 *     [ pos_err ]        [ jacp        ]
 *     [ w_rot·rot_err ]  [ w_rot·jacr ]      (≤6 × n)
 *
 * with singularity-adaptive damping, a Tikhonov pull to `q_rest`, an antipode
 * gate, joint-limit clamp, and a per-joint Δq cap. This is a 1:1 port of the
 * engine's `So101DlsSolver` generalized to any n (it already was), wrapped in
 * the `RetargetEngine` seams: robot-native units ↔ radians affine, gripper
 * from pinch, passthrough of undriven action indices from the seed, and the
 * warm-start-with-leash that lets the command run ahead of the (returned)
 * measured state without accumulating unbounded error.
 *
 * Per the design decision, the browser always uses this generic weighted DLS
 * regardless of the spec's `solver_type` (it subsumes 5/6/7-DOF); the
 * pod-side `decoupled_6dof` specialization is not ported.
 *
 * Wire conventions: positions in meters, quaternions xyzw. The output is the
 * full joint-target vector in action order + robot units (the `mode="targets"`
 * payload).
 */
import { Quat, Vec3 } from './quat';
import {
  KinematicSpec,
  forwardKinematics,
  siteJacobian,
  orientationError,
} from './kinematics';

// Warm-start leash (rad/joint): how far the integrated command may run ahead
// of measured state before it's pulled back. Mirrors engine.py _WARMSTART_LEASH_RAD.
const WARMSTART_LEASH_RAD = 0.35;

// --- small dense linear algebra (n ≤ ~7) ---

/** A (n×n) symmetric = Jᵀ·J for J given as rows×n. */
function jTj(J: number[][], n: number): number[][] {
  const rows = J.length;
  const out = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      let s = 0;
      for (let r = 0; r < rows; r++) s += J[r][i] * J[r][j];
      out[i][j] = s;
      out[j][i] = s;
    }
  }
  return out;
}

/** Jᵀ·e for J rows×n, e length rows. */
function jTvec(J: number[][], e: number[], n: number): number[] {
  const rows = J.length;
  const out = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let r = 0; r < rows; r++) s += J[r][i] * e[r];
    out[i] = s;
  }
  return out;
}

/** Solve A·x = b for symmetric-PD A (n×n) via Gaussian elimination w/ partial pivot. */
function solveLinear(A: number[][], b: number[]): number[] {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++)
      if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    if (piv !== col) [M[col], M[piv]] = [M[piv], M[col]];
    const d = M[col][col] || 1e-12;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = M[r][col] / d;
      for (let c = col; c <= n; c++) M[r][c] -= f * M[col][c];
    }
  }
  return M.map((row, i) => row[n] / (M[i][i] || 1e-12));
}

/**
 * Smallest singular value of J (rows×n) = sqrt(smallest eigenvalue of JᵀJ).
 * Eigenvalues of the small symmetric n×n JᵀJ via cyclic Jacobi — the
 * small-matrix stand-in for the numpy SVD the pod uses.
 */
function smallestSingularValue(J: number[][], n: number): number {
  const A = jTj(J, n);
  for (let sweep = 0; sweep < 24; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++)
      for (let q = p + 1; q < n; q++) off += A[p][q] * A[p][q];
    if (off < 1e-20) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(A[p][q]) < 1e-18) continue;
        const theta = (A[q][q] - A[p][p]) / (2 * A[p][q]);
        const t =
          Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = A[k][p];
          const akq = A[k][q];
          A[k][p] = c * akp - s * akq;
          A[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = A[p][k];
          const aqk = A[q][k];
          A[p][k] = c * apk - s * aqk;
          A[q][k] = s * apk + c * aqk;
        }
      }
    }
  }
  let lam = Infinity;
  for (let i = 0; i < n; i++) lam = Math.min(lam, A[i][i]);
  return Math.sqrt(Math.max(0, lam));
}

function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}

export class DlsSolver {
  private spec: KinematicSpec;
  private n: number;
  private qCmdPrev: number[] | null = null;

  constructor(spec: KinematicSpec) {
    this.spec = spec;
    this.n = spec.n_ik_joints;
  }

  /** Forget the integrated command; next solve re-seeds from measured state. */
  resetWarmstart(): void {
    this.qCmdPrev = null;
  }

  /** FK at an action-order/robot-units state → (pos_m, quat_xyzw). */
  fk(actionState: number[]): { pos: Vec3; quat: Quat } {
    const q = this.seedToRad(actionState);
    const r = forwardKinematics(this.spec, q);
    return { pos: r.pos, quat: r.quat };
  }

  /**
   * One IK step toward the absolute EE target. Returns the full joint-target
   * vector in action order + robot units.
   */
  solve(
    targetPos: Vec3,
    targetQuatXyzw: Quat,
    seedActionUnits: number[],
    pinch = 0,
  ): number[] {
    const qMeas = this.seedToRad(seedActionUnits);
    let qSeed: number[];
    if (this.qCmdPrev === null) {
      qSeed = qMeas;
    } else {
      qSeed = qMeas.map((qm, i) =>
        qm + clamp(this.qCmdPrev![i] - qm, -WARMSTART_LEASH_RAD, WARMSTART_LEASH_RAD),
      );
    }
    const qNew = this.step(targetPos, targetQuatXyzw, qSeed);
    this.qCmdPrev = [...qNew];
    return this.radToAction(qNew, seedActionUnits, pinch);
  }

  // --- units seam (mirrors RetargetEngine) ---
  private seedToRad(actionUnits: number[]): number[] {
    return this.spec.joints.map((j) =>
      j.affine.scale * (actionUnits[j.action_index] ?? 0) + j.affine.offset,
    );
  }

  private radToAction(qIk: number[], seedUnits: number[], pinch: number): number[] {
    const out = [...seedUnits]; // passthrough default
    this.spec.joints.forEach((j, i) => {
      out[j.action_index] = (qIk[i] - j.affine.offset) / (j.affine.scale || 1e-12);
    });
    const g = this.spec.gripper;
    if (g) {
      const [openCmd, closedCmd] = g.range;
      const p = clamp(pinch, 0, 1);
      out[g.action_index] = openCmd + p * (closedCmd - openCmd);
    }
    return out;
  }

  // --- one weighted-DLS Newton step (mirrors So101DlsSolver.solve) ---
  private step(targetPos: Vec3, targetQuat: Quat, qSeedIn: number[]): number[] {
    const n = this.n;
    const d = this.spec.damping;
    const wRot = this.spec.w_rot;
    const q = qSeedIn.slice(0, n);

    const fk = forwardKinematics(this.spec, q);
    const posErr: Vec3 = [
      targetPos[0] - fk.pos[0],
      targetPos[1] - fk.pos[1],
      targetPos[2] - fk.pos[2],
    ];
    const eRot = orientationError(fk.quat, targetQuat);
    const rotGated = Math.hypot(eRot[0], eRot[1], eRot[2]) > d.rot_err_hold;

    const { jacp, jacr } = siteJacobian(this.spec, fk);

    // Stacked weighted task [pos; w_rot·rot] (drop rot rows when gated).
    const J: number[][] = [jacp[0], jacp[1], jacp[2]];
    const e: number[] = [posErr[0], posErr[1], posErr[2]];
    if (!rotGated) {
      J.push(jacr[0].map((x) => wRot * x));
      J.push(jacr[1].map((x) => wRot * x));
      J.push(jacr[2].map((x) => wRot * x));
      e.push(wRot * eRot[0], wRot * eRot[1], wRot * eRot[2]);
    }

    const sigmaMin = smallestSingularValue(J, n);
    const ramp = Math.max(0, 1 - sigmaMin / Math.max(d.w0, 1e-12));
    const lam2 = d.lam_pos ** 2 + d.lam0 ** 2 * ramp ** 2;
    const mu2 = d.mu ** 2;

    // A = JᵀJ + (lam²+mu²)I ; b = Jᵀe + mu²(q_rest − q)
    const A = jTj(J, n);
    for (let i = 0; i < n; i++) A[i][i] += lam2 + mu2;
    const b = jTvec(J, e, n);
    for (let i = 0; i < n; i++) b[i] += mu2 * (this.spec.joints[i].q_rest - q[i]);

    const dq = solveLinear(A, b);

    // Joint-limit clamp, then per-joint Δq cap, then re-clamp.
    const qNew = q.map((qi, i) => {
      const lim = this.spec.joints[i].limit;
      return clamp(qi + dq[i], lim[0], lim[1]);
    });
    let dqTotal = qNew.map((qn, i) => qn - q[i]);
    dqTotal = dqTotal.map((v, i) => {
      const cap = this.spec.joints[i].max_dq;
      return cap == null ? v : clamp(v, -cap, cap);
    });
    return q.map((qi, i) => {
      const lim = this.spec.joints[i].limit;
      return clamp(qi + dqTotal[i], lim[0], lim[1]);
    });
  }
}
