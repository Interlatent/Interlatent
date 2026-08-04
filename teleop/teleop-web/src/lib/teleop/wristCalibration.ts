// Copied from Interlatent-Main site/src/lib/teleop/wristCalibration.ts @ f7e4bfb6 (2026-07-30). Upstream is the dashboard copy; sync fixes both ways.
/**
 * Wrist-pivot calibration — port of vr-teleop-kit's in-VR ritual
 * (relay/web/client.js).
 *
 * WebXR gripSpace's origin is at the palm, but the operator's actual wrist
 * pivot is offset from the grip — typically backward and slightly off-axis.
 * Feeding the raw palm pose into the clutch mapper means every pure wrist
 * rotation also translates the readout point in a wide arc, which the IK
 * dutifully chases as ghost translation. Shifting the readout by a
 * controller-local offset `o` so it coincides with the pivot makes pure
 * wrist twists produce ~zero translation.
 *
 * The solve: given ~5s of samples (p, R) captured while the wrist pivot was
 * held still but freely rotated, find `o` such that p + R·o is constant:
 *
 *   o = -(Σ dRᵀ dR)⁻¹ (Σ dRᵀ dp)     (mean-centered dR, dp)
 *
 * The RMS residual of the recovered pivot path validates the capture —
 * large residual means the arm actually moved.
 */
import { Mat3, Quat, Vec3 } from './quat';

export interface PivotSample {
  p: Vec3;
  R: Mat3;
}

export interface WristOffsets {
  left: Vec3;
  right: Vec3;
}

export const CALIB_DURATION_S = 5.0;
/** 15mm RMS — anything larger probably means the arm moved. */
export const CALIB_RESIDUAL_MAX = 0.015;
/** Sanity cap; a real wrist offset is a few cm. */
export const CALIB_OFFSET_MAX = 0.2;
/** Analog grip value that counts as "pressed" for the start trigger. */
export const CALIB_GRIP_THRESHOLD = 0.7;
/** Hand-tuned starting point used for a hand that wasn't calibrated. */
export const DEFAULT_WRIST_OFFSET: Vec3 = [0, 0, 0.05];

const STORAGE_KEY = 'interlatent:vrteleop:wrist_offset_v1';

export function quatToMat3(q: Quat): Mat3 {
  const [x, y, z, w] = q;
  const xx = x * x, yy = y * y, zz = z * z;
  const xy = x * y, xz = x * z, yz = y * z;
  const wx = w * x, wy = w * y, wz = w * z;
  return [
    [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
    [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
    [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
  ];
}

function solve3x3(M: Mat3, v: Vec3): { x: Vec3 | null; det: number } {
  const [m00, m01, m02] = M[0];
  const [m10, m11, m12] = M[1];
  const [m20, m21, m22] = M[2];
  const det =
    m00 * (m11 * m22 - m12 * m21) -
    m01 * (m10 * m22 - m12 * m20) +
    m02 * (m10 * m21 - m11 * m20);
  const tr = Math.abs(m00) + Math.abs(m11) + Math.abs(m22);
  if (!Number.isFinite(det) || Math.abs(det) < 1e-6 * Math.max(1, tr) ** 3) {
    return { x: null, det };
  }
  const dx =
    v[0] * (m11 * m22 - m12 * m21) -
    m01 * (v[1] * m22 - m12 * v[2]) +
    m02 * (v[1] * m21 - m11 * v[2]);
  const dy =
    m00 * (v[1] * m22 - m12 * v[2]) -
    v[0] * (m10 * m22 - m12 * m20) +
    m02 * (m10 * v[2] - v[1] * m20);
  const dz =
    m00 * (m11 * v[2] - v[1] * m21) -
    m01 * (m10 * v[2] - v[1] * m20) +
    v[0] * (m10 * m21 - m11 * m20);
  return { x: [dx / det, dy / det, dz / det], det };
}

export type PivotResult =
  | { ok: true; o: Vec3; rms: number; n: number }
  | { ok: false; reason: string };

export function solvePivot(samples: PivotSample[]): PivotResult {
  const N = samples.length;
  if (N < 30) return { ok: false, reason: `too few samples (${N})` };
  const pbar: Vec3 = [0, 0, 0];
  const Rbar: Mat3 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (const s of samples) {
    for (let i = 0; i < 3; i++) {
      pbar[i] += s.p[i];
      for (let j = 0; j < 3; j++) Rbar[i][j] += s.R[i][j];
    }
  }
  for (let i = 0; i < 3; i++) {
    pbar[i] /= N;
    for (let j = 0; j < 3; j++) Rbar[i][j] /= N;
  }
  const A: Mat3 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  const b: Vec3 = [0, 0, 0];
  for (const s of samples) {
    const dR: Mat3 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++) dR[i][j] = s.R[i][j] - Rbar[i][j];
    const dp: Vec3 = [s.p[0] - pbar[0], s.p[1] - pbar[1], s.p[2] - pbar[2]];
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        let aij = 0;
        for (let k = 0; k < 3; k++) aij += dR[k][i] * dR[k][j];
        A[i][j] += aij;
      }
      let bi = 0;
      for (let k = 0; k < 3; k++) bi += dR[k][i] * dp[k];
      b[i] += bi;
    }
  }
  const sol = solve3x3(A, [-b[0], -b[1], -b[2]]);
  if (!sol.x) {
    return { ok: false, reason: `ill-conditioned (det=${sol.det.toExponential(2)})` };
  }
  const o = sol.x;
  // Residual: spread of p + R·o around its mean — how still the recovered
  // pivot actually was.
  const pivots = samples.map((s): Vec3 => [
    s.p[0] + s.R[0][0] * o[0] + s.R[0][1] * o[1] + s.R[0][2] * o[2],
    s.p[1] + s.R[1][0] * o[0] + s.R[1][1] * o[1] + s.R[1][2] * o[2],
    s.p[2] + s.R[2][0] * o[0] + s.R[2][1] * o[1] + s.R[2][2] * o[2],
  ]);
  const cm: Vec3 = [0, 0, 0];
  for (const pv of pivots) for (let i = 0; i < 3; i++) cm[i] += pv[i] / N;
  let sumSq = 0;
  for (const pv of pivots) {
    const dx = pv[0] - cm[0], dy = pv[1] - cm[1], dz = pv[2] - cm[2];
    sumSq += dx * dx + dy * dy + dz * dz;
  }
  const rms = Math.sqrt(sumSq / N);
  return { ok: true, o, rms, n: N };
}

/** Validate one hand's solve against the residual + magnitude gates. */
export function pivotPasses(r: PivotResult): r is Extract<PivotResult, { ok: true }> {
  return (
    r.ok &&
    r.rms <= CALIB_RESIDUAL_MAX &&
    Math.abs(r.o[0]) <= CALIB_OFFSET_MAX &&
    Math.abs(r.o[1]) <= CALIB_OFFSET_MAX &&
    Math.abs(r.o[2]) <= CALIB_OFFSET_MAX
  );
}

export function loadWristOffsets(): WristOffsets | null {
  try {
    const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null');
    if (
      stored &&
      Array.isArray(stored.left) && stored.left.length === 3 &&
      Array.isArray(stored.right) && stored.right.length === 3
    ) {
      return { left: stored.left as Vec3, right: stored.right as Vec3 };
    }
  } catch {
    // corrupted / unavailable storage — treat as uncalibrated
  }
  return null;
}

export function saveWristOffsets(offsets: WristOffsets): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(offsets));
  } catch {
    // storage unavailable — calibration still applies for this session
  }
}

export function clearWristOffsets(): void {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    // ignore
  }
}
