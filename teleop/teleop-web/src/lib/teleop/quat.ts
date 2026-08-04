// Copied from Interlatent-Main site/src/lib/teleop/quat.ts @ f7e4bfb6 (2026-07-30). Upstream is the dashboard copy; sync fixes both ways.
/**
 * Quaternion + small matrix helpers for the browser-side clutch mapper.
 *
 * Convention: quaternions are **xyzw** ([x, y, z, w]) — the WebXR /
 * TeleopFrame wire convention. (The reference implementation in
 * vr-teleop-kit uses MuJoCo's wxyz; the math here is identical, only the
 * component bookkeeping differs. The pod converts at its own seam.)
 */

export type Quat = [number, number, number, number]; // x, y, z, w
export type Vec3 = [number, number, number];
export type Mat3 = [Vec3, Vec3, Vec3]; // row-major

export const QUAT_IDENTITY: Quat = [0, 0, 0, 1];

export function quatMul(a: Quat, b: Quat): Quat {
  const [ax, ay, az, aw] = a;
  const [bx, by, bz, bw] = b;
  return [
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
    aw * bw - ax * bx - ay * by - az * bz,
  ];
}

export function quatConj(q: Quat): Quat {
  return [-q[0], -q[1], -q[2], q[3]];
}

export function quatNormalize(q: Quat): Quat {
  const n = Math.hypot(q[0], q[1], q[2], q[3]);
  if (n < 1e-12) return [...QUAT_IDENTITY];
  return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
}

export function quatDot(a: Quat, b: Quat): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}

/**
 * Raise a quaternion to a scalar power: keep the axis, scale the angle by k.
 * quatPow(q, 1) = q, quatPow(q, 0) = identity.
 */
export function quatPow(q: Quat, k: number): Quat {
  const w = q[3];
  const vn = Math.hypot(q[0], q[1], q[2]);
  const halfAngle = Math.atan2(vn, w);
  if (halfAngle < 1e-9) return [...QUAT_IDENTITY];
  const ax = q[0] / vn, ay = q[1] / vn, az = q[2] / vn;
  const newHalf = k * halfAngle;
  const s = Math.sin(newHalf);
  return [s * ax, s * ay, s * az, Math.cos(newHalf)];
}

/** Rotation vector (axis · angle, rad) of a quaternion, shortest way. */
export function quatToRotvec(q: Quat): Vec3 {
  let [x, y, z, w] = q;
  if (w < 0) { x = -x; y = -y; z = -z; w = -w; } // shortest way
  const vn = Math.hypot(x, y, z);
  if (vn < 1e-12) return [0, 0, 0];
  const angle = 2 * Math.atan2(vn, w);
  return [(x / vn) * angle, (y / vn) * angle, (z / vn) * angle];
}

export function rotvecToQuat(v: Vec3): Quat {
  const angle = Math.hypot(v[0], v[1], v[2]);
  if (angle < 1e-12) return [...QUAT_IDENTITY];
  const s = Math.sin(angle / 2);
  return [
    (v[0] / angle) * s,
    (v[1] / angle) * s,
    (v[2] / angle) * s,
    Math.cos(angle / 2),
  ];
}

/** Rotate a vector by a quaternion: v' = q · v · q⁻¹. */
export function rotateVec(v: Vec3, q: Quat): Vec3 {
  const [qx, qy, qz, qw] = q;
  // t = 2 · (qv × v); v' = v + w·t + (qv × t)
  const tx = 2 * (qy * v[2] - qz * v[1]);
  const ty = 2 * (qz * v[0] - qx * v[2]);
  const tz = 2 * (qx * v[1] - qy * v[0]);
  return [
    v[0] + qw * tx + (qy * tz - qz * ty),
    v[1] + qw * ty + (qz * tx - qx * tz),
    v[2] + qw * tz + (qx * ty - qy * tx),
  ];
}

/** Row-major 3x3 → xyzw quaternion (Shepperd's method). */
export function matToQuat(m: Mat3): Quat {
  const t = m[0][0] + m[1][1] + m[2][2];
  let q: Quat;
  if (t > 0) {
    const s = Math.sqrt(t + 1) * 2;
    q = [(m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s, s / 4];
  } else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
    const s = Math.sqrt(1 + m[0][0] - m[1][1] - m[2][2]) * 2;
    q = [s / 4, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s, (m[2][1] - m[1][2]) / s];
  } else if (m[1][1] > m[2][2]) {
    const s = Math.sqrt(1 + m[1][1] - m[0][0] - m[2][2]) * 2;
    q = [(m[0][1] + m[1][0]) / s, s / 4, (m[1][2] + m[2][1]) / s, (m[0][2] - m[2][0]) / s];
  } else {
    const s = Math.sqrt(1 + m[2][2] - m[0][0] - m[1][1]) * 2;
    q = [(m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, s / 4, (m[1][0] - m[0][1]) / s];
  }
  return quatNormalize(q);
}

export function matVec(m: Mat3, v: Vec3): Vec3 {
  return [
    m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
    m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
    m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
  ];
}

export function matMul(a: Mat3, b: Mat3): Mat3 {
  const out: number[][] = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
  return out as Mat3;
}

/** Rotation about +Y by theta (radians), row-major. */
export function rotY(theta: number): Mat3 {
  const c = Math.cos(theta), s = Math.sin(theta);
  return [
    [c, 0, s],
    [0, 1, 0],
    [-s, 0, c],
  ];
}

/**
 * Headset yaw about WebXR +Y, from the viewer orientation quaternion.
 * 0 = facing -Z; positive = counterclockwise seen from above.
 */
export function yawFromQuat(q: Quat): number {
  const f = rotateVec([0, 0, -1], q);
  return Math.atan2(-f[0], -f[2]);
}
