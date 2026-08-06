// Tests for the quaternion helpers the clutch mapper and IK are built on.
//
// Everything downstream — pose mapping, FK, the Jacobian — is composed from
// these six or seven functions, so an error here is invisible at the call
// site and shows up as a robot that moves in the wrong direction. The
// convention is xyzw (WebXR/wire), NOT the wxyz of the vr-teleop-kit
// reference; component bookkeeping is exactly the kind of thing that
// survives review and fails in a headset.
import { describe, expect, it } from 'vitest';

import {
  Mat3,
  Quat,
  QUAT_IDENTITY,
  Vec3,
  matMul,
  matToQuat,
  matVec,
  quatConj,
  quatDot,
  quatMul,
  quatNormalize,
  quatPow,
  quatToRotvec,
  rotateVec,
  rotvecToQuat,
  rotY,
  yawFromQuat,
} from '../quat';

/** xyzw quaternion for a rotation of `angle` about unit `axis`. */
function axisAngle(axis: Vec3, angle: number): Quat {
  const s = Math.sin(angle / 2);
  const n = Math.hypot(...axis);
  return [(axis[0] / n) * s, (axis[1] / n) * s, (axis[2] / n) * s, Math.cos(angle / 2)];
}

function expectVecClose(got: Vec3, want: Vec3, eps = 1e-9): void {
  got.forEach((v, i) => expect(v).toBeCloseTo(want[i], Math.round(-Math.log10(eps))));
}

/** Quaternions double-cover SO(3): q and -q are the same rotation. */
function expectSameRotation(a: Quat, b: Quat, eps = 1e-9): void {
  const sign = quatDot(a, b) < 0 ? -1 : 1;
  a.forEach((v, i) => expect(v).toBeCloseTo(sign * b[i], Math.round(-Math.log10(eps))));
}

describe('quatMul', () => {
  it('leaves identity alone', () => {
    const q = axisAngle([0, 1, 0], 0.7);
    expectSameRotation(quatMul(q, QUAT_IDENTITY), q);
    expectSameRotation(quatMul(QUAT_IDENTITY, q), q);
  });

  it('composes the same way rotating twice does', () => {
    const a = axisAngle([0, 0, 1], Math.PI / 2);
    const b = axisAngle([1, 0, 0], Math.PI / 2);
    const v: Vec3 = [0.3, -0.2, 0.7];
    // q = a·b applies b first, then a.
    expectVecClose(rotateVec(v, quatMul(a, b)), rotateVec(rotateVec(v, b), a));
  });

  it('does not commute (which is the whole reason order matters here)', () => {
    const a = axisAngle([0, 0, 1], 1.1);
    const b = axisAngle([1, 0, 0], 0.9);
    expect(quatDot(quatMul(a, b), quatMul(b, a))).toBeLessThan(1 - 1e-6);
  });
});

describe('quatConj', () => {
  it('inverts a unit quaternion', () => {
    const q = axisAngle([0.3, 0.5, -0.8], 1.3);
    expectSameRotation(quatMul(q, quatConj(q)), QUAT_IDENTITY);
  });

  it('undoes a rotation of a vector', () => {
    const q = axisAngle([1, 1, 0], 0.4);
    const v: Vec3 = [1, 2, 3];
    expectVecClose(rotateVec(rotateVec(v, q), quatConj(q)), v);
  });
});

describe('quatNormalize', () => {
  it('scales to unit length', () => {
    const n = quatNormalize([2, 0, 0, 2]);
    expect(Math.hypot(...n)).toBeCloseTo(1, 12);
  });

  it('falls back to identity for a degenerate quaternion', () => {
    // A zero quat has no axis; returning NaNs would poison every downstream
    // pose for the rest of the session.
    expect(quatNormalize([0, 0, 0, 0])).toEqual(QUAT_IDENTITY);
    expect(quatNormalize([1e-20, 0, 0, 0])).toEqual(QUAT_IDENTITY);
  });
});

describe('quatPow', () => {
  const q = axisAngle([0, 0, 1], 1.0);

  it('is the identity at k=0 and a no-op at k=1', () => {
    expect(quatPow(q, 0)).toEqual(QUAT_IDENTITY);
    expectSameRotation(quatPow(q, 1), q);
  });

  it('scales the angle and keeps the axis', () => {
    const half = quatPow(q, 0.5);
    expectSameRotation(quatMul(half, half), q);
    expectVecClose(quatToRotvec(half), [0, 0, 0.5]);
  });

  it('returns identity for a rotation too small to have an axis', () => {
    expect(quatPow(QUAT_IDENTITY, 3)).toEqual(QUAT_IDENTITY);
  });
});

describe('quatToRotvec / rotvecToQuat', () => {
  it('round-trips', () => {
    const v: Vec3 = [0.2, -0.5, 0.9];
    expectVecClose(quatToRotvec(rotvecToQuat(v)), v);
  });

  it('takes the shortest way for a negative-w quaternion', () => {
    // q and -q are the same rotation; the rotvec must not come back as the
    // ~2π-complement, which is what makes a reach limit flip direction.
    const q = axisAngle([0, 0, 1], 0.5);
    const flipped: Quat = [-q[0], -q[1], -q[2], -q[3]];
    expectVecClose(quatToRotvec(flipped), quatToRotvec(q));
    expect(Math.hypot(...quatToRotvec(flipped))).toBeCloseTo(0.5, 9);
  });

  it('maps zero rotation to a zero vector both ways', () => {
    expect(quatToRotvec(QUAT_IDENTITY)).toEqual([0, 0, 0]);
    expect(rotvecToQuat([0, 0, 0])).toEqual(QUAT_IDENTITY);
  });
});

describe('rotateVec', () => {
  it('rotates x into y for a +90° turn about +Z', () => {
    expectVecClose(rotateVec([1, 0, 0], axisAngle([0, 0, 1], Math.PI / 2)), [0, 1, 0]);
  });

  it('leaves the rotation axis fixed', () => {
    const axis: Vec3 = [0, 1, 0];
    expectVecClose(rotateVec(axis, axisAngle(axis, 1.234)), axis);
  });

  it('preserves length', () => {
    const v: Vec3 = [1, -2, 3];
    const out = rotateVec(v, axisAngle([0.3, 0.4, 0.5], 2.0));
    expect(Math.hypot(...out)).toBeCloseTo(Math.hypot(...v), 12);
  });
});

describe('matToQuat', () => {
  // Shepperd's method branches on the trace and the largest diagonal; a
  // single happy-path case leaves three of the four branches unexercised.
  const cases: Array<[string, Mat3]> = [
    ['identity (trace branch)', [[1, 0, 0], [0, 1, 0], [0, 0, 1]]],
    ['180° about X', [[1, 0, 0], [0, -1, 0], [0, 0, -1]]],
    ['180° about Y', [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]],
    ['180° about Z', [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]],
    ['90° about Y', rotY(Math.PI / 2)],
    ['-30° about Y', rotY(-Math.PI / 6)],
  ];

  it.each(cases)('agrees with the matrix for %s', (_name, m) => {
    const q = matToQuat(m);
    expect(Math.hypot(...q)).toBeCloseTo(1, 9);
    for (const v of [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.3, -0.7, 0.2]] as Vec3[]) {
      expectVecClose(rotateVec(v, q), matVec(m, v), 1e-9);
    }
  });
});

describe('matVec / matMul / rotY', () => {
  it('applies a row-major matrix to a column vector', () => {
    const m: Mat3 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
    expectVecClose(matVec(m, [1, 0, 0]), [1, 4, 7]);
    expectVecClose(matVec(m, [1, 1, 1]), [6, 15, 24]);
  });

  it('composes matrices the same way applying them in sequence does', () => {
    const a = rotY(0.4);
    const b = rotY(-1.1);
    const v: Vec3 = [0.2, 0.9, -0.3];
    expectVecClose(matVec(matMul(a, b), v), matVec(a, matVec(b, v)));
    // Rotations about the same axis add.
    expectVecClose(matVec(matMul(a, b), v), matVec(rotY(0.4 - 1.1), v));
  });

  it('rotates +X toward -Z for a +90° turn about +Y', () => {
    expectVecClose(matVec(rotY(Math.PI / 2), [1, 0, 0]), [0, 0, -1]);
  });
});

describe('yawFromQuat', () => {
  it('is zero when facing -Z (the WebXR resting direction)', () => {
    expect(yawFromQuat(QUAT_IDENTITY)).toBeCloseTo(0, 12);
  });

  it('reads back a yaw applied about +Y', () => {
    for (const theta of [0.3, -0.75, Math.PI / 2, -Math.PI / 3]) {
      expect(yawFromQuat(matToQuat(rotY(theta)))).toBeCloseTo(theta, 9);
    }
  });

  it('ignores pitch about the facing axis', () => {
    // Rolling the headset about where it looks must not change yaw, or the
    // per-engage yaw correction would drift every time the operator tilts.
    const roll = axisAngle([0, 0, -1], 0.6);
    expect(yawFromQuat(roll)).toBeCloseTo(0, 9);
  });
});
