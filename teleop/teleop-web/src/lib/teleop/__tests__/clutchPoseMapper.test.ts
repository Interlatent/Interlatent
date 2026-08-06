// Tests for clutch-relative pose mapping (WebXR controller -> EE target).
//
// This is the operator-facing half of teleop: it decides where the arm is
// told to go every time the human moves their hand. Two behaviours here are
// deliberate deviations from the vr-teleop-kit reference and both are the
// kind of thing that only reveals itself in a headset, so they are pinned:
//
//   1. the swing-twist split — wrist ROLL must spin the gripper about the
//      TOOL's own pointing axis, in every pose, instead of leaking into
//      pitch/yaw the way a pure world-frame mapping does;
//   2. the slipping clutch — a demand pressed past the arm's reach is
//      ABSORBED, so reversing the hand bites immediately instead of first
//      retracing the overshoot, and error can never build to the ~180°
//      where shortest-way direction flips.
import { describe, expect, it } from 'vitest';

import {
  Mat3,
  Quat,
  Vec3,
  matToQuat,
  quatConj,
  quatMul,
  quatToRotvec,
  rotY,
  rotateVec,
} from '../quat';
import { ClutchPoseMapper } from '../clutchPoseMapper';

const I: Quat = [0, 0, 0, 1];
const ORIGIN: Vec3 = [0, 0, 0];

function axisAngle(axis: Vec3, angle: number): Quat {
  const s = Math.sin(angle / 2);
  const n = Math.hypot(...axis);
  return [(axis[0] / n) * s, (axis[1] / n) * s, (axis[2] / n) * s, Math.cos(angle / 2)];
}

function expectVecClose(got: Vec3, want: Vec3, digits = 9): void {
  got.forEach((v, i) => expect(v).toBeCloseTo(want[i], digits));
}

function unit(v: Vec3): Vec3 {
  const n = Math.hypot(...v);
  return [v[0] / n, v[1] / n, v[2] / n];
}

describe('clutch state', () => {
  it('produces nothing until engaged', () => {
    const m = new ClutchPoseMapper();
    expect(m.engaged).toBe(false);
    expect(m.target([1, 2, 3], I, ORIGIN, I)).toBeNull();
  });

  it('stops producing on release so the hand can be repositioned', () => {
    const m = new ClutchPoseMapper();
    m.engage(ORIGIN, I, ORIGIN, I);
    expect(m.engaged).toBe(true);
    expect(m.target([0.1, 0, 0], I, ORIGIN, I)).not.toBeNull();

    m.disengage();
    expect(m.engaged).toBe(false);
    expect(m.target([5, 5, 5], I, ORIGIN, I)).toBeNull();
  });

  it('holds the arm still at the instant of engage', () => {
    // The first frame after the rising clutch edge must be a no-op, or the
    // arm jumps by whatever offset the hand happened to have.
    const m = new ClutchPoseMapper();
    const eePos: Vec3 = [0.4, -0.1, 0.25];
    const eeQuat = axisAngle([0, 1, 0], 0.8);
    m.engage([1, 2, 3], axisAngle([1, 0, 0], 0.3), eePos, eeQuat);

    const t = m.target([1, 2, 3], axisAngle([1, 0, 0], 0.3), eePos, eeQuat)!;
    expectVecClose(t.pos, eePos);
    expectVecClose(quatToRotvec(quatMul(t.quat, quatConj(eeQuat))), [0, 0, 0], 8);
  });

  it('re-anchors on every engage', () => {
    const m = new ClutchPoseMapper();
    m.engage(ORIGIN, I, ORIGIN, I);
    m.target([0.1, 0, 0], I, ORIGIN, I);
    m.disengage();
    // Hand moved a long way while disengaged; the new engage zeroes it.
    m.engage([9, 9, 9], I, [0.5, 0, 0], I);
    expectVecClose(m.target([9, 9, 9], I, [0.5, 0, 0], I)!.pos, [0.5, 0, 0]);
  });
});

describe('translation', () => {
  it('is 1:1 in the arm frame at unit scale', () => {
    const m = new ClutchPoseMapper({ posReachLimit: 0 });
    m.engage(ORIGIN, I, [0.3, 0, 0], I);
    const t = m.target([0.1, -0.05, 0.2], I, null, null)!;
    expectVecClose(t.pos, [0.4, -0.05, 0.2]);
  });

  it('applies the linear gain', () => {
    const m = new ClutchPoseMapper({ scale: 0.5, posReachLimit: 0 });
    m.engage(ORIGIN, I, ORIGIN, I);
    expectVecClose(m.target([1, 0, 0], I, null, null)!.pos, [0.5, 0, 0]);
  });

  it('rotates the hand delta into the arm base frame through R', () => {
    // R takes Quest-world vectors to arm-base vectors; a +90° yaw means
    // pushing the hand along Quest +X drives the arm along its own -Z.
    const R: Mat3 = rotY(Math.PI / 2);
    const m = new ClutchPoseMapper({ R, posReachLimit: 0 });
    m.engage(ORIGIN, I, ORIGIN, I);
    expectVecClose(m.target([0.2, 0, 0], I, null, null)!.pos, [0, 0, -0.2]);
  });

  it('honours setR for the per-engage yaw correction', () => {
    const m = new ClutchPoseMapper({ posReachLimit: 0 });
    m.setR(rotY(Math.PI / 2));
    m.engage(ORIGIN, I, ORIGIN, I);
    expectVecClose(m.target([0.2, 0, 0], I, null, null)!.pos, [0, 0, -0.2]);
  });

  it('accumulated increments equal the absolute delta until the limit bites', () => {
    // The reach-limited path integrates per-tick increments; while nothing
    // is being absorbed it must be identical to the absolute mapping.
    const limited = new ClutchPoseMapper({ posReachLimit: 10, rotReachLimit: 0 });
    const absolute = new ClutchPoseMapper({ posReachLimit: 0, rotReachLimit: 0 });
    limited.engage(ORIGIN, I, ORIGIN, I);
    absolute.engage(ORIGIN, I, ORIGIN, I);

    for (const p of [[0.01, 0, 0], [0.03, 0.01, 0], [0.02, 0.04, -0.01]] as Vec3[]) {
      const a = limited.target(p, I, ORIGIN, I)!;
      const b = absolute.target(p, I, null, null)!;
      expectVecClose(a.pos, b.pos, 9);
    }
  });
});

describe('position reach limit (slipping clutch)', () => {
  it('clamps the target to within the limit of the arm', () => {
    const m = new ClutchPoseMapper({ posReachLimit: 0.1, rotReachLimit: 0 });
    m.engage(ORIGIN, I, ORIGIN, I);
    const t = m.target([1, 0, 0], I, ORIGIN, I)!;
    expectVecClose(t.pos, [0.1, 0, 0]);
  });

  it('absorbs the excess so reversing bites immediately', () => {
    // Without absorption the hand would have to retrace the whole 0.9 m of
    // overshoot before the arm moved at all.
    const m = new ClutchPoseMapper({ posReachLimit: 0.1, rotReachLimit: 0 });
    m.engage(ORIGIN, I, ORIGIN, I);
    m.target([1, 0, 0], I, ORIGIN, I);

    const back = m.target([0.95, 0, 0], I, ORIGIN, I)!;
    expectVecClose(back.pos, [0.05, 0, 0]);
  });

  it('keeps clamping as the arm follows the demand', () => {
    const m = new ClutchPoseMapper({ posReachLimit: 0.1, rotReachLimit: 0 });
    m.engage(ORIGIN, I, ORIGIN, I);
    let ee: Vec3 = ORIGIN;
    for (let i = 1; i <= 5; i++) {
      const t = m.target([0.2 * i, 0, 0], I, ee, I)!;
      // Never more than the limit ahead of where the arm actually is.
      expect(Math.hypot(t.pos[0] - ee[0], t.pos[1] - ee[1], t.pos[2] - ee[2]))
        .toBeLessThanOrEqual(0.1 + 1e-12);
      ee = t.pos; // arm tracks perfectly
    }
    // ...and it did keep making progress rather than sticking.
    expect(ee[0]).toBeGreaterThan(0.3);
  });

  it('falls back to absolute mapping when no EE pose is supplied', () => {
    const m = new ClutchPoseMapper({ posReachLimit: 0.1 });
    m.engage(ORIGIN, I, ORIGIN, I);
    expectVecClose(m.target([1, 0, 0], I, null, null)!.pos, [1, 0, 0]);
  });
});

describe('rotation', () => {
  it('maps a controller roll onto the tool pointing axis', () => {
    // The swing-twist deviation: with the gripper pointing somewhere other
    // than the hand, a pure wrist roll must still spin the gripper about
    // ITS axis. A pure world-frame mapping would swing it sideways.
    const eeQuat = axisAngle([1, 0, 0], -Math.PI / 2);
    const m = new ClutchPoseMapper({ rotReachLimit: 5, posReachLimit: 0 });
    m.engage(ORIGIN, I, ORIGIN, eeQuat);

    // Roll about the controller's own pointing axis (-Z).
    const roll = axisAngle([0, 0, -1], 0.2);
    const t = m.target(ORIGIN, roll, null, eeQuat)!;

    const delta = quatToRotvec(quatMul(t.quat, quatConj(eeQuat)));
    expect(Math.hypot(...delta)).toBeCloseTo(0.2, 6);

    const toolAxis = rotateVec([0, 0, -1], eeQuat);
    const dot = unit(delta)[0] * toolAxis[0]
      + unit(delta)[1] * toolAxis[1]
      + unit(delta)[2] * toolAxis[2];
    expect(Math.abs(dot)).toBeCloseTo(1, 6);

    // Sanity: the tool axis is NOT the controller axis here, so this is a
    // real distinction and not a vacuous check.
    expect(Math.abs(toolAxis[2])).toBeLessThan(0.5);
  });

  it('reduces to the reference world-frame mapping when hand and tool agree', () => {
    // Co-oriented => both decomposition axes coincide => the split is a
    // no-op, which is the compatibility claim in the module docstring.
    const m = new ClutchPoseMapper({ rotReachLimit: 5, posReachLimit: 0 });
    m.engage(ORIGIN, I, ORIGIN, I);
    const roll = axisAngle([0, 0, -1], 0.3);
    const t = m.target(ORIGIN, roll, null, I)!;
    expectVecClose(quatToRotvec(t.quat), [0, 0, -0.3], 6);
  });

  it('applies the rotation gain per increment', () => {
    const m = new ClutchPoseMapper({ scaleRotation: 0.5, rotReachLimit: 5,
                                     posReachLimit: 0 });
    m.engage(ORIGIN, I, ORIGIN, I);
    const t = m.target(ORIGIN, axisAngle([0, 0, -1], 0.4), null, I)!;
    expect(Math.hypot(...quatToRotvec(t.quat))).toBeCloseTo(0.2, 6);
  });

  it('clamps the orientation target to the reach limit', () => {
    const m = new ClutchPoseMapper({ rotReachLimit: 0.6, posReachLimit: 0 });
    m.engage(ORIGIN, I, ORIGIN, I);
    const t = m.target(ORIGIN, axisAngle([0, 0, -1], 1.0), ORIGIN, I)!;
    const e = quatToRotvec(quatMul(t.quat, quatConj(I)));
    expect(Math.hypot(...e)).toBeCloseTo(0.6, 6);
  });

  it('absorbs rotational overshoot too', () => {
    const m = new ClutchPoseMapper({ rotReachLimit: 0.6, posReachLimit: 0 });
    m.engage(ORIGIN, I, ORIGIN, I);
    m.target(ORIGIN, axisAngle([0, 0, -1], 1.5), ORIGIN, I);
    // Hand rotates back by 0.1 rad; the target must follow by 0.1, not sit
    // still while 0.9 rad of overshoot unwinds.
    const back = m.target(ORIGIN, axisAngle([0, 0, -1], 1.4), ORIGIN, I)!;
    expect(Math.hypot(...quatToRotvec(back.quat))).toBeCloseTo(0.5, 5);
  });

  it('handles a hemisphere flip in the controller quaternion', () => {
    // WebXR may report q or -q frame to frame; treating that as a ~360°
    // increment would snap the arm.
    const m = new ClutchPoseMapper({ rotReachLimit: 5, posReachLimit: 0 });
    m.engage(ORIGIN, I, ORIGIN, I);
    const q = axisAngle([0, 0, -1], 0.2);
    m.target(ORIGIN, q, null, I);
    const flipped: Quat = [-q[0], -q[1], -q[2], -q[3]];
    const t = m.target(ORIGIN, flipped, null, I)!;
    expect(Math.hypot(...quatToRotvec(t.quat))).toBeCloseTo(0.2, 6);
  });

  it('uses the aim orientation, not the grip, as the twist axis', () => {
    // On a Quest Touch the grip's -Z is the handle, not where the operator
    // points; decomposing about the handle misreads part of a roll as
    // swing, which leaks into pitch/yaw.
    const aim = axisAngle([1, 0, 0], Math.PI / 2);
    const withAim = new ClutchPoseMapper({ rotReachLimit: 5, posReachLimit: 0 });
    const withoutAim = new ClutchPoseMapper({ rotReachLimit: 5, posReachLimit: 0 });
    withAim.engage(ORIGIN, I, ORIGIN, I);
    withoutAim.engage(ORIGIN, I, ORIGIN, I);

    const inc = axisAngle([0, 1, 0], 0.2);
    const a = withAim.target(ORIGIN, inc, null, I, aim)!;
    const b = withoutAim.target(ORIGIN, inc, null, I)!;
    // Different decomposition axis => a different arm-frame rotation.
    expect(Math.hypot(...quatToRotvec(quatMul(a.quat, quatConj(b.quat)))))
      .toBeGreaterThan(1e-3);
  });

  it('falls back to absolute rotation mapping without an EE orientation', () => {
    const m = new ClutchPoseMapper({ rotReachLimit: 0.6, posReachLimit: 0 });
    m.engage(ORIGIN, axisAngle([0, 0, -1], 1.0), ORIGIN, I);
    // Absolute path: delta measured from the ENGAGE pose, unclamped.
    const t = m.target(ORIGIN, axisAngle([0, 0, -1], 2.0), null, null)!;
    expect(Math.hypot(...quatToRotvec(t.quat))).toBeCloseTo(1.0, 6);
  });

  it('always returns a unit quaternion', () => {
    const m = new ClutchPoseMapper();
    m.engage(ORIGIN, I, ORIGIN, I);
    for (let i = 1; i <= 40; i++) {
      const t = m.target([0.01 * i, 0, 0], axisAngle([0.3, 0.5, -0.8], 0.05 * i),
                         [0.005 * i, 0, 0], I)!;
      expect(Math.hypot(...t.quat)).toBeCloseTo(1, 9);
      expect(t.pos.every(Number.isFinite)).toBe(true);
    }
  });
});

describe('options', () => {
  it('defaults to the documented gains and limits', () => {
    const m = new ClutchPoseMapper();
    expect(m.scale).toBe(1.0);
    expect(m.scaleRotation).toBe(1.0);
    expect(m.rotReachLimit).toBe(0.6);
    expect(m.posReachLimit).toBe(0.25);
    expect(m.toolTwistAxis).toEqual([0, 0, -1]);
    expect(m.R).toEqual([[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
  });

  it('takes a flipped tool axis for a robot whose tool frame points +Z', () => {
    const eeQuat = axisAngle([1, 0, 0], -Math.PI / 2);
    const m = new ClutchPoseMapper({ toolTwistAxis: [0, 0, 1], rotReachLimit: 5,
                                     posReachLimit: 0 });
    m.engage(ORIGIN, I, ORIGIN, eeQuat);
    const t = m.target(ORIGIN, axisAngle([0, 0, -1], 0.2), null, eeQuat)!;
    const delta = quatToRotvec(quatMul(t.quat, quatConj(eeQuat)));
    const toolAxis = rotateVec([0, 0, 1], eeQuat);
    const d = unit(delta)[0] * toolAxis[0] + unit(delta)[1] * toolAxis[1]
      + unit(delta)[2] * toolAxis[2];
    expect(Math.abs(d)).toBeCloseTo(1, 6);
  });

  it('accepts an R supplied as a quaternion-equivalent matrix', () => {
    const R = rotY(0.7);
    const m = new ClutchPoseMapper({ R, posReachLimit: 0, rotReachLimit: 0 });
    m.engage(ORIGIN, I, ORIGIN, I);
    const t = m.target([0.3, 0, 0], I, null, null)!;
    expectVecClose(t.pos, rotateVec([0.3, 0, 0], matToQuat(R)), 9);
  });
});
