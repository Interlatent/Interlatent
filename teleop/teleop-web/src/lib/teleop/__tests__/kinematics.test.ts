// Tests for browser-side FK + the geometric Jacobian.
//
// These two functions replaced the only things MuJoCo provided pod-side
// (`mj_kinematics` and `mj_jacSite`) so IK could run in the headset with no
// WASM. The module claims to be "a 1:1 port ... verified against real
// MuJoCo at machine precision" — nothing in the repo checked that claim.
//
// The Jacobian is checked against a central finite difference of FK, which
// is an independent derivation: if the analytic columns are assembled from
// the wrong frame (the classic error — capturing the axis AFTER the joint's
// own rotation instead of before), the two disagree immediately.
import { describe, expect, it } from 'vitest';

import { Quat, Vec3, quatConj, quatMul, quatToRotvec, rotateVec } from '../quat';
import {
  KinematicSpec,
  cross,
  forwardKinematics,
  isChainsSpec,
  orientationError,
  siteJacobian,
} from '../kinematics';
import { axisAngle, joint, mixedChain, planar3R, spec } from './specFixtures';

function expectVecClose(got: Vec3, want: Vec3, digits = 9): void {
  got.forEach((v, i) => expect(v).toBeCloseTo(want[i], digits));
}

/** Central-difference Jacobian of FK — an independent check of the
 *  analytic one. Rotational columns use the local rotvec of the delta. */
function numericJacobian(s: KinematicSpec, q: number[], h = 1e-6) {
  const n = s.joints.length;
  const jacp = [new Array(n).fill(0), new Array(n).fill(0), new Array(n).fill(0)];
  const jacr = [new Array(n).fill(0), new Array(n).fill(0), new Array(n).fill(0)];
  for (let i = 0; i < n; i++) {
    const qp = [...q];
    const qm = [...q];
    qp[i] += h;
    qm[i] -= h;
    const fp = forwardKinematics(s, qp);
    const fm = forwardKinematics(s, qm);
    for (let r = 0; r < 3; r++) jacp[r][i] = (fp.pos[r] - fm.pos[r]) / (2 * h);
    // World-frame angular velocity column: d(q_p ⊗ q_m⁻¹) / 2h.
    const rv = quatToRotvec(quatMul(fp.quat, quatConj(fm.quat)));
    for (let r = 0; r < 3; r++) jacr[r][i] = rv[r] / (2 * h);
  }
  return { jacp, jacr };
}

describe('forwardKinematics', () => {
  const arm = planar3R();

  it('extends straight along +X at the home pose', () => {
    const fk = forwardKinematics(arm, [0, 0, 0]);
    expectVecClose(fk.pos, [3, 0, 0]);
    expectVecClose(quatToRotvec(fk.quat), [0, 0, 0]);
  });

  it('swings the whole chain when the base joint turns', () => {
    const fk = forwardKinematics(arm, [Math.PI / 2, 0, 0]);
    expectVecClose(fk.pos, [0, 3, 0]);
    expectVecClose(quatToRotvec(fk.quat), [0, 0, Math.PI / 2]);
  });

  it('folds only the outboard links when a later joint turns', () => {
    const fk = forwardKinematics(arm, [0, Math.PI / 2, 0]);
    expectVecClose(fk.pos, [1, 2, 0]);
  });

  it('accumulates orientation across joints', () => {
    const fk = forwardKinematics(arm, [0.3, 0.4, -0.2]);
    expectVecClose(quatToRotvec(fk.quat), [0, 0, 0.5]);
  });

  it('translates along the world axis for a slide joint', () => {
    const s = spec([
      joint({ name: 'rot', axis: [0, 0, 1], action_index: 0 }),
      joint({ name: 'lift', type: 'slide', axis: [1, 0, 0], action_index: 1,
              limit: [-2, 2] }),
    ]);
    // With the base turned 90°, the slide's +X becomes world +Y.
    expectVecClose(forwardKinematics(s, [Math.PI / 2, 0.5]).pos, [0, 0.5, 0]);
    expectVecClose(forwardKinematics(s, [0, 0.5]).pos, [0.5, 0, 0]);
  });

  it('applies the tool0 offset in the final frame', () => {
    const s = spec([joint({ action_index: 0 })], {
      tool0: { origin_pos: [0.1, 0, 0], origin_quat_xyzw: axisAngle([0, 0, 1], 0.25) },
    });
    const fk = forwardKinematics(s, [Math.PI / 2]);
    expectVecClose(fk.pos, [0, 0.1, 0]);
    expectVecClose(quatToRotvec(fk.quat), [0, 0, Math.PI / 2 + 0.25]);
  });

  it('honours a joint mounting rotation', () => {
    // The joint spins about its LOCAL axis, re-expressed by origin_quat.
    const s = spec([
      joint({
        axis: [0, 0, 1],
        origin_quat_xyzw: axisAngle([1, 0, 0], Math.PI / 2),
        action_index: 0,
      }),
    ], { tool0: { origin_pos: [1, 0, 0], origin_quat_xyzw: [0, 0, 0, 1] } });
    // The mount takes local +Z to world -Y, so a +90° turn about the local
    // axis is a -90° turn about world +Y — which carries +X to +Z.
    expectVecClose(forwardKinematics(s, [Math.PI / 2]).pos, [0, 0, 1], 9);
    // Sanity: without the mount the same joint would swing +X to +Y.
    const unmounted = spec([joint({ axis: [0, 0, 1], action_index: 0 })], {
      tool0: { origin_pos: [1, 0, 0], origin_quat_xyzw: [0, 0, 0, 1] },
    });
    expectVecClose(forwardKinematics(unmounted, [Math.PI / 2]).pos, [0, 1, 0], 9);
  });

  it('reports one axis and anchor per joint, captured before its own motion', () => {
    const fk = forwardKinematics(arm, [Math.PI / 2, 0, 0]);
    expect(fk.axes).toHaveLength(3);
    expect(fk.anchors).toHaveLength(3);
    // Joint 0's anchor is the base, and its axis is unaffected by its own q.
    expectVecClose(fk.anchors[0], [0, 0, 0]);
    expectVecClose(fk.axes[0], [0, 0, 1]);
    expectVecClose(fk.anchors[1], [0, 1, 0]);
  });
});

describe('siteJacobian', () => {
  const cases: Array<[string, KinematicSpec, number[]]> = [
    ['planar 3R at home', planar3R(), [0, 0, 0]],
    ['planar 3R folded', planar3R(), [0.4, -0.9, 1.3]],
    ['planar 3R near a stretched singularity', planar3R(), [0, 1e-4, 0]],
    ['mixed slide/hinge chain', mixedChain(), [0.2, 0.5, -0.7]],
    ['mixed chain at zero', mixedChain(), [0, 0, 0]],
  ];

  it.each(cases)('matches a finite-difference FK derivative: %s', (_n, s, q) => {
    const analytic = siteJacobian(s, forwardKinematics(s, q));
    const numeric = numericJacobian(s, q);
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < s.joints.length; c++) {
        expect(analytic.jacp[r][c]).toBeCloseTo(numeric.jacp[r][c], 5);
        expect(analytic.jacr[r][c]).toBeCloseTo(numeric.jacr[r][c], 5);
      }
    }
  });

  it('gives a slide joint linear velocity only', () => {
    const s = mixedChain();
    const { jacp, jacr } = siteJacobian(s, forwardKinematics(s, [0.1, 0.2, 0.3]));
    expectVecClose([jacr[0][0], jacr[1][0], jacr[2][0]], [0, 0, 0]);
    // ...and its linear column is exactly the world axis.
    expectVecClose([jacp[0][0], jacp[1][0], jacp[2][0]], [0, 0, 1]);
  });

  it('gives a revolute joint the lever-arm cross product', () => {
    const s = planar3R();
    const fk = forwardKinematics(s, [0.3, 0.2, 0.1]);
    const { jacp } = siteJacobian(s, fk);
    const lever: Vec3 = [
      fk.pos[0] - fk.anchors[1][0],
      fk.pos[1] - fk.anchors[1][1],
      fk.pos[2] - fk.anchors[1][2],
    ];
    expectVecClose([jacp[0][1], jacp[1][1], jacp[2][1]], cross(fk.axes[1], lever));
  });
});

describe('orientationError', () => {
  it('is zero for identical orientations', () => {
    const q = axisAngle([0.2, 0.5, 0.8], 1.1);
    expectVecClose(orientationError(q, q), [0, 0, 0]);
  });

  it('is the world-frame rotation that takes current onto target', () => {
    const cur = axisAngle([0, 0, 1], 0.2);
    const target = axisAngle([0, 0, 1], 0.7);
    expectVecClose(orientationError(cur, target), [0, 0, 0.5]);
  });

  it('applying the error to current reaches target', () => {
    const cur = axisAngle([1, 0, 0], 0.4);
    const target = axisAngle([0, 1, 0], -0.9);
    const e = orientationError(cur, target);
    const n = Math.hypot(...e);
    const applied = quatMul(axisAngle(e as Vec3, n), cur);
    // Same rotation up to the double cover.
    for (const v of [[1, 0, 0], [0, 1, 0], [0, 0, 1]] as Vec3[]) {
      expectVecClose(rotateVec(v, applied), rotateVec(v, target), 8);
    }
  });

  it('takes the short way round rather than the long one', () => {
    const cur: Quat = [0, 0, 0, 1];
    const target = axisAngle([0, 0, 1], 2 * Math.PI - 0.3);
    expect(Math.hypot(...orientationError(cur, target))).toBeCloseTo(0.3, 9);
  });
});

describe('isChainsSpec', () => {
  it('recognises a bimanual bundle and rejects a flat spec', () => {
    const flat = planar3R();
    expect(isChainsSpec(flat)).toBe(false);
    expect(isChainsSpec({ chains: { left: flat, right: flat } })).toBe(true);
  });
});
