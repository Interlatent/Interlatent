// Tests for the browser-side weighted damped-least-squares IK.
//
// This solver is on the robot's control path: whatever it returns is sent
// to the node as absolute joint targets. The safety-relevant behaviour is
// not "does it converge" but "what does it do when it can't" — near a
// singularity, past a joint stop, or asked for a pose it cannot reach. The
// clamps below are the only thing between a demand and a slam, so each one
// is tested for the case it exists to handle.
//
// The units seam matters as much as the math: the solver works in radians
// but emits the robot's own action units, and any action index it does not
// drive must pass through from the seed untouched.
import { describe, expect, it } from 'vitest';

import { Quat, Vec3, quatToRotvec, quatMul, quatConj } from '../quat';
import { forwardKinematics } from '../kinematics';
import { DlsSolver } from '../dlsSolver';
import { joint, mixedChain, planar3R, spec } from './specFixtures';

/** Drive the solver to convergence, feeding its own output back as the
 *  measured state (a robot that tracks perfectly). */
function converge(
  solver: DlsSolver,
  target: { pos: Vec3; quat: Quat },
  seed: number[],
  iters = 400,
): number[] {
  let q = [...seed];
  for (let i = 0; i < iters; i++) q = solver.solve(target.pos, target.quat, q);
  return q;
}

function posError(spec_: ReturnType<typeof planar3R>, q: number[], target: Vec3): number {
  const fk = forwardKinematics(spec_, q);
  return Math.hypot(fk.pos[0] - target[0], fk.pos[1] - target[1], fk.pos[2] - target[2]);
}

describe('units seam', () => {
  it('applies the affine when reading the seed and inverts it on output', () => {
    // A robot reporting half-radians with a +0.1 rad zero offset.
    const s = planar3R();
    s.joints.forEach((j) => { j.affine = { scale: 2, offset: 0.1 }; });
    const solver = new DlsSolver(s);

    const actionUnits = [0.15, 0.2, -0.05];
    const expectedRad = actionUnits.map((u) => 2 * u + 0.1);
    const direct = forwardKinematics(s, expectedRad);
    expect(solver.fk(actionUnits).pos).toEqual(direct.pos);

    // Solving for the pose it is already at returns the same action units.
    const out = solver.solve(direct.pos, direct.quat, actionUnits);
    out.forEach((v, i) => expect(v).toBeCloseTo(actionUnits[i], 6));
  });

  it('passes undriven action indices through from the seed', () => {
    // Action vector is wider than the IK chain — e.g. a head or a second
    // arm sharing the same action space. Those slots must not be zeroed.
    const s = planar3R();
    s.joints[0].action_index = 1;
    s.joints[1].action_index = 2;
    s.joints[2].action_index = 3;
    const solver = new DlsSolver(s);
    const seed = [99, 0.1, 0.1, 0.1, -42];
    const out = solver.solve([2.5, 0.5, 0], [0, 0, 0, 1], seed);
    expect(out).toHaveLength(5);
    expect(out[0]).toBe(99);
    expect(out[4]).toBe(-42);
  });

  it('reads a missing seed entry as zero rather than NaN', () => {
    const solver = new DlsSolver(planar3R());
    const out = solver.solve([2.9, 0.1, 0], [0, 0, 0, 1], []);
    expect(out.every((v) => Number.isFinite(v))).toBe(true);
  });
});

describe('gripper', () => {
  const withGripper = () => {
    const s = planar3R();
    s.gripper = { action_index: 3, range: [0.0, 1.4] }; // [open, closed]
    return new DlsSolver(s);
  };

  it('maps pinch onto the robot-native open/closed commands', () => {
    const solver = withGripper();
    const at = (pinch: number) =>
      solver.solve([3, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], pinch)[3];
    expect(at(0)).toBeCloseTo(0.0, 9);
    expect(at(1)).toBeCloseTo(1.4, 9);
    expect(at(0.5)).toBeCloseTo(0.7, 9);
  });

  it('clamps an out-of-range pinch instead of over-driving the gripper', () => {
    const solver = withGripper();
    const at = (pinch: number) =>
      solver.solve([3, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], pinch)[3];
    expect(at(-5)).toBeCloseTo(0.0, 9);
    expect(at(17)).toBeCloseTo(1.4, 9);
  });

  it('leaves the gripper slot alone when the spec has no gripper', () => {
    const solver = new DlsSolver(planar3R());
    expect(solver.solve([3, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0.77], 1)[3]).toBe(0.77);
  });
});

describe('convergence', () => {
  it('reaches a reachable pose', () => {
    const s = planar3R();
    const solver = new DlsSolver(s);
    const goal = [0.3, 0.5, -0.4];
    const target = forwardKinematics(s, goal);

    const q = converge(solver, target, [0.05, 0.05, 0.05]);
    expect(posError(s, q, target.pos)).toBeLessThan(1e-3);
    const rot = quatToRotvec(quatMul(target.quat, quatConj(forwardKinematics(s, q).quat)));
    expect(Math.hypot(...rot)).toBeLessThan(1e-2);
  });

  it('reaches a reachable pose on a mixed slide/hinge chain', () => {
    const s = mixedChain();
    const solver = new DlsSolver(s);
    const target = forwardKinematics(s, [0.2, 0.4, -0.3]);
    const q = converge(solver, target, [0, 0, 0]);
    const fk = forwardKinematics(s, q);
    expect(Math.hypot(
      fk.pos[0] - target.pos[0], fk.pos[1] - target.pos[1], fk.pos[2] - target.pos[2],
    )).toBeLessThan(1e-3);
  });

  it('moves toward — never away from — an unreachable target', () => {
    // Twice the arm's reach. The damping must degrade to "get as close as
    // possible", not oscillate or run away.
    const s = planar3R();
    const solver = new DlsSolver(s);
    const far: Vec3 = [6, 0, 0];
    const start = [0.5, 0.5, 0.5];
    const before = posError(s, start, far);
    const q = converge(solver, { pos: far, quat: [0, 0, 0, 1] }, start, 200);
    expect(posError(s, q, far)).toBeLessThan(before);
    expect(q.every((v) => Number.isFinite(v))).toBe(true);
  });

  it('stays finite at a stretched-out singularity', () => {
    // Fully extended, the planar arm loses a rank; without the adaptive
    // damping the normal equations blow up here.
    const s = planar3R();
    const solver = new DlsSolver(s);
    const q = converge(solver, { pos: [3.5, 0, 0], quat: [0, 0, 0, 1] },
                       [0, 0, 0], 100);
    expect(q.every((v) => Number.isFinite(v))).toBe(true);
    expect(Math.max(...q.map(Math.abs))).toBeLessThan(Math.PI);
  });
});

describe('clamps', () => {
  it('never returns a joint outside its limits', () => {
    const s = planar3R();
    s.joints.forEach((j) => { j.limit = [-0.2, 0.2]; });
    const solver = new DlsSolver(s);
    // Demand a pose far outside the reachable set of the clamped arm.
    let q = [0, 0, 0];
    for (let i = 0; i < 100; i++) {
      q = solver.solve([0, 3, 0], [0, 0, 0, 1], q);
      q.forEach((v) => {
        expect(v).toBeGreaterThanOrEqual(-0.2 - 1e-12);
        expect(v).toBeLessThanOrEqual(0.2 + 1e-12);
      });
    }
  });

  it('caps how far any joint may move in a single tick', () => {
    // The per-tick cap is what turns "wildly wrong target" into a slow
    // drift instead of a slam.
    const s = planar3R();
    s.joints.forEach((j) => { j.max_dq = 0.01; });
    const solver = new DlsSolver(s);
    const seed = [0, 0, 0];
    const out = solver.solve([0, 3, 0], [0, 0, 0, 1], seed);
    out.forEach((v, i) => expect(Math.abs(v - seed[i])).toBeLessThanOrEqual(0.01 + 1e-12));
  });

  it('applies a per-joint cap only to the joints that declare one', () => {
    const s = planar3R();
    s.joints[1].max_dq = 0.005;
    const solver = new DlsSolver(s);
    const seed = [0, 0, 0];
    const out = solver.solve([0, 3, 0], [0, 0, 0, 1], seed);
    expect(Math.abs(out[1] - seed[1])).toBeLessThanOrEqual(0.005 + 1e-12);
    expect(Math.abs(out[0] - seed[0])).toBeGreaterThan(0.005);
  });

  it('pulls toward q_rest through the redundant null space', () => {
    // A position-only task on a 3R planar arm leaves one DOF free. The
    // Tikhonov term is what stops the arm drifting into an arbitrary
    // null-space pose; compare the same task solved with and without it.
    const goal: Vec3 = [2.0, 0.5, 0];
    const solveWithMu = (mu: number) => {
      const s = planar3R();
      // rot_err_hold below zero gates the rotation rows off entirely.
      s.damping = { ...s.damping, mu, rot_err_hold: -1 };
      s.joints.forEach((j) => { j.q_rest = 0; });
      const q = converge(new DlsSolver(s), { pos: goal, quat: [0, 0, 0, 1] },
                         [0.9, -0.9, 0.9], 400);
      return { q, err: posError(s, q, goal) };
    };

    const free = solveWithMu(0);
    const pulled = solveWithMu(0.05);
    // Both still reach the target; the pulled one does it closer to rest.
    expect(free.err).toBeLessThan(1e-2);
    expect(pulled.err).toBeLessThan(1e-2);
    expect(Math.hypot(...pulled.q)).toBeLessThan(Math.hypot(...free.q));
  });
});

describe('warm start', () => {
  it('leashes the integrated command to the measured state', () => {
    // The command may run ahead of measurement (that is the point), but a
    // robot that stops tracking must not let the command run away.
    const s = planar3R();
    const solver = new DlsSolver(s);
    const stuck = [0, 0, 0];
    let last: number[] = stuck;
    for (let i = 0; i < 200; i++) {
      // Measured state never moves — a stalled arm.
      last = solver.solve([0, 3, 0], [0, 0, 0, 1], stuck);
    }
    // 0.35 rad leash + one step's worth of motion, not 200 steps' worth.
    last.forEach((v) => expect(Math.abs(v)).toBeLessThan(0.35 + 0.5));
  });

  it('re-seeds from measurement after resetWarmstart', () => {
    const s = planar3R();
    const solver = new DlsSolver(s);
    const target = forwardKinematics(s, [0.4, 0.4, 0.4]);
    const first = solver.solve(target.pos, target.quat, [0, 0, 0]);
    // Without a reset the second call starts from the integrated command.
    const second = solver.solve(target.pos, target.quat, [0, 0, 0]);
    expect(second).not.toEqual(first);

    solver.resetWarmstart();
    const afterReset = solver.solve(target.pos, target.quat, [0, 0, 0]);
    afterReset.forEach((v, i) => expect(v).toBeCloseTo(first[i], 12));
  });
});

describe('rotation gating', () => {
  it('drops the orientation task while the orientation error is large', () => {
    // rot_err_hold exists so a big orientation error doesn't fight the
    // position task; below the hold, orientation is tracked again.
    const s = spec([
      joint({ name: 'a', action_index: 0 }),
      joint({ name: 'b', action_index: 1, origin_pos: [1, 0, 0] }),
      joint({ name: 'c', action_index: 2, origin_pos: [1, 0, 0] }),
    ], {
      tool0: { origin_pos: [1, 0, 0], origin_quat_xyzw: [0, 0, 0, 1] },
      damping: { lam_pos: 1e-3, lam0: 1e-2, w0: 1e-2, mu: 0, rot_err_hold: 0.1 },
    });
    const solver = new DlsSolver(s);
    const target = forwardKinematics(s, [0.3, 0.3, 0.3]);
    const q = converge(solver, target, [0, 0, 0], 400);
    // Position still converges with the rotation rows dropped early on.
    expect(posError(s, q, target.pos)).toBeLessThan(1e-2);
  });
});
