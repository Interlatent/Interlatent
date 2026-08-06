// Shared kinematic-spec builders for the FK / Jacobian / IK tests.
//
// The real specs come from the engine's `kinematic_spec.py` export and are
// robot-sized; these are the smallest chains that still exercise every
// branch (hinge + slide, non-identity joint offsets, an affine units seam,
// a gripper, and undriven action indices).
import { Quat, Vec3 } from '../quat';
import { KinematicSpec, SpecJoint } from '../kinematics';

export function axisAngle(axis: Vec3, angle: number): Quat {
  const s = Math.sin(angle / 2);
  const n = Math.hypot(...axis);
  return [(axis[0] / n) * s, (axis[1] / n) * s, (axis[2] / n) * s, Math.cos(angle / 2)];
}

export function joint(over: Partial<SpecJoint> = {}): SpecJoint {
  return {
    name: 'j',
    type: 'hinge',
    axis: [0, 0, 1],
    origin_pos: [0, 0, 0],
    origin_quat_xyzw: [0, 0, 0, 1],
    limit: [-Math.PI, Math.PI],
    action_index: 0,
    affine: { scale: 1, offset: 0 },
    q_rest: 0,
    // Real exported specs cap per-tick motion; without a cap the DLS step is
    // a full Newton step and the linearisation is only valid for small
    // errors, so a far target diverges. Keep the fixture realistic.
    max_dq: 0.2,
    ...over,
  };
}

export function spec(joints: SpecJoint[], over: Partial<KinematicSpec> = {}): KinematicSpec {
  return {
    version: 1,
    solver_type: 'weighted_dls',
    n_ik_joints: joints.length,
    joints,
    tool0: { origin_pos: [0, 0, 0], origin_quat_xyzw: [0, 0, 0, 1] },
    gripper: null,
    // rot_err_hold is deliberately huge so the orientation task is always
    // included; tests that care about gating set it themselves.
    damping: { lam_pos: 0.05, lam0: 0.5, w0: 0.05, mu: 0, rot_err_hold: 100 },
    w_rot: 1,
    webxr_to_base_R: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    scale_translation: 1,
    scale_rotation: 1,
    pos_reach_limit: 0.25,
    rot_reach_limit: 0.6,
    ...over,
  };
}

/** Planar 3R arm in the XY plane: three +Z hinges, unit links, tool at +X. */
export function planar3R(over: Partial<KinematicSpec> = {}): KinematicSpec {
  return spec(
    [
      joint({ name: 'a', action_index: 0 }),
      joint({ name: 'b', action_index: 1, origin_pos: [1, 0, 0] }),
      joint({ name: 'c', action_index: 2, origin_pos: [1, 0, 0] }),
    ],
    { tool0: { origin_pos: [1, 0, 0], origin_quat_xyzw: [0, 0, 0, 1] }, ...over },
  );
}

/** Mixed chain: a slide along +X, a tilted hinge, and a hinge with a
 *  non-identity mounting rotation — enough to break a wrong Jacobian. */
export function mixedChain(): KinematicSpec {
  return spec([
    joint({ name: 'lift', type: 'slide', axis: [0, 0, 1], action_index: 0,
            limit: [-1, 1] }),
    joint({ name: 'shoulder', axis: [0, 1, 0], origin_pos: [0.2, 0, 0.1],
            action_index: 1 }),
    joint({
      name: 'elbow',
      axis: [0, 0, 1],
      origin_pos: [0.3, 0, 0],
      origin_quat_xyzw: axisAngle([1, 0, 0], Math.PI / 3),
      action_index: 2,
    }),
  ], {
    tool0: {
      origin_pos: [0.15, 0.05, 0],
      origin_quat_xyzw: axisAngle([0, 1, 0], Math.PI / 4),
    },
  });
}
