// Copied from Interlatent-Main site/src/lib/teleop/clutchPoseMapper.ts @ f7e4bfb6 (2026-07-30). Upstream is that copy; sync fixes both ways.
/**
 * Clutch-relative pose mapping: WebXR controller pose -> EE target in the
 * arm-base frame.
 *
 * TypeScript port of vr-teleop-kit's `ClutchPoseMapper`
 * (core/pose_mapping.py) — quats are xyzw here (WebXR/wire convention)
 * instead of MuJoCo wxyz. One deliberate deviation from the reference:
 * rotation increments use a **swing–twist split** instead of the
 * reference's pure world-frame mapping (see below).
 *
 * Swing–twist rotation mapping. The reference maps every rotation
 * increment extrinsically (world axis → base axis via R). That is
 * intuitive only while the hand and the gripper point roughly the same
 * way: with the gripper facing up/down and the hand held naturally
 * (horizontal), a wrist ROLL is a rotation about a horizontal world axis,
 * which swings the vertical gripper sideways — perceived as yaw. So each
 * increment is decomposed about the controller's own pointing axis:
 * the **twist** (wrist roll) is re-applied about the TOOL's current
 * pointing axis, while the **swing** (hand pitch/yaw) keeps the
 * reference's world mapping. Roll therefore always spins the gripper
 * about its own axis, in every pose; swing behavior is unchanged, and
 * when hand and gripper are co-oriented the split is a no-op.
 *
 * The mapper holds the "engage" state — the controller pose and the EE pose
 * captured at the moment the operator pressed grip (the EE anchor comes
 * from the pod's `ee_state` stream — FK of the live robot). While the
 * clutch is held, the EE target is the engaged EE pose composed with an
 * effective controller delta, rotated from Quest world into the arm-base
 * frame. The delta accumulates per-tick increments and is reach-limited to
 * within `rotReachLimit` / `posReachLimit` of the arm's CURRENT pose, with
 * the excess absorbed (slipping clutch): a demand pressed far past a joint
 * stop can never build up the ~180° error where the shortest-way direction
 * flips, and reversing the hand bites immediately instead of retracing the
 * overshoot.
 *
 * On clutch release the mapper disengages; `target()` returns null until
 * the next engage, so the operator can reposition their hand without
 * moving the robot.
 */
import {
  Mat3,
  Quat,
  QUAT_IDENTITY,
  Vec3,
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
} from './quat';

/** WebXR grip/aim spaces point along -Z: the controller's "forward". */
const CTRL_FORWARD: Vec3 = [0, 0, -1];

/**
 * Split a small rotation `q` into (swing, twistAngle) about unit axis `a`:
 * `q = swing ⊗ twist(a, angle)`, with the signed angle about +a. Standard
 * swing–twist decomposition — project the quat's vector part onto the axis.
 */
function swingTwist(q: Quat, a: Vec3): { swing: Quat; twistAngle: number } {
  const p = q[0] * a[0] + q[1] * a[1] + q[2] * a[2];
  const twist = quatNormalize([p * a[0], p * a[1], p * a[2], q[3]]);
  const swing = quatMul(q, quatConj(twist));
  const s = twist[0] * a[0] + twist[1] * a[1] + twist[2] * a[2];
  return { swing, twistAngle: 2 * Math.atan2(s, twist[3]) };
}

export interface ClutchPoseMapperOptions {
  /** 3x3 rotation taking Quest-world vectors to arm-base vectors. */
  R?: Mat3;
  /** Linear gain on translation (1.0 = 1:1 motion). */
  scale?: number;
  /** Gain on rotation delta (a rate gain on the reach-limited path). */
  scaleRotation?: number;
  /** Max angle (rad) the orientation target may run ahead of the arm's
   *  current orientation. 0/undefined disables (legacy absolute mapping). */
  rotReachLimit?: number;
  /** Max distance (m) the position target may run ahead of the arm's
   *  current EE position. 0/undefined disables. */
  posReachLimit?: number;
  /** Tool APPROACH axis in the tool0 BODY frame — the axis wrist twist is
   *  applied about, sign-matched to the controller's pointing direction.
   *  Default -Z (YAM: fingers + tool0 offset extend along body -Z). Flip
   *  if a robot's tool frame points the other way. */
  toolTwistAxis?: Vec3;
}

const IDENTITY_R: Mat3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];

export class ClutchPoseMapper {
  R: Mat3;
  scale: number;
  scaleRotation: number;
  rotReachLimit: number | null;
  posReachLimit: number | null;
  toolTwistAxis: Vec3;

  private _engaged = false;
  private _ctrlEngagePos: Vec3 | null = null;
  private _ctrlEngageQuat: Quat | null = null;
  private _eeEngagePos: Vec3 | null = null;
  private _eeEngageQuat: Quat | null = null;
  private _RQuat: Quat;
  private _RQuatConj: Quat;
  // Incremental reach-limit state: previous-tick controller pose (quest
  // frame) + accumulated effective deltas (arm-base frame). Reset per engage.
  private _ctrlPrevQuat: Quat | null = null;
  private _ctrlPrevPos: Vec3 | null = null;
  private _dQuatEff: Quat = [...QUAT_IDENTITY];
  private _dPosEff: Vec3 = [0, 0, 0];

  constructor(opts: ClutchPoseMapperOptions = {}) {
    this.R = opts.R ?? IDENTITY_R;
    this.scale = opts.scale ?? 1.0;
    this.scaleRotation = opts.scaleRotation ?? 1.0;
    this.rotReachLimit = opts.rotReachLimit ?? 0.6;
    this.posReachLimit = opts.posReachLimit ?? 0.25;
    this.toolTwistAxis = opts.toolTwistAxis ?? [0, 0, -1];
    this._RQuat = matToQuat(this.R);
    this._RQuatConj = quatConj(this._RQuat);
  }

  get engaged(): boolean {
    return this._engaged;
  }

  /** Replace the Quest→arm rotation. Typically called per engage with a
   *  yaw-corrected R so "controller forward" tracks "robot forward"
   *  regardless of where the operator's body faces. */
  setR(R: Mat3): void {
    this.R = R;
    this._RQuat = matToQuat(R);
    this._RQuatConj = quatConj(this._RQuat);
  }

  /** Capture the engage frame. Call on the rising clutch edge with the
   *  arm's current EE pose (from the pod's ee_state stream). */
  engage(
    controllerPosQuest: Vec3,
    controllerQuatQuest: Quat,
    eePosArmbase: Vec3,
    eeQuatArmbase: Quat,
  ): void {
    this._ctrlEngagePos = [...controllerPosQuest];
    this._ctrlEngageQuat = [...controllerQuatQuest];
    this._eeEngagePos = [...eePosArmbase];
    this._eeEngageQuat = [...eeQuatArmbase];
    this._ctrlPrevQuat = [...controllerQuatQuest];
    this._ctrlPrevPos = [...controllerPosQuest];
    this._dQuatEff = [...QUAT_IDENTITY];
    this._dPosEff = [0, 0, 0];
    this._engaged = true;
  }

  disengage(): void {
    this._engaged = false;
  }

  /**
   * Current EE target in arm-base frame, or null when disengaged.
   *
   * `eePosArmbase` / `eeQuatArmbase` are the arm's CURRENT EE pose (latest
   * ee_state); passing them enables the reach limits. Without them the
   * legacy absolute mapping applies.
   */
  target(
    controllerPosQuest: Vec3,
    controllerQuatQuest: Quat,
    eePosArmbase?: Vec3 | null,
    eeQuatArmbase?: Quat | null,
    aimQuatQuest?: Quat | null,
  ): { pos: Vec3; quat: Quat } | null {
    if (!this._engaged) return null;
    const ctrlEngagePos = this._ctrlEngagePos!;
    const ctrlEngageQuat = this._ctrlEngageQuat!;
    const eeEngagePos = this._eeEngagePos!;
    const eeEngageQuat = this._eeEngageQuat!;

    // Position delta. Reach-limited path: accumulate per-tick increments
    // (vectors commute, so increments with per-tick scale compose to
    // exactly the absolute scaled delta until the reach limit absorbs).
    const pNow = controllerPosQuest;
    const posLimited = !!eePosArmbase && !!this.posReachLimit;
    let dPosArm: Vec3;
    if (posLimited) {
      const prev = this._ctrlPrevPos!;
      const inc = matVec(this.R, [
        this.scale * (pNow[0] - prev[0]),
        this.scale * (pNow[1] - prev[1]),
        this.scale * (pNow[2] - prev[2]),
      ]);
      this._dPosEff = [
        this._dPosEff[0] + inc[0],
        this._dPosEff[1] + inc[1],
        this._dPosEff[2] + inc[2],
      ];
      dPosArm = this._dPosEff;
    } else {
      dPosArm = matVec(this.R, [
        this.scale * (pNow[0] - ctrlEngagePos[0]),
        this.scale * (pNow[1] - ctrlEngagePos[1]),
        this.scale * (pNow[2] - ctrlEngagePos[2]),
      ]);
    }
    this._ctrlPrevPos = [...pNow];

    let qNow: Quat = [...controllerQuatQuest];
    const rotLimited = !!eeQuatArmbase && !!this.rotReachLimit;
    let dQuatArm: Quat;
    if (rotLimited) {
      // Incremental path: accumulate this tick's controller rotation
      // increment (a degree or two — never direction-ambiguous) into the
      // effective delta. scaleRotation applies per increment (a RATE gain,
      // like mouse sensitivity); re-clutching realigns any drift.
      //
      // Swing–twist split (see module docstring): the increment's twist
      // about the controller's pointing axis becomes a rotation about the
      // TOOL's current pointing axis; only the swing takes the extrinsic
      // R-mapping. Co-oriented hand/gripper ⇒ both axes coincide and this
      // reduces exactly to the reference's world-frame mapping.
      const prev = this._ctrlPrevQuat!;
      if (quatDot(qNow, prev) < 0) {
        qNow = [-qNow[0], -qNow[1], -qNow[2], -qNow[3]]; // hemisphere-align
      }
      const inc = quatMul(qNow, quatConj(prev));
      this._ctrlPrevQuat = [...qNow];
      // Decomposition axis: the controller's POINTING direction in Quest
      // world. The control pose is gripSpace, whose -Z is the HANDLE, not
      // where the operator points — on a Quest Touch they differ by a
      // large fixed tilt, and decomposing about the handle misclassifies
      // part of a physical wrist roll as swing (world-mapped → pitch/yaw
      // leakage). The aim (targetRay) orientation, when supplied, gives
      // the true pointing axis; the grip quat is only a fallback.
      const aCtrl = rotateVec(CTRL_FORWARD, aimQuatQuest ?? prev);
      const { swing, twistAngle } = swingTwist(inc, aCtrl);
      // Tool pointing axis in arm-base world, at the current target.
      const targetPrevQuat = quatMul(this._dQuatEff, eeEngageQuat);
      const aTool = rotateVec(this.toolTwistAxis, targetPrevQuat);
      const th = twistAngle * this.scaleRotation;
      const twistArm = rotvecToQuat([aTool[0] * th, aTool[1] * th, aTool[2] * th]);
      let swingScaled = swing;
      if (this.scaleRotation !== 1.0) swingScaled = quatPow(swing, this.scaleRotation);
      const swingArm = quatMul(quatMul(this._RQuat, swingScaled), this._RQuatConj);
      const incArm = quatMul(swingArm, twistArm);
      dQuatArm = quatNormalize(quatMul(incArm, this._dQuatEff));
      this._dQuatEff = dQuatArm;
    } else {
      // Legacy absolute path: rotation delta in Quest world (world-frame
      // composition: now · engage⁻¹), conjugated by R to express in arm
      // base. Deliberately still pure-extrinsic (no swing–twist): absolute
      // deltas aren't per-tick, so the twist re-anchor doesn't apply. Keep
      // the incremental state fresh anyway so a live reach-limit toggle
      // doesn't see stale prev.
      if (this._ctrlPrevQuat) {
        if (quatDot(qNow, this._ctrlPrevQuat) < 0) {
          qNow = [-qNow[0], -qNow[1], -qNow[2], -qNow[3]];
        }
        this._ctrlPrevQuat = [...qNow];
      }
      let dQuatQuest = quatMul(qNow, quatConj(ctrlEngageQuat));
      if (this.scaleRotation !== 1.0) dQuatQuest = quatPow(dQuatQuest, this.scaleRotation);
      dQuatArm = quatMul(quatMul(this._RQuat, dQuatQuest), this._RQuatConj);
    }

    let targetQuat = quatMul(dQuatArm, eeEngageQuat);

    if (rotLimited) {
      // Rotation reach limit: clamp the target to within rotReachLimit of
      // the arm's current orientation and ABSORB the excess into the
      // effective delta (slipping clutch).
      const eeQ = eeQuatArmbase as Quat;
      const e = quatToRotvec(quatMul(targetQuat, quatConj(eeQ)));
      const eNorm = Math.hypot(e[0], e[1], e[2]);
      const limit = this.rotReachLimit as number;
      if (eNorm > limit) {
        const k = limit / eNorm;
        targetQuat = quatMul(rotvecToQuat([e[0] * k, e[1] * k, e[2] * k]), eeQ);
        this._dQuatEff = quatMul(targetQuat, quatConj(eeEngageQuat));
        dQuatArm = this._dQuatEff;
      }
    }

    let targetPos: Vec3 = [
      eeEngagePos[0] + dPosArm[0],
      eeEngagePos[1] + dPosArm[1],
      eeEngagePos[2] + dPosArm[2],
    ];

    // Position reach limit: clamp toward the current EE position and ABSORB
    // the excess (mouse-at-screen-edge semantics).
    if (posLimited) {
      const eeP = eePosArmbase as Vec3;
      const dp: Vec3 = [
        targetPos[0] - eeP[0],
        targetPos[1] - eeP[1],
        targetPos[2] - eeP[2],
      ];
      const dpNorm = Math.hypot(dp[0], dp[1], dp[2]);
      const limit = this.posReachLimit as number;
      if (dpNorm > limit) {
        const k = limit / dpNorm;
        const clamped: Vec3 = [eeP[0] + dp[0] * k, eeP[1] + dp[1] * k, eeP[2] + dp[2] * k];
        this._dPosEff = [
          this._dPosEff[0] + (clamped[0] - targetPos[0]),
          this._dPosEff[1] + (clamped[1] - targetPos[1]),
          this._dPosEff[2] + (clamped[2] - targetPos[2]),
        ];
        targetPos = clamped;
      }
    }

    return { pos: targetPos, quat: quatNormalize(targetQuat) };
  }
}
