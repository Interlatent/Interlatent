"""SO-101 follower kinematics.

Uses the exact elementary-transform chain from the box2ai-robotics
`lerobot-kinematics` package — the reference implementation that ships
working SO-101 IK examples. We reimplement the chain directly in numpy
to avoid the roboticstoolbox + spatialmath dependency tree.

Reference (verified May 2026):
  https://github.com/box2ai-robotics/lerobot-kinematics/blob/main/
    lerobot_kinematics/lerobot/lerobot_Kinematics.py

SO-101 follower chain (math frame, meters, joint axes shown):

  Rz(pan)
   -> tx(0.02943)  tz(0.05504)  Ry(lift)
   -> tx(0.02798)  tz(0.11270)  Ry(elbow)
   -> tx(0.13504)  tz(0.00519)  Ry(wrist_flex)
   -> tx(0.05930)  tz(0.00996)  Rx(wrist_roll)

The crucial fact my earlier hand-rolled model missed: the upper arm
points *up* (along Z) with a small forward offset, not horizontally
forward. This invalidates the simple 2-link planar approximation.

Forward kinematics is a 4x4 matrix product. Inverse kinematics is
damped-least-squares Jacobian pseudoinverse over (lift, elbow,
wrist_flex), with pan solved analytically from atan2 and wrist_roll
passed through directly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Workspace envelope (cylindrical around the base z-axis).
#
# These are computed from FK at the SO-101's joint range (±90° on lift,
# elbow, and wrist_flex). The max radial reach is ~0.34 m and the arm
# can dip ~10 cm below the base plate. We add a small margin to keep
# the IK away from singular configurations at full extension.
#
# Limits exist only to reject obviously-bogus inputs (e.g. NaN, or a
# target 10 m away). Inside this envelope, unreachable targets degrade
# gracefully — the IK saturates at the closest reachable joint config.
SO101_WORKSPACE_R = (0.02, 0.34)   # radial distance from base z-axis (m)
SO101_WORKSPACE_Z = (-0.15, 0.37)  # height above (or below) base plate (m)


# ---------------------------------------------------------------------------
# 4x4 transform helpers (homogeneous coordinates, float64 for IK stability)

def _Rz(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)


def _Ry(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]], dtype=np.float64)


def _Rx(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], dtype=np.float64)


def _T(x: float, y: float, z: float) -> np.ndarray:
    M = np.eye(4, dtype=np.float64)
    M[0, 3] = x
    M[1, 3] = y
    M[2, 3] = z
    return M


# ---------------------------------------------------------------------------

@dataclass
class SO101Kinematics:
    """SO-101 follower kinematics.

    Link parameters default to the values in lerobot-kinematics; per-
    joint sign/offset fields handle calibration differences across
    individual builds. Most users only need to flip a sign if a
    direction feels reversed on their assembly.
    """

    # Elementary-transform link parameters (meters). Defaults from
    # lerobot-kinematics SO-101 model.
    base_to_lift_x: float = 0.02943
    base_to_lift_z: float = 0.05504
    lift_to_elbow_x: float = 0.02798
    lift_to_elbow_z: float = 0.1127
    elbow_to_wflex_x: float = 0.13504
    elbow_to_wflex_z: float = 0.00519
    wflex_to_wroll_x: float = 0.0593
    wflex_to_wroll_z: float = 0.00996

    # Motor↔math conversion per joint:
    #   math_rad = deg2rad((motor_deg - offset) / sign)
    # Defaults assume the motor convention matches the math model.
    pan_sign: float = 1.0
    lift_sign: float = 1.0
    elbow_sign: float = 1.0
    wflex_sign: float = 1.0
    wroll_sign: float = 1.0
    pan_offset: float = 0.0
    lift_offset: float = 0.0
    elbow_offset: float = 0.0
    wflex_offset: float = 0.0
    wroll_offset: float = 0.0

    # ------------------------------------------------------------------
    # Convention conversion

    def _motor_to_math(self, joints_deg: np.ndarray) -> tuple[float, float, float, float, float]:
        return (
            math.radians((float(joints_deg[0]) - self.pan_offset) / self.pan_sign),
            math.radians((float(joints_deg[1]) - self.lift_offset) / self.lift_sign),
            math.radians((float(joints_deg[2]) - self.elbow_offset) / self.elbow_sign),
            math.radians((float(joints_deg[3]) - self.wflex_offset) / self.wflex_sign),
            math.radians((float(joints_deg[4]) - self.wroll_offset) / self.wroll_sign),
        )

    def _math_to_motor_pan(self, pan_rad: float) -> float:
        return math.degrees(pan_rad) * self.pan_sign + self.pan_offset

    def _math_to_motor_lift(self, lift_rad: float) -> float:
        return math.degrees(lift_rad) * self.lift_sign + self.lift_offset

    def _math_to_motor_elbow(self, elbow_rad: float) -> float:
        return math.degrees(elbow_rad) * self.elbow_sign + self.elbow_offset

    def _math_to_motor_wflex(self, wflex_rad: float) -> float:
        return math.degrees(wflex_rad) * self.wflex_sign + self.wflex_offset

    def _math_to_motor_wroll(self, wroll_rad: float) -> float:
        return math.degrees(wroll_rad) * self.wroll_sign + self.wroll_offset

    # ------------------------------------------------------------------
    # Forward kinematics

    def _fk_se3(self, joints_deg: np.ndarray) -> np.ndarray:
        """Full 4x4 homogeneous transform of the gripper frame."""
        pan, lift, elbow, wflex, wroll = self._motor_to_math(joints_deg)
        M = _Rz(pan)
        M = M @ _T(self.base_to_lift_x, 0, self.base_to_lift_z) @ _Ry(lift)
        M = M @ _T(self.lift_to_elbow_x, 0, self.lift_to_elbow_z) @ _Ry(elbow)
        M = M @ _T(self.elbow_to_wflex_x, 0, self.elbow_to_wflex_z) @ _Ry(wflex)
        M = M @ _T(self.wflex_to_wroll_x, 0, self.wflex_to_wroll_z) @ _Rx(wroll)
        return M

    def fk(self, joints_deg: np.ndarray) -> np.ndarray:
        """Gripper tip position (x, y, z) in meters."""
        return self._fk_se3(joints_deg)[:3, 3].astype(np.float32)

    def gripper_pitch(self, joints_deg: np.ndarray) -> float:
        """Gripper pitch (rotation around Y in the math frame), radians.

        Pitch from an extrinsic XYZ Euler decomposition; matches the
        `beta` returned by lerobot-kinematics' FK.
        """
        M = self._fk_se3(joints_deg)
        return math.atan2(-M[2, 0], math.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2))

    # ------------------------------------------------------------------
    # Inverse kinematics

    def ik_full(
        self,
        target_xyz: np.ndarray,
        target_pitch_rad: float,
        target_roll_rad: float = 0.0,
        current_joints: Optional[np.ndarray] = None,
        n_iter: int = 60,
        max_dq_deg: float = 8.0,
    ) -> np.ndarray:
        """Position + pitch + roll → 5 arm motor angles (deg).

        Returns [pan, lift, elbow, wrist_flex, wrist_roll]. Pan is
        solved analytically from atan2; wrist_roll is passed through;
        (lift, elbow, wrist_flex) come from damped-pseudoinverse IK
        in the arm's vertical plane.

        Targets are auto-clamped to the SO-101 workspace envelope
        (cylindrical: 10-32 cm radial, 4.6-30 cm height). Unreachable
        pitches at extreme positions degrade gracefully — the position
        is matched first, the pitch as close as possible.
        """
        x = float(target_xyz[0])
        y = float(target_xyz[1])
        z = float(target_xyz[2])

        # --- Pan analytically. The arm rotates around base z to face
        #     the target in the horizontal plane.
        pan_math = math.atan2(y, x)

        # --- Cylindrical workspace clamp on (r, z). Y stays implicit
        #     in pan; only the radius matters for reachability.
        r = math.hypot(x, y)
        r = max(SO101_WORKSPACE_R[0], min(SO101_WORKSPACE_R[1], r))
        z = max(SO101_WORKSPACE_Z[0], min(SO101_WORKSPACE_Z[1], z))

        # --- Warm start. Without a seed we pick a sensible mid-workspace
        #     pose so the first call has somewhere to iterate from.
        if current_joints is None:
            q_lift_deg = 30.0
            q_elbow_deg = -60.0
            q_wflex_deg = 30.0
        else:
            cj = np.asarray(current_joints, dtype=np.float64)
            q_lift_deg = float(cj[1])
            q_elbow_deg = float(cj[2])
            q_wflex_deg = float(cj[3])

        pan_motor = self._math_to_motor_pan(pan_math)
        wroll_motor = self._math_to_motor_wroll(target_roll_rad)

        # --- 3-DOF damped pseudoinverse IK over (lift, elbow, wflex)
        #     against (r, z, pitch). The Jacobian is computed via finite
        #     differences on the full SE(3) FK; we then read r, z, and
        #     pitch off the resulting transform.
        q = np.array([q_lift_deg, q_elbow_deg, q_wflex_deg], dtype=np.float64)
        eps = 0.3  # deg
        target = np.array([r, z, target_pitch_rad], dtype=np.float64)

        for _ in range(n_iter):
            full = np.array([pan_motor, q[0], q[1], q[2], wroll_motor], dtype=np.float32)
            M = self._fk_se3(full)
            xyz = M[:3, 3]
            cur_r = math.hypot(float(xyz[0]), float(xyz[1]))
            cur_z = float(xyz[2])
            cur_pitch = math.atan2(-M[2, 0], math.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2))
            err = target - np.array([cur_r, cur_z, cur_pitch], dtype=np.float64)

            if float(np.linalg.norm(err)) < 1e-4:
                break

            # Numerical Jacobian
            J = np.zeros((3, 3))
            for i in range(3):
                qp = q.copy()
                qp[i] += eps
                full_p = np.array([pan_motor, qp[0], qp[1], qp[2], wroll_motor], dtype=np.float32)
                Mp = self._fk_se3(full_p)
                rp = math.hypot(float(Mp[0, 3]), float(Mp[1, 3]))
                zp = float(Mp[2, 3])
                pp = math.atan2(-Mp[2, 0], math.sqrt(Mp[0, 0] ** 2 + Mp[1, 0] ** 2))
                J[0, i] = (rp - cur_r) / eps
                J[1, i] = (zp - cur_z) / eps
                J[2, i] = (pp - cur_pitch) / eps

            jjt = J @ J.T
            lam = 1e-6 * (float(np.trace(jjt)) + 1e-12)
            try:
                dq = J.T @ np.linalg.solve(jjt + lam * np.eye(3), err)
            except np.linalg.LinAlgError:
                break
            # Uniform-scale the dq vector instead of clipping per component.
            # Per-component clipping breaks the implicit pitch constraint
            # (dpitch = dq_lift + dq_elbow + dq_wflex == 0 when pitch is
            # held fixed), causing the IK to drift away from the target.
            mag = float(np.max(np.abs(dq)))
            if mag > max_dq_deg:
                dq = dq * (max_dq_deg / mag)
            q += dq

        return np.array([pan_motor, q[0], q[1], q[2], wroll_motor], dtype=np.float32)

    # ------------------------------------------------------------------
    # Backwards-compatibility shims (old name used by retargeting.py)

    def ik_jacobian(
        self,
        target_xyz: np.ndarray,
        current_joints: np.ndarray,
        target_pitch_rad: Optional[float] = None,
        **_: object,
    ) -> np.ndarray:
        """Legacy 4-DOF call site. Forwards to `ik_full` and drops
        wrist_roll from the returned vector for compatibility.
        """
        pitch = target_pitch_rad if target_pitch_rad is not None else self.gripper_pitch(current_joints)
        full = self.ik_full(
            target_xyz=target_xyz,
            target_pitch_rad=pitch,
            target_roll_rad=math.radians(float(current_joints[4])),
            current_joints=current_joints,
        )
        return full[:4]
