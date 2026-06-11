"""Hand-landmark -> SO-101 joint targets.

v0.4 retargeter: Cartesian "macbook-as-table" mapping.

Mental model: the macbook screen is the table. Your hand position in
front of the camera maps to a target position for the gripper tip on
that table.

  hand x         (left-right in image)  -> arm Y  (lateral on table)
  hand y         (up-down in image)     -> arm X  (forward/back on table)
  hand size      (depth proxy)          -> arm Z  (height; bigger hand = arm lower)

MediaPipe's wrist.z is always 0 (it's the origin for the other
landmarks), so we use apparent hand size as the camera-distance
proxy: closer hand fills more of the frame.

Calibration captures both the hand's current position and the arm's
current pose. From then on, hand *deltas* are scaled by
`workspace_scale` and applied to the calibrated home position, so a
still hand produces a still arm. A small deadband filters MediaPipe
jitter so the arm stops dead between intentional moves.

`--mirror-x` flips lateral if the camera is/isn't selfie-mirrored.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .hand_tracker import HandObservation
from .kinematics import SO101Kinematics


@dataclass
class RetargetCalibration:
    """Neutral-pose snapshot captured when the first confident hand is seen."""

    wrist_xyz: np.ndarray   # hand wrist landmark at calibration (3,)
    hand_size: float        # bbox diagonal of all 21 landmarks; proxy for camera distance
    pinch_dist: float       # pinch distance at calibration
    home_joints: np.ndarray # Pi-reported home pose (6,)
    home_xyz: np.ndarray    # gripper tip world position at home (3,), via FK
    home_pitch: float       # gripper pitch (math frame, rad) at home, via FK


@dataclass
class RetargetConfig:
    # Meters of arm motion per unit of normalized hand motion.
    # 0.20 means moving your hand fully across the camera frame moves
    # the arm ~20 cm. Lower = the arm tracks more precisely but you
    # need to move your hand farther.
    workspace_scale: float = 0.20

    # MediaPipe reports wrist.z == 0 (origin), so we can't use it
    # for camera distance. We use hand size in the image (wrist to
    # middle-finger MCP) as the depth proxy: closer hand = larger
    # in frame. depth_scale_boost converts that ratio change into
    # robot Z motion.
    depth_scale_boost: float = 5.0

    # Deadband in meters: hand deltas that translate to less than this
    # much robot motion (after scaling) are treated as jitter and the
    # arm holds its previous target. Set to 0 to disable.
    deadband_m: float = 0.008

    # Desired pitch of the gripper tip in the math frame, radians.
    # `None` (default) means "preserve the orientation the gripper had
    # at calibration" — usually what you want for toy-grabber feel.
    gripper_pitch_rad: Optional[float] = None

    # Mirror the lateral axis. Set True if "hand right" makes the arm
    # go left on your camera setup (selfie cams are usually mirrored
    # by the OS, non-selfie cams are not).
    mirror_x: bool = True

    # Gripper from thumb-index pinch distance.
    gripper_open_dist: float = 0.20
    gripper_closed_dist: float = 0.02

    # Low-pass filter on the final joint vector. 0 = no smoothing,
    # 1 = freeze. Higher values reduce remaining jitter at the cost
    # of a small lag.
    smoothing: float = 0.6


def _hand_size(landmarks: np.ndarray) -> float:
    """Bounding-box diagonal of all 21 landmarks in normalized image coords.

    Robust to finger curling, hand rotation, and pose changes — only
    changes meaningfully with actual camera distance.
    """
    xs = landmarks[:, 0]
    ys = landmarks[:, 1]
    dx = float(xs.max() - xs.min())
    dy = float(ys.max() - ys.min())
    return float(np.hypot(dx, dy))


@dataclass
class Retargeter:
    config: RetargetConfig = field(default_factory=RetargetConfig)
    kinematics: SO101Kinematics = field(default_factory=SO101Kinematics)
    calibration: Optional[RetargetCalibration] = None
    _last_target: Optional[np.ndarray] = None
    _last_target_xyz: Optional[np.ndarray] = None
    # Debug telemetry surfaced for the preview overlay.
    last_hand_size: float = 0.0
    last_d_size: float = 0.0
    last_target_delta: Optional[np.ndarray] = None  # target_xyz - home_xyz, before deadband

    def calibrate(self, obs: HandObservation, home_joints: np.ndarray) -> None:
        home = home_joints.astype(np.float32).copy()
        home_xyz = self.kinematics.fk(home)
        self.calibration = RetargetCalibration(
            wrist_xyz=obs.landmarks[0].copy(),
            hand_size=_hand_size(obs.landmarks),
            pinch_dist=float(np.linalg.norm(obs.landmarks[4] - obs.landmarks[8])),
            home_joints=home,
            home_xyz=home_xyz,
            home_pitch=self.kinematics.gripper_pitch(home),
        )
        self._last_target = home.copy()
        self._last_target_xyz = home_xyz.copy()

    def reset(self) -> None:
        self.calibration = None
        self._last_target = None
        self._last_target_xyz = None

    def map(self, obs: HandObservation) -> Optional[np.ndarray]:
        if self.calibration is None:
            return None

        c = self.calibration
        cfg = self.config

        # Hand delta in image-normalized coords.
        dx_hand = float(obs.landmarks[0][0] - c.wrist_xyz[0])
        dy_hand = float(obs.landmarks[0][1] - c.wrist_xyz[1])
        # MediaPipe's landmark[0].z is always 0 (wrist is the origin),
        # so it can't be used for camera distance. Use the bounding-
        # box diagonal of all 21 landmarks instead — closer hand
        # fills more of the frame, regardless of orientation.
        current_size = _hand_size(obs.landmarks)
        d_size = current_size - c.hand_size
        self.last_hand_size = current_size
        self.last_d_size = d_size

        # Macbook-as-table mapping. Robot world frame: X forward, Y left, Z up.
        #   hand x      -> arm Y (lateral)
        #   hand y      -> arm X (forward/back; hand up = arm forward)
        #   hand size   -> arm Z (height; bigger hand = closer to camera = arm lower)
        mirror = -1.0 if cfg.mirror_x else 1.0
        d_lateral = mirror * -dx_hand * cfg.workspace_scale
        d_forward = -dy_hand * cfg.workspace_scale
        d_up = -d_size * cfg.workspace_scale * cfg.depth_scale_boost

        target_xyz = c.home_xyz + np.array([d_forward, d_lateral, d_up], dtype=np.float32)
        self.last_target_delta = (target_xyz - c.home_xyz).astype(np.float32)

        # Deadband: if the new target is within deadband_m of the last
        # committed target, hold the old target. This stops MediaPipe
        # jitter from translating into constant arm motion when the
        # user is intentionally holding their hand still.
        if cfg.deadband_m > 0 and self._last_target_xyz is not None:
            if float(np.linalg.norm(target_xyz - self._last_target_xyz)) < cfg.deadband_m:
                target_xyz = self._last_target_xyz
        self._last_target_xyz = target_xyz

        # Jacobian-pseudoinverse IK warm-started from the last
        # commanded joints. With 4 arm DOFs and a 3-D position target,
        # the pseudoinverse minimum-norm solution picks the joints
        # closest to the warm start — which keeps the wrist near its
        # home value naturally, no explicit pitch constraint needed.
        # An explicit pitch constraint tends to lock the arm at the
        # workspace boundary when real geometry doesn't match the
        # kinematic model, so we drop it.
        warm = self._last_target if self._last_target is not None else c.home_joints
        ik_joints = self.kinematics.ik_jacobian(
            target_xyz=target_xyz,
            current_joints=warm,
            target_pitch_rad=None,
        )

        target = c.home_joints.copy()
        target[0] = ik_joints[0]   # shoulder_pan
        target[1] = ik_joints[1]   # shoulder_lift
        target[2] = ik_joints[2]   # elbow_flex
        target[3] = ik_joints[3]   # wrist_flex
        # target[4] (wrist_roll) stays at home for v0.4.

        # Gripper from pinch distance.
        pinch = float(np.linalg.norm(obs.landmarks[4] - obs.landmarks[8]))
        t = (pinch - cfg.gripper_closed_dist) / max(
            cfg.gripper_open_dist - cfg.gripper_closed_dist, 1e-6
        )
        target[5] = float(np.clip(t, 0.0, 1.0)) * 100.0

        # Low-pass filter against jitter.
        a = cfg.smoothing
        if self._last_target is not None:
            target = a * self._last_target + (1.0 - a) * target
        self._last_target = target.astype(np.float32)
        return self._last_target.copy()
