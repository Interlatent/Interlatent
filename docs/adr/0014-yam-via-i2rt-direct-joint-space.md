# YAM arms via the i2rt driver directly, joint-space only

We support I2RT's YAM bimanual arms (`--robot yam`) with a thin
`interlatent.adapters.yam` adapter that talks to the **`i2rt` driver directly**
(`get_yam_robot` → `command_joint_pos` / `get_joint_pos`) and exposes them through the
existing joint-space [action interface](0013-manual-action-interface-below-schedule.md),
reusing interlatent's own DRTC inference, recording, and safety. It is a vendor
subpackage selected by robot kind, per [ADR 0011](0011-vendor-robot-subpackage-via-robot-kind.md).

## Considered options

- **Wrap TRI's raiden `RobotController`.** Rejected: raiden bundles teleop, IK, learned
  depth, calibration, and a chiral WebSocket policy server — all redundant with
  interlatent — and pulls in jax/pyroki/chiral/torch. We need only the thin CAN path
  raiden's own `scripts/read_arm_poses.py` uses, so we depend on `i2rt` alone.
- **Support raiden's 20-D EE-pose action space (IK).** Rejected: it would break the
  robot-side joint-space invariant (ADR 0013) and reintroduce an IK dependency. YAM
  policies run in joint space (raiden's 14-D `joint` action mode); EE-pose stays out.

## Consequences

- No raiden dependency; `interlatent[yam]` pulls in `i2rt` + RGB camera libs only.
- The few raiden constants we want (CAN interface names, follower PD gains,
  `FOLLOWER_HOME_POS`, `GripperType.LINEAR_4310`) are reimplemented thin in the adapter.
- The YAM `RobotProfile` arm `joint_limits` are transcribed from the i2rt YAM URDF
  (`i2rt/robot_models/arm/yam/yam.urdf`); `max_velocity` is capped conservatively and
  the gripper range remains a placeholder until verified on hardware.
