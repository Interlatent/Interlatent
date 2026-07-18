"""Reference dimos session blueprints — ``dimos run interlatent.<kind>``.

These are registered under dimos's ``"dimos.blueprints"`` entry-point group (see
pyproject), so a box with both packages installed can run a known-good session
stack in one command. Each blueprint is the OTHER half of the adapter's
connect-time contract (ADR 0018, verified fail-closed either way):

- a ``ControlCoordinator`` with the kind's hardware,
- a **servo task** claiming exactly the kind's joints with a non-zero timeout —
  the piece stock dimos coordinator blueprints lack, without which dimos
  SILENTLY IGNORES streamed ``joint_command``,
- no other task claiming those joints (strict exclusivity),
- a camera publishing ``color_image`` (real webcam off-sim; in ``--simulation``
  the MujocoSimModule publishes it already),
- ``publish_joint_state`` left on.

Operator-authored blueprints satisfying the same contract are equally valid —
this module is a convenience, not a requirement. A dimos-side memory2 recorder
for low-level streams (go2_base pattern) is documented in CONFIG.md, not baked
in here.

Import guard: dimos is required at import time BY DESIGN — dimos itself resolves
the entry point lazily, so a base install never touches this module, and a
half-installed state produces one actionable error instead of a deep stack.
"""
from __future__ import annotations

try:
    from dimos.control.coordinator import TaskConfig
    from dimos.core.coordination.blueprints import autoconnect
    from dimos.core.global_config import global_config
    from dimos.hardware.sensors.camera.module import CameraModule
    from dimos.robot.manipulators.common.blueprints import coordinator
    from dimos.robot.manipulators.common.sim import mujoco_if_sim
    from dimos.robot.manipulators.xarm.config import XARM7_SIM_PATH, xarm7_hardware
except ImportError as exc:  # pragma: no cover - exercised only when half-installed
    raise ImportError(
        "interlatent's dimos session blueprints require the dimos stack: "
        "pip install 'interlatent[dimos]' (python 3.11-3.12). "
        f"Underlying import failure: {exc}"
    ) from exc

# Servo-task knobs, mirrored by the adapter's connect-time verification:
# timeout MUST be non-zero (0 = hold-forever on a stalled session) and the
# task must be the SOLE claimant of the arm joints (strict exclusivity).
_SERVO_TIMEOUT_S = 0.5
_SERVO_PRIORITY = 10


def _servo_task(hardware) -> TaskConfig:
    # Claim ALL joints including the gripper: dimos's per-tick hardware write
    # re-sends `_last_commanded` for every gripper joint (hardware_interface
    # write_command), so a gripper left unclaimed is stomped back to its
    # startup value at tick rate the moment any task streams to this hardware.
    # The gripper therefore rides joint_command like any other joint; the
    # coordinator's set_gripper_position RPC is only safe on an idle stack.
    return TaskConfig(
        name=f"servo_{hardware.hardware_id}",
        type="servo",
        joint_names=list(hardware.joints) + list(hardware.gripper_joints),
        priority=_SERVO_PRIORITY,
        params={"timeout": _SERVO_TIMEOUT_S},
    )


def _camera_if_real() -> tuple:
    """Webcam camera module off-sim only (dimos learning-blueprint pattern):
    in ``--simulation`` the MujocoSimModule already publishes color_image and a
    real device would be redundant (and fail with none connected)."""
    if global_config.simulation:
        return ()
    return (CameraModule.blueprint(),)


# UFACTORY xArm7 + gripper. `mock_without_address=True`: with no xarm7_ip
# configured this runs the mock adapter — the hardware-free path the
# integration tests (and first-time operators) use.
_xarm7_hw = xarm7_hardware("arm", gripper=True, mock_without_address=True)

xarm7 = autoconnect(
    coordinator(
        hardware=[_xarm7_hw],
        tasks=[_servo_task(_xarm7_hw)],
    ),
    *_camera_if_real(),
    *mujoco_if_sim(XARM7_SIM_PATH, len(_xarm7_hw.joints)),
)


__all__ = ["xarm7"]
