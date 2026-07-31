"""Reference dimos session blueprints — ``dimos run interlatent.<kind>``.

These are registered under dimos's ``"dimos.blueprints"`` entry-point group (see
pyproject), so a box with both packages installed can run a known-good session
stack in one command. Each blueprint is the OTHER half of the adapter's
connect-time contract (ADR 0018, verified fail-closed either way):

- a ``ControlCoordinator`` with the kind's hardware,
- a ``ManipulationModule`` carrying the robot model and a Viser viewer — the
  mock hardware path therefore gives developers a useful, browser-visible
  robot without requiring either physical hardware or MuJoCo,
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

Every dimos-shipped, per-vendor blueprint (xarm7's, A1Z's, ...) configures a
``trajectory`` task instead of the exclusive ``servo`` task this contract
needs, so this module has always had to author its own composition per kind
rather than reuse dimos's. ``_streaming_blueprint`` + ``_mock_hardware`` below
are the *generic* halves of that composition — genuinely vendor-agnostic,
built once and shared by every kind. What's left per kind is real,
irreducible vendor knowledge (each vendor's own hardware/model factory has a
different signature and mock/sim fork behavior — see CONFIG.md and the
per-kind blocks below) — now a few lines of binding, not a duplicated block.

Import guard: dimos is required at import time BY DESIGN — dimos itself resolves
the entry point lazily, so a base install never touches this module, and a
half-installed state produces one actionable error instead of a deep stack.
"""
from __future__ import annotations

from functools import partial

try:
    from dimos.control.components import (
        HardwareComponent,
        HardwareType,
        make_gripper_joints,
        make_joints,
    )
    from dimos.control.coordinator import TaskConfig
    from dimos.core.coordination.blueprints import autoconnect
    from dimos.core.global_config import global_config
    from dimos.hardware.sensors.camera.module import CameraModule
    from dimos.robot.manipulators.common.blueprints import coordinator, planner
    from dimos.robot.manipulators.common.sim import mujoco_if_sim
    from dimos.robot.manipulators.a1z.config import a1z_hardware, make_a1z_model_config
    from dimos.robot.manipulators.xarm.config import (
        XARM7_SIM_PATH,
        make_xarm7_model_config,
        xarm7_hardware,
    )
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


def _mock_hardware(
    hw_id: str,
    dof: int,
    *,
    has_gripper: bool,
    gripper_open_position: float | None = None,
    gripper_closed_position: float | None = None,
) -> HardwareComponent:
    """Vendor-independent hardware-free dev path.

    Built directly from dimos's own uniform primitives (``HardwareComponent``,
    ``make_joints``, ``make_gripper_joints`` — imported identically by every
    manipulator vendor dimos ships), NOT via any vendor's own mock-fallback
    logic. That logic is inconsistent across vendors (xarm7 has an
    address-optional ``mock_without_address`` knob; A1Z has none; a750 has its
    own differently-defaulted knob; openyam is always mock) — this gives every
    kind the same hardware-free path regardless of what its vendor factory
    happens to support.
    """
    return HardwareComponent(
        hardware_id=hw_id,
        hardware_type=HardwareType.MANIPULATOR,
        joints=make_joints(hw_id, dof),
        adapter_type="mock",
        gripper_joints=make_gripper_joints(hw_id) if has_gripper else [],
        gripper_open_position=gripper_open_position,
        gripper_closed_position=gripper_closed_position,
    )


def _resolve_hardware(
    real_factory,
    hw_id: str,
    dof: int,
    *,
    has_gripper: bool,
    address_configured: bool,
    gripper_open_position: float | None = None,
    gripper_closed_position: float | None = None,
) -> HardwareComponent:
    """Real hardware if an address is configured, or if ``--simulation`` is set
    (the vendor's own factory already knows how to build its own sim/mock
    adapter in that case — e.g. xarm7's MuJoCo path). Otherwise
    :func:`_mock_hardware` — the vendor-independent hardware-free fallback,
    given uniformly regardless of whether the vendor's own factory supports an
    address-optional mock knob.
    """
    if global_config.simulation or address_configured:
        return real_factory(hw_id)
    return _mock_hardware(
        hw_id,
        dof,
        has_gripper=has_gripper,
        gripper_open_position=gripper_open_position,
        gripper_closed_position=gripper_closed_position,
    )


def _without_coordinator_task(model):
    """Clear ``coordinator_task_name`` so the ManipulationModule's planner
    doesn't also attach a trajectory task claiming the session's joints
    (would violate this adapter's strict-exclusivity contract) -- but only if
    that field genuinely exists on the installed dimos version's
    ``RobotModelConfig``. A blind ``model_copy(update=...)`` is a pydantic v2
    no-op for an unknown key, which is exactly how this went unnoticed before:
    check first, rather than silently maybe-doing-nothing.
    """
    if "coordinator_task_name" in type(model).model_fields:
        return model.model_copy(update={"coordinator_task_name": None})
    return model


def _streaming_blueprint(hardware: HardwareComponent, *, model=None, sim_path=None):
    """Compose the full session contract (ADR 0018) from an already-built
    ``HardwareComponent`` (+ optional ``RobotModelConfig`` for the Viser/
    planning preview, + optional MuJoCo ``sim_path``). Fully vendor-agnostic —
    only ``hardware``/``model``/``sim_path`` vary per kind; every kind's
    blueprint reduces to one call.
    """
    parts: list = []
    if model is not None:
        parts.append(
            planner(
                robots=[_without_coordinator_task(model)],
                visualization={"backend": "viser"},
            )
        )
    parts.append(coordinator(hardware=[hardware], tasks=[_servo_task(hardware)]))
    parts.extend(_camera_if_real())
    if sim_path is not None:
        parts.extend(mujoco_if_sim(sim_path, len(hardware.joints)))
    return autoconnect(*parts)


# ---------------------------------------------------------------------------
# UFACTORY xArm7 + gripper
# ---------------------------------------------------------------------------

_xarm7_real_hardware = partial(xarm7_hardware, gripper=True)

xarm7 = _streaming_blueprint(
    _resolve_hardware(
        _xarm7_real_hardware,
        "arm",
        7,
        has_gripper=True,
        address_configured=bool(global_config.xarm7_ip),
        # xarm7's own gripper convention: raw meters passthrough (both None).
    ),
    model=make_xarm7_model_config(name="arm", add_gripper=True),
    sim_path=XARM7_SIM_PATH,
)


# ---------------------------------------------------------------------------
# Galaxea A1Z + gripper
# ---------------------------------------------------------------------------

_a1z_real_hardware = partial(a1z_hardware, has_gripper=True)

a1z = _streaming_blueprint(
    _resolve_hardware(
        _a1z_real_hardware,
        "arm",
        6,
        has_gripper=True,
        address_configured=bool(global_config.can_port),
        # A1Z's own gripper convention: normalized [0,1] fraction (both set) --
        # matches a1z_hardware()'s own gripper_open_position/closed_position.
        gripper_open_position=0.1,
        gripper_closed_position=0.0,
    ),
    model=make_a1z_model_config(name="arm", has_gripper=True),
    # No sim_path: dimos ships no MuJoCo scene for A1Z today. Unlike before,
    # A1Z now gets a hardware-free dev path WITHOUT --simulation too (just
    # don't configure --can-port) -- see _resolve_hardware above.
)


__all__ = ["xarm7", "a1z"]
