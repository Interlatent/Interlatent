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

import inspect
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
except ImportError as exc:  # pragma: no cover - exercised only when half-installed
    raise ImportError(
        "interlatent's dimos session blueprints require the dimos stack: "
        "pip install 'interlatent[dimos]' (python 3.11-3.12). "
        f"Underlying import failure: {exc}"
    ) from exc

# Only the SHARED dimos surface is imported above. Each kind's vendor imports
# live in its own builder below, reached through this module's ``__getattr__``,
# so one kind's incompatibility cannot take another down. That is not
# hypothetical: A1Z support was authored against a dimos branch carrying a
# Galaxea driver, and on a stock release its `a1z_hardware` import raised at
# module scope — which also killed `dimos run interlatent.xarm7`, a kind that
# had nothing to do with it. The only visible symptom was the tier-2 xarm7
# integration test failing.


def _component_accepts(*names: str) -> bool:
    """Whether ``HardwareComponent`` takes all of ``names`` as constructor args.

    dimos's ``HardwareComponent`` is not one fixed shape across the versions
    this SDK must run against: the Galaxea branch's carries gripper
    open/closed range fields that stock 0.0.14b1 has none of, and passing them
    blindly is a ``TypeError`` on every hardware-free start. ``signature`` works
    for a dataclass and a pydantic model alike, so this does not assume which.
    """
    try:
        params = inspect.signature(HardwareComponent).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic/builtin ctor
        return False
    return all(n in params for n in names)

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

    The gripper range is passed only when the installed ``HardwareComponent``
    declares those fields (see :func:`_component_accepts`); on a release that
    lacks them the mock simply carries no normalization, and the SDK-side range
    in ``robots/<kind>.toml`` — which the adapter clamps against, and which is
    the only limit in the whole path — is unaffected either way.
    """
    kwargs: dict = {}
    if gripper_open_position is not None and _component_accepts(
        "gripper_open_position", "gripper_closed_position"
    ):
        kwargs["gripper_open_position"] = gripper_open_position
        kwargs["gripper_closed_position"] = gripper_closed_position
    return HardwareComponent(
        hardware_id=hw_id,
        hardware_type=HardwareType.MANIPULATOR,
        joints=make_joints(hw_id, dof),
        adapter_type="mock",
        gripper_joints=make_gripper_joints(hw_id) if has_gripper else [],
        **kwargs,
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

    ``real_factory`` must be a vendor *policy* wrapper — one that reads
    ``global_config`` and picks its own adapter_type/address, like
    ``xarm7_hardware`` — not a raw component builder, which would return a mock
    no matter what this branch decided. Not every dimos build ships one per
    vendor, so callers feature-detect before routing through here; see the A1Z
    block below.
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


def _build_xarm7():
    from dimos.robot.manipulators.xarm.config import (
        XARM7_SIM_PATH,
        make_xarm7_model_config,
        xarm7_hardware,
    )

    return _streaming_blueprint(
        _resolve_hardware(
            partial(xarm7_hardware, gripper=True),
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

#: A1Z's own gripper convention: a normalized [0, 1] fraction, the opposite of
#: xarm7's raw-meters passthrough. Applied only where the installed
#: HardwareComponent declares the fields (see _mock_hardware).
_A1Z_GRIPPER_OPEN = 1.0
_A1Z_GRIPPER_CLOSED = 0.0


def _build_a1z():
    """A1Z, across two incompatible dimos lineages.

    dimos ships A1Z differently depending on which build you are on, and this
    SDK has to run on both:

    - **A Galaxea-enabled branch** exports ``a1z_hardware``, a vendor *policy*
      wrapper in the mould of ``xarm7_hardware``: it reads ``global_config``
      and binds the real CAN adapter. That is the build A1Z was originally
      developed and hardware-verified against, and `--can-port` is meaningful
      there, so we route it through :func:`_resolve_hardware` exactly as before.
    - **Every published release** (0.0.14b1 and earlier) ships A1Z as a
      *planning model* only — ``a1z/config.py`` is URDF paths, joint names and
      collision pairs, and its lone hardware helper ``make_a1z_hardware`` is a
      raw builder defaulting to ``adapter_type="mock", address=None``. There is
      no Galaxea entry in ``dimos.hardware.manipulators.registry`` to bind to,
      so no flag reaches real motors and ``--can-port`` (piper's knob) is inert.
      dimos's own A1Z blueprints call the builder bare for the same reason.

    Feature-detecting the wrapper rather than pinning either lineage is what
    lets one source tree serve both; hardcoding the release shape would have
    silently downgraded a real arm to a mock, and hardcoding the branch shape
    is what made this module unimportable on a stock install.

    No ``sim_path`` in either case: dimos ships no MuJoCo scene for A1Z.
    """
    from dimos.robot.manipulators.a1z import config as a1z_config

    model = a1z_config.make_a1z_model_config(name="arm", has_gripper=True)
    real_factory = getattr(a1z_config, "a1z_hardware", None)

    if real_factory is not None:
        hardware = _resolve_hardware(
            partial(real_factory, has_gripper=True),
            "arm",
            6,
            has_gripper=True,
            address_configured=bool(global_config.can_port),
            gripper_open_position=_A1Z_GRIPPER_OPEN,
            gripper_closed_position=_A1Z_GRIPPER_CLOSED,
        )
    else:
        # Mock-only lineage. Prefer dimos's own builder over _mock_hardware so
        # we inherit whatever it learns later (an adapter_type, an address).
        hardware = a1z_config.make_a1z_hardware("arm", has_gripper=True)

    return _streaming_blueprint(hardware, model=model)


# ---------------------------------------------------------------------------
# Entry points, resolved per kind
# ---------------------------------------------------------------------------

_BUILDERS = {"xarm7": _build_xarm7, "a1z": _build_a1z}
_CACHE: dict = {}


def __getattr__(name: str):
    """Build a kind's blueprint on first access (PEP 562).

    dimos resolves ``blueprints:xarm7`` / ``blueprints:a1z`` with ``getattr``,
    so this is the isolation seam: one kind's vendor imports run only when that
    kind is requested, and a kind that cannot build on the installed dimos
    raises an error naming *itself* instead of taking the whole module — and
    every other kind's entry point — down with it at import time.
    """
    builder = _BUILDERS.get(name)
    if builder is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name not in _CACHE:
        try:
            _CACHE[name] = builder()
        except Exception as exc:
            raise ImportError(
                f"interlatent's dimos blueprint for kind {name!r} could not be "
                f"built against the installed dimos: {exc}. Other kinds are "
                "unaffected — this is a per-kind failure. Check that your dimos "
                f"build supports {name!r} (for a1z, real hardware needs a "
                "Galaxea-enabled dimos; stock releases are mock-only)."
            ) from exc
    return _CACHE[name]


def __dir__() -> list:
    return sorted([*globals(), *_BUILDERS])


__all__ = ["xarm7", "a1z"]
