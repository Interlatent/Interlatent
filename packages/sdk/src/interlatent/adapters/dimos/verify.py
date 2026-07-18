"""Declare-then-verify: prove the declared kind matches the running dimos stack.

The operator declares an embodiment (``--robot-arg kind=xarm7``); nothing on the
dimos bus states what robot is running (identity is authoring-time config), so
``connect()`` verifies the declaration against live evidence and FAIL-CLOSES,
accumulating every mismatch into one :class:`DimosVerificationError` (the nori
handshake-validation pattern). Checks, in order:

1. **Liveness** — ``Coordinator/ping`` (via ``CoordinatorRPC.connect``). The one
   non-accumulated hard fail; its message names the #1 field failure mode, a
   transport-backend (lcm/zenoh) mismatch between the two processes.
2. **ControlCoordinator present** — ``Coordinator/list_modules``; captures the
   deployed ``rpc_name`` so instance-named deploys still resolve.
3. **Joints** — declared joints ⊆ ``list_joints()``.
4. **Servo task** — a servo-type task must claim EXACTLY the declared joints
   — arm joints AND gripper, because dimos's per-tick hardware write re-sends
   its last-commanded gripper value, so a gripper left unclaimed is stomped
   back to its startup value the moment streaming starts — with ``timeout !=
   0``, and no other configured task may claim any of them (strict
   exclusivity, ADR 0018). This is the trap check: a stock dimos coordinator
   blueprint has no servo task and SILENTLY IGNORES joint_command.
   Task introspection uses ``get_task`` when its result survives the pickled
   RPC (servo tasks hold locks and may not); otherwise a no-op probe publishes
   the CURRENT measured positions as a ``joint_command`` and watches
   ``get_active_tasks`` — an end-to-end proof that a servo task consumes the
   stream. In probe mode exact claims are unknowable, so exclusivity can only
   be proven for single-task stacks; anything else fails with guidance.
5. **First JointState** — declared joints present in the stream, order
   preserved (a policy binds to order).
6. **Gripper** — declared gripper hardware id exists and reports a position.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional

from .bus import DimosBus
from .config import DimosAdapterConfig
from .kinds import DimosKind

_logger = logging.getLogger(__name__)

_PROBE_WINDOW_S = 1.5
_SERVO_MODE_HINTS = ("SERVO",)  # ControlMode.SERVO_POSITION


class DimosVerificationError(RuntimeError):
    """The declared kind does not match the running dimos stack (fail closed)."""


def _default_coordinator_rpc_factory(timeout: float) -> Any:
    from dimos.core.coordination.coordinator_rpc import CoordinatorRPC

    return CoordinatorRPC.connect(timeout=timeout)


def _default_control_client_factory(rpc_name: str) -> Any:
    from dimos.control.coordinator import ControlCoordinator
    from dimos.core.rpc_client import RPCClient

    return RPCClient(None, ControlCoordinator, remote_name=rpc_name)


def verify_connect(
    cfg: DimosAdapterConfig,
    bus: DimosBus,
    *,
    coordinator_rpc_factory: Callable[[float], Any] | None = None,
    control_client_factory: Callable[[str], Any] | None = None,
) -> None:
    """Run every check; raise :class:`DimosVerificationError` listing ALL
    problems, or return None on a clean pass."""
    kind = cfg.kind
    rpc_factory = coordinator_rpc_factory or _default_coordinator_rpc_factory
    client_factory = control_client_factory or _default_control_client_factory

    # 1. Liveness (hard fail — nothing else is checkable without it).
    try:
        rpc = rpc_factory(cfg.connect_timeout_s)
    except TimeoutError as exc:
        raise DimosVerificationError(
            f"no running dimos stack answered Coordinator/ping within "
            f"{cfg.connect_timeout_s:.0f}s. Is `dimos run ...` up on this host, "
            "and do BOTH processes use the same transport backend "
            "(DIMOS_TRANSPORT=lcm|zenoh — a mismatch means they silently "
            "cannot see each other)?"
        ) from exc

    problems: list[str] = []
    control: Any = None
    try:
        # 2. ControlCoordinator module present.
        rpc_name = None
        try:
            modules = rpc.call("list_modules") or []
        except Exception as exc:  # noqa: BLE001
            modules = []
            problems.append(f"Coordinator/list_modules failed: {exc!r}")
        for m in modules:
            if getattr(m, "class_name", None) == "ControlCoordinator":
                rpc_name = getattr(m, "rpc_name", None) or "ControlCoordinator"
                break
        if rpc_name is None:
            problems.append(
                "no ControlCoordinator module is deployed in the running "
                f"blueprint (modules: {[getattr(m, 'class_name', m) for m in modules]}). "
                "The dimos adapter drives robots through the coordinator seam — "
                "run a coordinator blueprint (e.g. `dimos run interlatent."
                f"{kind.name}`)."
            )
        else:
            control = client_factory(rpc_name)
            problems += _verify_joints(control, kind)
            problems += _verify_servo_task(control, bus, kind)
            problems += _verify_gripper(control, kind)

        # 5. First JointState on the stream (independent of RPC health).
        problems += _verify_joint_state_stream(bus, cfg, kind)
    finally:
        for closer in ("stop",):
            try:
                getattr(rpc, closer)()
            except Exception:  # noqa: BLE001
                pass
        if control is not None:
            try:
                control.stop_rpc_client()
            except Exception:  # noqa: BLE001
                pass

    if problems:
        bullet = "\n  - ".join(problems)
        raise DimosVerificationError(
            f"declared kind={kind.name!r} does not match the running dimos "
            f"stack (fail closed; {len(problems)} problem(s)):\n  - {bullet}"
        )


# ---------------------------------------------------------------------------
# individual checks (each returns a list of problem strings)
# ---------------------------------------------------------------------------


def _verify_joints(control: Any, kind: DimosKind) -> list[str]:
    try:
        live = list(control.list_joints() or [])
    except Exception as exc:  # noqa: BLE001
        return [f"list_joints RPC failed: {exc!r}"]
    missing = [j for j in kind.dimos_arm_joints if j not in live]
    if missing:
        return [
            f"declared joint(s) not present in the running stack: {missing} "
            f"(live joints: {live})"
        ]
    return []


def _verify_servo_task(control: Any, bus: DimosBus, kind: DimosKind) -> list[str]:
    # The full joint set INCLUDING the gripper — see module docstring check 4.
    declared = set(kind.dimos_joint_names)
    try:
        task_names = list(control.list_tasks() or [])
    except Exception as exc:  # noqa: BLE001
        return [f"list_tasks RPC failed: {exc!r}"]

    if not task_names:
        return [_no_servo_task_message(kind)]

    # Preferred path: inspect real task objects (works when they survive the
    # pickled RPC). ANY failure drops us to the end-to-end probe.
    try:
        tasks = {name: control.get_task(name) for name in task_names}
        if any(t is None for t in tasks.values()):
            raise RuntimeError("get_task returned None for a listed task")
        return _check_task_claims(tasks, declared, kind)
    except Exception as exc:  # noqa: BLE001
        _logger.info(
            "task introspection over RPC unavailable (%r); falling back to the "
            "joint_command probe", exc,
        )
        return _probe_servo_task(control, bus, task_names, kind)


def _check_task_claims(
    tasks: dict[str, Any], declared: set, kind: DimosKind
) -> list[str]:
    problems: list[str] = []
    servo: Optional[str] = None
    for name, task in tasks.items():
        claim = task.claim()
        joints = set(claim.joints)
        mode = getattr(getattr(claim, "mode", None), "name", "")
        is_servo = any(h in str(mode).upper() for h in _SERVO_MODE_HINTS)
        if is_servo:
            if joints == declared:
                timeout = _task_timeout(task)
                if timeout == 0:
                    problems.append(
                        f"servo task {name!r} has timeout=0 (hold-forever): a "
                        "stalled session would leave the arm chasing its last "
                        "setpoint indefinitely. Configure a non-zero timeout."
                    )
                servo = name
            else:
                problems.append(
                    f"servo task {name!r} claims {sorted(joints)} but the "
                    f"declared kind needs exactly {sorted(declared)}"
                )
        elif joints & declared:
            problems.append(
                f"task {name!r} also claims joint(s) {sorted(joints & declared)} — "
                "strict exclusivity (ADR 0018): the session's servo task must be "
                "the sole claimant. Remove the competing task from the blueprint "
                "or run the reference blueprint."
            )
    if servo is None and not problems:
        problems.append(_no_servo_task_message(kind))
    elif servo is None:
        problems.insert(0, _no_servo_task_message(kind))
    return problems


def _task_timeout(task: Any) -> float | None:
    cfg = getattr(task, "_config", None)
    return getattr(cfg, "timeout", None) if cfg is not None else None


def _probe_servo_task(
    control: Any, bus: DimosBus, task_names: list[str], kind: DimosKind
) -> list[str]:
    """End-to-end proof: publish the CURRENT measured positions as a
    joint_command (a no-op motion-wise) and watch a task go active.

    The probe must carry the FULL declared joint set: dimos's servo task
    rejects any command missing even one claimed joint (set_target_by_name
    returns False without updating). The gripper position is read from
    get_joint_positions when the stack folds it in, else the gripper RPC.
    """
    problems: list[str] = []
    try:
        positions = dict(control.get_joint_positions() or {})
        probe_names = list(kind.dimos_joint_names)
        if (
            kind.dimos_gripper_joint
            and kind.dimos_gripper_joint not in positions
            and kind.gripper_hardware_id
        ):
            g = control.get_gripper_position(kind.gripper_hardware_id)
            if g is not None:
                positions[kind.dimos_gripper_joint] = float(g)
        missing = [j for j in probe_names if j not in positions]
        if missing:
            return [
                "probe aborted: could not read current position(s) for "
                f"declared joint(s) {missing}"
            ]
        before = set(control.get_active_tasks() or [])
        bus.publish_joint_command(probe_names, [positions[j] for j in probe_names])
        deadline = time.monotonic() + _PROBE_WINDOW_S
        activated: set = set()
        while time.monotonic() < deadline:
            activated = set(control.get_active_tasks() or []) - before
            if activated:
                break
            time.sleep(0.05)
        if not activated:
            problems.append(_no_servo_task_message(kind))
    except Exception as exc:  # noqa: BLE001
        problems.append(f"servo-task probe failed: {exc!r}")
        return problems

    if len(task_names) > 1:
        problems.append(
            "cannot verify strict joint exclusivity: task introspection is "
            f"unavailable over RPC and {len(task_names)} tasks are configured "
            f"({task_names}). Use a single-task session blueprint (e.g. `dimos "
            f"run interlatent.{kind.name}`) or a dimos version exposing task info."
        )
    return problems


def _no_servo_task_message(kind: DimosKind) -> str:
    return (
        "no servo task consumes joint_command for the declared joints — dimos "
        "SILENTLY IGNORES streamed joint commands without one (stock coordinator "
        "blueprints configure only a trajectory task). Run the reference session "
        f"blueprint (`dimos run interlatent.{kind.name}`) or add a TaskConfig("
        "type=\"servo\", joint_names=<the kind's joints>, params={\"timeout\": 0.5}) "
        "to yours."
    )


def _verify_joint_state_stream(
    bus: DimosBus, cfg: DimosAdapterConfig, kind: DimosKind
) -> list[str]:
    deadline = time.monotonic() + cfg.connect_timeout_s
    cached = bus.latest_joint_state()
    while cached is None and time.monotonic() < deadline:
        time.sleep(0.05)
        cached = bus.latest_joint_state()
    if cached is None:
        return [
            f"no JointState arrived on {cfg.joint_state_topic!r} within "
            f"{cfg.connect_timeout_s:.0f}s (is the coordinator's "
            "publish_joint_state enabled?)"
        ]
    live = list(cached.msg.name)
    missing = [j for j in kind.dimos_arm_joints if j not in live]
    if missing:
        return [
            f"joint state stream is missing declared joint(s): {missing} "
            f"(stream: {live})"
        ]
    indices = [live.index(j) for j in kind.dimos_arm_joints]
    if indices != sorted(indices):
        return [
            "joint state stream orders the declared joints as "
            f"{[live[i] for i in sorted(indices)]} but the kind declares "
            f"{list(kind.dimos_arm_joints)} — a policy binds to order; the "
            "profile/kind order must equal the stream's."
        ]
    return []


def _verify_gripper(control: Any, kind: DimosKind) -> list[str]:
    if kind.gripper_hardware_id is None:
        return []
    try:
        hardware = list(control.list_hardware() or [])
    except Exception as exc:  # noqa: BLE001
        return [f"list_hardware RPC failed: {exc!r}"]
    if kind.gripper_hardware_id not in hardware:
        return [
            f"gripper hardware id {kind.gripper_hardware_id!r} not registered "
            f"(hardware: {hardware})"
        ]
    try:
        pos = control.get_gripper_position(kind.gripper_hardware_id)
    except Exception as exc:  # noqa: BLE001
        return [f"get_gripper_position RPC failed: {exc!r}"]
    if pos is None:
        return [
            f"gripper on {kind.gripper_hardware_id!r} reports no position — is "
            "the gripper enabled/connected?"
        ]
    return []


__all__ = ["verify_connect", "DimosVerificationError"]
