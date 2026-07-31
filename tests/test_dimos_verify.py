"""Table-driven tests for the dimos adapter's declare-then-verify connect check.

Everything fake-injected: a fake CoordinatorRPC, a fake ControlCoordinator
client, and a fake bus. Each broken axis must surface its message; several at
once must accumulate into ONE raise (fail-closed, nori pattern).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from interlatent.adapters.dimos.config import build_adapter_config
from interlatent.adapters.dimos.verify import (
    DimosVerificationError,
    verify_connect,
)

ARM_JOINTS = [f"arm/joint{i}" for i in range(1, 8)]
# The servo task must claim arm joints AND the gripper — dimos's per-tick
# hardware write stomps an unclaimed gripper back to its startup value.
ALL_JOINTS = ARM_JOINTS + ["arm/gripper"]


@dataclass
class FakeClaim:
    joints: frozenset
    priority: int = 10
    mode: SimpleNamespace = field(
        default_factory=lambda: SimpleNamespace(name="SERVO_POSITION")
    )


class FakeTask:
    def __init__(self, joints, mode="SERVO_POSITION", timeout=0.5):
        self._joints = frozenset(joints)
        self._mode = mode
        self._config = SimpleNamespace(timeout=timeout)

    def claim(self):
        return FakeClaim(self._joints, mode=SimpleNamespace(name=self._mode))


class FakeControl:
    """The ControlCoordinator RPC-proxy duck type verify.py consumes."""

    def __init__(
        self,
        joints=None,
        hardware=("arm",),
        tasks=None,
        gripper_pos=0.4,
        picklable_tasks=True,
        activate_on_probe=True,
    ):
        self.joints = list(ARM_JOINTS) if joints is None else list(joints)
        self.hardware = list(hardware)
        self.tasks = tasks if tasks is not None else {
            "servo_arm": FakeTask(ALL_JOINTS)
        }
        self.gripper_pos = gripper_pos
        self.picklable_tasks = picklable_tasks
        self.activate_on_probe = activate_on_probe
        self._probed = False
        self.stopped = False

    def list_joints(self):
        return list(self.joints)

    def list_hardware(self):
        return list(self.hardware)

    def list_tasks(self):
        return list(self.tasks)

    def get_task(self, name):
        if not self.picklable_tasks:
            raise RuntimeError("cannot pickle '_thread.lock' object")
        return self.tasks.get(name)

    def get_active_tasks(self):
        return list(self.tasks) if self._probed else []

    def get_joint_positions(self):
        return {j: 0.0 for j in self.joints}

    def get_gripper_position(self, hardware_id):
        return self.gripper_pos

    def stop_rpc_client(self):
        self.stopped = True

    def note_probe(self):
        if self.activate_on_probe:
            self._probed = True


class FakeRpc:
    def __init__(self, modules=None, fail_ping=False):
        self.modules = (
            [SimpleNamespace(class_name="ControlCoordinator", rpc_name="ControlCoordinator")]
            if modules is None
            else modules
        )
        self.fail_ping = fail_ping
        self.stopped = False

    def call(self, method, *a, **k):
        if method == "list_modules":
            return self.modules
        raise AssertionError(f"unexpected coordinator call {method}")

    def stop(self):
        self.stopped = True


class FakeBus:
    def __init__(self, stream_names=None, deliver=True):
        self.stream_names = list(ARM_JOINTS) if stream_names is None else stream_names
        self.deliver = deliver
        self.commands = []
        self.control: FakeControl | None = None  # probe coupling

    def latest_joint_state(self):
        if not self.deliver:
            return None
        return SimpleNamespace(
            msg=SimpleNamespace(
                name=list(self.stream_names),
                position=[0.0] * len(self.stream_names),
            ),
            arrival_monotonic=time.monotonic(),
            producer_ts=time.time(),
        )

    def publish_joint_command(self, names, positions):
        self.commands.append((list(names), list(positions)))
        if self.control is not None:
            self.control.note_probe()


def run_verify(control=None, rpc=None, bus=None, **cfg_extra):
    cfg = build_adapter_config(
        {"kind": "xarm7", "connect_timeout_s": "0.3", **cfg_extra}, {}
    )
    control = control if control is not None else FakeControl()
    rpc = rpc if rpc is not None else FakeRpc()
    bus = bus if bus is not None else FakeBus()
    bus.control = control
    verify_connect(
        cfg,
        bus,
        coordinator_rpc_factory=lambda timeout: rpc,
        control_client_factory=lambda name: control,
    )
    return rpc, control


def expect_problems(match_all, control=None, rpc=None, bus=None, **cfg_extra):
    with pytest.raises(DimosVerificationError) as err:
        run_verify(control=control, rpc=rpc, bus=bus, **cfg_extra)
    text = str(err.value)
    for fragment in match_all:
        assert fragment in text, f"missing {fragment!r} in:\n{text}"
    return text


# ---------------------------------------------------------------------------


def test_healthy_stack_passes_and_cleans_up():
    rpc, control = run_verify()
    assert rpc.stopped and control.stopped


def test_ping_timeout_is_hard_fail_naming_transport_mismatch():
    def factory(timeout):
        raise TimeoutError

    cfg = build_adapter_config({"kind": "xarm7", "connect_timeout_s": "0.3"}, {})
    with pytest.raises(DimosVerificationError, match="DIMOS_TRANSPORT"):
        verify_connect(
            cfg, FakeBus(), coordinator_rpc_factory=factory,
            control_client_factory=lambda n: FakeControl(),
        )


def test_no_control_coordinator_module():
    rpc = FakeRpc(modules=[SimpleNamespace(class_name="OtherModule", rpc_name="x")])
    expect_problems(["no ControlCoordinator module"], rpc=rpc)


def test_missing_declared_joints():
    control = FakeControl(joints=ARM_JOINTS[:5])
    expect_problems(["declared joint(s) not present"], control=control)


def test_stock_blueprint_no_servo_task_message():
    # The trap this whole check exists for: trajectory-only stock blueprint.
    control = FakeControl(
        tasks={"trajectory_arm": FakeTask(ARM_JOINTS, mode="POSITION")}
    )
    expect_problems(
        ["SILENTLY IGNORES", "dimos run interlatent.xarm7", "also claims"],
        control=control,
    )


def test_no_tasks_at_all():
    control = FakeControl(tasks={})
    expect_problems(["no servo task consumes joint_command"], control=control)


def test_servo_task_wrong_claim():
    control = FakeControl(tasks={"servo": FakeTask(ARM_JOINTS[:3])})
    expect_problems(["claims", "exactly"], control=control)


def test_servo_task_missing_gripper_claim_rejected():
    """Regression: a servo task claiming only the arm joints leaves the
    gripper to be stomped at tick rate — must fail verification."""
    control = FakeControl(tasks={"servo": FakeTask(ARM_JOINTS)})
    expect_problems(["claims", "exactly"], control=control)


def test_servo_timeout_zero_rejected():
    control = FakeControl(tasks={"servo": FakeTask(ALL_JOINTS, timeout=0)})
    expect_problems(["timeout=0", "hold-forever"], control=control)


def test_competing_claimant_rejected():
    control = FakeControl(
        tasks={
            "servo": FakeTask(ALL_JOINTS),
            "teleop": FakeTask(ARM_JOINTS[:2], mode="POSITION"),
        }
    )
    expect_problems(["strict exclusivity", "sole claimant"], control=control)


def test_unpicklable_tasks_fall_back_to_probe_success():
    control = FakeControl(picklable_tasks=False)  # single task, probe activates
    bus = FakeBus()
    run_verify(control=control, bus=bus)
    # The probe published the FULL declared joint set (dimos's servo task
    # rejects partial commands); gripper position came from the gripper RPC.
    (names, positions) = bus.commands[0]
    assert names == ALL_JOINTS
    assert positions == [0.0] * 7 + [0.4]


def test_probe_failure_reports_no_servo_task():
    control = FakeControl(picklable_tasks=False, activate_on_probe=False)
    expect_problems(["no servo task consumes joint_command"], control=control)


def test_probe_multi_task_cannot_prove_exclusivity():
    control = FakeControl(
        picklable_tasks=False,
        tasks={"servo": FakeTask(ALL_JOINTS), "other": FakeTask([], mode="POSITION")},
    )
    expect_problems(["cannot verify strict joint exclusivity"], control=control)


def test_stream_missing_joints_and_order():
    expect_problems(
        ["joint state stream is missing"],
        bus=FakeBus(stream_names=ARM_JOINTS[:4]),
    )
    reordered = list(reversed(ARM_JOINTS))
    expect_problems(["binds to order"], bus=FakeBus(stream_names=reordered))


def test_no_joint_state_stream():
    expect_problems(["no JointState arrived"], bus=FakeBus(deliver=False))


def test_gripper_hardware_missing_and_position_none():
    expect_problems(
        ["gripper hardware id"], control=FakeControl(hardware=("base",))
    )
    expect_problems(
        ["reports no position"], control=FakeControl(gripper_pos=None)
    )


def test_multiple_problems_accumulate_into_one_raise():
    control = FakeControl(
        joints=ARM_JOINTS[:5],
        hardware=("base",),
        tasks={},
        gripper_pos=None,
    )
    text = expect_problems(
        [
            "declared joint(s) not present",
            "no servo task consumes joint_command",
            "gripper hardware id",
            "fail closed",
        ],
        control=control,
    )
    assert text.count("\n  - ") >= 3
