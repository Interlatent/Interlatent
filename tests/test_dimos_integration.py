"""Live-stack conformance test for the dimos adapter (ADR 0018).

The unit suites fake every wire; ADR 0018 deliberately vendors no fixtures
("the protocol is dimos's Python API") and names THIS as the conformance
check: the real adapter against `dimos run interlatent.<kind>` — mock
hardware, real bus, real coordinator, real servo task.

Parameterized over every kind that ships a blueprint entry point — read from
the entry points themselves, so a newly added kind is covered here the moment
it is declared and can never be silently left out of the one test that talks
to a real stack. One stack at a time (module-scoped, torn down between params:
the kinds share the `arm` hardware id and the default topic names, so they
must never overlap on the bus).

Every kind reaches this the same way regardless of vendor lineage — with no
vendor address configured, each blueprint resolves to a hardware-free mock
(A1Z included, on both its released mock-only and Galaxea-enabled shapes).

Runs only where the [dimos] extra is importable (python 3.11-3.12); a base
install skips. It spawns its own stack and assumes an otherwise-quiet bus —
kill any `dimos run` you have open before running it. Both processes inherit
this environment, so they agree on DIMOS_TRANSPORT by construction. If the
bus itself is unreachable (no multicast route — macOS needs 224.0.0.0/4 on
lo0, and VPNs love to steal it; see adapters/dimos/CONFIG.md failure modes)
the test SKIPS with the fix spelled out: an unreachable bus is an
environment problem, not an adapter regression. Contract violations fail.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("dimos")

from interlatent.adapters.dimos.config import build_adapter_config
from interlatent.adapters.dimos.kinds import get_kind
from interlatent.adapters.dimos.robot import DimosNativeRobot
from interlatent.adapters.dimos.verify import DimosVerificationError

_STARTUP_DEADLINE_S = 90.0  # cold start is ~30 s; leave slack for CI boxes
_PING_UNREACHABLE = "no running dimos stack answered Coordinator/ping"
# zenoh's dimos session is process-global: the first kind's disconnect() closes
# it, and every later connect() in the same process raises `ZError: session
# closed` from declare_subscriber. That is a transport property, not an adapter
# regression — LCM has no such constraint and covers all kinds in one process —
# so later params skip rather than fail. Matched on the message because zenoh
# need not be importable on an LCM-only box.
_SESSION_CLOSED = "session closed"


def _blueprint_kinds() -> list[str]:
    """Kinds this SDK ships a `dimos run interlatent.<kind>` blueprint for."""
    from importlib.metadata import entry_points

    return sorted(ep.name for ep in entry_points(group="dimos.blueprints"))


@pytest.fixture(scope="module", params=_blueprint_kinds())
def mock_stack(request, tmp_path_factory):
    """`dimos run interlatent.<kind>` as a child process group, torn down hard."""
    kind = request.param
    dimos_cli = Path(sys.executable).parent / "dimos"
    if not dimos_cli.exists():
        pytest.skip(f"dimos CLI not found next to {sys.executable}")

    log_path = tmp_path_factory.mktemp(f"dimos-{kind}") / "stack.log"
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(
            [str(dimos_cli), "run", f"interlatent.{kind}"],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # dimos deploys worker subprocesses; kill the group
        )
    try:
        yield kind, proc, log_path
    finally:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                break
            try:
                proc.wait(timeout=10)
                break
            except subprocess.TimeoutExpired:
                continue


def _connect_when_ready(stack) -> DimosNativeRobot:
    """Retry connect() through stack startup; skip only if the bus never answers."""
    kind, proc, log_path = stack
    cfg = build_adapter_config({"kind": kind, "connect_timeout_s": "5"}, None)
    deadline = time.monotonic() + _STARTUP_DEADLINE_S
    last_error: DimosVerificationError | None = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            pytest.fail(
                f"dimos stack exited rc={proc.returncode} during startup — "
                f"see {log_path}"
            )
        robot = DimosNativeRobot(cfg)
        try:
            robot.connect()
            return robot
        except DimosVerificationError as exc:
            if _PING_UNREACHABLE not in str(exc):
                raise  # a real contract violation — that IS the test failing
            last_error = exc  # not up yet (or bus dead); keep trying
        except Exception as exc:  # noqa: BLE001 - narrowed by message below
            if _SESSION_CLOSED not in str(exc).lower():
                raise
            pytest.skip(
                f"transport session is process-global and was closed by an "
                f"earlier kind's disconnect(), so {kind!r} cannot connect in "
                "this process (zenoh only). Re-run with DIMOS_TRANSPORT=lcm to "
                f"cover every kind in one pass. Underlying error: {exc}"
            )
    pytest.skip(
        "dimos stack never became reachable over the bus: likely a multicast "
        "problem, not an adapter bug (macOS: route 224.0.0.0/4 must point at "
        "lo0 — VPNs steal it; both sides must share DIMOS_TRANSPORT). "
        f"Last error: {last_error}"
    )


def test_connect_verify_move_readback(mock_stack):
    """The whole tier-2 path: fail-closed verify, clamped move, observation."""
    kind = mock_stack[0]
    robot = _connect_when_ready(mock_stack)
    try:
        # Declared embodiment survived live verification; order is the
        # contract. Sourced from the kind's own TOML declaration, so a stack
        # whose real DOF disagrees with it fails here rather than being
        # rubber-stamped by an assertion derived from the same live stack.
        assert robot.action_features == list(get_kind(kind).feature_keys)

        before = robot.get_observation()
        assert set(robot.action_features) <= set(before)

        # One clamp-ramped arm move + one full-range gripper step, held-missing —
        # the same seam interlatent-act and the control loop use.
        robot.action(
            arm_joint2=0.25, arm_gripper=0.4, hold_missing=True, timeout=30.0
        )
        after = robot.get_observation()
        assert after["arm_joint2.pos"] == pytest.approx(0.25, abs=0.06)
        assert after["arm_gripper.pos"] == pytest.approx(0.4, abs=0.06)
        # Held joints stayed put.
        assert after["arm_joint5.pos"] == pytest.approx(
            before["arm_joint5.pos"], abs=0.06
        )
    finally:
        robot.disconnect()
