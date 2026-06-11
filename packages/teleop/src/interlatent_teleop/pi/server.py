"""gRPC server: accepts OpenTeleop + bidirectional Stream + CloseTeleop."""
from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent import futures
from typing import Iterator, Optional

import grpc
import numpy as np

from ..common.config import RobotProfile, SO101_PROFILE
from ..protocol import teleop_pb2, teleop_pb2_grpc
from .control_loop import ControlLoop
from .safety import SafetyConfig, SafetyGate, TargetSample
from .so101_driver import RobotDriver

_LOG = logging.getLogger("interlatent_teleop.pi.server")


class TeleopServicer(teleop_pb2_grpc.TeleopServiceServicer):
    """Single-session servicer.

    For MVP we hold at most one active session — a real arm can only
    be teleoperated by one producer at a time anyway. New OpenTeleop
    calls while a session is active are rejected with FAILED_PRECONDITION.
    """

    def __init__(
        self,
        driver: RobotDriver,
        profile: RobotProfile = SO101_PROFILE,
        control_hz: int = 50,
        session_token: Optional[str] = None,
    ) -> None:
        self.driver = driver
        self.profile = profile
        self.control_hz = control_hz
        self._expected_token = session_token
        self._lock = threading.Lock()
        self._session_id: Optional[str] = None
        self._gate: Optional[SafetyGate] = None
        self._loop: Optional[ControlLoop] = None

    # ------------------------------------------------------------------
    # RPCs

    def OpenTeleop(self, request, context):  # type: ignore[override]
        if self._expected_token and request.session_token != self._expected_token:
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "invalid session_token")

        with self._lock:
            if self._session_id is not None:
                context.abort(
                    grpc.StatusCode.FAILED_PRECONDITION,
                    f"another session is active: {self._session_id}",
                )
            control_hz = int(request.control_hz) if request.control_hz > 0 else self.control_hz
            control_hz = max(10, min(control_hz, 200))
            gate = SafetyGate(
                profile=self.profile,
                control_dt=1.0 / control_hz,
                config=SafetyConfig(),
            )
            loop = ControlLoop(driver=self.driver, gate=gate, control_hz=control_hz)
            loop.start()
            self._session_id = uuid.uuid4().hex
            self._gate = gate
            self._loop = loop
            current = loop.state.current_joints

        _LOG.info(
            "OpenTeleop client=%s robot=%s control_hz=%d session=%s",
            request.client_id, request.robot_id, control_hz, self._session_id,
        )

        return teleop_pb2.OpenTeleopResponse(
            session_id=self._session_id,
            control_hz=control_hz,
            joint_names=list(self.profile.joint_names),
            joint_min=[lim[0] for lim in self.profile.joint_limits],
            joint_max=[lim[1] for lim in self.profile.joint_limits],
            home_joints=list(map(float, current.tolist())),
            max_velocity=list(self.profile.max_velocity),
        )

    def Stream(self, request_iterator, context):  # type: ignore[override]
        gate = self._gate
        loop = self._loop
        if gate is None or loop is None:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "no session; call OpenTeleop first")

        stop = threading.Event()
        n_joints = len(self.profile.joint_names)

        def reader() -> None:
            try:
                for msg in request_iterator:
                    if len(msg.joint_targets) != n_joints:
                        _LOG.warning("dropping target with %d joints (expected %d)",
                                     len(msg.joint_targets), n_joints)
                        continue
                    sample = TargetSample(
                        joints=np.array(msg.joint_targets, dtype=np.float32),
                        deadman_active=bool(msg.deadman_active),
                        confidence=float(msg.confidence),
                        received_at=time.monotonic(),
                        producer_timestamp_ns=int(msg.control_timestamp),
                    )
                    gate.submit(sample)
            except grpc.RpcError:
                pass
            finally:
                stop.set()

        threading.Thread(target=reader, name="teleop-reader", daemon=True).start()

        # Emit acks at control rate. Cheaper than tying acks to the
        # control loop directly via a queue; the producer doesn't need
        # tick-exact alignment.
        interval = 1.0 / loop.control_hz
        next_tick = time.monotonic()
        try:
            while not stop.is_set() and context.is_active():
                state = loop.state
                yield teleop_pb2.TeleopAck(
                    control_timestamp=state.last_commanded_ts_ns,
                    server_timestamp_ns=time.monotonic_ns(),
                    current_joints=list(map(float, state.current_joints.tolist())),
                    estopped=state.estopped,
                    status_message=state.last_status,
                )
                next_tick += interval
                sleep = next_tick - time.monotonic()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    next_tick = time.monotonic()
        finally:
            stop.set()

    def CloseTeleop(self, request, context):  # type: ignore[override]
        with self._lock:
            if self._session_id != request.session_id:
                context.abort(grpc.StatusCode.NOT_FOUND, "session not found")
            self._teardown_locked()
        return teleop_pb2.CloseTeleopResponse()

    # ------------------------------------------------------------------

    def _teardown_locked(self) -> None:
        if self._loop is not None:
            self._loop.stop()
        self._loop = None
        self._gate = None
        self._session_id = None
        _LOG.info("session closed")


def serve(
    driver: RobotDriver,
    host: str = "0.0.0.0",
    port: int = 50061,
    control_hz: int = 50,
    session_token: Optional[str] = None,
    connect_driver: bool = True,
) -> grpc.Server:
    """Build, start, and return the gRPC server.

    If `connect_driver` is True (default) the driver is connected here;
    pass False if the caller has already connected it (e.g. to ramp the
    arm to a home pose before opening the gRPC port).

    Caller is responsible for `server.wait_for_termination()` /
    `server.stop(grace)`.
    """
    if connect_driver:
        driver.connect()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    servicer = TeleopServicer(
        driver=driver,
        control_hz=control_hz,
        session_token=session_token,
    )
    teleop_pb2_grpc.add_TeleopServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    _LOG.info("teleop server listening on %s:%d", host, port)
    return server
