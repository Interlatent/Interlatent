"""gRPC client wrapper: OpenTeleop + bidi Stream + CloseTeleop."""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Iterator, Optional

import grpc
import numpy as np

from ..protocol import teleop_pb2, teleop_pb2_grpc

_LOG = logging.getLogger("interlatent_teleop.laptop.client")


@dataclass
class SessionInfo:
    session_id: str
    control_hz: int
    joint_names: tuple[str, ...]
    joint_min: np.ndarray
    joint_max: np.ndarray
    home_joints: np.ndarray
    max_velocity: np.ndarray


class TeleopClient:
    def __init__(
        self,
        address: str,
        client_id: str = "laptop",
        robot_id: str = "so101",
        session_token: str = "",
        control_hz: int = 50,
    ) -> None:
        self.address = address
        self.client_id = client_id
        self.robot_id = robot_id
        self.session_token = session_token
        self.control_hz = control_hz
        self._channel: Optional[grpc.Channel] = None
        self._stub: Optional[teleop_pb2_grpc.TeleopServiceStub] = None
        self._session: Optional[SessionInfo] = None
        self._send_q: "queue.Queue[teleop_pb2.TeleopTarget]" = queue.Queue(maxsize=8)
        self._stop = threading.Event()
        self._stream_thread: Optional[threading.Thread] = None
        self._latest_ack: Optional[teleop_pb2.TeleopAck] = None
        self._ack_lock = threading.Lock()
        self._seq = 0

    @property
    def session(self) -> SessionInfo:
        if self._session is None:
            raise RuntimeError("not open")
        return self._session

    @property
    def latest_ack(self) -> Optional[teleop_pb2.TeleopAck]:
        with self._ack_lock:
            return self._latest_ack

    def open(self, connect_timeout_s: float = 5.0) -> SessionInfo:
        self._channel = grpc.insecure_channel(self.address)
        self._stub = teleop_pb2_grpc.TeleopServiceStub(self._channel)
        try:
            grpc.channel_ready_future(self._channel).result(timeout=connect_timeout_s)
        except grpc.FutureTimeoutError as e:
            raise ConnectionError(
                f"could not reach teleop server at {self.address} "
                f"within {connect_timeout_s:.1f}s. Is `interlatent-teleop-pi` "
                f"running and is the Pi reachable (try `tailscale ping`)?"
            ) from e
        resp = self._stub.OpenTeleop(teleop_pb2.OpenTeleopRequest(
            robot_id=self.robot_id,
            client_id=self.client_id,
            control_hz=self.control_hz,
            session_token=self.session_token,
        ), timeout=connect_timeout_s)
        info = SessionInfo(
            session_id=resp.session_id,
            control_hz=resp.control_hz,
            joint_names=tuple(resp.joint_names),
            joint_min=np.array(resp.joint_min, dtype=np.float32),
            joint_max=np.array(resp.joint_max, dtype=np.float32),
            home_joints=np.array(resp.home_joints, dtype=np.float32),
            max_velocity=np.array(resp.max_velocity, dtype=np.float32),
        )
        self._session = info
        _LOG.info("OpenTeleop ok: session=%s control_hz=%d joints=%s",
                  info.session_id, info.control_hz, info.joint_names)

        # Start bidi stream in a background thread; the main thread
        # only enqueues targets.
        self._stream_thread = threading.Thread(
            target=self._run_stream, name="teleop-stream", daemon=True,
        )
        self._stream_thread.start()
        return info

    def close(self) -> None:
        if self._session is None:
            return
        self._stop.set()
        try:
            if self._stub is not None:
                self._stub.CloseTeleop(teleop_pb2.CloseTeleopRequest(
                    session_id=self._session.session_id,
                ), timeout=2.0)
        except grpc.RpcError as e:
            _LOG.warning("CloseTeleop raised: %s", e)
        if self._stream_thread is not None:
            self._stream_thread.join(timeout=2.0)
        if self._channel is not None:
            self._channel.close()
        self._session = None

    def send_target(self, joints: np.ndarray, *, deadman: bool, confidence: float = 1.0,
                    ts_ns: Optional[int] = None) -> None:
        if self._session is None:
            raise RuntimeError("not open")
        if ts_ns is None:
            ts_ns = time.monotonic_ns()
        self._seq += 1
        msg = teleop_pb2.TeleopTarget(
            control_timestamp=ts_ns,
            sequence=self._seq,
            joint_targets=list(map(float, joints.tolist())),
            confidence=float(confidence),
            deadman_active=bool(deadman),
        )
        # Drop oldest if the send queue is full — fresher targets matter
        # more than complete history for streaming control.
        try:
            self._send_q.put_nowait(msg)
        except queue.Full:
            try:
                self._send_q.get_nowait()
            except queue.Empty:
                pass
            self._send_q.put_nowait(msg)

    # ------------------------------------------------------------------

    def _outgoing(self) -> Iterator[teleop_pb2.TeleopTarget]:
        while not self._stop.is_set():
            try:
                msg = self._send_q.get(timeout=0.1)
            except queue.Empty:
                continue
            yield msg

    def _run_stream(self) -> None:
        if self._stub is None:
            return
        try:
            for ack in self._stub.Stream(self._outgoing()):
                with self._ack_lock:
                    self._latest_ack = ack
                if self._stop.is_set():
                    break
        except grpc.RpcError as e:
            if not self._stop.is_set():
                _LOG.warning("stream ended: %s", e)
