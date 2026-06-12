"""End-to-end teleop: `interlatent-teleop-pi --driver mock` driven over gRPC.

Covers OpenTeleop, the 50 Hz control loop, the safety gate, and the ack
stream — everything except a physical motor bus.
"""
import socket
import subprocess
import sys
import time

import grpc
import numpy as np
import pytest

from interlatent_teleop.protocol import teleop_pb2 as pb
from interlatent_teleop.protocol import teleop_pb2_grpc as pbg


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def pi_server():
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "interlatent_teleop.pi",
         "--driver", "mock", "--grpc-port", str(port), "--no-home"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            out = proc.stdout.read().decode(errors="replace")
            pytest.fail(f"interlatent-teleop-pi died on startup:\n{out}")
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
            break
        except OSError:
            time.sleep(0.2)
    else:
        proc.terminate()
        pytest.fail("teleop pi server never came up")
    yield f"127.0.0.1:{port}"
    proc.terminate()
    proc.wait(timeout=10)


def test_open_stream_close(pi_server):
    channel = grpc.insecure_channel(pi_server)
    stub = pbg.TeleopServiceStub(channel)

    resp = stub.OpenTeleop(pb.OpenTeleopRequest(
        robot_id="so101", client_id="pytest", control_hz=50,
    ), timeout=5)
    assert resp.session_id
    n_joints = len(resp.joint_names)
    assert n_joints > 0
    assert len(resp.joint_min) == n_joints
    assert len(resp.joint_max) == n_joints

    home = np.array(resp.home_joints, dtype=np.float32)

    def targets():
        for i in range(25):
            yield pb.TeleopTarget(
                control_timestamp=time.monotonic_ns(),
                sequence=i,
                joint_targets=(home + 1.0).tolist(),
                confidence=1.0,
                deadman_active=True,
            )
            time.sleep(0.02)

    acks = []
    for ack in stub.Stream(targets(), timeout=15):
        acks.append(ack)
        if len(acks) >= 20:
            break

    assert len(acks) >= 20
    last = acks[-1]
    assert not last.estopped
    assert len(last.current_joints) == n_joints

    closed = stub.CloseTeleop(pb.CloseTeleopRequest(session_id=resp.session_id), timeout=5)
    assert closed is not None


def test_session_token_rejected_when_not_required(pi_server):
    """Without --session-token the server accepts any client; with a wrong
    token against a token-protected server the open must fail (covered in
    unit form by the server's token check — here we just assert the open
    path stays permissive by default)."""
    channel = grpc.insecure_channel(pi_server)
    stub = pbg.TeleopServiceStub(channel)
    resp = stub.OpenTeleop(pb.OpenTeleopRequest(robot_id="so101", client_id="x"), timeout=5)
    assert resp.session_id
