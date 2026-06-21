"""End-to-end: real `interlatent-serve` subprocess driven by the DRTC client.

No GPU, no robot — echo and tiny_torch backends. This is the same loop a
real robot runs, so it covers session open, the sender/receiver threads,
chunk merging, and clean shutdown.
"""
import io
import socket
import subprocess
import sys
import time

import numpy as np
import pytest

from interlatent.inference.integration import connect_drtc

ACTION_DIM = 4


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server():
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "interlatent_server.server.app",
         "--host", "127.0.0.1", "--port", str(port)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            out = proc.stdout.read().decode(errors="replace")
            pytest.fail(f"interlatent-serve died on startup:\n{out}")
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
            break
        except OSError:
            time.sleep(0.2)
    else:
        proc.terminate()
        pytest.fail("interlatent-serve never came up")
    yield f"127.0.0.1:{port}"
    proc.terminate()
    proc.wait(timeout=10)


def _obs(rng: np.random.Generator) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **{
        "observation.state": rng.standard_normal(ACTION_DIM).astype(np.float32),
    })
    return buf.getvalue()


@pytest.mark.parametrize("backend", ["echo", "tiny_torch"])
def test_full_loop_receives_actions(server, backend):
    client = connect_drtc(
        environment="pytest",
        policy_backend=backend,
        server_address=server,
        chunk_size=16,
        action_dim=ACTION_DIM,
        min_execution_horizon=4,
        cooldown_steps=4,
        fps=30,
    )
    assert client.session_id
    rng = np.random.default_rng(0)
    received = 0
    try:
        for _ in range(60):
            action = client.step(_obs(rng), codec="npz")
            if action is not None:
                received += 1
                assert action.shape == (ACTION_DIM,)
                assert np.isfinite(action).all()
            time.sleep(1 / 30)
    finally:
        client.close()
    # Cold start swallows the first few ticks; the vast majority must land.
    assert received >= 40, f"only {received}/60 ticks had an action"


def test_one_session_per_box(server):
    """The box enforces one session at a time (ADR-0001): a second concurrent
    OpenSession is refused, but the box is reusable once the first closes."""
    import grpc

    a = connect_drtc(environment="pytest-a", policy_backend="echo",
                     server_address=server, chunk_size=8, action_dim=ACTION_DIM, fps=30)
    try:
        # A second session while A is live must be rejected, not queued.
        with pytest.raises(grpc.RpcError) as exc:
            connect_drtc(environment="pytest-b", policy_backend="echo",
                         server_address=server, chunk_size=8, action_dim=ACTION_DIM, fps=30)
        assert exc.value.code() == grpc.StatusCode.FAILED_PRECONDITION
        assert a.session_id in exc.value.details()
    finally:
        a.close()

    # Once A is closed the box is free again — a fresh session opens cleanly
    # and gets a distinct id.
    b = connect_drtc(environment="pytest-b", policy_backend="echo",
                     server_address=server, chunk_size=8, action_dim=ACTION_DIM, fps=30)
    try:
        assert b.session_id and b.session_id != a.session_id
    finally:
        b.close()
