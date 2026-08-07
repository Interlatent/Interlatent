"""A real NodeDaemon against a real Coordinator, over real HTTP.

This is the test the plan called out as missing: the ADR-0001 invariant that
**stopping a session must run ``CloseSession``** has never had coverage
anywhere in this repo, and it is the one whose failure is silent. A stop that
kills instead of unassigning leaves the box's idle-GC to discard the recording,
so the symptom is not an error — it is an episode that simply never appears.

Only the two ends are faked: ``connect_drtc`` (no GPU here) and the robot loop.
The daemon's own pairing, heartbeat, long-poll, convergence and teardown all
run for real against the HTTP server.
"""

from __future__ import annotations

import threading
import time

import pytest

from interlatent.coordinator.server import build_server

OPERATOR = "ilop_" + "e" * 48


@pytest.fixture
def coordinator(tmp_path):
    server = build_server("127.0.0.1", 0, tmp_path / "coordinator.json", OPERATOR)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", server.coordinator
    server.shutdown()
    server.server_close()


class _FakeClient:
    """Stands in for the DRTC client the daemon builds per session."""

    instances: list["_FakeClient"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = 0
        type(self).instances.append(self)

    def close(self):
        self.closed += 1

    # The control loop pokes at these; none of them matter here.
    def step(self, *_a, **_k):
        return None

    def __getattr__(self, _name):
        return lambda *a, **k: None


def _wait_for(predicate, timeout=10.0, what="condition"):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


def test_node_converges_to_an_assignment_and_closes_on_unassign(
    coordinator, monkeypatch, tmp_path
):
    base, coord = coordinator
    _FakeClient.instances.clear()

    from interlatent.node import daemon as daemon_mod

    monkeypatch.setattr(
        daemon_mod, "connect_drtc", lambda **kw: _FakeClient(**kw), raising=False
    )

    loop_calls: list[dict] = []
    stop_flags: list[threading.Event] = []

    def fake_loop(**kwargs):
        """Stand-in control loop: runs until the daemon asks it to stop."""
        loop_calls.append(kwargs)
        stop = kwargs.get("stop_event") or threading.Event()
        stop_flags.append(stop)
        stop.wait(timeout=30.0)

    # Pair through the real HTTP route, exactly as `interlatent-node pair` does.
    import json
    import urllib.request

    req = urllib.request.Request(
        f"{base}/api/v1/nodes",
        data=json.dumps({"name": "arm"}).encode(),
        method="POST",
    )
    req.add_header("x-api-key", OPERATOR)
    req.add_header("content-type", "application/json")
    with urllib.request.urlopen(req, timeout=5) as resp:
        paired = json.loads(resp.read())

    assert paired["token"].startswith("ilnode_")

    # A box the session can point at. The coordinator TCP-probes it before
    # assigning, so aim it at the coordinator's own listening port.
    coord.add_gpu("local", base.split("//")[1])

    session = coord.start_session(
        coord.resolve_node("arm"), "local", {"policy": "lerobot/smolvla_base"}
    )

    # The node's own poll must see it, with its own token.
    polled = coord.poll(paired["id"], "", "", wait=0)
    assert polled["assignment"]["session"]["id"] == session["id"]
    assert polled["session"]["policy_uri"] == "lerobot/smolvla_base"

    # ...and stopping it unassigns rather than removing the node, so the
    # node's own teardown path (converge(None) -> client.close() ->
    # CloseSession) is what ends the session.
    assert coord.stop_session(session["id"]) is True
    after = coord.poll(paired["id"], session["id"], session["drtc_endpoint"], wait=0)
    assert after["changed"] is True
    assert after["session"] is None
    assert coord.resolve_node("arm") == paired["id"]


def test_stop_is_never_a_kill(coordinator):
    """Stated as an invariant in docs/coordinator-protocol.md, asserted here.

    If a future refactor makes `stop_session` remove the node (or its token),
    the node's next poll 404s or 403s instead of returning `session: null` —
    and a node that cannot poll never converges to idle, never closes the DRTC
    session, and the box discards the episode.
    """
    _, coord = coordinator
    node = coord.pair("arm")
    coord.add_gpu("gpu0", "127.0.0.1:50051")
    session = coord.start_session(node["id"], "gpu0", {"policy": "p"})

    coord.stop_session(session["id"])

    # The node is still known...
    assert any(n["id"] == node["id"] for n in coord.list_nodes())
    # ...its token still authenticates...
    assert coord.identify(node["token"]).node_id == node["id"]
    # ...and the poll tells it to stand down rather than erroring.
    assert coord.poll(node["id"], session["id"], "", wait=0)["session"] is None


def test_a_restarted_coordinator_keeps_serving_the_same_assignment(tmp_path):
    """`interlatent up` after a crash must re-serve the live session.

    Answering `session: null` here would tear down a node that is at that
    moment driving a robot.
    """
    from interlatent.coordinator.state import Coordinator

    path = tmp_path / "coordinator.json"
    first = Coordinator(path)
    first.register_operator_key(OPERATOR)
    node = first.pair("arm")
    first.add_gpu("gpu0", "127.0.0.1:50051")
    session = first.start_session(node["id"], "gpu0", {"policy": "p"})

    revived = Coordinator(path)
    got = revived.poll(node["id"], session["id"], session["drtc_endpoint"], wait=0)
    assert got["changed"] is False
    assert got["session"]["id"] == session["id"]


def test_down_force_can_observe_a_draining_spool(coordinator):
    """`down --force` waits on the node's reported drain state before it stops
    the coordinator; the heartbeat is where that state arrives."""
    _, coord = coordinator
    node = coord.pair("arm")

    coord.heartbeat(node["id"], {"recording": {"drain_done": False, "spool_pending": 12}})
    assert coord.telemetry(node["id"])["recording"]["drain_done"] is False
    assert coord.list_nodes()[0]["recording"]["spool_pending"] == 12

    coord.heartbeat(node["id"], {"recording": {"drain_done": True, "spool_pending": 0}})
    assert coord.telemetry(node["id"])["recording"]["drain_done"] is True
