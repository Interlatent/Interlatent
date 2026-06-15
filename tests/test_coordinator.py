"""Coordinator control-plane: assignment, long-poll, busy guard, HTTP wiring."""
import socket
import threading
import time
from http.server import ThreadingHTTPServer

import pytest

from interlatent.cli.client import CoordinatorClient, CoordinatorError
from interlatent.coordinator.server import Coordinator, _Handler


# ----------------------------------------------------------------------
# In-process logic
# ----------------------------------------------------------------------


def _coord(tmp_path):
    return Coordinator(tmp_path / "state.json")


def test_pair_and_resolve_by_name(tmp_path):
    c = _coord(tmp_path)
    a = c.pair("arm0")
    assert a["id"].startswith("node_") and a["token"].startswith("ilnode_")
    # resolve by id and by unique live name
    assert c.resolve_node(a["id"]) == a["id"]
    assert c.resolve_node("arm0") == a["id"]
    # duplicate name -> ambiguous -> None
    c.pair("arm0")
    assert c.resolve_node("arm0") is None
    assert c.resolve_node("nope") is None


def test_session_lifecycle_and_busy_guard(tmp_path):
    c = _coord(tmp_path)
    node = c.pair("arm0")["id"]
    c.add_gpu("gpu0", "127.0.0.1:50051")

    sess = c.start_session(node, "gpu0", {"policy": "lerobot/smolvla", "task": "pick"})
    assert sess["drtc_endpoint"] == "127.0.0.1:50051"
    assert sess["policy_uri"] == "lerobot/smolvla"
    assert sess["collection_context"]["env_slug"] == "default"
    # Route descriptor stamped for the node's connector (direct = dial as-is).
    assert sess["route"] == {"method": "direct", "address": "127.0.0.1:50051"}

    # busy guard
    with pytest.raises(ValueError):
        c.start_session(node, "gpu0", {"policy": "x"})

    # poll: fresh node (knows nothing) sees the change immediately
    res = c.poll(node, known_session_id="", known_endpoint="", wait=0)
    assert res["changed"] is True and res["session"]["id"] == sess["id"]

    # poll: node already on this session+endpoint -> no change
    res = c.poll(node, known_session_id=sess["id"],
                 known_endpoint=sess["drtc_endpoint"], wait=0)
    assert res["changed"] is False

    # stop -> unassign -> poll reports change to None
    assert c.stop_session(sess["id"]) is True
    res = c.poll(node, known_session_id=sess["id"],
                 known_endpoint=sess["drtc_endpoint"], wait=0)
    assert res["changed"] is True and res["session"] is None
    assert c.list_sessions() == []


def test_add_gpu_rejects_unknown_method(tmp_path):
    c = _coord(tmp_path)
    with pytest.raises(ValueError):
        c.add_gpu("gpu0", "127.0.0.1:50051", method="relay")
    # direct is accepted and stored.
    gpu = c.add_gpu("gpu0", "127.0.0.1:50051", method="direct")
    assert gpu["method"] == "direct"


def test_onboard_policy_guard(tmp_path):
    from interlatent.coordinator.server import PolicyChangeError

    c = _coord(tmp_path)
    node = c.pair("arm0")["id"]
    c.add_gpu("gpu0", "127.0.0.1:50051", warm_policy="lerobot/smolvla_base")
    assert c.list_gpus()[0]["warm_policy"] == "lerobot/smolvla_base"

    # Matching policy -> proceeds (fast path, reuses the compiled runtime).
    s1 = c.start_session(node, "gpu0", {"policy": "lerobot/smolvla_base"})
    assert s1["policy_uri"] == "lerobot/smolvla_base"
    c.stop_session(s1["id"])

    # Mismatch without confirm -> refused; onboard policy unchanged.
    with pytest.raises(PolicyChangeError):
        c.start_session(node, "gpu0", {"policy": "lerobot/act_xyz"})
    assert c.list_gpus()[0]["warm_policy"] == "lerobot/smolvla_base"

    # Mismatch WITH confirm -> proceeds and updates the onboard policy.
    s2 = c.start_session(
        node, "gpu0", {"policy": "lerobot/act_xyz", "confirm_policy_change": True}
    )
    assert s2["policy_uri"] == "lerobot/act_xyz"
    assert c.list_gpus()[0]["warm_policy"] == "lerobot/act_xyz"
    c.stop_session(s2["id"])

    # act_xyz is now the onboard policy -> matches, no confirm needed.
    s3 = c.start_session(node, "gpu0", {"policy": "lerobot/act_xyz"})
    assert s3["policy_uri"] == "lerobot/act_xyz"


def test_unknown_onboard_policy_is_recorded_on_first_session(tmp_path):
    from interlatent.coordinator.server import PolicyChangeError

    c = _coord(tmp_path)
    node = c.pair("arm0")["id"]
    c.add_gpu("gpu0", "127.0.0.1:50051")  # no warm policy -> guard off initially
    s = c.start_session(node, "gpu0", {"policy": "lerobot/smolvla_base"})
    c.stop_session(s["id"])
    # First session established the onboard policy; a different one now prompts.
    assert c.list_gpus()[0]["warm_policy"] == "lerobot/smolvla_base"
    with pytest.raises(PolicyChangeError):
        c.start_session(node, "gpu0", {"policy": "other"})


def test_recording_destination_injected(tmp_path):
    c = _coord(tmp_path)
    node = c.pair("arm0")["id"]
    c.add_gpu("gpu0", "127.0.0.1:50051")
    c.set_destination({"output_dir": "/data/run"})
    sess = c.start_session(node, "gpu0", {"policy": "p"})
    assert sess["recording"] == {"output_dir": "/data/run"}


def test_start_session_unknown_gpu_or_node(tmp_path):
    c = _coord(tmp_path)
    node = c.pair("arm0")["id"]
    with pytest.raises(ValueError):
        c.start_session(node, "ghost", {"policy": "p"})
    with pytest.raises(ValueError):
        c.start_session("node_missing", "gpu0", {"policy": "p"})


def test_longpoll_wakes_on_assignment(tmp_path):
    c = _coord(tmp_path)
    node = c.pair("arm0")["id"]
    c.add_gpu("gpu0", "127.0.0.1:50051")
    result = {}

    def _poller():
        result["res"] = c.poll(node, known_session_id="", known_endpoint="", wait=5)

    t = threading.Thread(target=_poller)
    t.start()
    time.sleep(0.2)  # ensure the poll is blocked-waiting
    sess = c.start_session(node, "gpu0", {"policy": "p"})
    t.join(timeout=3)
    assert not t.is_alive(), "long-poll did not wake on assignment"
    assert result["res"]["changed"] is True
    assert result["res"]["session"]["id"] == sess["id"]


def test_state_persists_across_reload(tmp_path):
    c = _coord(tmp_path)
    node = c.pair("arm0")["id"]
    c.add_gpu("gpu0", "127.0.0.1:50051")
    c.start_session(node, "gpu0", {"policy": "p"})
    # New instance from the same file re-serves the same assignment.
    c2 = Coordinator(tmp_path / "state.json")
    assert len(c2.list_sessions()) == 1
    assert c2.list_gpus()[0]["name"] == "gpu0"


# ----------------------------------------------------------------------
# HTTP wiring (pair + gpu + session over the wire) with the thin client
# ----------------------------------------------------------------------


def test_http_roundtrip(tmp_path):
    # A real listening socket stands in for the GPU box so the start-session
    # reachability probe succeeds.
    gpu_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    gpu_sock.bind(("127.0.0.1", 0))
    gpu_sock.listen()
    gpu_port = gpu_sock.getsockname()[1]

    _Handler.coordinator = Coordinator(tmp_path / "state.json")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        client = CoordinatorClient(f"http://127.0.0.1:{port}")
        assert client.ping()

        # Node self-registers over the node-facing API.
        paired = client._req("POST", "/api/v1/nodes", {"name": "arm0"})
        assert paired["id"].startswith("node_")

        client.add_gpu("gpu0", f"127.0.0.1:{gpu_port}")
        nodes = client.list_nodes()
        assert any(n["name"] == "arm0" for n in nodes)

        resp = client.start_session({
            "node": "arm0", "gpu": "gpu0", "policy": "lerobot/smolvla", "task": "pick",
        })
        sid = resp["session"]["id"]
        assert resp["session"]["drtc_endpoint"] == f"127.0.0.1:{gpu_port}"
        # No destination configured -> warning surfaced.
        assert resp["warning"]

        assert len(client.list_sessions()) == 1
        client.stop_session(sid)
        assert client.list_sessions() == []
    finally:
        srv.shutdown()
        srv.server_close()
        gpu_sock.close()


def test_http_policy_change_requires_confirm(tmp_path):
    gpu_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    gpu_sock.bind(("127.0.0.1", 0))
    gpu_sock.listen()
    gpu_port = gpu_sock.getsockname()[1]

    _Handler.coordinator = Coordinator(tmp_path / "state.json")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        client = CoordinatorClient(f"http://127.0.0.1:{port}")
        client._req("POST", "/api/v1/nodes", {"name": "arm0"})
        client.add_gpu("gpu0", f"127.0.0.1:{gpu_port}", warm_policy="lerobot/smolvla_base")

        # Mismatch without confirm -> structured 409 the CLI can prompt on.
        with pytest.raises(CoordinatorError) as ei:
            client.start_session({"node": "arm0", "gpu": "gpu0", "policy": "lerobot/act_xyz"})
        assert ei.value.payload.get("needs_policy_confirm") is True
        assert ei.value.payload.get("warm_policy") == "lerobot/smolvla_base"

        # With confirm -> succeeds.
        resp = client.start_session({
            "node": "arm0", "gpu": "gpu0", "policy": "lerobot/act_xyz",
            "confirm_policy_change": True,
        })
        assert resp["session"]["policy_uri"] == "lerobot/act_xyz"
    finally:
        srv.shutdown()
        srv.server_close()
        gpu_sock.close()


def test_http_unreachable_gpu_rejected(tmp_path):
    _Handler.coordinator = Coordinator(tmp_path / "state.json")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        client = CoordinatorClient(f"http://127.0.0.1:{port}")
        client._req("POST", "/api/v1/nodes", {"name": "arm0"})
        # Point at a (closed) port nothing listens on -> probe fails.
        client.add_gpu("gpu0", "127.0.0.1:1")
        with pytest.raises(CoordinatorError) as ei:
            client.start_session({"node": "arm0", "gpu": "gpu0", "policy": "p"})
        assert "unreachable" in str(ei.value)
    finally:
        srv.shutdown()
        srv.server_close()
