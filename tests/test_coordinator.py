"""The coordinator: state machine, long-poll, and the HTTP surface.

Restored from the suite deleted in ``347e9d1`` and adapted: every route moved
from the old bespoke ``/admin/*`` onto ``/api/v1/*`` (ADR 0038), and every
request now carries a key.

The cases carried over verbatim in intent are the ones that were load-bearing
then and still are: ``test_longpoll_wakes_on_assignment``,
``test_state_persists_across_reload``, ``test_http_roundtrip``,
``test_one_session_per_gpu_box`` and the onboard-policy guard.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from interlatent.coordinator import auth
from interlatent.coordinator.server import build_server
from interlatent.coordinator.state import Coordinator, PolicyChangeError

OPERATOR = "ilop_" + "a" * 48


@pytest.fixture
def coord(tmp_path):
    c = Coordinator(tmp_path / "coordinator.json")
    c.register_operator_key(OPERATOR)
    return c


def _gpu(c, name="gpu0", url="127.0.0.1:50051", warm=""):
    return c.add_gpu(name, url, warm_policy=warm)


# ----------------------------------------------------------------------
# Node plane
# ----------------------------------------------------------------------


def test_pair_mints_a_scoped_token_and_resolves_by_name(coord):
    out = coord.pair("arm")
    assert out["token"].startswith("ilnode_")
    assert coord.resolve_node("arm") == out["id"]
    assert coord.resolve_node(out["id"]) == out["id"]
    assert coord.resolve_node("nope") is None

    principal = coord.identify(out["token"])
    assert principal.kind == auth.KIND_NODE
    assert principal.node_id == out["id"]


def test_state_file_never_holds_a_plaintext_token(coord, tmp_path):
    out = coord.pair("arm")
    raw = (tmp_path / "coordinator.json").read_text()
    assert out["token"] not in raw
    assert auth.hash_key(out["token"]) in raw


def test_heartbeat_payload_is_kept_for_drain_aware_shutdown(coord):
    node = coord.pair("arm")
    coord.heartbeat(node["id"], {"recording": {"drain_done": False, "spool_pending": 3}})
    assert coord.telemetry(node["id"])["recording"]["spool_pending"] == 3
    coord.heartbeat(node["id"], {"recording": {"drain_done": True}})
    assert coord.telemetry(node["id"])["recording"]["drain_done"] is True


def test_poll_returns_both_the_typed_envelope_and_the_flat_session(coord):
    node = coord.pair("arm")
    _gpu(coord)
    coord.start_session(node["id"], "gpu0", {"policy": "p"})

    got = coord.poll(node["id"], "", "", wait=0)
    assert got["changed"] is True
    assert got["assignment"]["type"] == "inference_session"
    # Same payload both ways: a node that predates the envelope still works.
    assert got["assignment"]["session"] == got["session"]


def test_longpoll_wakes_on_assignment(coord):
    node = coord.pair("arm")
    _gpu(coord)
    result: dict = {}

    def poller():
        result.update(coord.poll(node["id"], "", "", wait=5.0))

    t = threading.Thread(target=poller)
    t.start()
    time.sleep(0.1)
    started = time.time()
    coord.start_session(node["id"], "gpu0", {"policy": "lerobot/smolvla_base"})
    t.join(timeout=5.0)

    assert not t.is_alive()
    # Woken by the notify, not by the 5s timeout expiring.
    assert time.time() - started < 2.0
    assert result["changed"] is True
    assert result["session"]["policy_uri"] == "lerobot/smolvla_base"


def test_longpoll_times_out_without_a_change(coord):
    node = coord.pair("arm")
    started = time.time()
    got = coord.poll(node["id"], "", "", wait=0.3)
    assert got["changed"] is False
    assert got["session"] is None
    assert 0.2 < time.time() - started < 3.0


# ----------------------------------------------------------------------
# Sessions
# ----------------------------------------------------------------------


def test_session_lifecycle_and_busy_guard(coord):
    node = coord.pair("arm")
    _gpu(coord)
    sess = coord.start_session(node["id"], "gpu0", {"policy": "p", "env_slug": "kitchen"})

    assert sess["collection_context"]["env_slug"] == "kitchen"
    assert coord.list_sessions() == [sess]
    with pytest.raises(ValueError, match="already has an active session"):
        coord.start_session(node["id"], "gpu0", {"policy": "p"})

    assert coord.stop_session(sess["id"]) is True
    assert coord.list_sessions() == []
    assert coord.stop_session(sess["id"]) is False


def test_stop_session_unassigns_rather_than_deleting_the_node(coord):
    """The ADR-0001 invariant: stop is an unassign, so the node's own teardown
    runs CloseSession. If stop removed the node the poll would 404 instead."""
    node = coord.pair("arm")
    _gpu(coord)
    sess = coord.start_session(node["id"], "gpu0", {"policy": "p"})
    coord.stop_session(sess["id"])

    got = coord.poll(node["id"], sess["id"], sess["drtc_endpoint"], wait=0)
    assert got["changed"] is True
    assert got["session"] is None       # converge to idle...
    assert got["assignment"] is None
    assert coord.resolve_node("arm") == node["id"]   # ...node still paired


def test_one_session_per_gpu_box(coord):
    a, b = coord.pair("a"), coord.pair("b")
    _gpu(coord)
    coord.start_session(a["id"], "gpu0", {"policy": "p"})
    with pytest.raises(ValueError, match="one session per box"):
        coord.start_session(b["id"], "gpu0", {"policy": "p"})


def test_onboard_policy_guard(coord):
    node = coord.pair("arm")
    _gpu(coord, warm="policy/a")
    with pytest.raises(PolicyChangeError) as exc:
        coord.start_session(node["id"], "gpu0", {"policy": "policy/b"})
    assert exc.value.warm == "policy/a"
    assert exc.value.requested == "policy/b"

    sess = coord.start_session(
        node["id"], "gpu0", {"policy": "policy/b", "confirm_policy_change": True}
    )
    assert sess["policy_uri"] == "policy/b"
    # The switch is now the box's onboard policy, so the next one is free.
    coord.stop_session(sess["id"])
    coord.start_session(node["id"], "gpu0", {"policy": "policy/b"})


def test_unknown_onboard_policy_is_recorded_on_first_session(coord):
    node = coord.pair("arm")
    _gpu(coord)  # warm unknown
    coord.start_session(node["id"], "gpu0", {"policy": "policy/a"})
    assert coord.list_gpus()[0]["warm_policy"] == "policy/a"


def test_recording_destination_is_stamped_onto_every_session(coord):
    """The seam that makes the whole inbox tier optional: the node forwards
    this block verbatim into OpenSession metadata."""
    node = coord.pair("arm")
    _gpu(coord)
    coord.set_destination({"output_dir": "/data/lerobot"})
    sess = coord.start_session(node["id"], "gpu0", {"policy": "p"})
    assert sess["recording"] == {"output_dir": "/data/lerobot"}


def test_synchronous_and_task_id_reach_the_session_payload(coord):
    node = coord.pair("arm")
    _gpu(coord)
    sess = coord.start_session(
        node["id"], "gpu0",
        {"policy": "p", "synchronous": True, "task_id": "task-42"},
    )
    assert sess["synchronous"] is True
    assert sess["task_id"] == "task-42"


def test_start_session_rejects_unknown_node_or_gpu(coord):
    node = coord.pair("arm")
    with pytest.raises(ValueError, match="unknown gpu"):
        coord.start_session(node["id"], "nope", {"policy": "p"})
    _gpu(coord)
    with pytest.raises(ValueError, match="unknown node"):
        coord.start_session("node_missing", "gpu0", {"policy": "p"})


def test_add_gpu_rejects_an_unknown_routing_method(coord):
    with pytest.raises(ValueError, match="unknown routing method"):
        coord.add_gpu("g", "host:1", method="carrier-pigeon")


# ----------------------------------------------------------------------
# Boxes and environments
# ----------------------------------------------------------------------


def test_register_box_is_idempotent_on_box_id(coord):
    first = coord.register_box({"box_id": "b1", "name": "rig", "endpoint": "h:50051"})
    second = coord.register_box({"box_id": "b1", "name": "rig", "endpoint": "h:50051"})
    assert first["box_id"] == second["box_id"] == "b1"
    assert len(coord.list_gpus()) == 1        # does not orphan a row

    # Re-registering rotates the box key and revokes the old one. The box
    # takes the new key from the response on every boot, so nothing has to
    # store a credential that outlives the process.
    assert first["key"] != second["key"]
    assert coord.identify(first["key"]) is None
    assert coord.identify(second["key"]).box_id == "b1"


def test_box_key_is_scoped_and_never_listed(coord):
    out = coord.register_box({"box_id": "b1", "name": "rig", "endpoint": "h:1"})
    principal = coord.identify(out["key"])
    assert principal.kind == auth.KIND_BOX
    assert principal.box_id == "b1"
    assert all("key" not in row for row in coord.list_gpus())
    assert coord.identify("ilbox_nope") is None


def test_warmup_target_is_none_until_a_policy_is_known(coord):
    coord.register_box({"box_id": "b1", "name": "rig", "endpoint": "h:1"})
    assert coord.warmup_target("b1") is None
    coord.register_box({
        "box_id": "b1", "name": "rig", "endpoint": "h:1", "warmup_policy": "p/a",
    })
    assert coord.warmup_target("b1")["policy_uri"] == "p/a"


def test_environment_round_trip_by_slug_or_id(coord):
    env = coord.create_environment({
        "slug": "kitchen", "action_dim": 7, "camera_names": ["top", "wrist"],
    })
    assert coord.get_environment("kitchen") == env
    assert coord.get_environment(env["id"]) == env
    assert coord.get_environment("nope") is None
    assert coord.list_environments() == [env]


def test_session_action_dim_falls_back_to_the_environment(coord):
    node = coord.pair("arm")
    _gpu(coord)
    coord.create_environment({"slug": "kitchen", "action_dim": 14})
    sess = coord.start_session(
        node["id"], "gpu0", {"policy": "p", "env_slug": "kitchen"}
    )
    assert sess["action_dim"] == 14


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------


def test_state_persists_across_reload(tmp_path):
    """Load-bearing: a coordinator that forgets its assignments answers
    session:null on the next poll and tears down a node mid-rollout."""
    path = tmp_path / "coordinator.json"
    first = Coordinator(path)
    first.register_operator_key(OPERATOR)
    node = first.pair("arm")
    first.add_gpu("gpu0", "127.0.0.1:50051")
    sess = first.start_session(node["id"], "gpu0", {"policy": "p"})

    second = Coordinator(path)
    assert [s["id"] for s in second.list_sessions()] == [sess["id"]]
    assert second.resolve_node(node["id"]) == node["id"]
    # A still-running node re-polls and is told to keep going, not to stop.
    assert second.poll(
        node["id"], sess["id"], sess["drtc_endpoint"], wait=0
    )["changed"] is False
    # Credentials survive too, or every node would be locked out by a restart.
    assert second.identify(node["token"]).node_id == node["id"]
    assert second.identify(OPERATOR).is_operator


def test_corrupt_state_file_starts_fresh_rather_than_crashing(tmp_path):
    path = tmp_path / "coordinator.json"
    path.write_text("{not json")
    c = Coordinator(path)
    assert c.list_nodes() == []


# ----------------------------------------------------------------------
# HTTP surface
# ----------------------------------------------------------------------


@pytest.fixture
def http(tmp_path):
    server = build_server("127.0.0.1", 0, tmp_path / "coordinator.json", OPERATOR)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    yield base, server.coordinator
    server.shutdown()
    server.server_close()


def call(base, method, path, key=OPERATOR, body=None, timeout=10.0):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method=method)
    if key:
        req.add_header("x-api-key", key)
    if data:
        req.add_header("content-type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return resp.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as e:
        raw = e.read()
        return e.code, (json.loads(raw) if raw else {})


def test_http_roundtrip(http):
    base, c = http
    status, node = call(base, "POST", "/api/v1/nodes", body={"name": "arm"})
    assert status == 200 and node["token"].startswith("ilnode_")

    status, _ = call(base, "POST", "/api/v1/compute/boxes/register",
                     body={"box_id": "b1", "name": "gpu0", "endpoint": "127.0.0.1:1"})
    assert status == 200

    # The node reports hardware with its own token.
    status, _ = call(base, "POST", f"/api/v1/nodes/{node['id']}/hardware",
                     key=node["token"], body={"robot_type": "so101"})
    assert status == 200

    status, listed = call(base, "GET", "/api/v1/nodes")
    assert status == 200 and listed["nodes"][0]["robot_type"] == "so101"

    # Sessions need a reachable box, so point at the coordinator's own port.
    c.add_gpu("local", base.split("//")[1], warm_policy="")
    status, made = call(base, "POST", "/api/v1/inference/sessions",
                        body={"node": "arm", "pod": "local", "policy": "p/a"})
    assert status == 200, made
    session_id = made["id"]

    status, sessions = call(base, "GET", "/api/v1/inference/sessions")
    assert [s["id"] for s in sessions["sessions"]] == [session_id]

    status, polled = call(base, "GET",
                          f"/api/v1/nodes/{node['id']}/poll?wait=0",
                          key=node["token"])
    assert polled["assignment"]["session"]["id"] == session_id

    status, _ = call(base, "DELETE", f"/api/v1/inference/sessions/{session_id}")
    assert status == 200
    assert call(base, "GET", "/api/v1/inference/sessions")[1]["sessions"] == []


def test_trailing_slash_is_accepted_on_sessions(http):
    """The CLI has always sent the trailing-slash form and the teleop web app
    the bare one. A coordinator honouring only one breaks a shipped caller."""
    base, _ = http
    assert call(base, "GET", "/api/v1/inference/sessions")[0] == 200
    assert call(base, "GET", "/api/v1/inference/sessions/")[0] == 200


def test_http_policy_change_requires_confirmation(http):
    base, c = http
    call(base, "POST", "/api/v1/nodes", body={"name": "arm"})
    c.add_gpu("local", base.split("//")[1], warm_policy="p/a")

    status, body = call(base, "POST", "/api/v1/inference/sessions",
                        body={"node": "arm", "pod": "local", "policy": "p/b"})
    assert status == 409 and body["code"] == "policy_change"

    status, _ = call(base, "POST", "/api/v1/inference/sessions",
                     body={"node": "arm", "pod": "local", "policy": "p/b",
                           "confirm_policy_change": True})
    assert status == 200


def test_http_unreachable_gpu_is_rejected_before_the_node_dials(http):
    base, c = http
    call(base, "POST", "/api/v1/nodes", body={"name": "arm"})
    c.add_gpu("dead", "127.0.0.1:1")  # nothing listens there
    status, body = call(base, "POST", "/api/v1/inference/sessions",
                        body={"node": "arm", "pod": "dead", "policy": "p"})
    assert status == 400 and "not reachable" in body["error"]


def test_unknown_route_is_a_clean_404(http):
    base, _ = http
    status, body = call(base, "GET", "/api/v1/nope")
    assert status == 404 and "no route" in body["error"]


def test_no_admin_surface_exists(http):
    """ADR 0038: one surface. The old /admin/* is what forked every operator
    flow into a dashboard spelling and a coordinator spelling."""
    base, _ = http
    for path in ("/admin/gpus", "/admin/nodes", "/admin/sessions",
                 "/admin/destination"):
        assert call(base, "GET", path)[0] == 404
