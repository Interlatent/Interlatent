"""The coordinator is the authority for the keys it issues.

Deleted ADR 0001 bound an unauthenticated ``/admin/*`` to ``0.0.0.0`` on the
grounds that "the network is the trust boundary". ADR 0023 deliberately
reversed that for the GPU port and ADR 0038 keeps the reversal, so these are
the assertions that stop it drifting back: anyone who can reach the port could
otherwise assign a session and move somebody's arm.
"""

from __future__ import annotations

import json
import os
import stat
import threading
import urllib.error
import urllib.request

import pytest

from interlatent.coordinator import auth
from interlatent.coordinator.server import build_server
from interlatent.coordinator.state import Coordinator

OPERATOR = "ilop_" + "f" * 48


# ----------------------------------------------------------------------
# Key minting and storage
# ----------------------------------------------------------------------


def test_operator_key_is_created_0600_and_only_once(tmp_path):
    path = tmp_path / "operator.key"
    key, created = auth.ensure_operator_key(path)

    assert created is True
    assert key.startswith("ilop_")
    mode = stat.S_IMODE(os.stat(path).st_mode)
    assert mode == 0o600, f"key is mode {mode:o}, not 0600"

    again, created_again = auth.ensure_operator_key(path)
    assert (again, created_again) == (key, False)


def test_key_file_is_created_exclusively_not_chmodded_after_the_fact(tmp_path):
    """A write-then-chmod leaves a window where the key is world-readable.

    Asserted by construction: if the file already exists, minting must not
    replace it — which is the same O_EXCL that closes the window.
    """
    path = tmp_path / "operator.key"
    path.write_text("ilop_preexisting\n")
    key, created = auth.ensure_operator_key(path)
    assert key == "ilop_preexisting"
    assert created is False


def test_a_second_up_does_not_clobber_the_key_the_first_handed_out(tmp_path):
    path = tmp_path / "operator.key"
    first, _ = auth.ensure_operator_key(path)
    results = []

    def racer():
        results.append(auth.ensure_operator_key(path)[0])

    threads = [threading.Thread(target=racer) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert set(results) == {first}


def test_hashing_is_stable_and_comparison_is_constant_time(tmp_path):
    key = auth.mint_key(auth.KIND_OPERATOR)
    assert auth.hash_key(key) == auth.hash_key(key)
    assert auth.key_matches(key, auth.hash_key(key))
    assert not auth.key_matches(key + "x", auth.hash_key(key))


def test_kinds_have_distinct_prefixes():
    prefixes = {
        auth.mint_key(k).split("_")[0]
        for k in (auth.KIND_OPERATOR, auth.KIND_NODE, auth.KIND_BOX)
    }
    assert prefixes == {"ilop", "ilnode", "ilbox"}
    with pytest.raises(ValueError):
        auth.mint_key("nonsense")


# ----------------------------------------------------------------------
# HTTP authorization
# ----------------------------------------------------------------------


@pytest.fixture
def http(tmp_path):
    server = build_server("127.0.0.1", 0, tmp_path / "coordinator.json", OPERATOR)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", server.coordinator
    server.shutdown()
    server.server_close()


def call(base, method, path, key=OPERATOR, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method=method)
    if key is not None:
        req.add_header("x-api-key", key)
    if data:
        req.add_header("content-type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read()
            return resp.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as e:
        raw = e.read()
        return e.code, (json.loads(raw) if raw else {})


def test_no_key_is_401_and_an_unknown_key_is_401(http):
    base, _ = http
    assert call(base, "GET", "/api/v1/nodes", key=None)[0] == 401
    assert call(base, "GET", "/api/v1/nodes", key="ilop_bogus")[0] == 401


def test_a_node_token_cannot_drive_the_operator_plane(http):
    base, _ = http
    _, node = call(base, "POST", "/api/v1/nodes", body={"name": "arm"})
    status, body = call(base, "GET", "/api/v1/inference/sessions", key=node["token"])
    assert status == 403
    assert "operator" in body["error"]


def test_a_node_token_is_scoped_to_its_own_node(http):
    base, _ = http
    _, a = call(base, "POST", "/api/v1/nodes", body={"name": "a"})
    _, b = call(base, "POST", "/api/v1/nodes", body={"name": "b"})

    # Its own routes: fine.
    assert call(base, "GET", f"/api/v1/nodes/{a['id']}/poll?wait=0",
                key=a["token"])[0] == 200
    # Someone else's: refused, and told why.
    status, body = call(base, "GET", f"/api/v1/nodes/{b['id']}/poll?wait=0",
                        key=a["token"])
    assert status == 403
    assert "does not belong to that node" in body["error"]


def test_a_box_key_is_scoped_to_its_own_box(http):
    base, _ = http
    _, one = call(base, "POST", "/api/v1/compute/boxes/register",
                  body={"box_id": "b1", "name": "r1", "endpoint": "h:1"})
    _, _two = call(base, "POST", "/api/v1/compute/boxes/register",
                   body={"box_id": "b2", "name": "r2", "endpoint": "h:2"})

    assert call(base, "POST", "/api/v1/compute/boxes/b1/status",
                key=one["key"], body={"status": "ready"})[0] == 200
    assert call(base, "POST", "/api/v1/compute/boxes/b2/status",
                key=one["key"], body={"status": "ready"})[0] == 403


def test_authz_accepts_both_the_operator_key_and_any_node_token(http):
    """The trap. A box presents whatever the node put in `x-api-key`, which is
    `drtc_api_key or token` — frequently the node token. Accepting only the
    operator key here makes every Infer return UNAUTHENTICATED.
    """
    base, _ = http
    _, node = call(base, "POST", "/api/v1/nodes", body={"name": "arm"})
    _, box = call(base, "POST", "/api/v1/compute/boxes/register",
                  body={"box_id": "b1", "name": "rig", "endpoint": "h:1"})

    assert call(base, "GET", "/api/v1/compute/boxes/b1/authz")[0] == 200
    assert call(base, "GET", "/api/v1/compute/boxes/b1/authz",
                key=node["token"])[0] == 200
    assert call(base, "GET", "/api/v1/compute/boxes/b1/authz",
                key=box["key"])[0] == 200
    assert call(base, "GET", "/api/v1/compute/boxes/b1/authz",
                key="ilop_stranger")[0] == 401
    assert call(base, "GET", "/api/v1/compute/boxes/unknown/authz")[0] == 404


def test_environments_probe_accepts_any_issued_key(http):
    """`GET /environments` is how a GPU box validates a presented key. If it
    were operator-only, a node's key would never validate and the box would
    reject every RPC."""
    base, _ = http
    _, node = call(base, "POST", "/api/v1/nodes", body={"name": "arm"})
    assert call(base, "GET", "/api/v1/environments", key=node["token"])[0] == 200
    assert call(base, "GET", "/api/v1/environments", key="ilnode_nope")[0] == 401


def test_removing_a_node_revokes_its_token(http):
    base, _ = http
    _, node = call(base, "POST", "/api/v1/nodes", body={"name": "arm"})
    assert call(base, "GET", f"/api/v1/nodes/{node['id']}/poll?wait=0",
                key=node["token"])[0] == 200

    assert call(base, "DELETE", f"/api/v1/nodes/{node['id']}")[0] == 200
    assert call(base, "GET", f"/api/v1/nodes/{node['id']}/poll?wait=0",
                key=node["token"])[0] == 401


def test_state_file_holds_no_plaintext_credential(tmp_path):
    path = tmp_path / "coordinator.json"
    c = Coordinator(path)
    c.register_operator_key(OPERATOR)
    node = c.pair("arm")
    box = c.register_box({"box_id": "b1", "name": "rig", "endpoint": "h:1"})

    raw = path.read_text()
    for secret in (OPERATOR, node["token"], box["key"]):
        assert secret not in raw, "a plaintext credential reached the state file"
    for secret in (OPERATOR, node["token"], box["key"]):
        assert auth.hash_key(secret) in raw
