"""Teleop on a self-hosted coordinator: tokens, certificates, relay pairing.

The 2026-06 coordinator hard-404'd teleop (``{"error": "teleop disabled
(offline coordinator)"}``), so none of this is a restoration — it is the part
that makes "managed by the CLI alone" true rather than nearly true.

The aioquic wire glue itself is not exercised here: it was validated live on
the upstream deployment and needs a real UDP socket and a real headset. What
*is* covered is everything that decides whether a connection is allowed and
where the browser is told to dial — which is where a self-hosted deployment
differs from the hosted one.
"""

from __future__ import annotations

import datetime as _dt
import json
import threading
import urllib.error
import urllib.request

import pytest

from interlatent.coordinator.server import build_server
from interlatent.coordinator.state import Coordinator

OPERATOR = "ilop_" + "7" * 48


@pytest.fixture
def coord(tmp_path):
    c = Coordinator(tmp_path / "coordinator.json")
    c.register_operator_key(OPERATOR)
    return c


def _session(c):
    node = c.pair("arm")
    c.add_gpu("gpu0", "127.0.0.1:50051")
    return c.start_session(node["id"], "gpu0", {"policy": "p"})


# ----------------------------------------------------------------------
# Tokens
# ----------------------------------------------------------------------


def test_token_is_scoped_to_one_session_and_role(coord):
    session = _session(coord)
    minted = coord.mint_teleop_token(session["id"], "browser")

    ok, _ = coord.verify_teleop_token(minted["token"], session["id"], "browser")
    assert ok

    bad_role, why = coord.verify_teleop_token(
        minted["token"], session["id"], "node"
    )
    assert not bad_role and "role" in why

    bad_session, why = coord.verify_teleop_token(
        minted["token"], "sess_other", "browser"
    )
    assert not bad_session and "session" in why


def test_unknown_token_is_refused(coord):
    session = _session(coord)
    ok, why = coord.verify_teleop_token("iltel_nope", session["id"], "node")
    assert not ok and why == "unknown token"
    ok, why = coord.verify_teleop_token("", session["id"], "node")
    assert not ok


def test_token_for_an_unknown_session_is_not_minted(coord):
    assert coord.mint_teleop_token("sess_nope", "node") is None


def test_role_must_be_node_or_browser(coord):
    session = _session(coord)
    with pytest.raises(ValueError):
        coord.mint_teleop_token(session["id"], "admin")


def test_tokens_are_hashed_at_rest(coord, tmp_path):
    session = _session(coord)
    minted = coord.mint_teleop_token(session["id"], "node")
    assert minted["token"] not in (tmp_path / "coordinator.json").read_text()


def test_tokens_survive_a_restart(coord, tmp_path):
    """The node re-mints on every reconnect, but the *browser* mints exactly
    once per overlay open. A coordinator restart that invalidated tokens would
    end a live VR session with no way for it to recover."""
    session = _session(coord)
    minted = coord.mint_teleop_token(session["id"], "browser")

    revived = Coordinator(tmp_path / "coordinator.json")
    ok, _ = revived.verify_teleop_token(
        minted["token"], session["id"], "browser"
    )
    assert ok


def test_stopping_a_recording_revokes_its_tokens(coord):
    coord.pair("arm")
    rec = coord.create_teleop_recording({"node_id": "arm", "environment_id": "e"})
    minted = coord.mint_teleop_token(rec["id"], "browser")
    assert coord.verify_teleop_token(minted["token"], rec["id"], "browser")[0]

    coord.stop_teleop_recording(rec["id"])
    coord.revoke_teleop_tokens(rec["id"])
    assert not coord.verify_teleop_token(minted["token"], rec["id"], "browser")[0]


# ----------------------------------------------------------------------
# Teleop recordings behave like sessions
# ----------------------------------------------------------------------


def test_a_recording_is_assigned_and_polled_like_a_session(coord):
    node = coord.pair("arm")
    rec = coord.create_teleop_recording({"node_id": "arm", "environment_id": "e"})

    got = coord.poll(node["id"], "", "", wait=0)
    assert got["assignment"]["type"] == "teleop_recording"
    assert got["assignment"]["recording"]["id"] == rec["id"]
    # The flat key is still populated for the daemon's fallback path.
    assert got["session"]["id"] == rec["id"]


def test_stopping_a_recording_unassigns_the_node(coord):
    node = coord.pair("arm")
    rec = coord.create_teleop_recording({"node_id": "arm", "environment_id": "e"})
    assert coord.stop_teleop_recording(rec["id"]) is True

    got = coord.poll(node["id"], rec["id"], "", wait=0)
    assert got["session"] is None
    assert coord.list_teleop_recordings()[0]["status"] == "stopped"


def test_a_recording_and_a_session_cannot_share_a_node(coord):
    coord.pair("arm")
    coord.add_gpu("gpu0", "127.0.0.1:50051")
    coord.create_teleop_recording({"node_id": "arm", "environment_id": "e"})
    with pytest.raises(ValueError, match="already has an active session"):
        coord.start_session(coord.resolve_node("arm"), "gpu0", {"policy": "p"})


# ----------------------------------------------------------------------
# The mint response the shipped clients require
# ----------------------------------------------------------------------


class _FakeRelay:
    def __init__(self, hashes=None):
        self.descriptor = {
            "base": "https://10.0.0.5:4433",
            "certificate_hashes": hashes or [],
        }


@pytest.fixture
def http(tmp_path):
    def _make(relay=None):
        server = build_server(
            "127.0.0.1", 0, tmp_path / "coordinator.json", OPERATOR, relay
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, f"http://127.0.0.1:{server.server_address[1]}"

    made = []
    yield lambda relay=None: made.append(_make(relay)) or made[-1]
    for server, _ in made:
        server.shutdown()
        server.server_close()


def call(base, method, path, key=OPERATOR, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method=method)
    if key:
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


def test_mint_response_satisfies_both_shipped_clients(http):
    """node/teleop/factory.py disables teleop unless `transport == "quic"` and
    `webtransport_url` is present; the browser gates on the same pair. A
    response missing either is silently no teleop."""
    hashes = [{"algorithm": "sha-256", "value": "ab" * 32}]
    server, base = http(_FakeRelay(hashes))
    session = _session(server.coordinator)

    status, body = call(
        base, "POST",
        f"/api/v1/inference/sessions/{session['id']}/teleop-token?role=node",
    )
    assert status == 200
    assert body["transport"] == "quic"
    assert body["webtransport_url"] == (
        f"https://10.0.0.5:4433/teleop/node/{session['id']}"
    )
    assert body["token"].startswith("iltel_")
    assert body["server_certificate_hashes"] == hashes
    assert "expires_at" in body  # typed by both clients


def test_the_url_carries_the_role_the_relay_pairs_on(http):
    server, base = http(_FakeRelay())
    session = _session(server.coordinator)
    urls = {}
    for role in ("node", "browser"):
        _, body = call(
            base, "POST",
            f"/api/v1/inference/sessions/{session['id']}/teleop-token?role={role}",
        )
        urls[role] = body["webtransport_url"]
    assert urls["node"].endswith(f"/teleop/node/{session['id']}")
    assert urls["browser"].endswith(f"/teleop/browser/{session['id']}")


def test_no_relay_is_a_definitive_404_not_a_retry_loop(http):
    """factory.py treats 401/403/404 as final and stops asking. Anything else
    makes the node retry forever against a coordinator that will never serve
    teleop."""
    server, base = http(None)
    session = _session(server.coordinator)
    status, body = call(
        base, "POST",
        f"/api/v1/inference/sessions/{session['id']}/teleop-token?role=node",
    )
    assert status == 404
    assert "relay" in body["error"]


def test_cert_hashes_are_omitted_when_the_relay_has_a_real_certificate(http):
    server, base = http(_FakeRelay(hashes=[]))
    session = _session(server.coordinator)
    _, body = call(
        base, "POST",
        f"/api/v1/inference/sessions/{session['id']}/teleop-token?role=node",
    )
    assert "server_certificate_hashes" not in body


def test_capabilities_reports_teleop_only_when_a_relay_runs(http):
    with_relay, base_with = http(_FakeRelay())
    _, body = call(base_with, "GET", "/api/v1/capabilities")
    assert any("teleop-token" in p for p in body["optional_supported"])

    _, base_without = http(None)
    _, body = call(base_without, "GET", "/api/v1/capabilities")
    assert not any("teleop" in p for p in body["optional_supported"])


def test_a_node_token_can_mint_its_own_teleop_token(http):
    """The node mints with the key it already has; requiring the operator key
    would mean shipping the root credential to every robot."""
    server, base = http(_FakeRelay())
    session = _session(server.coordinator)
    node_token = [
        k for k in [server.coordinator.pair("second")["token"]]
    ][0]
    status, _ = call(
        base, "POST",
        f"/api/v1/inference/sessions/{session['id']}/teleop-token?role=node",
        key=node_token,
    )
    assert status == 200


# ----------------------------------------------------------------------
# Certificates
# ----------------------------------------------------------------------


def test_cert_meets_the_constraints_chromium_imposes(tmp_path):
    """serverCertificateHashes is not a general escape hatch: Chromium accepts
    it only for an ECDSA P-256 certificate valid for at most 14 days."""
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives.asymmetric import ec

    from interlatent.coordinator import certs

    cert = certs.ensure(tmp_path, ["10.0.0.5", "localhost"])
    assert len(cert.sha256) == 64
    assert cert.hashes_for_browser == [
        {"algorithm": "sha-256", "value": cert.sha256}
    ]

    lifetime = cert.not_after - _dt.datetime.now(_dt.timezone.utc)
    assert lifetime <= _dt.timedelta(days=certs.MAX_VALIDITY_DAYS)

    from cryptography import x509
    parsed = x509.load_pem_x509_certificate(cert.cert_path.read_bytes())
    key = parsed.public_key()
    assert isinstance(key, ec.EllipticCurvePublicKey)
    assert key.curve.name == "secp256r1"


def test_cert_is_reused_until_it_nears_expiry(tmp_path):
    pytest.importorskip("cryptography")
    from interlatent.coordinator import certs

    first = certs.ensure(tmp_path, ["localhost"])
    second = certs.ensure(tmp_path, ["localhost"])
    assert first.sha256 == second.sha256, "rotated a cert that was still fresh"


def test_private_key_is_written_0600(tmp_path):
    pytest.importorskip("cryptography")
    import os
    import stat

    from interlatent.coordinator import certs

    cert = certs.ensure(tmp_path, ["localhost"])
    mode = stat.S_IMODE(os.stat(cert.key_path).st_mode)
    assert mode == 0o600, f"relay private key is mode {mode:o}"


def test_a_corrupt_cert_is_replaced_rather_than_fatal(tmp_path):
    pytest.importorskip("cryptography")
    from interlatent.coordinator import certs

    (tmp_path / "relay-cert.pem").write_text("not a certificate")
    (tmp_path / "relay-key.pem").write_text("nor a key")
    cert = certs.ensure(tmp_path, ["localhost"])
    assert len(cert.sha256) == 64


# ----------------------------------------------------------------------
# Relay pairing logic (transport-agnostic half)
# ----------------------------------------------------------------------


def test_relay_paths_carry_role_and_session():
    from interlatent.coordinator.relay_core import parse_teleop_path

    assert parse_teleop_path("/teleop/node/sess_1") == ("node", "sess_1")
    assert parse_teleop_path("/teleop/browser/sess_1?token=x") == (
        "browser", "sess_1",
    )
    assert parse_teleop_path("/teleop/admin/sess_1") is None
    assert parse_teleop_path("/nope/node/sess_1") is None
    assert parse_teleop_path("/teleop/node") is None


def test_relay_pairs_the_two_sides_by_session_id():
    from interlatent.coordinator.relay_core import Registry

    reg = Registry()
    # peer_of(session, role) is "who should this role's bytes go to".
    assert reg.peer_of("s1", "node") is None

    reg.attach("s1", "node", "NODE")
    # Node attached, browser absent: the node's datagrams have nowhere to go,
    # so they are dropped rather than queued. The browser's 250ms request_spec
    # retry exists to cover exactly this window.
    assert reg.peer_of("s1", "node") is None

    reg.attach("s1", "browser", "BROWSER")
    assert reg.peer_of("s1", "node") == "BROWSER"
    assert reg.peer_of("s1", "browser") == "NODE"

    # A different session never crosses over.
    assert reg.peer_of("s2", "node") is None


def test_advertised_capabilities_are_all_actually_routed(http):
    """A capability list that lies is worse than none: callers use it to decide
    what not to attempt. An earlier substring filter advertised
    `cancel-processing` (it does not contain "/process") and
    `/environments/{id}/episodes` (it does not start with "/episodes")."""
    from interlatent.coordinator.server import ROUTES

    for relay in (None, _FakeRelay()):
        server, base = http(relay)
        _, body = call(base, "GET", "/api/v1/capabilities")
        for advertised in body["optional_supported"]:
            probe = (
                advertised
                .replace("{session_id}", "x")
                .replace("{recording_id}", "x")
                .replace("{env_id}", "x")
                .replace("{episode_id}", "x")
            )
            assert any(r.regex.match(probe) for r in ROUTES), (
                f"capabilities advertises {advertised}, which no route serves"
            )
