"""The coordinator's HTTP surface.

A stdlib ``ThreadingHTTPServer`` — no web framework, matching the SDK's
deliberately small dependency surface.

**One surface: ``/api/v1/*``.** The 2026-06 coordinator additionally served a
bespoke ``/admin/*`` consumed by its own client class, which meant every
operator flow had two spellings, one per client. ADR 0023
identifies that fork as what collapsed the stack. Serving only the documented
protocol is what keeps "one protocol, one code path" true rather than
aspirational: the same ``interlatent`` CLI, the same node, the same
``interlatent-serve`` and the same teleop web app talk to this and to anything
else that implements the contract, with no branch on which it is.

Every route is authenticated — see :mod:`interlatent.coordinator.auth` for the
principals, and ``docs/coordinator-protocol.md`` for the contract.
"""

from __future__ import annotations

import json
import logging
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import parse_qs, urlparse

from . import auth
from .state import Coordinator, PolicyChangeError, probe_reachable

_LOG = logging.getLogger("interlatent.coordinator")

#: Who may call what. ``operator`` is always allowed; the other values name an
#: additional principal that is allowed *for its own resource only*.
OWNER_NODE = "own-node"
OWNER_BOX = "own-box"
ANY_ISSUED = "any-issued"


class Route:
    """One (method, pattern) -> handler binding."""

    __slots__ = ("method", "regex", "handler", "principal")

    def __init__(
        self,
        method: str,
        pattern: str,
        handler: Callable[..., Any],
        principal: str = "operator",
    ) -> None:
        self.method = method
        # ``{name}`` -> a named group that does not cross a path separator.
        self.regex = re.compile(
            "^" + re.sub(r"\{(\w+)\}", r"(?P<\1>[^/]+)", pattern) + "/?$"
        )
        self.handler = handler
        self.principal = principal


class _Handler(BaseHTTPRequestHandler):
    coordinator: Coordinator = None  # set by run_server
    server_relay = None  # RelayHandle, when an embedded relay is running
    protocol_version = "HTTP/1.1"

    # -- plumbing -------------------------------------------------------

    def log_message(self, fmt, *args):  # quiet the default access log
        _LOG.debug("%s - %s", self.address_string(), fmt % args)

    def _send(self, code: int, body: Optional[dict | list] = None) -> None:
        data = json.dumps(body if body is not None else {}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        # The teleop web app is served from a different origin than the
        # coordinator in every realistic deployment.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "x-api-key, content-type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return {}
        return body if isinstance(body, dict) else {}

    def _query(self) -> dict:
        return {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}

    # -- dispatch -------------------------------------------------------

    def do_OPTIONS(self):  # CORS preflight from the teleop web app
        self._send(204)

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_PUT(self):
        self._dispatch("PUT")

    def do_DELETE(self):
        self._dispatch("DELETE")

    def _dispatch(self, method: str) -> None:
        path = urlparse(self.path).path.rstrip("/") or "/"
        for route in ROUTES:
            if route.method != method:
                continue
            m = route.regex.match(path)
            if not m:
                continue
            params = m.groupdict()
            principal = self.coordinator.identify(
                self.headers.get("x-api-key") or ""
            )
            allowed, why = _authorize(route, principal, params)
            if not allowed:
                return self._send(
                    401 if principal is None else 403, {"error": why}
                )
            try:
                return route.handler(self, params)
            except PolicyChangeError as e:
                return self._send(409, {
                    "error": str(e), "code": "policy_change",
                    "gpu": e.gpu, "warm": e.warm, "requested": e.requested,
                })
            except ValueError as e:
                return self._send(400, {"error": str(e)})
            except Exception as e:  # never take down the handler thread
                _LOG.exception("%s %s failed", method, path)
                return self._send(500, {"error": str(e)})
        self._send(404, {"error": f"no route for {method} {path}"})


def _authorize(route, principal, params) -> tuple[bool, str]:
    """The principal table from ``docs/coordinator-protocol.md``."""
    if principal is None:
        return False, "missing or unknown x-api-key"
    if principal.is_operator:
        return True, ""
    if route.principal == ANY_ISSUED:
        return True, ""
    if route.principal == OWNER_NODE:
        if principal.kind == auth.KIND_NODE and principal.node_id == params.get(
            "node_id"
        ):
            return True, ""
        return False, "this token does not belong to that node"
    if route.principal == OWNER_BOX:
        if principal.kind == auth.KIND_BOX and principal.box_id == params.get("box_id"):
            return True, ""
        return False, "this key does not belong to that box"
    return False, "operator key required"


# ----------------------------------------------------------------------
# Handlers
# ----------------------------------------------------------------------


def _h_pair(h: _Handler, _p):
    body = h._read_json()
    return h._send(200, h.coordinator.pair((body.get("name") or "").strip()))


def _h_heartbeat(h: _Handler, p):
    h.coordinator.heartbeat(p["node_id"], h._read_json())
    return h._send(200, {"ok": True})


def _h_poll(h: _Handler, p):
    q = h._query()
    try:
        wait = float(q.get("wait", "25"))
    except ValueError:
        wait = 25.0
    # Bound the hold: an unbounded `wait` lets a client pin a handler thread.
    wait = max(0.0, min(wait, 60.0))
    return h._send(200, h.coordinator.poll(
        p["node_id"],
        known_session_id=q.get("known_session_id", ""),
        known_endpoint=q.get("known_endpoint", ""),
        wait=wait,
    ))


def _h_hardware(h: _Handler, p):
    h.coordinator.set_hardware(p["node_id"], h._read_json())
    return h._send(200, {"ok": True})


def _h_features(h: _Handler, p):
    h.coordinator.set_features(p["node_id"], h._read_json())
    return h._send(200, {"ok": True})


def _h_list_nodes(h: _Handler, _p):
    return h._send(200, {"nodes": h.coordinator.list_nodes()})


def _h_delete_node(h: _Handler, p):
    ok = h.coordinator.remove_node(p["node_id"])
    return h._send(200 if ok else 404, {"ok": ok})


def _h_register_box(h: _Handler, _p):
    return h._send(200, h.coordinator.register_box(h._read_json()))


def _h_box_status(h: _Handler, p):
    body = h._read_json()
    ok = h.coordinator.set_box_status(
        p["box_id"], body.get("status") or "", body.get("endpoint") or ""
    )
    return h._send(200 if ok else 404, {"ok": ok})


def _h_warmup_target(h: _Handler, p):
    target = h.coordinator.warmup_target(p["box_id"])
    # 404 is the documented "no target" answer; the box then falls back to its
    # own --warmup-policy / --warmup-image-keys.
    return h._send(404, {"error": "no warmup target"}) if target is None else h._send(
        200, target
    )


def _h_authz(h: _Handler, p):
    """The per-RPC probe a GPU box makes for every gRPC call.

    Deliberately accepts the operator key *and any node token this coordinator
    issued*: the node presents ``drtc_api_key or token`` on the box's gRPC
    metadata, which is frequently the node token. Accepting only the operator
    key here returns UNAUTHENTICATED for every Infer and nothing works.
    """
    if h.coordinator.find_box(p["box_id"]) is None:
        return h._send(404, {"error": "unknown box"})
    return h._send(200, {"ok": True})


def _h_list_gpus(h: _Handler, _p):
    return h._send(200, {"gpus": h.coordinator.list_gpus()})


def _h_delete_gpu(h: _Handler, p):
    ok = h.coordinator.remove_gpu(p["name"])
    return h._send(200 if ok else 404, {"ok": ok})


def _h_list_envs(h: _Handler, _p):
    return h._send(200, {"environments": h.coordinator.list_environments()})


def _h_create_env(h: _Handler, _p):
    return h._send(200, h.coordinator.create_environment(h._read_json()))


def _h_env_config(h: _Handler, p):
    env = h.coordinator.get_environment(p["env_id"])
    return h._send(404, {"error": "unknown environment"}) if env is None else h._send(
        200, env
    )


def _h_list_sessions(h: _Handler, _p):
    return h._send(200, {"sessions": h.coordinator.list_sessions()})


def _h_create_session(h: _Handler, _p):
    c = h.coordinator
    body = h._read_json()
    node_ref = (body.get("node") or body.get("node_id") or "").strip()
    node_id = c.resolve_node(node_ref)
    if not node_id:
        raise ValueError(f"unknown or ambiguous node {node_ref!r}")
    gpu_name = (body.get("pod") or body.get("gpu") or "").strip()
    gpu = next(
        (g for g in c.list_gpus() if g.get("name") == gpu_name), None
    )
    if gpu is None:
        raise ValueError(f"unknown gpu {gpu_name!r}")
    if gpu.get("url") and not probe_reachable(gpu["url"]):
        raise ValueError(
            f"gpu {gpu_name!r} at {gpu['url']} is not reachable; "
            "start interlatent-serve there, or check the address"
        )
    params = {
        "policy": body.get("policy") or body.get("policy_uri") or "",
        "backend": body.get("backend"),
        "task": body.get("task"),
        "task_id": body.get("task_id"),
        "env_slug": body.get("env_slug") or body.get("environment"),
        "fps": body.get("fps"),
        "chunk_size": body.get("chunk_size"),
        "action_dim": body.get("action_dim"),
        "synchronous": body.get("synchronous"),
        "confirm_policy_change": body.get("confirm_policy_change"),
    }
    if not params["policy"]:
        raise ValueError("policy is required")
    session = c.start_session(node_id, gpu_name, params)
    return h._send(200, {"session": session, "id": session["id"]})


def _h_delete_session(h: _Handler, p):
    ok = h.coordinator.stop_session(p["session_id"])
    return h._send(200 if ok else 404, {"ok": ok})


#: The protocol has one tier and this coordinator routes all of it, so the
#: capability list is down to a single remaining question: is teleop usable?
#: Teleop needs an embedded relay, and without one the mint route answers a
#: definitive 404 (`node/teleop/factory.py` treats that as "teleop off for this
#: session") — so a caller can and should ask in advance.
#:
#: Spelled out rather than derived by pattern. A substring filter once claimed
#: `cancel-processing` was served because it does not contain "/process", and
#: `/environments/{id}/episodes` because it does not start with "/episodes".
#: A capability list that lies is worse than none — callers use it to decide
#: what not to attempt, so it names exactly what this process is serving.
_SERVED_WITH_RELAY = {
    "/inference/sessions/{session_id}/teleop-token",
    "/teleop-recordings/{recording_id}/teleop-token",
    "/teleop-recordings",
    "/teleop-recordings/{recording_id}/stop",
}


def _h_capabilities(h: _Handler, _p):
    from .protocol import API_PREFIX, PROTOCOL_VERSION

    served = set(_SERVED_WITH_RELAY) if h.server_relay is not None else set()
    # The key keeps its name from protocol/1: it is what the shipped callers
    # read, and renaming a field they parse would cost more than the stale
    # word "optional" does. It now means "conditionally served, and live".
    return h._send(200, {
        "protocol": PROTOCOL_VERSION,
        "optional_supported": sorted(API_PREFIX + p for p in served),
    })


def _h_mint_teleop_token(h: _Handler, p):
    """Mint a teleop token and tell the caller where the relay is.

    The response shape is fixed by two shipped clients: the node refuses
    unless `transport == "quic"` and `webtransport_url` is present
    (node/teleop/factory.py), and the browser gates on the same pair.
    """
    c = h.coordinator
    session_id = p.get("session_id") or p.get("recording_id") or ""
    role = h._query().get("role", "node")
    try:
        minted = c.mint_teleop_token(session_id, role)
    except ValueError as e:
        return h._send(400, {"error": str(e)})
    if minted is None:
        return h._send(404, {"error": "unknown session"})

    relay = getattr(h.server_relay, "descriptor", None) if h.server_relay else None
    if relay is None:
        # A definitive 404 turns teleop off for the session rather than
        # leaving the node retrying forever (factory.py treats 401/403/404 as
        # final). That is the honest answer when no relay is running.
        return h._send(404, {"error": "no teleop relay on this coordinator"})

    body = {
        "token": minted["token"],
        "transport": "quic",
        "webtransport_url": f"{relay['base']}/teleop/{role}/{session_id}",
        # Neither endpoint reads this; present because the shipped clients
        # type it and a missing key has bitten JSON consumers before.
        "expires_at": None,
    }
    if relay.get("certificate_hashes"):
        # No public CA exists for a LAN address, so Chromium pins the cert by
        # digest instead. Sent on every mint so a rotation cannot strand a
        # browser on a stale hash.
        body["server_certificate_hashes"] = relay["certificate_hashes"]
    return h._send(200, body)


def _h_list_teleop_recordings(h: _Handler, _p):
    return h._send(200, {"recordings": h.coordinator.list_teleop_recordings()})


def _h_create_teleop_recording(h: _Handler, _p):
    return h._send(200, h.coordinator.create_teleop_recording(h._read_json()))


def _h_stop_teleop_recording(h: _Handler, p):
    ok = h.coordinator.stop_teleop_recording(p["recording_id"])
    h.coordinator.revoke_teleop_tokens(p["recording_id"])
    return h._send(200 if ok else 404, {"ok": ok})


def _h_get_destination(h: _Handler, _p):
    return h._send(200, {"recording": h.coordinator.get_destination()})


def _h_set_destination(h: _Handler, _p):
    body = h._read_json()
    h.coordinator.set_destination(body.get("recording") or body)
    return h._send(200, {"recording": h.coordinator.get_destination()})


ROUTES: list[Route] = [
    # Node plane
    Route("POST", "/api/v1/nodes", _h_pair),
    Route("GET", "/api/v1/nodes", _h_list_nodes),
    Route("POST", "/api/v1/nodes/{node_id}/heartbeat", _h_heartbeat, OWNER_NODE),
    Route("GET", "/api/v1/nodes/{node_id}/poll", _h_poll, OWNER_NODE),
    Route("POST", "/api/v1/nodes/{node_id}/hardware", _h_hardware, OWNER_NODE),
    Route("POST", "/api/v1/nodes/{node_id}/robot-features", _h_features, OWNER_NODE),
    Route("DELETE", "/api/v1/nodes/{node_id}", _h_delete_node),
    # Box plane
    Route("POST", "/api/v1/compute/boxes/register", _h_register_box),
    Route("POST", "/api/v1/compute/boxes/{box_id}/status", _h_box_status, OWNER_BOX),
    Route("GET", "/api/v1/compute/boxes/{box_id}/warmup-target",
          _h_warmup_target, OWNER_BOX),
    Route("GET", "/api/v1/compute/boxes/{box_id}/authz", _h_authz, ANY_ISSUED),
    Route("GET", "/api/v1/gpus", _h_list_gpus),
    Route("DELETE", "/api/v1/gpus/{name}", _h_delete_gpu),
    # Environments. GET /environments doubles as the box's auth probe, so it
    # must accept any key this coordinator issued, not just the operator's.
    Route("GET", "/api/v1/environments", _h_list_envs, ANY_ISSUED),
    Route("POST", "/api/v1/environments", _h_create_env),
    Route("GET", "/api/v1/environments/{env_id}/config", _h_env_config, ANY_ISSUED),
    # Sessions
    Route("GET", "/api/v1/inference/sessions", _h_list_sessions),
    Route("POST", "/api/v1/inference/sessions", _h_create_session),
    Route("DELETE", "/api/v1/inference/sessions/{session_id}", _h_delete_session),
    # Teleop. Tokens are minted for a role and pinned to one session; the
    # relay verifies them at CONNECT.
    Route("POST", "/api/v1/inference/sessions/{session_id}/teleop-token",
          _h_mint_teleop_token, ANY_ISSUED),
    Route("POST", "/api/v1/teleop-recordings/{recording_id}/teleop-token",
          _h_mint_teleop_token, ANY_ISSUED),
    Route("GET", "/api/v1/teleop-recordings", _h_list_teleop_recordings),
    Route("POST", "/api/v1/teleop-recordings", _h_create_teleop_recording),
    Route("POST", "/api/v1/teleop-recordings/{recording_id}/stop",
          _h_stop_teleop_recording),
    # Discovery + the coordinator's own administration surface
    Route("GET", "/api/v1/capabilities", _h_capabilities, ANY_ISSUED),
    Route("GET", "/api/v1/coordinator/recording", _h_get_destination),
    Route("PUT", "/api/v1/coordinator/recording", _h_set_destination),
]


# ----------------------------------------------------------------------
# Serving
# ----------------------------------------------------------------------


def build_server(
    host: str,
    port: int,
    state_path: Path,
    operator_key: str,
    relay=None,
    coordinator: Coordinator | None = None,
) -> ThreadingHTTPServer:
    # One Coordinator per process. The relay verifies teleop tokens against
    # the same object the HTTP surface mints them from — two instances over
    # one state file would each hold their own in-memory copy, and a token
    # minted through HTTP would be invisible to the relay.
    coordinator = coordinator or Coordinator(state_path)
    coordinator.register_operator_key(operator_key)

    handler = type(
        "_BoundHandler", (_Handler,),
        {"coordinator": coordinator, "server_relay": relay},
    )
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    server.coordinator = coordinator  # type: ignore[attr-defined]
    return server


def run_server(
    host: str,
    port: int,
    state_path: Path,
    operator_key: str,
    relay=None,
    coordinator: Coordinator | None = None,
) -> None:
    server = build_server(
        host, port, state_path, operator_key, relay, coordinator
    )
    _LOG.info("Coordinator listening on http://%s:%d", host, port)
    try:
        server.serve_forever()
    finally:
        server.server_close()
