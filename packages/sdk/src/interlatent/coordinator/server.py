"""Coordinator HTTP control plane.

A stdlib ``ThreadingHTTPServer`` (no web-framework dependency, matching the
SDK's deliberately small dep surface). Two request surfaces:

* **Node-facing** ``/api/v1/nodes/*`` — byte-for-byte what
  :mod:`interlatent.node.daemon` calls: ``pair``, ``heartbeat``, and a real
  blocking ``poll`` long-poll. Plus ``hardware`` / ``robot-features``
  (200-noop) and a ``teleop-token`` stub (404 — teleop is disabled offline).
* **Admin** ``/admin/*`` — the thin client in :mod:`interlatent.cli`: register
  GPU boxes, list nodes, start/stop/list sessions, set the recording
  destination.

State (gpus, nodes, active session assignments, recording destination) is
persisted atomically to JSON so ``interlatent up`` after a crash/``down``
re-serves the *same* assignment to a still-running node — see
docs/adr/0001-offline-coordinator-control-plane.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import secrets
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

from .. import routing

_LOG = logging.getLogger("interlatent.coordinator")

DEFAULT_STATE_PATH = Path(
    os.environ.get("INTERLATENT_COORDINATOR_STATE", "~/.interlatent/coordinator.json")
).expanduser()

# A node is "live" if it heartbeated within this window.
_LIVE_WINDOW_S = 30.0


class PolicyChangeError(Exception):
    """Raised when a session would switch a GPU box's onboard policy.

    A GPU box pre-warms (loads + torch.compiles) one policy; running a
    *different* one recompiles (slow) and loads alongside the warm policy
    (possible OOM). We refuse unless the caller explicitly confirms.
    """

    def __init__(self, gpu: str, warm: str, requested: str) -> None:
        super().__init__(
            f"gpu {gpu} is warmed for {warm}; switching to {requested} recompiles "
            f"(slow) and may OOM. Confirm to change the onboard policy."
        )
        self.gpu = gpu
        self.warm = warm
        self.requested = requested


class Coordinator:
    """In-memory control-plane state with atomic JSON persistence.

    All mutating/reading methods take ``self._lock``; the long-poll waits on
    ``self._cond`` (the same lock) and is woken by any assignment change.
    """

    def __init__(self, state_path: Path = DEFAULT_STATE_PATH) -> None:
        self.state_path = Path(state_path)
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        # Liveness is runtime-only (not persisted): node_id -> last heartbeat.
        self._last_seen: dict[str, float] = {}
        self._state: dict[str, Any] = {
            "gpus": {},        # name -> {"name", "url"}
            "nodes": {},       # node_id -> {"id", "name", "token", "hardware"}
            "sessions": {},    # node_id -> session dict (current assignment)
            "recording": {},   # destination block injected into sessions
        }
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self.state_path.exists():
            try:
                self._state.update(json.loads(self.state_path.read_text()))
            except Exception:
                _LOG.warning("Could not read state %s; starting fresh", self.state_path,
                             exc_info=True)

    def _persist(self) -> None:
        """Atomically write state (tmp + os.replace). Caller holds the lock."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._state, indent=2))
        os.replace(tmp, self.state_path)

    # ------------------------------------------------------------------
    # Node-facing API
    # ------------------------------------------------------------------

    def pair(self, name: str) -> dict:
        node_id = "node_" + secrets.token_hex(8)
        token = "ilnode_" + secrets.token_hex(16)
        with self._lock:
            self._state["nodes"][node_id] = {
                "id": node_id, "name": name or node_id, "token": token, "hardware": {},
            }
            self._last_seen[node_id] = time.time()
            self._persist()
        _LOG.info("Paired node %s (name=%r)", node_id, name)
        return {"id": node_id, "token": token, "name": name or node_id}

    def heartbeat(self, node_id: str) -> None:
        with self._lock:
            self._last_seen[node_id] = time.time()

    def set_hardware(self, node_id: str, payload: dict) -> None:
        with self._lock:
            node = self._state["nodes"].get(node_id)
            if node is not None:
                node["hardware"] = payload
                self._persist()

    def poll(
        self, node_id: str, known_session_id: str, known_endpoint: str, wait: float
    ) -> dict:
        """Block until this node's assignment changes or ``wait`` elapses."""
        deadline = time.time() + max(0.0, wait)
        with self._cond:
            self._last_seen[node_id] = time.time()
            while True:
                desired = self._state["sessions"].get(node_id)
                desired_id = desired.get("id", "") if desired else ""
                desired_endpoint = desired.get("drtc_endpoint", "") if desired else ""
                changed = (
                    known_session_id != desired_id or known_endpoint != desired_endpoint
                )
                remaining = deadline - time.time()
                if changed or remaining <= 0:
                    return {"changed": changed, "session": desired}
                self._cond.wait(timeout=min(remaining, 5.0))

    # ------------------------------------------------------------------
    # Admin API
    # ------------------------------------------------------------------

    def add_gpu(self, name: str, url: str, method: str = "direct", warm_policy: str = "") -> dict:
        if method not in routing.known_methods():
            raise ValueError(
                f"unknown routing method {method!r}; known: {routing.known_methods()}"
            )
        with self._lock:
            # ``warm_policy`` tracks the box's onboard (compiled) policy. Seeded
            # from the operator's DRTC_WARMUP_POLICY here; updated on a confirmed
            # policy switch. Empty = unknown, so the switch guard stays off until
            # the first session establishes it.
            gpu = {"name": name, "url": url, "method": method, "warm_policy": warm_policy}
            self._state["gpus"][name] = gpu
            self._persist()
        return gpu

    def remove_gpu(self, name: str) -> bool:
        with self._lock:
            existed = self._state["gpus"].pop(name, None) is not None
            if existed:
                self._persist()
        return existed

    def list_gpus(self) -> list[dict]:
        with self._lock:
            return list(self._state["gpus"].values())

    def list_nodes(self) -> list[dict]:
        now = time.time()
        with self._lock:
            out = []
            for node in self._state["nodes"].values():
                last = self._last_seen.get(node["id"])
                out.append({
                    "id": node["id"],
                    "name": node["name"],
                    "hardware": node.get("hardware", {}),
                    "last_seen_age_s": (now - last) if last else None,
                    "live": bool(last and (now - last) <= _LIVE_WINDOW_S),
                    "busy": self._state["sessions"].get(node["id"]) is not None,
                })
            return out

    def remove_node(self, node_id: str) -> bool:
        with self._lock:
            existed = self._state["nodes"].pop(node_id, None) is not None
            self._state["sessions"].pop(node_id, None)
            self._last_seen.pop(node_id, None)
            if existed:
                self._persist()
        return existed

    def resolve_node(self, ref: str) -> Optional[str]:
        """Resolve ``ref`` to a node_id: exact id, else a unique *live* name."""
        now = time.time()
        with self._lock:
            if ref in self._state["nodes"]:
                return ref
            matches = [
                n["id"] for n in self._state["nodes"].values()
                if n["name"] == ref
                and (self._last_seen.get(n["id"], 0) and now - self._last_seen[n["id"]] <= _LIVE_WINDOW_S)
            ]
            if len(matches) == 1:
                return matches[0]
            # Ambiguous or all-stale: fall back to a unique name match (any liveness).
            any_matches = [n["id"] for n in self._state["nodes"].values() if n["name"] == ref]
            if len(any_matches) == 1:
                return any_matches[0]
            return None

    def set_destination(self, recording: dict) -> None:
        with self._lock:
            self._state["recording"] = dict(recording or {})
            self._persist()

    def get_destination(self) -> dict:
        with self._lock:
            return dict(self._state.get("recording") or {})

    def start_session(self, node_id: str, gpu_name: str, params: dict) -> dict:
        """Assign a session to ``node_id``. Raises ValueError on conflicts."""
        with self._cond:
            if node_id not in self._state["nodes"]:
                raise ValueError(f"unknown node {node_id}")
            if self._state["sessions"].get(node_id) is not None:
                raise ValueError(f"node {node_id} already has an active session")
            gpu = self._state["gpus"].get(gpu_name)
            if gpu is None:
                raise ValueError(f"unknown gpu {gpu_name!r} (register with `interlatent gpu add`)")
            # Onboard-policy guard: a matching policy reuses the box's compiled
            # runtime (instant); a mismatch recompiles + may OOM, so refuse
            # unless the caller confirmed. ``warm_policy`` empty = unknown, so
            # the first session just records the policy without prompting.
            warm = (gpu.get("warm_policy") or "").strip()
            requested = params["policy"]
            if warm and requested != warm and not params.get("confirm_policy_change"):
                raise PolicyChangeError(gpu_name, warm, requested)
            env_slug = params.get("env_slug") or "default"
            task = params.get("task") or env_slug
            fps = float(params.get("fps") or 30.0)
            # Resolve the GPU's route descriptor into the session ``route`` the
            # node connects through. ``drtc_endpoint`` is kept = the resolved
            # address for back-compat with nodes that predate routing.
            route = routing.resolve(
                routing.make_descriptor(gpu["url"], method=gpu.get("method", "direct"))
            )
            session = {
                "id": "sess_" + secrets.token_hex(8),
                "policy_uri": params["policy"],
                "policy_backend": params.get("backend") or "lerobot",
                "task": task,
                "fps": fps,
                "chunk_size": int(params.get("chunk_size") or 50),
                "action_dim": int(params.get("action_dim") or 6),
                "drtc_endpoint": route["address"],
                "route": route,
                "environment_id": env_slug,
                "collection_context": {"env_slug": env_slug, "task": task, "fps": fps},
                "recording": dict(self._state.get("recording") or {}),
                "node_id": node_id,
                "gpu": gpu_name,
            }
            self._state["sessions"][node_id] = session
            # Track the (now) onboard policy so the next session for it is the
            # fast path and a later switch prompts again. Covers both a confirmed
            # switch and the first-session establish-from-unknown case.
            if requested and requested != warm:
                gpu["warm_policy"] = requested
            self._persist()
            self._cond.notify_all()
        _LOG.info("Started session %s on node %s (gpu=%s policy=%s)",
                  session["id"], node_id, gpu_name, session["policy_uri"])
        return session

    def stop_session(self, session_id: str) -> bool:
        with self._cond:
            for nid, sess in list(self._state["sessions"].items()):
                if sess and sess.get("id") == session_id:
                    self._state["sessions"][nid] = None
                    self._persist()
                    self._cond.notify_all()
                    _LOG.info("Stopped session %s (node %s unassigned)", session_id, nid)
                    return True
        return False

    def stop_all(self) -> list[str]:
        with self._cond:
            stopped = []
            for nid, sess in list(self._state["sessions"].items()):
                if sess:
                    stopped.append(sess.get("id"))
                    self._state["sessions"][nid] = None
            if stopped:
                self._persist()
                self._cond.notify_all()
        return stopped

    def list_sessions(self) -> list[dict]:
        with self._lock:
            return [s for s in self._state["sessions"].values() if s]


# ----------------------------------------------------------------------
# HTTP handler
# ----------------------------------------------------------------------


def _probe_reachable(url: str, timeout: float = 2.0) -> bool:
    """Fast TCP-connect probe of a ``host:port`` (or ``scheme://host:port``)."""
    parsed = urlparse(url if "://" in url else f"//{url}", scheme="")
    host = parsed.hostname
    port = parsed.port or (443 if (parsed.scheme or "").endswith("s") else 50051)
    if not host:
        return False
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


class _Handler(BaseHTTPRequestHandler):
    coordinator: Coordinator = None  # set in run_server
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet default access logging
        _LOG.debug("%s - %s", self.address_string(), fmt % args)

    # -- helpers --------------------------------------------------------

    def _send(self, code: int, body: Optional[dict] = None) -> None:
        data = json.dumps(body if body is not None else {}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return {}

    # -- dispatch -------------------------------------------------------

    def do_GET(self):
        c = self.coordinator
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path.startswith("/api/v1/nodes/") and path.endswith("/poll"):
                node_id = path[len("/api/v1/nodes/"):-len("/poll")]
                q = parse_qs(parsed.query)
                result = c.poll(
                    node_id,
                    known_session_id=(q.get("known_session_id", [""])[0]),
                    known_endpoint=(q.get("known_endpoint", [""])[0]),
                    wait=float(q.get("wait", ["25"])[0]),
                )
                return self._send(200, result)
            if path == "/admin/gpus":
                return self._send(200, {"gpus": c.list_gpus()})
            if path == "/admin/nodes":
                return self._send(200, {"nodes": c.list_nodes()})
            if path == "/admin/sessions":
                return self._send(200, {"sessions": c.list_sessions()})
            if path == "/admin/destination":
                return self._send(200, {"recording": c.get_destination()})
            return self._send(404, {"error": "not found"})
        except Exception as e:  # never crash the handler thread
            _LOG.exception("GET %s failed", path)
            return self._send(500, {"error": str(e)})

    def do_POST(self):
        c = self.coordinator
        path = urlparse(self.path).path
        body = self._read_json()
        try:
            if path == "/api/v1/nodes":
                return self._send(200, c.pair(body.get("name", "")))
            if path.startswith("/api/v1/nodes/") and path.endswith("/heartbeat"):
                c.heartbeat(path[len("/api/v1/nodes/"):-len("/heartbeat")])
                return self._send(200, {})
            if path.startswith("/api/v1/nodes/") and path.endswith("/hardware"):
                c.set_hardware(path[len("/api/v1/nodes/"):-len("/hardware")], body)
                return self._send(200, {})
            if path.startswith("/api/v1/nodes/") and path.endswith("/robot-features"):
                return self._send(200, {})  # accepted, not used offline
            if path.startswith("/api/v1/inference/sessions/") and path.endswith("/teleop-token"):
                return self._send(404, {"error": "teleop disabled (offline coordinator)"})
            if path == "/admin/gpus":
                return self._send(200, c.add_gpu(
                    body["name"], body["url"],
                    body.get("method", "direct"), body.get("warm_policy", ""),
                ))
            if path == "/admin/destination":
                c.set_destination(body.get("recording") or {})
                return self._send(200, {"recording": c.get_destination()})
            if path == "/admin/sessions":
                return self._start_session(body)
            return self._send(404, {"error": "not found"})
        except KeyError as e:
            return self._send(400, {"error": f"missing field {e}"})
        except ValueError as e:
            return self._send(400, {"error": str(e)})
        except Exception as e:
            _LOG.exception("POST %s failed", path)
            return self._send(500, {"error": str(e)})

    def do_DELETE(self):
        c = self.coordinator
        path = urlparse(self.path).path
        try:
            if path.startswith("/admin/gpus/"):
                name = path[len("/admin/gpus/"):]
                return self._send(200 if c.remove_gpu(name) else 404, {})
            if path.startswith("/admin/nodes/"):
                node_id = path[len("/admin/nodes/"):]
                return self._send(200 if c.remove_node(node_id) else 404, {})
            if path.startswith("/admin/sessions/"):
                sid = path[len("/admin/sessions/"):]
                return self._send(200 if c.stop_session(sid) else 404, {})
            return self._send(404, {"error": "not found"})
        except Exception as e:
            _LOG.exception("DELETE %s failed", path)
            return self._send(500, {"error": str(e)})

    # -- admin: start session (with node lookup + reachability probe) ---

    def _start_session(self, body: dict):
        c = self.coordinator
        node_ref = body.get("node", "")
        node_id = c.resolve_node(node_ref)
        if node_id is None:
            return self._send(
                404, {"error": f"node {node_ref!r} not found (ambiguous name or unknown id)"}
            )
        gpu_name = body.get("gpu", "")
        gpu = {g["name"]: g for g in c.list_gpus()}.get(gpu_name)
        if gpu is None:
            return self._send(404, {"error": f"gpu {gpu_name!r} not registered"})
        # The TCP reachability probe only makes sense for the ``direct`` method
        # (a fixed address). Other methods manage their own connectivity.
        method = gpu.get("method", "direct")
        if method == "direct" and not body.get("no_probe") and not _probe_reachable(gpu["url"]):
            return self._send(
                502, {"error": f"gpu {gpu_name} ({gpu['url']}) unreachable (TCP probe failed)"}
            )
        if not body.get("policy"):
            return self._send(400, {"error": "policy is required"})
        try:
            session = c.start_session(node_id, gpu_name, body)
        except PolicyChangeError as e:
            return self._send(409, {
                "error": str(e),
                "needs_policy_confirm": True,
                "warm_policy": e.warm,
                "requested": e.requested,
            })
        except ValueError as e:
            return self._send(409, {"error": str(e)})
        warn = None
        if not c.get_destination():
            warn = ("no recording destination configured on the coordinator — "
                    "this session will not be saved")
        return self._send(200, {"session": session, "warning": warn})


# ----------------------------------------------------------------------
# Entry point (spawned detached by `interlatent up`)
# ----------------------------------------------------------------------


def run_server(host: str, port: int, state_path: Path) -> None:
    _Handler.coordinator = Coordinator(state_path)
    httpd = ThreadingHTTPServer((host, port), _Handler)
    _LOG.info("Coordinator listening on %s:%d (state=%s)", host, port, state_path)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="interlatent-coordinator")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8900)
    p.add_argument("--state", default=str(DEFAULT_STATE_PATH))
    p.add_argument("--output-dir", default="")
    p.add_argument("--s3-uri", default="")
    p.add_argument("--s3-endpoint-url", default="")
    p.add_argument("--s3-access-key", default="")
    p.add_argument("--s3-secret-key", default="")
    p.add_argument("--s3-region", default="")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    coordinator = Coordinator(Path(args.state).expanduser())
    recording = _recording_from_args(args)
    if recording:
        coordinator.set_destination(recording)
    _Handler.coordinator = coordinator
    httpd = ThreadingHTTPServer((args.host, args.port), _Handler)
    _LOG.info("Coordinator listening on %s:%d (state=%s, destination=%s)",
              args.host, args.port, args.state, recording or "none (inference-only)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


def _recording_from_args(args) -> dict:
    if args.output_dir:
        return {"output_dir": args.output_dir}
    if args.s3_uri:
        rec = {"s3_uri": args.s3_uri}
        if args.s3_endpoint_url:
            rec["s3_endpoint_url"] = args.s3_endpoint_url
        if args.s3_access_key:
            rec["s3_access_key"] = args.s3_access_key
        if args.s3_secret_key:
            rec["s3_secret_key"] = args.s3_secret_key
        if args.s3_region:
            rec["s3_region"] = args.s3_region
        return rec
    return {}


if __name__ == "__main__":
    raise SystemExit(main())
