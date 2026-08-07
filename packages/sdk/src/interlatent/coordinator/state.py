"""Coordinator state: nodes, boxes, environments, sessions, destination.

Forward-ported from the implementation deleted in ``347e9d1`` (last live tree
``8695afe``). The core is unchanged and was already right: one lock, a
``threading.Condition`` the long-poll waits on, and atomic tmp+``os.replace``
persistence.

What is new here relative to that version:

* **Boxes, environments and teleop recordings** — the 2026-06 coordinator
  served only the node plane, so a GPU box could not register and
  ``interlatent-serve`` could not boot against it.
* **Hashed credentials** — the old state file held ``ilnode_`` tokens in
  plaintext.
* **A heartbeat payload** — the node now reports spool depth and safety state,
  and ``down --force`` has to be able to wait for ``drain_done``.
* **``synchronous`` and ``task_id``** on the session payload.

The state file is load-bearing: a coordinator that forgets its assignments
answers ``session: null`` on the next poll and tears down a node that was
happily driving a robot.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import socket
import threading
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from .. import routing
from . import auth

_LOG = logging.getLogger("interlatent.coordinator")

DEFAULT_STATE_PATH = Path(
    os.environ.get("INTERLATENT_COORDINATOR_STATE", "~/.interlatent/coordinator.json")
).expanduser()

#: A node is "live" if it heartbeated within this window.
_LIVE_WINDOW_S = 30.0


class PolicyChangeError(Exception):
    """Raised when a session would switch a GPU box's onboard policy.

    A box pre-warms (loads + torch.compiles) one policy; running a *different*
    one recompiles (slow) and loads alongside the warm policy (possible OOM).
    Refuse unless the caller explicitly confirms.
    """

    def __init__(self, gpu: str, warm: str, requested: str) -> None:
        super().__init__(
            f"gpu {gpu} is warmed for {warm}; switching to {requested} recompiles "
            f"(slow) and may OOM. Confirm to change the onboard policy."
        )
        self.gpu = gpu
        self.warm = warm
        self.requested = requested


def probe_reachable(url: str, timeout: float = 2.0) -> bool:
    """Fast TCP-connect probe of ``host:port`` (or ``scheme://host:port``).

    Cheap pre-flight so assigning an unreachable box fails at the CLI rather
    than as a node-side ``UNAVAILABLE`` two seconds into a rollout.
    """
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


class Coordinator:
    """In-memory control-plane state with atomic JSON persistence.

    Every mutating and reading method takes ``self._lock``; the long-poll waits
    on ``self._cond`` (the same lock) and is woken by any assignment change.
    """

    def __init__(self, state_path: Path = DEFAULT_STATE_PATH) -> None:
        self.state_path = Path(state_path)
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        # Liveness is runtime-only, deliberately not persisted: a coordinator
        # that restarts should not claim a node is live because it was live
        # before the restart.
        self._last_seen: dict[str, float] = {}
        self._telemetry: dict[str, dict] = {}
        self._state: dict[str, Any] = {
            "keys": {},          # sha256(key) -> {kind, node_id?, box_id?, created}
            "gpus": {},          # name -> box record
            "nodes": {},         # node_id -> {id, name, token_hash, hardware, features}
            "sessions": {},      # node_id -> session dict or None
            "environments": {},  # slug -> env record
            "recordings": {},    # recording_id -> teleop recording
            "recording": {},     # destination block stamped onto every session
        }
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            loaded = json.loads(self.state_path.read_text())
        except Exception:
            _LOG.warning(
                "Could not read state %s; starting fresh", self.state_path,
                exc_info=True,
            )
            return
        if isinstance(loaded, dict):
            self._state.update(loaded)

    def _persist(self) -> None:
        """Atomically write state. Caller holds the lock."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._state, indent=2))
        os.replace(tmp, self.state_path)

    # ------------------------------------------------------------------
    # Credentials
    # ------------------------------------------------------------------

    def register_operator_key(self, key: str) -> None:
        """Record the operator key's hash so ``identify`` recognises it."""
        with self._lock:
            self._state["keys"][auth.hash_key(key)] = {
                "kind": auth.KIND_OPERATOR,
                "created": time.time(),
            }
            self._persist()

    def identify(self, presented: str) -> Optional[auth.Principal]:
        """Map a presented ``x-api-key`` to a principal, or None."""
        if not presented:
            return None
        digest = auth.hash_key(presented)
        with self._lock:
            record = self._state["keys"].get(digest)
        if not record:
            return None
        return auth.Principal(
            kind=record.get("kind", ""),
            node_id=record.get("node_id"),
            box_id=record.get("box_id"),
        )

    def _issue_key(self, kind: str, **binding: str) -> str:
        """Mint a key and record only its hash. Caller holds the lock."""
        key = auth.mint_key(kind)
        self._state["keys"][auth.hash_key(key)] = {
            "kind": kind, "created": time.time(), **binding,
        }
        return key

    # ------------------------------------------------------------------
    # Node plane
    # ------------------------------------------------------------------

    def pair(self, name: str) -> dict:
        node_id = "node_" + secrets.token_hex(8)
        with self._lock:
            token = self._issue_key(auth.KIND_NODE, node_id=node_id)
            self._state["nodes"][node_id] = {
                "id": node_id,
                "name": name or node_id,
                "token_hash": auth.hash_key(token),
                "hardware": {},
                "features": {},
            }
            self._last_seen[node_id] = time.time()
            self._persist()
        _LOG.info("Paired node %s (name=%r)", node_id, name)
        return {"id": node_id, "token": token, "name": name or node_id}

    def heartbeat(self, node_id: str, payload: Optional[dict] = None) -> None:
        """Liveness, plus whatever the node chose to report.

        The payload is not decoration: ``interlatent down --force`` waits on
        ``recording.drain_done`` before unassigning, so that a forced shutdown
        does not cut off a spool that is still draining.
        """
        with self._lock:
            self._last_seen[node_id] = time.time()
            if payload:
                self._telemetry[node_id] = dict(payload)

    def telemetry(self, node_id: str) -> dict:
        with self._lock:
            return dict(self._telemetry.get(node_id) or {})

    def set_hardware(self, node_id: str, payload: dict) -> None:
        with self._lock:
            node = self._state["nodes"].get(node_id)
            if node is not None:
                node["hardware"] = payload
                self._persist()

    def set_features(self, node_id: str, payload: dict) -> None:
        with self._lock:
            node = self._state["nodes"].get(node_id)
            if node is not None:
                node["features"] = payload
                self._persist()

    def poll(
        self, node_id: str, known_session_id: str, known_endpoint: str, wait: float
    ) -> dict:
        """Block until this node's assignment changes or ``wait`` elapses.

        Returns both the typed envelope the current node reads and the flat
        ``session`` key an older node falls back to.
        """
        deadline = time.time() + max(0.0, wait)
        with self._cond:
            self._last_seen[node_id] = time.time()
            while True:
                desired = self._state["sessions"].get(node_id)
                desired_id = desired.get("id", "") if desired else ""
                desired_endpoint = desired.get("drtc_endpoint", "") if desired else ""
                changed = (
                    known_session_id != desired_id
                    or known_endpoint != desired_endpoint
                )
                remaining = deadline - time.time()
                if changed or remaining <= 0:
                    return self._assignment_envelope(changed, desired)
                self._cond.wait(timeout=min(remaining, 5.0))

    @staticmethod
    def _assignment_envelope(changed: bool, desired: Optional[dict]) -> dict:
        kind = (desired or {}).get("assignment_type") or "inference_session"
        envelope: dict[str, Any] = {"changed": changed, "session": desired}
        if desired is None:
            envelope["assignment"] = None
        elif kind == "teleop_recording":
            envelope["assignment"] = {"type": kind, "recording": desired}
        else:
            envelope["assignment"] = {"type": kind, "session": desired}
        return envelope

    # ------------------------------------------------------------------
    # Box plane
    # ------------------------------------------------------------------

    def register_box(self, payload: dict) -> dict:
        """Idempotent on ``box_id`` — a restart re-registers the same box
        rather than leaving an orphan row."""
        box_id = (payload.get("box_id") or "").strip()
        if not box_id:
            raise ValueError("box_id is required")
        name = (payload.get("name") or box_id).strip()
        endpoint = (payload.get("endpoint") or "").strip()
        with self._lock:
            existing = None
            for box in self._state["gpus"].values():
                if box.get("box_id") == box_id:
                    existing = box
                    break
            # Mint fresh every time and keep only the hash. `interlatent-serve`
            # registers on every boot and takes the key from the response, so
            # rotating here costs nothing and keeps the state file free of a
            # plaintext credential. The previous key is revoked with it.
            for digest, rec in list(self._state["keys"].items()):
                if rec.get("box_id") == box_id:
                    self._state["keys"].pop(digest, None)
            key = self._issue_key(auth.KIND_BOX, box_id=box_id)
            box = {
                "name": name,
                "box_id": box_id,
                "url": endpoint,
                "method": payload.get("method") or "direct",
                "gpu_model": payload.get("gpu_model") or "",
                "gpu_count": payload.get("gpu_count") or 1,
                "vram_gb": payload.get("vram_gb") or None,
                "provider": payload.get("provider") or "byo",
                # Tracks the box's onboard (compiled) policy. Empty = unknown,
                # so the switch guard stays off until the first session
                # establishes it.
                "warm_policy": payload.get("warmup_policy") or (
                    existing.get("warm_policy") if existing else ""
                ),
                "status": "ready",
                "registered_at": time.time(),
            }
            if existing and existing.get("name") != name:
                self._state["gpus"].pop(existing["name"], None)
            self._state["gpus"][name] = box
            self._persist()
        _LOG.info("Registered box %s (%s) at %s", name, box_id, endpoint or "?")
        return {"name": name, "box_id": box_id, "status": "ready", "key": key}

    def find_box(self, box_id: str) -> Optional[dict]:
        with self._lock:
            for box in self._state["gpus"].values():
                if box.get("box_id") == box_id:
                    return dict(box)
        return None

    def set_box_status(self, box_id: str, status: str, endpoint: str = "") -> bool:
        with self._lock:
            for box in self._state["gpus"].values():
                if box.get("box_id") == box_id:
                    box["status"] = status
                    if endpoint:
                        box["url"] = endpoint
                    self._persist()
                    return True
        return False

    def warmup_target(self, box_id: str) -> Optional[dict]:
        """What this box should pre-warm, or None (the box then falls back to
        its own ``--warmup-policy`` / ``--warmup-image-keys``)."""
        box = self.find_box(box_id)
        if not box:
            return None
        policy = (box.get("warm_policy") or "").strip()
        if not policy:
            return None
        env = self.get_environment(box.get("environment") or "")
        return {
            "policy_uri": policy,
            "image_keys": (env or {}).get("camera_names") or [],
            "num_inference_steps": box.get("num_inference_steps"),
            "inference_action_mode": box.get("inference_action_mode") or "continuous",
        }

    def add_gpu(
        self, name: str, url: str, method: str = "direct", warm_policy: str = ""
    ) -> dict:
        if method not in routing.known_methods():
            raise ValueError(
                f"unknown routing method {method!r}; known: {routing.known_methods()}"
            )
        with self._lock:
            gpu = {
                "name": name,
                "box_id": name,
                "url": url,
                "method": method,
                "warm_policy": warm_policy,
                "status": "ready",
            }
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
            return [dict(box) for box in self._state["gpus"].values()]

    # ------------------------------------------------------------------
    # Environments
    # ------------------------------------------------------------------

    def create_environment(self, payload: dict) -> dict:
        slug = (payload.get("slug") or "").strip()
        if not slug:
            raise ValueError("slug is required")
        with self._lock:
            env = dict(self._state["environments"].get(slug) or {})
            env.update({
                "id": env.get("id") or ("env_" + secrets.token_hex(6)),
                "slug": slug,
                "display_name": payload.get("display_name") or slug,
                "robot_type": payload.get("robot_type") or "",
                "action_dim": payload.get("action_dim"),
                "camera_names": payload.get("camera_names") or [],
                "num_cameras": payload.get("num_cameras"),
                "task_description": payload.get("task_description") or "",
            })
            self._state["environments"][slug] = env
            self._persist()
        return env

    def get_environment(self, ref: str) -> Optional[dict]:
        """By slug or by id — callers use both interchangeably."""
        with self._lock:
            return self._get_environment_locked(ref)

    def _get_environment_locked(self, ref: str) -> Optional[dict]:
        """Caller holds the lock.

        ``self._lock`` is a plain Lock, not an RLock, so a locked section that
        called the public method above would deadlock against itself —
        ``start_session`` did exactly that.
        """
        if not ref:
            return None
        env = self._state["environments"].get(ref)
        if env:
            return dict(env)
        for candidate in self._state["environments"].values():
            if candidate.get("id") == ref:
                return dict(candidate)
        return None

    def list_environments(self) -> list[dict]:
        with self._lock:
            return list(self._state["environments"].values())

    # ------------------------------------------------------------------
    # Nodes (operator views)
    # ------------------------------------------------------------------

    def list_nodes(self) -> list[dict]:
        now = time.time()
        with self._lock:
            out = []
            for node in self._state["nodes"].values():
                last = self._last_seen.get(node["id"])
                session = self._state["sessions"].get(node["id"])
                out.append({
                    "id": node["id"],
                    "name": node["name"],
                    "hardware": node.get("hardware", {}),
                    "robot_type": (node.get("hardware") or {}).get("robot_type"),
                    "last_seen_age_s": (now - last) if last else None,
                    "live": bool(last and (now - last) <= _LIVE_WINDOW_S),
                    "online": bool(last and (now - last) <= _LIVE_WINDOW_S),
                    "busy": session is not None,
                    "current_session_id": (session or {}).get("id"),
                    "status": "online" if (
                        last and (now - last) <= _LIVE_WINDOW_S
                    ) else "offline",
                    "recording": (self._telemetry.get(node["id"]) or {}).get(
                        "recording", {}
                    ),
                })
            return out

    def remove_node(self, node_id: str) -> bool:
        with self._lock:
            existed = self._state["nodes"].pop(node_id, None) is not None
            self._state["sessions"].pop(node_id, None)
            self._last_seen.pop(node_id, None)
            self._telemetry.pop(node_id, None)
            # Revoke the node's token along with the node.
            for digest, rec in list(self._state["keys"].items()):
                if rec.get("node_id") == node_id:
                    self._state["keys"].pop(digest, None)
            if existed:
                self._persist()
        return existed

    def resolve_node(self, ref: str) -> Optional[str]:
        """Resolve ``ref`` to a node_id: exact id, else a unique *live* name."""
        now = time.time()
        with self._lock:
            if ref in self._state["nodes"]:
                return ref
            live = [
                n["id"] for n in self._state["nodes"].values()
                if n["name"] == ref
                and self._last_seen.get(n["id"], 0)
                and now - self._last_seen[n["id"]] <= _LIVE_WINDOW_S
            ]
            if len(live) == 1:
                return live[0]
            # Ambiguous or all stale: fall back to a unique name match.
            any_match = [
                n["id"] for n in self._state["nodes"].values() if n["name"] == ref
            ]
            return any_match[0] if len(any_match) == 1 else None

    # ------------------------------------------------------------------
    # Recording destination
    # ------------------------------------------------------------------

    def set_destination(self, recording: dict) -> None:
        with self._lock:
            self._state["recording"] = dict(recording or {})
            self._persist()

    def get_destination(self) -> dict:
        with self._lock:
            return dict(self._state.get("recording") or {})

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    def start_session(self, node_id: str, gpu_name: str, params: dict) -> dict:
        """Assign a session to ``node_id``. Raises ValueError on conflicts."""
        with self._cond:
            if node_id not in self._state["nodes"]:
                raise ValueError(f"unknown node {node_id}")
            if self._state["sessions"].get(node_id) is not None:
                raise ValueError(f"node {node_id} already has an active session")
            gpu = self._state["gpus"].get(gpu_name)
            if gpu is None:
                raise ValueError(
                    f"unknown gpu {gpu_name!r} (register with `interlatent gpu add`)"
                )
            # One session per GPU box. The box enforces this too — it is the
            # trust boundary — but rejecting here gives a clean error before
            # the node ever dials.
            for other in self._state["sessions"].values():
                if other and other.get("gpu") == gpu_name:
                    raise ValueError(
                        f"gpu {gpu_name!r} is already serving session "
                        f"{other.get('id')}; one session per box. Stop it first."
                    )
            warm = (gpu.get("warm_policy") or "").strip()
            requested = params["policy"]
            if warm and requested != warm and not params.get("confirm_policy_change"):
                raise PolicyChangeError(gpu_name, warm, requested)

            env_slug = params.get("env_slug") or "default"
            env = self._get_environment_locked(env_slug) or {}
            task = params.get("task") or env_slug
            fps = float(params.get("fps") or 30.0)
            route = routing.resolve(
                routing.make_descriptor(gpu["url"], method=gpu.get("method", "direct"))
            )
            session = {
                "id": "sess_" + secrets.token_hex(8),
                "assignment_type": "inference_session",
                "policy_uri": requested,
                "policy_backend": params.get("backend") or "lerobot",
                "task": task,
                "task_id": params.get("task_id"),
                "fps": fps,
                "chunk_size": int(params.get("chunk_size") or 50),
                "action_dim": int(
                    params.get("action_dim") or env.get("action_dim") or 6
                ),
                # Sequential rather than overlapping chunking. A per-policy
                # fact, and a session pins one policy, so this is its home.
                "synchronous": bool(params.get("synchronous")),
                "drtc_endpoint": route["address"],
                "route": route,
                "environment_id": env.get("id") or env_slug,
                "collection_context": {
                    "env_slug": env_slug, "task": task, "fps": fps,
                },
                "recording": dict(self._state.get("recording") or {}),
                "node_id": node_id,
                "gpu": gpu_name,
                "status": "running",
            }
            self._state["sessions"][node_id] = session
            if requested and requested != warm:
                gpu["warm_policy"] = requested
            self._persist()
            self._cond.notify_all()
        _LOG.info(
            "Started session %s on node %s (gpu=%s policy=%s)",
            session["id"], node_id, gpu_name, session["policy_uri"],
        )
        return session

    def stop_session(self, session_id: str) -> bool:
        """Unassign — never kill.

        The node's own convergence then runs ``_converge(None)`` ->
        ``client.close()`` -> gRPC ``CloseSession``, which is the *only*
        trigger for the box's dataset build; the box's idle-GC discards any
        recording whose session was never closed.
        """
        with self._cond:
            for nid, sess in list(self._state["sessions"].items()):
                if sess and sess.get("id") == session_id:
                    self._state["sessions"][nid] = None
                    self._persist()
                    self._cond.notify_all()
                    _LOG.info(
                        "Stopped session %s (node %s unassigned)", session_id, nid
                    )
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

    def find_session(self, session_id: str) -> Optional[dict]:
        with self._lock:
            for sess in self._state["sessions"].values():
                if sess and sess.get("id") == session_id:
                    return dict(sess)
        return None

    # ------------------------------------------------------------------
    # Teleop
    # ------------------------------------------------------------------

    def mint_teleop_token(self, session_id: str, role: str) -> Optional[dict]:
        """A token for one (session, role), or None if the session is unknown.

        Deliberately long-lived and *not* expiry-checked at the relay: neither
        endpoint ever reads ``expires_at``, and the browser mints exactly once
        per overlay open, so expiring a paired session would kill a live VR
        session with no way for it to recover.
        """
        if role not in ("node", "browser"):
            raise ValueError(f"role must be node|browser, got {role!r}")
        with self._lock:
            known = any(
                s and s.get("id") == session_id
                for s in self._state["sessions"].values()
            ) or session_id in self._state["recordings"]
            if not known:
                return None
            token = "iltel_" + secrets.token_hex(24)
            self._state.setdefault("teleop_tokens", {})[auth.hash_key(token)] = {
                "session_id": session_id,
                "role": role,
                "created": time.time(),
            }
            self._persist()
        return {"token": token, "session_id": session_id, "role": role}

    def verify_teleop_token(
        self, token: str, session_id: str, role: str
    ) -> tuple[bool, str]:
        """The relay's admission check. Returns ``(ok, reason)``."""
        if not token:
            return False, "no token"
        with self._lock:
            record = (self._state.get("teleop_tokens") or {}).get(
                auth.hash_key(token)
            )
        if record is None:
            return False, "unknown token"
        if record.get("session_id") != session_id:
            return False, "token is for a different session"
        if record.get("role") != role:
            return False, "token is for a different role"
        return True, ""

    def revoke_teleop_tokens(self, session_id: str) -> None:
        with self._lock:
            tokens = self._state.get("teleop_tokens") or {}
            for digest, rec in list(tokens.items()):
                if rec.get("session_id") == session_id:
                    tokens.pop(digest, None)
            self._persist()

    def create_teleop_recording(self, payload: dict) -> dict:
        """A policy-less VR demonstration, assigned to a node like a session."""
        node_ref = (payload.get("node_id") or payload.get("node") or "").strip()
        node_id = self.resolve_node(node_ref)
        if not node_id:
            raise ValueError(f"unknown or ambiguous node {node_ref!r}")
        env_slug = (payload.get("environment_id")
                    or payload.get("environment") or "default")
        env = self.get_environment(env_slug) or {}
        with self._cond:
            if self._state["sessions"].get(node_id) is not None:
                raise ValueError(f"node {node_id} already has an active session")
            recording = {
                "id": "trec_" + secrets.token_hex(8),
                "assignment_type": "teleop_recording",
                "status": "running",
                "node_id": node_id,
                "environment_id": env.get("id") or env_slug,
                "collection_context": {"env_slug": env_slug},
                "task": payload.get("task") or "",
                "robot_kind": payload.get("robot_kind") or "",
                "recording": dict(self._state.get("recording") or {}),
                "policy_uri": "",
            }
            self._state["recordings"][recording["id"]] = recording
            self._state["sessions"][node_id] = recording
            self._persist()
            self._cond.notify_all()
        return recording

    def stop_teleop_recording(self, recording_id: str) -> bool:
        with self._cond:
            recording = self._state["recordings"].get(recording_id)
            if not recording:
                return False
            recording["status"] = "stopped"
            node_id = recording.get("node_id")
            if node_id and (self._state["sessions"].get(node_id) or {}).get(
                "id"
            ) == recording_id:
                # Unassign, same as an inference session: the node's own
                # teardown is what closes the recording out.
                self._state["sessions"][node_id] = None
            self._persist()
            self._cond.notify_all()
        return True

    def list_teleop_recordings(self) -> list[dict]:
        with self._lock:
            return list(self._state["recordings"].values())

    def has_active_sessions(self) -> bool:
        with self._lock:
            return any(s for s in self._state["sessions"].values() if s)
