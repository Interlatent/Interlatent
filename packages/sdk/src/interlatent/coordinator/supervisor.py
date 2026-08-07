"""Lifecycle for a locally-run coordinator: ``up``, ``down``, ``status``, ``logs``.

Background daemon rather than a foreground process, so an operator is not
holding a terminal hostage for the life of a robot deployment. Runtime facts
(pid, port, log path) live in ``~/.interlatent/coordinator.runtime.json``;
persistent control-plane state is separate, in ``coordinator.json``.

``down`` refuses while a session is active unless ``--force``, and ``--force``
unassigns and then *waits*. That is not politeness: the node's teardown is what
sends gRPC ``CloseSession``, which is the only trigger for the box's dataset
build, and the box discards any recording whose session never closed. Killing
the coordinator out from under a live session loses the episode.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

HOME = Path(
    os.environ.get("INTERLATENT_HOME", "~/.interlatent")
).expanduser()
RUNTIME_PATH = HOME / "coordinator.runtime.json"
LOG_PATH = HOME / "coordinator.log"
STATE_PATH = Path(
    os.environ.get("INTERLATENT_COORDINATOR_STATE", str(HOME / "coordinator.json"))
).expanduser()

DEFAULT_PORT = 8900


def read_runtime() -> Optional[dict]:
    try:
        return json.loads(RUNTIME_PATH.read_text())
    except (OSError, ValueError):
        return None


def write_runtime(data: dict) -> None:
    RUNTIME_PATH.parent.mkdir(parents=True, exist_ok=True)
    RUNTIME_PATH.write_text(json.dumps(data, indent=2))


def clear_runtime() -> None:
    try:
        RUNTIME_PATH.unlink()
    except OSError:
        pass


def pid_alive(pid: int) -> bool:
    if not pid or pid < 1:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _request(
    url: str, key: str, method: str = "GET", body: Any = None, timeout: float = 5.0
) -> tuple[int, Any]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("x-api-key", key)
    if data is not None:
        req.add_header("content-type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return resp.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except Exception:
            return e.code, {}
    except (urllib.error.URLError, OSError, ValueError):
        return 0, {}


def ping(base: str, key: str) -> bool:
    status, _ = _request(f"{base}/api/v1/capabilities", key, timeout=2.0)
    return status == 200


def active_sessions(base: str, key: str) -> list[dict]:
    status, body = _request(f"{base}/api/v1/inference/sessions", key)
    if status != 200:
        return []
    return body.get("sessions") or []


def local_base(port: int) -> str:
    return f"http://127.0.0.1:{port}"


def start(port: int, host: str = "0.0.0.0", state_path: Path = None) -> tuple[int, str]:
    """Spawn the coordinator in the background. Returns ``(exit_code, message)``."""
    from . import auth

    key, created = auth.ensure_operator_key()
    rt = read_runtime()
    if rt and pid_alive(rt.get("pid", -1)) and ping(local_base(rt["port"]), key):
        return 0, (
            f"Coordinator already running (pid {rt['pid']}, port {rt['port']})."
        )

    HOME.mkdir(parents=True, exist_ok=True)
    state = Path(state_path) if state_path else STATE_PATH
    cmd = [
        sys.executable, "-m", "interlatent.coordinator",
        "--host", host, "--port", str(port), "--state", str(state),
    ]
    logf = open(LOG_PATH, "a")
    proc = subprocess.Popen(
        cmd, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True
    )
    write_runtime({
        "pid": proc.pid, "host": host, "port": port,
        "log": str(LOG_PATH), "state": str(state),
    })

    base = local_base(port)
    for _ in range(50):  # up to ~5 s
        if proc.poll() is not None:
            return 1, f"coordinator exited immediately; see {LOG_PATH}"
        if ping(base, key):
            break
        time.sleep(0.1)
    else:
        return 1, f"coordinator did not come up; see {LOG_PATH}"

    lines = [f"✓ Coordinator up (pid {proc.pid}) on port {port}."]
    if created:
        lines.append(f"  Operator key: {key}")
        lines.append(f"  Stored 0600 at {auth.default_operator_key_path()}.")
    else:
        lines.append(f"  Operator key: {auth.default_operator_key_path()}")
    status, body = _request(f"{base}/api/v1/coordinator/recording", key)
    dest = (body or {}).get("recording") or {}
    lines.append(
        "  Recording destination: "
        + (json.dumps(dest) if dest else
           "none — sessions will run but NOT be saved (`interlatent config`)")
    )
    lines.append(
        f"  Point a node at it: interlatent-node pair --name <name> "
        f"--coordinator http://<this-host>:{port} --api-key {key[:9]}…"
    )
    return 0, "\n".join(lines)


def stop(force: bool, grace: float) -> tuple[int, str]:
    from . import auth

    rt = read_runtime()
    if not rt:
        return 0, "Coordinator not running."
    key = auth.load_operator_key() or ""
    base = local_base(rt["port"])

    if ping(base, key):
        sessions = active_sessions(base, key)
        if sessions and not force:
            detail = "\n".join(
                f"  {s['id']}  node={s.get('node_id')}  policy={s.get('policy_uri')}"
                for s in sessions
            )
            return 2, (
                "active sessions — stop them first, or use --force:\n" + detail
            )
        if sessions and force:
            out = [f"Stopping {len(sessions)} active session(s) gracefully..."]
            for s in sessions:
                _request(
                    f"{base}/api/v1/inference/sessions/{s['id']}", key, method="DELETE"
                )
            # Wait for the nodes to converge to idle. This is the window in
            # which CloseSession -> dataset build -> publish happens.
            deadline = time.time() + grace
            while time.time() < deadline and active_sessions(base, key):
                time.sleep(0.25)
            if active_sessions(base, key):
                out.append(
                    f"warning: sessions still assigned after {grace:.0f}s; "
                    "shutting down anyway"
                )
            print("\n".join(out))

    pid = rt.get("pid", -1)
    if pid_alive(pid):
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
        deadline = time.time() + 10.0
        while time.time() < deadline and pid_alive(pid):
            time.sleep(0.1)
        if pid_alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
    clear_runtime()
    return 0, "✓ Coordinator down."


def status() -> tuple[int, str]:
    from . import auth

    rt = read_runtime()
    if not rt:
        return 1, "Coordinator not running."
    key = auth.load_operator_key() or ""
    base = local_base(rt["port"])
    alive = pid_alive(rt.get("pid", -1))
    reachable = ping(base, key)
    lines = [
        f"pid       {rt.get('pid')} ({'alive' if alive else 'DEAD'})",
        f"address   http://{rt.get('host')}:{rt.get('port')}"
        f" ({'reachable' if reachable else 'NOT reachable'})",
        f"state     {rt.get('state')}",
        f"log       {rt.get('log')}",
    ]
    if reachable:
        _, nodes = _request(f"{base}/api/v1/nodes", key)
        _, gpus = _request(f"{base}/api/v1/gpus", key)
        sessions = active_sessions(base, key)
        lines.append(
            f"inventory {len(nodes.get('nodes') or [])} node(s), "
            f"{len(gpus.get('gpus') or [])} gpu(s), "
            f"{len(sessions)} active session(s)"
        )
    return (0 if alive and reachable else 1), "\n".join(lines)


def logs(follow: bool, lines: int) -> int:
    rt = read_runtime()
    path = Path(rt["log"]) if rt and rt.get("log") else LOG_PATH
    if not path.exists():
        print(f"no log at {path}", file=sys.stderr)
        return 1
    if follow:
        return subprocess.call(["tail", "-f", str(path)])
    return subprocess.call(["tail", "-n", str(lines), str(path)])
