"""`interlatent` CLI: coordinator daemon management + thin admin client."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .client import CoordinatorClient, CoordinatorError

_DIR = Path(os.environ.get("INTERLATENT_HOME", "~/.interlatent")).expanduser()
_RUNTIME = _DIR / "coordinator.runtime.json"
_LOG = _DIR / "coordinator.log"
_STATE = _DIR / "coordinator.json"


# ----------------------------------------------------------------------
# Runtime file + client helpers
# ----------------------------------------------------------------------


def _read_runtime() -> dict | None:
    if _RUNTIME.exists():
        try:
            return json.loads(_RUNTIME.read_text())
        except Exception:
            return None
    return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _client_or_exit() -> CoordinatorClient:
    rt = _read_runtime()
    if not rt:
        print("error: coordinator not running. Start it with `interlatent up`.",
              file=sys.stderr)
        raise SystemExit(2)
    return CoordinatorClient(f"http://127.0.0.1:{rt['port']}")


# ----------------------------------------------------------------------
# up / down / status / logs
# ----------------------------------------------------------------------


def cmd_up(args: argparse.Namespace) -> int:
    rt = _read_runtime()
    if rt and _pid_alive(rt.get("pid", -1)) and CoordinatorClient(
        f"http://127.0.0.1:{rt['port']}"
    ).ping():
        print(f"Coordinator already running (pid {rt['pid']}, port {rt['port']}).")
        return 0

    _DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "interlatent.coordinator.server",
        "--host", "0.0.0.0", "--port", str(args.port), "--state", str(_STATE),
    ]
    if args.output_dir:
        cmd += ["--output-dir", args.output_dir]
    if args.s3_uri:
        cmd += ["--s3-uri", args.s3_uri]
        for flag, val in (
            ("--s3-endpoint-url", args.s3_endpoint_url),
            ("--s3-access-key", args.s3_access_key),
            ("--s3-secret-key", args.s3_secret_key),
            ("--s3-region", args.s3_region),
        ):
            if val:
                cmd += [flag, val]

    logf = open(_LOG, "a")
    proc = subprocess.Popen(
        cmd, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True
    )
    _RUNTIME.write_text(json.dumps(
        {"pid": proc.pid, "host": "0.0.0.0", "port": args.port,
         "log": str(_LOG), "state": str(_STATE)}
    ))

    client = CoordinatorClient(f"http://127.0.0.1:{args.port}")
    for _ in range(50):  # up to ~5s
        if proc.poll() is not None:
            print(f"error: coordinator exited immediately; see {_LOG}", file=sys.stderr)
            return 1
        if client.ping():
            break
        time.sleep(0.1)
    else:
        print(f"error: coordinator did not come up; see {_LOG}", file=sys.stderr)
        return 1

    print(f"✓ Coordinator up (pid {proc.pid}) on port {args.port}.")
    dest = client.get_destination()
    print(f"  Recording destination: {dest or 'none (inference-only — sessions will NOT be saved)'}")
    print(f"  Point nodes at it: interlatent-node pair --name <name> "
          f"--api-base http://<this-host>:{args.port}")
    return 0


def cmd_down(args: argparse.Namespace) -> int:
    rt = _read_runtime()
    if not rt:
        print("Coordinator not running.")
        return 0
    client = CoordinatorClient(f"http://127.0.0.1:{rt['port']}")

    if client.ping():
        sessions = client.list_sessions()
        if sessions and not args.force:
            print("error: active sessions — stop them first or use --force:", file=sys.stderr)
            for s in sessions:
                print(f"  {s['id']}  node={s.get('node_id')}  policy={s.get('policy_uri')}",
                      file=sys.stderr)
            return 2
        if sessions and args.force:
            print(f"Stopping {len(sessions)} active session(s) gracefully...")
            for s in sessions:
                try:
                    client.stop_session(s["id"])
                except CoordinatorError:
                    pass
            # Wait for nodes to converge to idle (CloseSession -> publish).
            deadline = time.time() + args.grace
            while time.time() < deadline and client.list_sessions():
                time.sleep(0.5)
            still = client.list_sessions()
            if still:
                print(f"warning: {len(still)} session(s) did not confirm teardown within "
                      f"{args.grace}s; shutting down anyway.", file=sys.stderr)

    pid = rt.get("pid", -1)
    if _pid_alive(pid):
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
        for _ in range(50):
            if not _pid_alive(pid):
                break
            time.sleep(0.1)
    _RUNTIME.unlink(missing_ok=True)
    print("✓ Coordinator stopped.")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    rt = _read_runtime()
    if not rt:
        print("Coordinator: not running.")
        return 0
    alive = _pid_alive(rt.get("pid", -1))
    client = CoordinatorClient(f"http://127.0.0.1:{rt['port']}")
    reachable = client.ping()
    print(f"Coordinator: pid {rt['pid']} ({'alive' if alive else 'dead'}), "
          f"port {rt['port']} ({'reachable' if reachable else 'unreachable'})")
    if reachable:
        print(f"  destination: {client.get_destination() or 'none'}")
        print(f"  nodes: {len(client.list_nodes())}  gpus: {len(client.list_gpus())}  "
              f"active sessions: {len(client.list_sessions())}")
    return 0


def cmd_logs(args: argparse.Namespace) -> int:
    if not _LOG.exists():
        print("No coordinator log yet.")
        return 0
    lines = _LOG.read_text().splitlines()
    for line in lines[-args.n:]:
        print(line)
    return 0


# ----------------------------------------------------------------------
# gpu / node / session
# ----------------------------------------------------------------------


def cmd_gpu(args: argparse.Namespace) -> int:
    client = _client_or_exit()
    if args.gpu_cmd == "add":
        client.add_gpu(args.name, args.url, args.method)
        suffix = "" if args.method == "direct" else f" (method={args.method})"
        print(f"✓ Registered gpu {args.name} -> {args.url}{suffix}")
    elif args.gpu_cmd == "rm":
        client.remove_gpu(args.name)
        print(f"✓ Removed gpu {args.name}")
    else:  # ls
        gpus = client.list_gpus()
        if not gpus:
            print("(no gpus registered)")
        for g in gpus:
            method = g.get("method", "direct")
            suffix = "" if method == "direct" else f"  [{method}]"
            print(f"{g['name']:20}  {g['url']}{suffix}")
    return 0


def cmd_node(args: argparse.Namespace) -> int:
    client = _client_or_exit()
    if args.node_cmd == "rm":
        client.remove_node(args.node_id)
        print(f"✓ Removed node {args.node_id}")
    else:  # ls
        nodes = client.list_nodes()
        if not nodes:
            print("(no nodes paired)")
        for n in nodes:
            age = n.get("last_seen_age_s")
            age_s = f"{age:.0f}s ago" if age is not None else "never"
            flags = []
            if n.get("live"):
                flags.append("live")
            if n.get("busy"):
                flags.append("busy")
            print(f"{n['id']}  name={n['name']:16}  seen={age_s:10}  {' '.join(flags)}")
    return 0


def cmd_session(args: argparse.Namespace) -> int:
    client = _client_or_exit()
    if args.session_cmd == "start":
        params = {
            "node": args.node, "gpu": args.gpu, "policy": args.policy,
            "backend": args.backend, "task": args.task, "env_slug": args.env_slug,
            "no_probe": args.no_probe,
        }
        for key, val in (("fps", args.fps), ("chunk_size", args.chunk_size),
                         ("action_dim", args.action_dim)):
            if val is not None:
                params[key] = val
        try:
            resp = client.start_session(params)
        except CoordinatorError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        sess = resp["session"]
        print(f"✓ Started session {sess['id']} on node {sess.get('node_id')} "
              f"(gpu={sess.get('gpu')}, policy={sess['policy_uri']})")
        if resp.get("warning"):
            print(f"  warning: {resp['warning']}", file=sys.stderr)
    elif args.session_cmd == "stop":
        try:
            client.stop_session(args.session_id)
        except CoordinatorError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        print(f"✓ Stopping session {args.session_id} (node unassigns; dataset publishes "
              f"on its next converge).")
    else:  # ls
        sessions = client.list_sessions()
        if not sessions:
            print("(no active sessions)")
        for s in sessions:
            print(f"{s['id']}  node={s.get('node_id')}  gpu={s.get('gpu')}  "
                  f"policy={s['policy_uri']}  task={s.get('task')!r}")
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    client = _client_or_exit()
    if args.output_dir:
        rec = {"output_dir": args.output_dir}
    elif args.s3_uri:
        rec = {"s3_uri": args.s3_uri}
        for k, v in (("s3_endpoint_url", args.s3_endpoint_url),
                     ("s3_access_key", args.s3_access_key),
                     ("s3_secret_key", args.s3_secret_key),
                     ("s3_region", args.s3_region)):
            if v:
                rec[k] = v
    else:
        print(f"destination: {client.get_destination() or 'none'}")
        return 0
    client.set_destination(rec)
    print(f"✓ Recording destination set: {rec}")
    return 0


# ----------------------------------------------------------------------
# argparse wiring
# ----------------------------------------------------------------------


def _add_s3_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--output-dir", default="")
    p.add_argument("--s3-uri", default="")
    p.add_argument("--s3-endpoint-url", default="")
    p.add_argument("--s3-access-key", default="")
    p.add_argument("--s3-secret-key", default="")
    p.add_argument("--s3-region", default="")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="interlatent",
        description="Local coordinator for running VLA inference sessions without "
        "the hosted dashboard.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_up = sub.add_parser("up", help="Start the coordinator (background daemon).")
    p_up.add_argument("--port", type=int, default=8900)
    _add_s3_flags(p_up)
    p_up.set_defaults(func=cmd_up)

    p_down = sub.add_parser("down", help="Stop the coordinator.")
    p_down.add_argument("--force", action="store_true",
                        help="Stop active sessions gracefully, then shut down.")
    p_down.add_argument("--grace", type=float, default=30.0,
                        help="Seconds to wait for sessions to tear down under --force.")
    p_down.set_defaults(func=cmd_down)

    p_status = sub.add_parser("status", help="Show coordinator status.")
    p_status.set_defaults(func=cmd_status)

    p_logs = sub.add_parser("logs", help="Print the coordinator log.")
    p_logs.add_argument("-n", type=int, default=200, help="Lines to show.")
    p_logs.set_defaults(func=cmd_logs)

    p_gpu = sub.add_parser("gpu", help="Manage registered GPU boxes.")
    gpu_sub = p_gpu.add_subparsers(dest="gpu_cmd", required=True)
    g_add = gpu_sub.add_parser("add", help="Register a GPU box URL.")
    g_add.add_argument("name")
    g_add.add_argument("url", help="DRTC endpoint the node dials, e.g. localhost:50051, "
                       "100.x.y.z:50051, or https://…modal.run (any reachable address)")
    g_add.add_argument("--method", default="direct",
                       help="Routing method (default: direct = dial the address as-is). "
                       "Future methods (relay/tunnel) plug in here.")
    gpu_sub.add_parser("ls", help="List GPU boxes.")
    g_rm = gpu_sub.add_parser("rm", help="Forget a GPU box.")
    g_rm.add_argument("name")
    p_gpu.set_defaults(func=cmd_gpu)

    p_node = sub.add_parser("node", help="Inspect paired nodes.")
    node_sub = p_node.add_subparsers(dest="node_cmd", required=True)
    node_sub.add_parser("ls", help="List paired nodes.")
    n_rm = node_sub.add_parser("rm", help="Forget a node.")
    n_rm.add_argument("node_id")
    p_node.set_defaults(func=cmd_node)

    p_sess = sub.add_parser("session", help="Start/stop/list inference sessions.")
    sess_sub = p_sess.add_subparsers(dest="session_cmd", required=True)
    s_start = sess_sub.add_parser("start", help="Assign a session to a node.")
    s_start.add_argument("--node", required=True, help="Node name or id.")
    s_start.add_argument("--gpu", required=True, help="Registered GPU name.")
    s_start.add_argument("--policy", required=True, help="Policy URI.")
    s_start.add_argument("--backend", default="lerobot")
    s_start.add_argument("--task", default="")
    s_start.add_argument("--env-slug", default="")
    s_start.add_argument("--fps", type=float, default=None)
    s_start.add_argument("--chunk-size", type=int, default=None)
    s_start.add_argument("--action-dim", type=int, default=None)
    s_start.add_argument("--no-probe", action="store_true",
                         help="Skip the GPU reachability probe.")
    s_stop = sess_sub.add_parser("stop", help="Stop (unassign) a session.")
    s_stop.add_argument("session_id")
    sess_sub.add_parser("ls", help="List active sessions.")
    p_sess.set_defaults(func=cmd_session)

    p_cfg = sub.add_parser("config", help="Get/set the recording destination.")
    _add_s3_flags(p_cfg)
    p_cfg.set_defaults(func=cmd_config)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except CoordinatorError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
