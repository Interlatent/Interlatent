"""`interlatent` CLI: a thin command-line client for the Interlatent dashboard.

Inference is provisioned and run in the cloud through the Interlatent
dashboard. This CLI is a small utility view of that dashboard — it resolves
the caller from an Interlatent API key (``ilat_…``) and lets you, from a
terminal:

    interlatent pods ls                 # GPU pods available to your account
    interlatent nodes ls                # robot nodes paired to your account
    interlatent session ls              # active inference sessions
    interlatent session start ...       # assign a node+pod+policy session
    interlatent session stop <id>       # cancel a session

Auth: pass ``--api-key`` or set ``INTERLATENT_API_KEY``. The base URL defaults
to https://interlatent.com (override with ``--api-base`` / ``INTERLATENT_API_BASE``).

The robot-side daemon (``interlatent-node``) polls the dashboard directly and
needs no coordinator — this CLI never sits in the inference data path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from .._exceptions import APIError, AuthenticationError, NotFoundError
from .._http import HTTPClient

DEFAULT_API_BASE = os.environ.get(
    "INTERLATENT_API_BASE", "https://interlatent.com"
).rstrip("/")


# ----------------------------------------------------------------------
# Client construction
# ----------------------------------------------------------------------


def _make_client(args: argparse.Namespace) -> HTTPClient:
    """Build an authenticated dashboard client or exit with a clear error."""
    api_key = getattr(args, "api_key", None) or os.environ.get("INTERLATENT_API_KEY", "")
    if not api_key:
        print(
            "error: no Interlatent API key. Pass --api-key or set "
            "INTERLATENT_API_KEY (get one from the dashboard).",
            file=sys.stderr,
        )
        raise SystemExit(2)
    base = (getattr(args, "api_base", None) or DEFAULT_API_BASE).rstrip("/")
    return HTTPClient(base_url=base, api_key=api_key)


def _rows(payload: Any, key: str) -> list[dict]:
    """Normalize a list response.

    The dashboard may return either a bare JSON array or an object wrapping
    the array under ``key`` (e.g. ``{"pods": [...]}``). Accept both.
    """
    if isinstance(payload, dict):
        val = payload.get(key)
        return val if isinstance(val, list) else []
    return payload if isinstance(payload, list) else []


def _print_table(rows: list[dict], columns: list[tuple[str, str]], empty: str) -> None:
    """Print ``rows`` as a simple aligned table.

    ``columns`` is a list of (header, dict-key) pairs.
    """
    if not rows:
        print(empty)
        return
    headers = [h for h, _ in columns]
    cells = [[str(r.get(k, "")) for _, k in columns] for r in rows]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in cells)) for i in range(len(columns))
    ]
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    for row in cells:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(columns))))


# ----------------------------------------------------------------------
# pods
# ----------------------------------------------------------------------


def cmd_pods(args: argparse.Namespace) -> int:
    client = _make_client(args)
    # TODO(api): confirm the pods listing endpoint + response shape with the
    # dashboard backend. Expected: GET /api/v1/pods -> [{id, name, status,
    # gpu, region, ...}] (or {"pods": [...]}).
    payload = client.request("GET", "/api/v1/pods")
    rows = _rows(payload, "pods")
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    _print_table(
        rows,
        [("ID", "id"), ("NAME", "name"), ("STATUS", "status"), ("GPU", "gpu"),
         ("REGION", "region")],
        empty="(no pods available)",
    )
    return 0


# ----------------------------------------------------------------------
# nodes
# ----------------------------------------------------------------------


def cmd_nodes(args: argparse.Namespace) -> int:
    client = _make_client(args)
    # The nodes API already exists (the node daemon pairs/heartbeats here).
    # TODO(api): confirm the GET listing shape returned to an api-key caller.
    payload = client.request("GET", "/api/v1/nodes")
    rows = _rows(payload, "nodes")
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    _print_table(
        rows,
        [("ID", "id"), ("NAME", "name"), ("STATUS", "status"),
         ("ROBOT", "robot_type")],
        empty="(no nodes paired)",
    )
    return 0


# ----------------------------------------------------------------------
# session
# ----------------------------------------------------------------------

_SESSIONS_PATH = "/api/v1/inference/sessions/"


def cmd_session(args: argparse.Namespace) -> int:
    client = _make_client(args)
    if args.session_cmd == "start":
        # TODO(api): confirm the create-session payload + response. Expected:
        # POST /api/v1/inference/sessions/ with the node + pod + policy and
        # optional control knobs; returns the created session object.
        body: dict[str, Any] = {
            "node": args.node,
            "pod": args.pod,
            "policy": args.policy,
            "backend": args.backend,
        }
        for key, val in (("task", args.task), ("env_slug", args.env_slug),
                         ("fps", args.fps), ("chunk_size", args.chunk_size),
                         ("action_dim", args.action_dim)):
            if val not in (None, ""):
                body[key] = val
        resp = client.request("POST", _SESSIONS_PATH, json_body=body)
        sess = resp.get("session", resp) if isinstance(resp, dict) else resp
        sid = sess.get("id") if isinstance(sess, dict) else sess
        print(f"✓ Started session {sid} (node={args.node}, pod={args.pod}, "
              f"policy={args.policy})")
        return 0
    if args.session_cmd == "stop":
        # TODO(api): confirm the cancel endpoint (DELETE vs POST .../cancel).
        client.request("DELETE", f"{_SESSIONS_PATH}{args.session_id}")
        print(f"✓ Stopped session {args.session_id}")
        return 0
    # ls
    payload = client.request("GET", _SESSIONS_PATH)
    rows = _rows(payload, "sessions")
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    _print_table(
        rows,
        [("ID", "id"), ("NODE", "node"), ("POD", "pod"), ("POLICY", "policy_uri"),
         ("STATUS", "status")],
        empty="(no active sessions)",
    )
    return 0


# ----------------------------------------------------------------------
# argparse wiring
# ----------------------------------------------------------------------


def _add_auth_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--api-key", default=None,
                   help="Interlatent API key (ilat_…). Falls back to INTERLATENT_API_KEY.")
    p.add_argument("--api-base", default=DEFAULT_API_BASE,
                   help=f"Dashboard base URL (default: {DEFAULT_API_BASE}).")
    p.add_argument("--json", action="store_true", help="Emit raw JSON instead of a table.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="interlatent",
        description="Command-line client for the Interlatent dashboard: list "
        "your GPU pods and robot nodes, and start/stop cloud inference sessions.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_pods = sub.add_parser("pods", help="List GPU pods available to your account.")
    pods_sub = p_pods.add_subparsers(dest="pods_cmd", required=True)
    p_pods_ls = pods_sub.add_parser("ls", help="List pods.")
    _add_auth_flags(p_pods_ls)
    p_pods.set_defaults(func=cmd_pods)

    p_nodes = sub.add_parser("nodes", help="List robot nodes paired to your account.")
    nodes_sub = p_nodes.add_subparsers(dest="nodes_cmd", required=True)
    p_nodes_ls = nodes_sub.add_parser("ls", help="List nodes.")
    _add_auth_flags(p_nodes_ls)
    p_nodes.set_defaults(func=cmd_nodes)

    p_sess = sub.add_parser("session", help="List/start/stop inference sessions.")
    sess_sub = p_sess.add_subparsers(dest="session_cmd", required=True)

    s_ls = sess_sub.add_parser("ls", help="List active sessions.")
    _add_auth_flags(s_ls)

    s_start = sess_sub.add_parser("start", help="Start an inference session.")
    s_start.add_argument("--node", required=True, help="Node name or id.")
    s_start.add_argument("--pod", required=True, help="GPU pod name or id.")
    s_start.add_argument("--policy", required=True, help="Policy URI.")
    s_start.add_argument("--backend", default="lerobot")
    s_start.add_argument("--task", default="")
    s_start.add_argument("--env-slug", default="")
    s_start.add_argument("--fps", type=float, default=None)
    s_start.add_argument("--chunk-size", type=int, default=None)
    s_start.add_argument("--action-dim", type=int, default=None)
    _add_auth_flags(s_start)

    s_stop = sess_sub.add_parser("stop", help="Stop (cancel) a session.")
    s_stop.add_argument("session_id")
    _add_auth_flags(s_stop)

    p_sess.set_defaults(func=cmd_session)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except AuthenticationError:
        print("error: authentication failed — check your INTERLATENT_API_KEY.",
              file=sys.stderr)
        return 1
    except NotFoundError as e:
        print(f"error: not found ({e}).", file=sys.stderr)
        return 1
    except APIError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
