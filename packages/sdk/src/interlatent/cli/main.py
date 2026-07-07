"""`interlatent` CLI: a thin command-line client for the Interlatent dashboard.

Inference is provisioned and run in the cloud through the Interlatent
dashboard. This CLI is a small utility view of that dashboard — it resolves
the caller from an Interlatent API key (``ilat_…``) and lets you, from a
terminal:

    interlatent gpus ls                 # GPU boxes available to your account
    interlatent nodes ls                # robot nodes paired to your account
    interlatent env create --slug ...   # create an environment to collect into
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
# gpus
# ----------------------------------------------------------------------


def cmd_gpus(args: argparse.Namespace) -> int:
    client = _make_client(args)
    # GET /api/v1/gpus -> [{id, name, status, gpu, region, ...}]
    # (or {"gpus": [...]}). A flat projection of the user's ComputeBox rows.
    payload = client.request("GET", "/api/v1/gpus")
    rows = _rows(payload, "gpus")
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    _print_table(
        rows,
        [("ID", "id"), ("NAME", "name"), ("STATUS", "status"), ("GPU", "gpu"),
         ("REGION", "region")],
        empty="(no GPUs available)",
    )
    return 0


# ----------------------------------------------------------------------
# nodes
# ----------------------------------------------------------------------


def cmd_nodes(args: argparse.Namespace) -> int:
    client = _make_client(args)
    # GET /api/v1/nodes -> [{id, name, status, robot_type, ...}]
    # (or {"nodes": [...]}). Same resource the node daemon pairs against.
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
        # POST /api/v1/inference/sessions/ with the node + pod + policy and
        # optional control knobs; returns the created session object ({id}).
        body: dict[str, Any] = {
            "node": args.node,
            # The backend session body field is "pod" (its word for a GPU box);
            # the CLI flag is --gpu for symmetry with `interlatent gpus ls`.
            "pod": args.gpu,
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
        print(f"✓ Started session {sid} (node={args.node}, gpu={args.gpu}, "
              f"policy={args.policy})")
        return 0
    if args.session_cmd == "stop":
        # DELETE /api/v1/inference/sessions/{id} — any 2xx is success; the
        # node converges to idle on its next poll.
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
# env
# ----------------------------------------------------------------------


def cmd_env(args: argparse.Namespace) -> int:
    client = _make_client(args)
    if args.env_cmd == "create":
        # POST /api/v1/environments -> the created environment config.
        # `session start` requires the env to already exist; this is how you
        # create one from the terminal (the dashboard is the other way).
        body: dict[str, Any] = {
            "slug": args.slug,
            "display_name": args.display_name or args.slug,
        }
        for key, val in (("robot_type", args.robot_type),
                         ("task_description", args.task)):
            if val not in (None, ""):
                body[key] = val
        resp = client.request("POST", "/api/v1/environments", json_body=body)
        if isinstance(resp, dict):
            slug = resp.get("slug", args.slug)
            env_id = resp.get("environment_id") or resp.get("id") or ""
        else:
            slug, env_id = args.slug, ""
        print(f"✓ Created environment {slug}" + (f" ({env_id})" if env_id else ""))
        return 0
    return 1


# ----------------------------------------------------------------------
# behavior (offline — no API key, no cloud)
# ----------------------------------------------------------------------


def _robot_arg_dict(pairs: "list[str] | None") -> dict[str, str]:
    """Parse repeated ``--robot-arg key=value`` flags into a dict."""
    out: dict[str, str] = {}
    for item in pairs or []:
        if "=" not in item:
            raise SystemExit(f"error: --robot-arg expects key=value, got: {item!r}")
        k, _, v = item.partition("=")
        out[k.strip()] = v.strip()
    return out


def cmd_behavior(args: argparse.Namespace) -> int:
    """List, validate, or run named behaviors — fully offline (no API key)."""
    # Imported lazily so `interlatent gpus ls` etc. never pay the behaviors import.
    from ..behaviors.registry import BehaviorRegistry
    from ..behaviors.schema import BehaviorError

    if args.behavior_cmd == "ls":
        try:
            reg = BehaviorRegistry.for_robot(args.robot)
        except BehaviorError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        rows = [{"name": n, "type": t, "duration": d} for n, t, d in reg.summaries()]
        if args.json:
            print(json.dumps(rows, indent=2))
            return 0
        _print_table(
            rows,
            [("NAME", "name"), ("TYPE", "type"), ("DURATION", "duration")],
            empty="(no behaviors)",
        )
        return 0

    if args.behavior_cmd == "validate":
        try:
            # Building the registry validates the built-ins; the explicit path (if any)
            # is validated as it loads and overrides by name.
            reg = BehaviorRegistry.for_robot(args.robot, explicit=args.path)
        except BehaviorError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        where = f" + {args.path}" if args.path else ""
        print(f"✓ behaviors valid for {args.robot!r}{where}: {', '.join(reg.names())}")
        return 0

    # run
    from ..robot import Robot

    try:
        robot = Robot(
            args.robot,
            port=args.port,
            behaviors=args.behaviors,
            robot_arg=_robot_arg_dict(args.robot_arg),
            control_hz=args.control_hz,
            force=args.force,
        )
    except BehaviorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 — clean message, not a traceback
        print(f"error: could not open {args.robot!r}: {exc}", file=sys.stderr)
        return 1
    try:
        result = robot.act(args.name, speed=args.speed)
        worst = max(result.joint_error.items(), key=lambda kv: abs(kv[1]), default=(None, 0.0))
        status = "reached" if result.reached else f"aborted ({result.reason})"
        print(
            f"{args.name}: {status} in {result.elapsed:.2f}s"
            + (f"; worst joint error {worst[0]}={worst[1]:+.3f}" if worst[0] else "")
        )
        return 0 if result.reached else 1
    except BehaviorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        robot.close()


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
        "your GPU boxes and robot nodes, and start/stop cloud inference sessions.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_gpus = sub.add_parser("gpus", help="List GPU boxes available to your account.")
    gpus_sub = p_gpus.add_subparsers(dest="gpus_cmd", required=True)
    p_gpus_ls = gpus_sub.add_parser("ls", help="List GPUs.")
    _add_auth_flags(p_gpus_ls)
    p_gpus.set_defaults(func=cmd_gpus)

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
    s_start.add_argument("--gpu", required=True, help="GPU box name or id.")
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

    # behavior — offline named moves/trajectories (no API key).
    p_beh = sub.add_parser(
        "behavior", help="List/validate/run named behaviors offline (no cloud, no API key)."
    )
    beh_sub = p_beh.add_subparsers(dest="behavior_cmd", required=True)

    b_ls = beh_sub.add_parser("ls", help="List available behaviors for a robot.")
    b_ls.add_argument("--robot", default="so101", help="Robot kind (default: so101).")
    b_ls.add_argument("--json", action="store_true", help="Emit raw JSON instead of a table.")

    b_val = beh_sub.add_parser(
        "validate", help="Validate a behaviors TOML against a robot profile (no hardware)."
    )
    b_val.add_argument("path", nargs="?", default=None, help="Behaviors TOML to validate.")
    b_val.add_argument("--robot", default="so101", help="Robot kind (default: so101).")

    b_run = beh_sub.add_parser("run", help="Run a named behavior on a connected robot.")
    b_run.add_argument("name", help="Behavior name (e.g. home, hello).")
    b_run.add_argument("--robot", default="so101", help="Robot kind (default: so101).")
    b_run.add_argument("--port", default=None, help="Serial port (e.g. /dev/ttyACM0).")
    b_run.add_argument("--speed", type=float, default=1.0, help="Time-scale factor (default: 1.0).")
    b_run.add_argument("--behaviors", default=None, help="Extra behaviors TOML to load.")
    b_run.add_argument(
        "--robot-arg", action="append", metavar="key=value",
        help="Extra key=value passed to the robot config (repeatable).",
    )
    b_run.add_argument("--control-hz", type=float, default=30.0, help="Control rate (default: 30).")
    b_run.add_argument(
        "--force", action="store_true",
        help="Override bus arbitration (dangerous — can corrupt a live node session).",
    )
    p_beh.set_defaults(func=cmd_behavior)

    p_env = sub.add_parser("env", help="Manage environments (data collections).")
    env_sub = p_env.add_subparsers(dest="env_cmd", required=True)
    e_create = env_sub.add_parser("create", help="Create an environment.")
    e_create.add_argument("--slug", required=True, help="Environment slug/name.")
    e_create.add_argument("--display-name", default="",
                          help="Human-readable name (defaults to the slug).")
    e_create.add_argument("--robot-type", default="", help="Robot type, e.g. so101.")
    e_create.add_argument("--task", default="", help="Task description.")
    _add_auth_flags(e_create)
    p_env.set_defaults(func=cmd_env)

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
