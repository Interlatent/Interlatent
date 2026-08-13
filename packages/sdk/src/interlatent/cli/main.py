"""`interlatent` CLI: run a coordinator, and drive it from a terminal.

``interlatent up`` starts the coordinator on this machine; every other command
is a client of it. It resolves the caller from the operator key that
coordinator minted (``ilop_…``) and lets you:

    interlatent up                      # start a coordinator on this machine
    interlatent gpus ls                 # GPU boxes registered with it
    interlatent nodes ls                # robot nodes paired to it
    interlatent env create --slug ...   # create an environment to collect into
    interlatent session ls              # active inference sessions
    interlatent session start ...       # assign a node+box+policy session
    interlatent session stop <id>       # cancel a session

Auth: pass ``--api-key`` or set ``INTERLATENT_API_KEY``; with neither, the key
``interlatent up`` wrote to disk is used. There is no default coordinator
address — name yours with ``--coordinator`` or ``INTERLATENT_COORDINATOR``.

The robot-side daemon (``interlatent-node``) polls the coordinator directly —
this CLI never sits in the inference data path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from .._exceptions import APIError, AuthenticationError, NotFoundError
from .._coordinator import CoordinatorNotConfigured, resolve
from .._http import HTTPClient


def _local_coordinator() -> str | None:
    """The coordinator running on *this* machine, if one is.

    ``interlatent up`` spawns a daemon and records its port in
    ``coordinator.runtime.json``, but it cannot export an env var back into
    the shell that invoked it. Without this lookup the documented first
    session — ``interlatent up`` and then ``interlatent gpus ls`` — fails with
    a "no coordinator" error whose remedy is to run the command you just ran.

    Fed to :func:`resolve` as ``config``, i.e. *below* ``--coordinator`` and
    ``INTERLATENT_COORDINATOR`` in precedence: pointing this CLI at a remote
    control plane must keep working on a host that also runs its own.

    A dead pid does not count. A stale runtime file outlives a crashed daemon
    (and one was sitting on this machine), so trusting it blindly would aim
    every command at a closed port instead of saying nothing is running.
    """
    from ..coordinator import supervisor

    rt = supervisor.read_runtime()
    if not rt or not supervisor.pid_alive(rt.get("pid", -1)):
        return None
    port = rt.get("port")
    return supervisor.local_base(port) if port else None


#: Resolved per-invocation, not at import: the env var must be readable after
#: this module loads (tests set it, and the runtime file appears only once
#: `interlatent up` has run).
def _default_api_base() -> str:
    return resolve(config=_local_coordinator(), purpose="cli")


# ----------------------------------------------------------------------
# Client construction
# ----------------------------------------------------------------------


def _make_client(args: argparse.Namespace) -> HTTPClient:
    """Build an authenticated coordinator client or exit with a clear error."""
    api_key = getattr(args, "api_key", None) or os.environ.get("INTERLATENT_API_KEY", "")
    if not api_key:
        # A locally-run coordinator already minted one; use it rather than
        # making the operator copy a key from `interlatent up`'s output.
        from ..coordinator.auth import load_operator_key
        api_key = load_operator_key() or ""
    if not api_key:
        print(
            "error: no API key. Pass --api-key, set INTERLATENT_API_KEY, or "
            "run `interlatent up` (which mints an operator key for you).",
            file=sys.stderr,
        )
        raise SystemExit(2)
    base = resolve(
        getattr(args, "api_base", None),
        config=_local_coordinator(),
        purpose="cli",
    )
    return HTTPClient(base_url=base, api_key=api_key)


def _rows(payload: Any, key: str) -> list[dict]:
    """Normalize a list response.

    The coordinator may return either a bare JSON array or an object wrapping
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
    # Keys as the coordinator actually emits them (see Coordinator.add_gpu /
    # register_box): there is no `id`, `gpu` or `region` on a box row, so the
    # columns named for them were blank in every listing, and `url` — the one
    # field that tells you whether a registration points anywhere real — was
    # not shown at all. Boxes registered before the fuller schema carry only
    # name/url/method, hence the blanks the table already tolerates.
    _print_table(
        rows,
        [("NAME", "name"), ("URL", "url"), ("STATUS", "status"),
         ("GPU", "gpu_model"), ("PROVIDER", "provider")],
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
        # create one.
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
# behavior (offline — no API key, no coordinator)
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
# Running a coordinator of your own
# ----------------------------------------------------------------------


def cmd_up(args: argparse.Namespace) -> int:
    from ..coordinator import supervisor
    code, message = supervisor.start(args.port, host=args.host)
    print(message, file=sys.stderr if code else sys.stdout)
    return code


def cmd_down(args: argparse.Namespace) -> int:
    from ..coordinator import supervisor
    code, message = supervisor.stop(args.force, args.grace)
    print(message, file=sys.stderr if code else sys.stdout)
    return code


def cmd_status(args: argparse.Namespace) -> int:
    from ..coordinator import supervisor
    code, message = supervisor.status()
    print(message)
    return code


def cmd_logs(args: argparse.Namespace) -> int:
    from ..coordinator import supervisor
    return supervisor.logs(args.follow, args.lines)


def cmd_gpu(args: argparse.Namespace) -> int:
    """Register/remove a GPU box. `add` is POST compute/boxes/register --
    the same route interlatent-serve calls, not a coordinator-only verb."""
    client = _make_client(args)
    if args.gpu_cmd == "add":
        body = {
            "box_id": args.box_id or args.name,
            "name": args.name,
            "endpoint": args.url,
            "provider": "manual",
        }
        if args.warm_policy:
            body["warmup_policy"] = args.warm_policy
        out = client.request(
            "POST", "/api/v1/compute/boxes/register", json_body=body
        )
        print(f"✓ registered {out.get('name')} at {args.url}")
        return 0
    if args.gpu_cmd == "rm":
        client.request("DELETE", f"/api/v1/gpus/{args.name}")
        print(f"✓ removed {args.name}")
        return 0
    return cmd_gpus(args)


def cmd_config(args: argparse.Namespace) -> int:
    """Set the recording destination stamped onto every session.

    A coordinator that does not implement the route 404s, and this says so
    by name rather than printing a bare HTTP error.
    """
    client = _make_client(args)
    recording: dict = {}
    if args.output_dir:
        recording["output_dir"] = args.output_dir
    if args.s3_uri:
        recording["s3_uri"] = args.s3_uri
        for key, val in (
            ("s3_endpoint_url", args.s3_endpoint_url),
            ("s3_access_key", args.s3_access_key),
            ("s3_secret_key", args.s3_secret_key),
            ("s3_region", args.s3_region),
        ):
            if val:
                recording[key] = val

    try:
        if not recording:
            out = client.request("GET", "/api/v1/coordinator/recording")
        else:
            out = client.request(
                "PUT", "/api/v1/coordinator/recording",
                json_body={"recording": recording},
            )
    except NotFoundError:
        print(
            "error: this coordinator does not manage recording destinations; "
            "set one on the GPU box instead (serve_gpu's --output-dir, or "
            "DRTC_OUTPUT_DIR in its environment).",
            file=sys.stderr,
        )
        return 2
    dest = out.get("recording") or {}
    print(json.dumps(dest, indent=2) if dest else
          "none — sessions will run but NOT be saved")
    return 0


# ----------------------------------------------------------------------


def _add_auth_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--api-key", default=None,
                   help="Operator key from `interlatent up` (ilop_…). Falls "
                        "back to INTERLATENT_API_KEY, then to the key on disk.")
    p.add_argument("--coordinator", "--api-base", dest="api_base", default=None,
                   help="Coordinator base URL, e.g. http://10.0.0.5:8900. "
                        "Env: INTERLATENT_COORDINATOR.")
    p.add_argument("--json", action="store_true", help="Emit raw JSON instead of a table.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="interlatent",
        description="Session manager for Interlatent robot fleets. Run a "
        "coordinator of your own (`up`), register GPU boxes and nodes "
        "against it, and start/stop inference sessions — or point the same "
        "commands at a coordinator running elsewhere with --coordinator.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # -- running a coordinator of your own ------------------------------
    p_up = sub.add_parser(
        "up", help="Start a coordinator on this machine (background daemon)."
    )
    p_up.add_argument("--port", type=int, default=8900)
    p_up.add_argument(
        "--host", default="0.0.0.0",
        help="Bind address. Default binds all interfaces so robots on the "
             "LAN can reach it; use 127.0.0.1 to keep it local.",
    )
    p_up.set_defaults(func=cmd_up)

    p_down = sub.add_parser("down", help="Stop the local coordinator.")
    p_down.add_argument(
        "--force", action="store_true",
        help="Stop even with active sessions: unassigns each one and waits "
             "for the nodes to tear down (which is what publishes datasets).",
    )
    p_down.add_argument(
        "--grace", type=float, default=30.0,
        help="Seconds to wait for nodes to converge to idle under --force.",
    )
    p_down.set_defaults(func=cmd_down)

    p_status = sub.add_parser("status", help="Is the local coordinator up?")
    p_status.set_defaults(func=cmd_status)

    p_logs = sub.add_parser("logs", help="Tail the local coordinator's log.")
    p_logs.add_argument("-f", "--follow", action="store_true")
    p_logs.add_argument("-n", "--lines", type=int, default=50)
    p_logs.set_defaults(func=cmd_logs)

    p_config = sub.add_parser(
        "config",
        help="Get/set the recording destination stamped onto every session.",
    )
    p_config.add_argument("--output-dir", default=None,
                          help="Publish datasets into this local directory.")
    p_config.add_argument("--s3-uri", default=None,
                          help="Publish to s3://bucket/prefix.")
    p_config.add_argument("--s3-endpoint-url", default=None,
                          help="For S3-compatible stores (MinIO, R2, ...).")
    p_config.add_argument("--s3-access-key", default=None)
    p_config.add_argument("--s3-secret-key", default=None)
    p_config.add_argument("--s3-region", default=None)
    _add_auth_flags(p_config)
    p_config.set_defaults(func=cmd_config)

    p_gpu = sub.add_parser("gpu", help="Register or remove a GPU box.")
    gpu_sub = p_gpu.add_subparsers(dest="gpu_cmd", required=True)
    g_add = gpu_sub.add_parser("add", help="Register a GPU box by address.")
    g_add.add_argument("--name", required=True)
    g_add.add_argument("--url", required=True,
                       help="host:port your nodes can reach the box at.")
    g_add.add_argument("--box-id", default=None,
                       help="Stable id; defaults to --name.")
    g_add.add_argument("--warm-policy", default=None,
                       help="Policy the box is pre-warmed for, if known.")
    _add_auth_flags(g_add)
    g_ls = gpu_sub.add_parser("ls", help="List GPU boxes.")
    _add_auth_flags(g_ls)
    g_rm = gpu_sub.add_parser("rm", help="Remove a GPU box.")
    g_rm.add_argument("name")
    _add_auth_flags(g_rm)
    p_gpu.set_defaults(func=cmd_gpu)

    p_gpus = sub.add_parser("gpus", help="List GPU boxes registered with the coordinator.")
    gpus_sub = p_gpus.add_subparsers(dest="gpus_cmd", required=True)
    p_gpus_ls = gpus_sub.add_parser("ls", help="List GPUs.")
    _add_auth_flags(p_gpus_ls)
    p_gpus.set_defaults(func=cmd_gpus)

    p_nodes = sub.add_parser("nodes", help="List robot nodes paired with the coordinator.")
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
        "behavior",
        help="List/validate/run named behaviors offline (no coordinator, no API key).",
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
    except CoordinatorNotConfigured as e:
        # Carries its own remediation sentence; a traceback would bury it.
        print(f"error: {e}", file=sys.stderr)
        return 2
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
