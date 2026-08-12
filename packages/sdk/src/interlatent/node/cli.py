"""`interlatent-node` console script.

Two subcommands:

    interlatent-node pair --coordinator <URL> --name <NAME> [--api-key ilop_...]
        Registers this machine as a Node with your coordinator (the one
        `interlatent up` runs). Writes ~/.interlatent/node.toml with the
        coordinator address, node id + minted token. Run once per machine.

    interlatent-node run --robot <NAME> [--port <PATH>] [...]
        Boots the daemon. Heartbeats every 10s, long-polls for
        assignment changes, and converges to whatever
        InferenceSession the coordinator has assigned. Run under
        systemd / tmux on the Pi.

The daemon itself lives in `daemon.py`; this file is just argparse
plumbing + pair-time HTTP.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import requests

from .._clamp_log import LOGGER_NAME as _CLAMP_LOGGER_NAME
from .._coordinator import resolve

def _default_api_base() -> str:
    """Resolved per-invocation; see interlatent._coordinator."""
    return resolve(purpose="node")
DEFAULT_CONFIG_PATH = Path(
    os.environ.get("INTERLATENT_NODE_CONFIG", "~/.interlatent/node.toml")
).expanduser()

_LOG = logging.getLogger("interlatent.node")


# ---------------------------------------------------------------------------
# Config file
# ---------------------------------------------------------------------------
#
# Format is intentionally trivial TOML so we don't drag in `tomli_w`. Keys:
#   node_id     = "..."
#   token       = "ilnode_..."
#   coordinator = "http://10.0.0.5:8900"
#   api_base    = "..."   # compat alias for `coordinator`, written too
#   name        = "..."
#
# We hand-format / hand-parse so the SDK stays free of optional deps.


def _write_config(path: Path, data: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for k, v in data.items():
        escaped = v.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'{k} = "{escaped}"')
    path.write_text("\n".join(lines) + "\n")
    # Limit token visibility — TOML contains a long-lived credential.
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _read_config(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Node config not found at {path}. Run "
            f"`interlatent-node pair --name <name>` first."
        )
    out: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1].replace('\\"', '"').replace("\\\\", "\\")
        out[key.strip()] = value
    return out


# ---------------------------------------------------------------------------
# `pair` subcommand
# ---------------------------------------------------------------------------


def cmd_pair(args: argparse.Namespace) -> int:
    api_key = args.api_key or os.environ.get("INTERLATENT_API_KEY", "")
    # The node pairs against your coordinator, which resolves the caller from
    # the operator key (ilop_...) it minted. Without a key there is nothing to
    # authenticate against.
    if not api_key:
        print(
            "error: an operator API key is required to pair. Pass "
            "--api-key or set INTERLATENT_API_KEY (`interlatent up` mints "
            "one and writes it to disk).",
            file=sys.stderr,
        )
        return 2

    # DRTC endpoint — normally inherited per session from whichever GPU
    # box the coordinator assigns, so we do NOT prompt at pair time.
    # --drtc-url and INTERLATENT_DRTC_URL remain available for fleets
    # that want a fixed endpoint baked into the node config.
    drtc_url = (
        (args.drtc_url or "").strip()
        or os.environ.get("INTERLATENT_DRTC_URL", "").strip()
    )

    coordinator = resolve(args.api_base, purpose="node")
    url = f"{coordinator}/api/v1/nodes"
    try:
        resp = requests.post(
            url,
            headers={
                "x-api-key": api_key,
                "content-type": "application/json",
            },
            data=json.dumps({"name": args.name}),
            timeout=30,
        )
    except requests.RequestException as e:
        print(f"error: failed to reach {url}: {e}", file=sys.stderr)
        return 1

    if resp.status_code >= 400:
        print(
            f"error: pair failed ({resp.status_code}): {resp.text}",
            file=sys.stderr,
        )
        return 1
    payload: dict[str, Any] = resp.json()

    cfg_path = Path(args.config).expanduser()
    cfg_data = {
        "node_id": payload["id"],
        "token": payload["token"],
        # Stored under the new key; "api_base" stays as a
        # compat alias so a node.toml written by an older SDK
        # keeps working without a re-pair.
        "coordinator": coordinator,
        "api_base": coordinator,
        "name": payload["name"],
    }
    # Persist the operator API key: the node token authenticates heartbeat/poll,
    # but DRTC inference auth needs the ilop_ key.
    cfg_data["api_key"] = api_key
    if drtc_url:
        cfg_data["drtc_url"] = drtc_url
    _write_config(cfg_path, cfg_data)

    print(f"✓ Paired '{payload['name']}' as node_id={payload['id']}")
    print(f"✓ Saved credentials to {cfg_path}")
    if drtc_url:
        print(f"✓ DRTC endpoint (fixed): {drtc_url}")
    else:
        print(
            "DRTC endpoint will be set per session from whichever GPU "
            "box you assign with `interlatent session start`."
        )
    print(
        "  Run `interlatent-node run --robot <name> --port <path>` to "
        "start the daemon."
    )
    return 0


# ---------------------------------------------------------------------------
# `run` subcommand — defers all heavy work to daemon.py
# ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    cfg_path = Path(args.config).expanduser()
    cfg = _read_config(cfg_path)
    missing = [
        k for k in ("node_id", "token") if k not in cfg
    ]
    if not (cfg.get("coordinator") or cfg.get("api_base")):
        missing.append("coordinator")
    if missing:
        print(
            f"error: config {cfg_path} is missing keys: {missing}. "
            f"Re-run `interlatent-node pair`.",
            file=sys.stderr,
        )
        return 2

    # --verbose is a shortcut for the most verbose level; otherwise honor an
    # explicit --log-level, defaulting to INFO.
    if args.verbose:
        level = logging.DEBUG
    else:
        level = getattr(logging, (args.log_level or "info").upper())
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # --quiet-clamp silences only the per-tick delta-clamp warnings (their own
    # logger), leaving the INFO latency reports and every other warning intact.
    if args.quiet_clamp:
        logging.getLogger(_CLAMP_LOGGER_NAME).setLevel(logging.ERROR)

    # DRTC inference auth needs the ilop_ operator key, not the node token.
    # Resolve from CLI > env > config saved at pair time.
    drtc_api_key = (
        args.api_key
        or os.environ.get("INTERLATENT_API_KEY")
        or cfg.get("api_key")
    )
    if not drtc_api_key:
        _LOG.warning(
            "No operator API key available for DRTC inference. The node "
            "token alone is rejected by the DRTC server. Pass --api-key, set "
            "INTERLATENT_API_KEY, or re-run `interlatent-node pair` to save it."
        )

    # Lazy import so `pair` doesn't require asyncio/lerobot at all.
    from .daemon import NodeDaemon, NodeDaemonConfig

    # DRTC endpoint resolution at run time: CLI flag > env var > pair-time
    # config. The daemon refuses to launch a session if none of these is
    # set and the coordinator's session payload carries no route — there
    # is no default endpoint. Re-running pair with --drtc-url is the
    # persistent fix.
    drtc_url = (
        (args.drtc_url or "").strip()
        or os.environ.get("INTERLATENT_DRTC_URL", "").strip()
        or cfg.get("drtc_url", "").strip()
        or None
    )

    # VLA latency knobs. CLI > env > unset (lets the GPU side or daemon
    # pick a sane default per backend, e.g. MolmoAct2 → 5 / 256).
    def _resolve_int(cli_val, env_var):
        if cli_val is not None:
            return int(cli_val)
        env_val = os.environ.get(env_var)
        return int(env_val) if env_val else None

    num_inference_steps = _resolve_int(
        getattr(args, "num_inference_steps", None),
        "INTERLATENT_NUM_INFERENCE_STEPS",
    )
    image_resize = _resolve_int(
        getattr(args, "image_resize", None),
        "INTERLATENT_IMAGE_RESIZE",
    )

    daemon = NodeDaemon(
        NodeDaemonConfig(
            node_id=cfg["node_id"],
            token=cfg["token"],
            drtc_api_key=drtc_api_key,
            drtc_url=drtc_url,
            api_base=resolve(
                config=cfg.get("coordinator") or cfg.get("api_base"),
                purpose="node",
            ),
            robot_kind=args.robot,
            robot_port=args.port,
            robot_extra=dict(args.robot_arg or []),
            robot_cameras=dict(args.camera or []),
            loop_override=args.loop,
            num_inference_steps=num_inference_steps,
            image_resize=image_resize,
            synchronous=(
                bool(getattr(args, "synchronous", False))
                or os.environ.get("INTERLATENT_SYNCHRONOUS", "").strip().lower()
                in ("1", "true", "yes", "on")
            ),
        )
    )
    daemon.run_forever()
    return 0


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------


def _kv(s: str) -> tuple[str, str]:
    if "=" not in s:
        raise argparse.ArgumentTypeError(
            f"--robot-arg expects key=value, got: {s!r}"
        )
    k, _, v = s.partition("=")
    return k.strip(), v.strip()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="interlatent-node",
        description="Run a Pi-side daemon that executes inference sessions "
        "assigned by your coordinator.",
    )
    p.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the node config file (default: ~/.interlatent/node.toml)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_pair = sub.add_parser("pair", help="Register this machine as a Node.")
    p_pair.add_argument(
        "--name", required=True,
        help="Display name for this node, shown by `interlatent nodes ls`.",
    )
    p_pair.add_argument(
        "--api-key",
        default=None,
        help="Operator API key, ilop_... (or set INTERLATENT_API_KEY). "
             "`interlatent up` mints one.",
    )
    p_pair.add_argument(
        "--coordinator",
        "--api-base",
        dest="api_base",
        default=None,
        help="Coordinator base URL, e.g. http://10.0.0.5:8900. No default. "
             "Env: INTERLATENT_COORDINATOR.",
    )
    p_pair.add_argument(
        "--drtc-url",
        default=None,
        help="DRTC inference endpoint to persist (e.g. "
        "203.0.113.7:50051 for a GPU box's public IP:port, or an "
        "https:// URL when the box sits behind a gRPC-Web proxy). "
        "Optional: when omitted, the endpoint is inherited per session "
        "from the GPU box your coordinator assigns.",
    )
    p_pair.set_defaults(func=cmd_pair)

    p_run = sub.add_parser("run", help="Start the daemon (long-running).")
    p_run.add_argument(
        "--robot",
        required=False,
        default=None,
        help="LeRobot robot type, e.g. 'so101', 'koch', 'aloha'. Required "
        "unless --loop is given.",
    )
    p_run.add_argument(
        "--port",
        default=None,
        help="Serial port for the robot (e.g. /dev/ttyUSB0).",
    )
    p_run.add_argument(
        "--robot-arg",
        type=_kv,
        action="append",
        help="Extra key=value passed to the LeRobot robot config "
        "(repeatable). e.g. --robot-arg cameras=front,wrist",
    )
    p_run.add_argument(
        "--loop",
        default=None,
        help="Override the control loop with a custom callable "
        "(module:function). Bypasses the LeRobot wrapper.",
    )
    p_run.add_argument(
        "--camera",
        type=_kv,
        action="append",
        help="Attach a camera as name=device (repeatable). `name` becomes "
        "the observation.images.<name> key the policy sees — match it to "
        "the policy's expected image keys. e.g. --camera top=/dev/video0",
    )
    p_run.add_argument(
        "--api-key",
        default=None,
        help="Operator API key (ilop_...) for DRTC inference auth. "
        "Falls back to INTERLATENT_API_KEY, then the key saved at pair time.",
    )
    p_run.add_argument(
        "--drtc-url",
        default=None,
        help="DRTC inference endpoint for this run. Overrides "
        "INTERLATENT_DRTC_URL and the value saved at pair time.",
    )
    p_run.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Flow-matching denoising steps for VLA policies "
        "(currently MolmoAct2). Lower = faster, slightly noisier "
        "actions. Range 3-10; MolmoAct2 default is 5. Also "
        "settable via INTERLATENT_NUM_INFERENCE_STEPS.",
    )
    p_run.add_argument(
        "--image-resize",
        type=int,
        default=None,
        help="Resize camera frames to this square edge (pixels) "
        "before JPEG-encoding for the GPU. None keeps native "
        "resolution; 256 is the right default for MolmoAct2 "
        "(its image processor resizes to ~224 anyway). Also "
        "settable via INTERLATENT_IMAGE_RESIZE.",
    )
    p_run.add_argument(
        "--synchronous",
        action="store_true",
        help="Sequential (request-response) chunking: send one observation only "
        "when the action queue is fully drained, wait for the whole chunk, "
        "execute it, then re-observe. The robot holds (~one inference round-trip) "
        "at each chunk seam, but no fresh chunk ever overwrites an unexecuted tail "
        "— eliminates mid-chunk overwrite thrash for high-latency policies "
        "(MolmoAct2). Default off (async overlapping chunking). Also settable via "
        "INTERLATENT_SYNCHRONOUS.",
    )
    p_run.add_argument(
        "--log-level",
        default=None,
        choices=["debug", "info", "warning", "error", "critical"],
        help="Daemon log verbosity (default: info; --verbose forces debug). "
        "Note: raising this to 'warning'+ still shows the per-tick delta-clamp "
        "warnings, and 'error'+ also hides the periodic latency reports — to "
        "quiet just the clamp warnings use --quiet-clamp.",
    )
    p_run.add_argument(
        "--quiet-clamp",
        action="store_true",
        help="Suppress the per-tick delta-clamp warnings (execution-safety "
        "clamp) without hiding anything else. Use when a policy overshoots "
        "max_step every tick and the warnings drown out the latency reports.",
    )
    p_run.add_argument(
        "-v", "--verbose", action="store_true",
        help="Shortcut for --log-level debug.",
    )
    p_run.set_defaults(func=cmd_run)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
