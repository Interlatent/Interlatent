"""`interlatent-node` console script.

Two subcommands:

    interlatent-node pair --name <NAME> [--api-key ilat_...]
        Registers this machine as a Node under the user's account.
        Writes ~/.interlatent/node.toml with the node id + minted
        token. Run once per machine.

    interlatent-node run --robot <NAME> [--port <PATH>] [...]
        Boots the daemon. Heartbeats every 10s, long-polls for
        assignment changes, and converges to whatever
        InferenceSession the dashboard has assigned. Run under
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

DEFAULT_API_BASE = os.environ.get(
    "INTERLATENT_API_BASE", "https://interlatent.com"
).rstrip("/")
DEFAULT_CONFIG_PATH = Path(
    os.environ.get("INTERLATENT_NODE_CONFIG", "~/.interlatent/node.toml")
).expanduser()

_LOG = logging.getLogger("interlatent.node")


# ---------------------------------------------------------------------------
# Config file
# ---------------------------------------------------------------------------
#
# Format is intentionally trivial TOML so we don't drag in `tomli_w`. Keys:
#   node_id   = "..."
#   token     = "ilnode_..."
#   api_base  = "https://interlatent.com"
#   name      = "..."
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
    # An API key is only required when pairing against Interlatent Cloud. A
    # self-hosted coordinator (`interlatent up`) accepts pairing without one.
    # In that case DRTC inference falls back to the node token (ignored by an
    # unguarded self-hosted server) and the teleop channel is auto-disabled
    # (the daemon gates teleop on a non-empty user key).
    if not api_key:
        print(
            "No API key provided — pairing without one (self-hosted "
            "coordinator mode). DRTC auth uses the node token and DAgger "
            "teleop is disabled. Pass --api-key / INTERLATENT_API_KEY to "
            "pair against Interlatent Cloud instead.",
            file=sys.stderr,
        )

    # DRTC endpoint — normally inherited per session from whichever
    # compute box is attached to the env in the dashboard, so we do
    # NOT prompt at pair time. --drtc-url and INTERLATENT_DRTC_URL
    # remain available for legacy / operator-managed fleets that want
    # a fixed endpoint baked into the node config.
    drtc_url = (
        (args.drtc_url or "").strip()
        or os.environ.get("INTERLATENT_DRTC_URL", "").strip()
    )

    url = f"{args.api_base.rstrip('/')}/api/v1/nodes"
    try:
        resp = requests.post(
            url,
            headers={"x-api-key": api_key, "content-type": "application/json"},
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
        "api_base": args.api_base.rstrip("/"),
        "name": payload["name"],
    }
    # Persist the user API key only when one was given: the node token
    # authenticates heartbeat/poll, but Cloud DRTC inference auth needs the
    # ilat_ key. Omitted entirely in self-hosted coordinator mode.
    if api_key:
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
            "DRTC endpoint will be set per session from whichever "
            "compute box you attach through the cli/dashboard."
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
    missing = [k for k in ("node_id", "token", "api_base") if k not in cfg]
    if missing:
        print(
            f"error: config {cfg_path} is missing keys: {missing}. "
            f"Re-run `interlatent-node pair`.",
            file=sys.stderr,
        )
        return 2

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # DRTC inference auth needs the ilat_ user key, not the node token.
    # Resolve from CLI > env > config saved at pair time.
    drtc_api_key = (
        args.api_key
        or os.environ.get("INTERLATENT_API_KEY")
        or cfg.get("api_key")
    )
    if not drtc_api_key:
        _LOG.warning(
            "No Interlatent API key available for DRTC inference. The node "
            "token alone is rejected by the DRTC server. Pass --api-key, set "
            "INTERLATENT_API_KEY, or re-run `interlatent-node pair` to save it."
        )

    # Lazy import so `pair` doesn't require asyncio/lerobot at all.
    from .daemon import NodeDaemon, NodeDaemonConfig

    # DRTC endpoint resolution at run time: CLI flag > env var > pair-time
    # config. The daemon refuses to launch a session if none of these is
    # set — there is no hosted default. Re-running pair with --drtc-url
    # is the persistent fix.
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
            api_base=cfg["api_base"],
            robot_kind=args.robot,
            robot_port=args.port,
            robot_extra=dict(args.robot_arg or []),
            robot_cameras=dict(args.camera or []),
            loop_override=args.loop,
            num_inference_steps=num_inference_steps,
            image_resize=image_resize,
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
        "assigned from the Interlatent dashboard.",
    )
    p.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the node config file (default: ~/.interlatent/node.toml)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_pair = sub.add_parser("pair", help="Register this machine as a Node.")
    p_pair.add_argument("--name", required=True, help="Display name shown in the dashboard.")
    p_pair.add_argument(
        "--api-key",
        default=None,
        help="Interlatent user API key (or set INTERLATENT_API_KEY).",
    )
    p_pair.add_argument(
        "--api-base",
        default=DEFAULT_API_BASE,
        help=f"Backend base URL (default: {DEFAULT_API_BASE}).",
    )
    p_pair.add_argument(
        "--drtc-url",
        default=None,
        help="DRTC inference endpoint to persist (e.g. "
        "http://100.x.y.z:8000 for a Runpod box reached over Tailscale, "
        "or https://<workspace>--interlatent-drtc-inference-web.modal.run "
        "for a Modal deployment). If omitted on a TTY, you'll be prompted.",
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
        help="Interlatent user API key (ilat_...) for DRTC inference auth. "
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
    p_run.add_argument("-v", "--verbose", action="store_true")
    p_run.set_defaults(func=cmd_run)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
