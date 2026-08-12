"""``python -m interlatent.coordinator`` — serve the control plane.

The foreground entry point. Operators normally reach it through
``interlatent up``, which spawns this in the background and manages the pidfile;
running it directly is what you want under systemd or in a container, where the
supervisor already belongs to something else.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from . import auth
from .relay_supervisor import DEFAULT_RELAY_PORT, start_relay
from .server import run_server
from .state import Coordinator
from .supervisor import DEFAULT_PORT, STATE_PATH


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m interlatent.coordinator",
        description="Serve the Interlatent coordinator control plane.",
    )
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--state", default=str(STATE_PATH))
    p.add_argument(
        "--relay-port", type=int, default=DEFAULT_RELAY_PORT,
        help="UDP port for the embedded teleop relay.",
    )
    p.add_argument(
        "--relay-advertise", default="",
        help="Address a headset can reach this machine at. Auto-detected "
             "from the outbound interface if unset — never the bind address, "
             "which a browser cannot dial.",
    )
    p.add_argument(
        "--no-relay", action="store_true",
        help="Do not serve teleop. The token route then 404s, which turns "
             "teleop off cleanly rather than leaving nodes retrying.",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    key, created = auth.ensure_operator_key()
    if created:
        print(f"Operator key minted: {key}")
        print(f"Stored 0600 at {auth.default_operator_key_path()}")

    # One instance, shared: the relay verifies the tokens the HTTP surface
    # mints, so they must be looking at the same object.
    coordinator = Coordinator(Path(args.state))

    relay = None
    if not args.no_relay:
        relay = start_relay(
            coordinator=coordinator,
            cert_dir=Path(args.state).parent,
            host=args.host,
            port=args.relay_port,
            advertise=args.relay_advertise,
        )

    run_server(
        args.host, args.port, Path(args.state), key, relay,
        coordinator=coordinator,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
