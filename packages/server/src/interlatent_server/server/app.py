"""Entry point for the self-hosted DRTC inference server.

Run:
    interlatent-serve --policy lerobot/smolvla_base

or, equivalently:
    python -m interlatent_server.server.app --policy lerobot/smolvla_base

``--policy`` is optional: when given, the policy is loaded (and
torch.compile'd, for policies that use it) before the server accepts
traffic, so the first OpenSession does not stall on a multi-minute
compile. Without it, the first session for each policy pays that cost
once per process.

The optional teleop relay (browser/laptop DAgger takeover) starts only
when ``INTERLATENT_TELEOP_SECRET`` is set.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor

log = logging.getLogger("interlatent_server")


def _warmup(policy_uri: str, backend: str) -> None:
    """Pre-load a policy so the first session is instant.

    Mirrors the routing the transport applies at OpenSession time so the
    pre-warm and the first real session share the same cache entry.
    """
    from .molmoact2_backend import resolve_backend
    from .policy_runtime import PolicyRuntime

    if "molmoact" in policy_uri.lower():
        log.warning(
            "Pre-warm skipped for MolmoAct2 policy %s — released MolmoAct2 "
            "checkpoints need per-session image_keys metadata, so the first "
            "OpenSession will load it instead.",
            policy_uri,
        )
        return

    log.info("Pre-warming policy %s ...", policy_uri)
    try:
        PolicyRuntime.load(
            backend=resolve_backend(backend, policy_uri),
            policy_uri=policy_uri,
        )
        log.info("Pre-warm complete")
    except Exception:
        log.warning(
            "Pre-warm failed — the first real session will load the policy "
            "instead",
            exc_info=True,
        )


async def _serve(host: str, port: int, *, teleop_port: int, teleop_secret: str) -> None:
    import grpc

    from ..protocol import messages_pb2_grpc as pb_grpc
    from .transport import InferenceServicer

    # Keepalive options: clients send HTTP/2 pings every 10s of idle to
    # keep cloud TCP proxies from half-closing long-lived streams. gRPC's
    # server-side defaults reject pings faster than every 5 minutes with
    # a GOAWAY, so relax that here or the keepalive pings get the client
    # kicked off.
    server = grpc.aio.server(
        options=[
            ("grpc.keepalive_time_ms", 30000),
            ("grpc.keepalive_timeout_ms", 5000),
            ("grpc.keepalive_permit_without_calls", 1),
            ("grpc.http2.max_pings_without_data", 0),
            ("grpc.http2.min_ping_interval_without_data_ms", 5000),
            ("grpc.http2.min_time_between_pings_ms", 5000),
        ],
    )

    # A single-worker executor keeps the blocking policy.forward() off
    # the event loop (so recording ingest never stalls behind a slow VLA
    # preprocess) while preserving per-session chunk-buffer ordering.
    inference_executor = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="drtc-infer"
    )

    pb_grpc.add_InferenceServiceServicer_to_server(
        InferenceServicer(inference_executor=inference_executor), server
    )
    server.add_insecure_port(f"{host}:{port}")
    await server.start()
    log.info("DRTC inference server listening on %s:%d (gRPC)", host, port)

    if teleop_secret:
        from .teleop_relay import serve as serve_teleop

        await serve_teleop(host=host, port=teleop_port, secret=teleop_secret)
        log.info("Teleop relay listening on %s:%d (WebSocket)", host, teleop_port)
    else:
        log.info(
            "Teleop relay disabled (INTERLATENT_TELEOP_SECRET unset). "
            "Set the secret to enable DAgger takeover."
        )

    await server.wait_for_termination()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="interlatent-serve",
        description="Self-hosted low-latency inference server for VLA / "
        "action-chunking policies (DRTC protocol).",
    )
    parser.add_argument("--host", default="0.0.0.0", help="bind address")
    parser.add_argument("--port", type=int, default=50051, help="gRPC port")
    parser.add_argument(
        "--policy",
        default=os.environ.get("DRTC_WARMUP_POLICY", ""),
        help="policy URI to pre-warm at startup, e.g. lerobot/smolvla_base "
        "(HF repo id or local checkpoint path). Env: DRTC_WARMUP_POLICY",
    )
    parser.add_argument(
        "--backend",
        default="lerobot",
        help="policy backend for --policy (default: lerobot)",
    )
    parser.add_argument(
        "--teleop-port",
        type=int,
        default=int(os.environ.get("INTERLATENT_TELEOP_WS_PORT", "50052")),
        help="WebSocket teleop relay port (active only when "
        "INTERLATENT_TELEOP_SECRET is set)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if args.policy:
        _warmup(args.policy, args.backend)

    asyncio.run(
        _serve(
            args.host,
            args.port,
            teleop_port=args.teleop_port,
            teleop_secret=os.environ.get("INTERLATENT_TELEOP_SECRET", ""),
        )
    )


if __name__ == "__main__":
    main()
    sys.exit(0)
