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


def _reachable_addresses() -> tuple[str, list[str]]:
    """Best-effort list of this host's non-loopback addresses (no extra deps).

    The bind address is usually ``0.0.0.0``; this is what a client actually
    dials. Tailnet / other-interface addresses may not all be discoverable
    here, so this is a hint, not an exhaustive list.
    """
    import socket

    ips: list[str] = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))  # no packets sent — just selects a route
            ips.append(s.getsockname()[0])
        finally:
            s.close()
    except OSError:
        pass
    host = socket.gethostname()
    try:
        for info in socket.getaddrinfo(host, None):
            ip = info[4][0]
            if ip not in ips and not ip.startswith("127.") and ip != "::1":
                ips.append(ip)
    except OSError:
        pass
    return host, ips


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


async def _serve(
    host: str,
    port: int,
    *,
    teleop_port: int,
    teleop_secret: str,
    dataset_sink=None,
) -> None:
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
        InferenceServicer(
            inference_executor=inference_executor, dataset_sink=dataset_sink
        ),
        server,
    )
    server.add_insecure_port(f"{host}:{port}")
    await server.start()
    log.info("DRTC inference server listening on %s:%d (gRPC)", host, port)
    _host, _ips = _reachable_addresses()
    if _ips:
        log.info(
            "Reachable for clients (e.g. `interlatent gpu add <name> <addr>`) at: %s "
            "(hostname=%s). Tailnet / other-interface addresses may differ.",
            ", ".join(f"{ip}:{port}" for ip in _ips), _host,
        )

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
    # Standalone recording destination. Only used when a session's
    # OpenSession metadata doesn't already specify one (the coordinator
    # path sets it per-session). Without either, recording falls back to
    # the hosted inbox (needs an API key). See ADR-0002.
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("INTERLATENT_OUTPUT_DIR", ""),
        help="Write recorded episodes into one flat LeRobot dataset at this "
        "local directory (merge-on-stop). Env: INTERLATENT_OUTPUT_DIR",
    )
    parser.add_argument(
        "--s3-uri",
        default=os.environ.get("INTERLATENT_S3_URI", ""),
        help="Write recorded episodes to an S3-compatible bucket, e.g. "
        "s3://bucket/prefix (merge-on-stop). Env: INTERLATENT_S3_URI",
    )
    parser.add_argument(
        "--s3-endpoint-url",
        default=os.environ.get("INTERLATENT_S3_ENDPOINT_URL", ""),
        help="Custom S3 endpoint (R2/MinIO). Env: INTERLATENT_S3_ENDPOINT_URL",
    )
    parser.add_argument(
        "--s3-access-key",
        default=os.environ.get("AWS_ACCESS_KEY_ID", ""),
        help="S3 access key. Env: AWS_ACCESS_KEY_ID",
    )
    parser.add_argument(
        "--s3-secret-key",
        default=os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
        help="S3 secret key. Env: AWS_SECRET_ACCESS_KEY",
    )
    parser.add_argument(
        "--s3-region",
        default=os.environ.get("AWS_REGION", ""),
        help="S3 region. Env: AWS_REGION",
    )
    args = parser.parse_args(argv)

    if args.output_dir and args.s3_uri:
        parser.error("--output-dir and --s3-uri are mutually exclusive")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if args.policy:
        _warmup(args.policy, args.backend)

    dataset_sink = _resolve_sink(args)

    asyncio.run(
        _serve(
            args.host,
            args.port,
            teleop_port=args.teleop_port,
            teleop_secret=os.environ.get("INTERLATENT_TELEOP_SECRET", ""),
            dataset_sink=dataset_sink,
        )
    )


def _resolve_sink(args):
    """Build the standalone default sink from CLI flags, or None for inbox."""
    from .sinks import LocalDirSink, S3Sink

    if args.output_dir:
        log.info("Recording destination: local dir %s", args.output_dir)
        return LocalDirSink(args.output_dir)
    if args.s3_uri:
        log.info("Recording destination: %s", args.s3_uri)
        return S3Sink.from_uri(
            args.s3_uri,
            endpoint_url=args.s3_endpoint_url or None,
            access_key=args.s3_access_key or None,
            secret_key=args.s3_secret_key or None,
            region=args.s3_region or None,
        )
    return None


if __name__ == "__main__":
    main()
    sys.exit(0)
