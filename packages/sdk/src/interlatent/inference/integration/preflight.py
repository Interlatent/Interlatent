"""Backend for the ``interlatent-preflight`` console script.

A non-destructive connectivity check for the hosted cloud inference
path. It rides the *real* flow — exactly what a robot does — without
any hardware:

    1. Fetch the environment's observation schema from the dashboard
       (camera names, action_dim) so synthetic obs match the policy.
    2. Open a real DRTC session against a managed GPU pod running the
       requested policy (``connect_drtc``).
    3. Push synthetic observations (a random state vector + gray camera
       frames) for a few seconds, confirming action chunks come back.
    4. Report the measured network-vs-compute latency split and a
       PASS / WARN / FAIL verdict.

It answers "is the cloud inference path healthy and fast enough from
where my robot sits?" — it does NOT exercise cameras, joints, or the
motor bus, so a green preflight is not "my robot is ready".

All the interesting machinery lives in ``inference/client/`` and
``inference/integration/connect.py``; this file is just the harness.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import numpy as np

from .connect import connect_drtc

log = logging.getLogger("interlatent.preflight")

# Fallback obs schema when the dashboard config is unavailable or sparse.
# Sized for an SO-101-class arm: 6 joints, a single camera.
_DEFAULT_ACTION_DIM = 6
_DEFAULT_CAMERAS = ["front"]
_DEFAULT_IMAGE_SIZE = 256
# How long to wait for the first action chunk before declaring failure.
_FIRST_CHUNK_TIMEOUT_S = 30.0
# Schedule-starvation fraction above which we WARN about stutter risk.
_STARVATION_WARN_PCT = 25.0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="interlatent-preflight",
        description="Connectivity check for the Interlatent cloud inference "
        "path: opens a real DRTC session against a managed GPU pod with "
        "synthetic observations and reports a latency verdict. Does NOT test "
        "robot hardware.",
    )
    p.add_argument("--environment", required=True,
                   help="Interlatent environment slug registered in the dashboard.")
    p.add_argument("--policy", required=True,
                   help="Policy URI to load on the pod (e.g. lerobot/smolvla_base).")
    p.add_argument("--task", default="preflight connectivity check",
                   help="Natural-language task string passed to the policy.")
    p.add_argument("--fps", type=float, default=30.0,
                   help="Control rate to simulate (default 30).")
    p.add_argument("--steps", type=int, default=60,
                   help="How many synthetic control steps to run (default 60).")
    p.add_argument("--state-dim", type=int, default=None,
                   help="Length of the synthetic observation.state vector. "
                        "Defaults to the env's action_dim.")
    p.add_argument("--cameras", default=None,
                   help="Comma-separated camera names. Defaults to the env's "
                        "registered cameras.")
    p.add_argument("--image-size", type=int, default=_DEFAULT_IMAGE_SIZE,
                   help="Square edge (px) of the synthetic camera frames.")
    p.add_argument("--api-key", default=None,
                   help="Interlatent API key (ilat_…). Falls back to INTERLATENT_API_KEY.")
    p.add_argument("--api-base", default=None,
                   help="Dashboard base URL (default https://interlatent.com / "
                        "INTERLATENT_API_BASE) — used to fetch the env schema.")
    p.add_argument("--server", default=None,
                   help="DRTC endpoint override (default INTERLATENT_DRTC_URL / hosted).")
    p.add_argument("--grpc-web", action="store_true",
                   help="Force gRPC-Web transport (auto-inferred from an https URL).")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def _fetch_schema(environment: str, api_key: str, api_base: str | None) -> dict:
    """Pull the env's obs schema from the dashboard; empty dict on failure."""
    try:
        from ..._http import HTTPClient
        from ..._resources import EnvironmentsResource
        base = api_base or os.environ.get("INTERLATENT_API_BASE") or "https://interlatent.com"
        http = HTTPClient(base_url=base, api_key=api_key)
        cfg = EnvironmentsResource(http).get(environment)
        return cfg if isinstance(cfg, dict) else {}
    except Exception as e:  # noqa: BLE001 — schema fetch is best-effort
        log.warning("Could not fetch env schema (%s); using defaults.", e)
        return {}


def _resolve_cameras(args: argparse.Namespace, schema: dict) -> list[str]:
    if args.cameras:
        return [c.strip() for c in args.cameras.split(",") if c.strip()]
    names = schema.get("camera_names")
    if isinstance(names, list) and names:
        return [str(n) for n in names]
    num = schema.get("num_cameras")
    if isinstance(num, int) and num > 0:
        return [f"cam{i}" for i in range(num)]
    return list(_DEFAULT_CAMERAS)


def _make_synthetic_obs(state_dim: int, cameras: list[str], image_size: int) -> dict:
    """A policy-schema observation: a random state vector + gray frames."""
    obs: dict = {
        "observation.state": np.random.randn(state_dim).astype(np.float32),
    }
    gray = np.full((image_size, image_size, 3), 128, dtype=np.uint8)
    for cam in cameras:
        obs[f"observation.images.{cam}"] = gray
    return obs


def _verdict(stats: dict, got_action: bool, chunk_size: int, control_period_s: float,
             est_latency_s: float) -> tuple[str, list[str]]:
    """Return (PASS|WARN|FAIL, notes)."""
    if not got_action:
        return "FAIL", ["No action chunk was received — the cloud path is not serving."]
    notes: list[str] = []
    # DRTC tolerates high latency by chunking; the risk is the round-trip
    # outrunning the chunk horizon, which starves the schedule.
    horizon_s = chunk_size * control_period_s
    if est_latency_s > horizon_s > 0:
        notes.append(
            f"Estimated latency {est_latency_s * 1000:.0f}ms exceeds the chunk "
            f"horizon {horizon_s * 1000:.0f}ms — control may stutter at this fps."
        )
    starv = float(stats.get("starvation_pct", 0.0))
    if starv > _STARVATION_WARN_PCT:
        notes.append(f"Schedule starved {starv:.0f}% of ticks — stutter risk.")
    return ("WARN" if notes else "PASS"), notes


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(message)s")

    api_key = args.api_key or os.environ.get("INTERLATENT_API_KEY", "")
    if not api_key and not args.server:
        log.error("error: an Interlatent API key is required "
                  "(pass --api-key or set INTERLATENT_API_KEY).")
        return 2

    schema = _fetch_schema(args.environment, api_key, args.api_base)
    action_dim = int(schema.get("action_dim") or _DEFAULT_ACTION_DIM)
    state_dim = args.state_dim or action_dim
    cameras = _resolve_cameras(args, schema)
    control_period_s = 1.0 / args.fps if args.fps > 0 else 1.0 / 30
    chunk_size = 50  # connect_drtc default

    log.info("Preflight → env=%s policy=%s", args.environment, args.policy)
    log.info("  obs: state_dim=%d cameras=%s image=%dpx",
             state_dim, cameras, args.image_size)

    # Reused npz encoder (lazy import keeps this module light).
    from ...node.control import _encode_npz

    obs = _make_synthetic_obs(state_dim, cameras, args.image_size)

    try:
        client = connect_drtc(
            api_key=api_key,
            environment=args.environment,
            policy_uri=args.policy,
            policy_backend="lerobot",
            server_address=args.server,
            action_dim=action_dim,
            chunk_size=chunk_size,
            fps=args.fps,
            task=args.task,
            payload_codec="npz",
            stats_interval_s=0,  # we own the stats() window; no background thread
        )
    except Exception as e:  # noqa: BLE001 — surface any open failure as FAIL
        log.error("FAIL: could not open a session (%s)", e)
        return 1

    got_action = False
    stats: dict = {}
    est_latency_s = 0.0
    try:
        start = time.monotonic()
        for i in range(args.steps):
            t0 = time.monotonic()
            action = client.step(lambda o=obs: _encode_npz(o, image_resize=args.image_size),
                                  codec="npz")
            if action is not None and not got_action:
                got_action = True
                log.info("  first action chunk at step %d (%.1fs)", i, time.monotonic() - start)
            if not got_action and (time.monotonic() - start) > _FIRST_CHUNK_TIMEOUT_S:
                log.error("FAIL: no action chunk within %.0fs.", _FIRST_CHUNK_TIMEOUT_S)
                break
            dt = time.monotonic() - t0
            if dt < control_period_s:
                time.sleep(control_period_s - dt)

        stats = client.stats()
        est_latency_s = client.estimated_latency_s
    except Exception as e:  # noqa: BLE001 — any drive-loop error is a FAIL
        log.error("FAIL: inference loop errored (%s)", e)
    finally:
        client.close()

    verdict, notes = _verdict(stats, got_action, chunk_size, control_period_s, est_latency_s)
    log.info("")
    log.info("Result: %s", verdict)
    if got_action:
        log.info("  round-trip:  %.0f ms  (compute %.0f ms + network %.0f ms)",
                 stats.get("infer_ms", 0.0), stats.get("compute_ms", 0.0),
                 stats.get("net_ms", 0.0))
        log.info("  est latency: %.0f ms   control: %.1f Hz   chunks: %d",
                 est_latency_s * 1000, stats.get("control_hz", 0.0),
                 int(stats.get("chunks_recv", 0)))
    for n in notes:
        log.info("  ⚠ %s", n)
    log.info("  (tests the cloud inference path only — NOT cameras, joints, or motors.)")

    return 0 if verdict in ("PASS", "WARN") else 1


if __name__ == "__main__":
    sys.exit(main())
