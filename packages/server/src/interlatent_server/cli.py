"""``interlatent-serve`` — run a self-hosted DRTC policy server.

The self-hosted counterpart of a dashboard-provisioned GPU box: the
operator brings the GPU, the Interlatent dashboard stays the control
plane. On start the CLI

1. mints (once) and persists a box UUID at ``~/.interlatent/box-id``,
2. detects the GPU model,
3. registers the box with the dashboard —
   ``POST /api/v1/compute/boxes/register`` authenticated with the
   operator's own ``ilat_`` API key — so it appears on the Compute page
   as a launchable self-hosted box, and
4. execs the standard :mod:`interlatent_server.serve_gpu` server with
   the owner-key identity wired in (warmup-target fetch, status
   reports, and the default-on owner-checked RPC auth all use it).

The box only ever dials *out* to the backend; nodes dial *in* to the
gRPC port at ``--advertise-address``. Registration is idempotent: a
restart re-registers the same box row instead of orphaning a new one.

Typical run (Docker image or bare metal):

    INTERLATENT_API_KEY=ilat_... interlatent-serve \\
        --advertise-address 203.0.113.7 --port 50051

``--no-register`` skips the dashboard entirely (pure-local smoke test —
equivalent to running ``python -m interlatent_server.serve_gpu``).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import subprocess
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

log = logging.getLogger("interlatent-serve")

DEFAULT_API_BASE = "https://interlatent.com"
BOX_ID_PATH = Path.home() / ".interlatent" / "box-id"


def _resolve_box_id(override: str) -> str:
    """A stable per-machine box UUID. ``INTERLATENT_BOX_ID``/--box-id
    overrides; otherwise mint once and persist so restarts re-register
    the same dashboard row (the backend upserts on it)."""
    if override.strip():
        return override.strip()
    if BOX_ID_PATH.exists():
        existing = BOX_ID_PATH.read_text().strip()
        if existing:
            return existing
    box_id = str(uuid.uuid4())
    BOX_ID_PATH.parent.mkdir(parents=True, exist_ok=True)
    BOX_ID_PATH.write_text(box_id + "\n")
    log.info("Minted box id %s (persisted at %s)", box_id, BOX_ID_PATH)
    return box_id


def _detect_gpu_model() -> str:
    """Human GPU label for the dashboard card. Best-effort, never raises."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10.0,
        )
        name = (out.stdout or "").strip().splitlines()
        if out.returncode == 0 and name:
            return name[0].strip()
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "unknown"


def _detect_gpu_capacity() -> tuple[int, int | None]:
    """``(gpu_count, vram_gb)`` for this box. Best-effort, never raises.

    ``gpu_model`` cannot carry this: it is ``nvidia-smi``'s first line only, so
    a 2xH100 box and a 1xH100 box have always looked identical to the backend.
    World-action models need at least 2 GPUs (ADR 0037), and without a real
    count the launch gate can only fail late — the session opens, the sidecar
    dies at ``torchrun`` spawn, and the operator sees a dead process instead of
    a reason. Reporting it turns that into a 409 they can act on.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10.0,
        )
        rows = [r.strip() for r in (out.stdout or "").splitlines() if r.strip()]
        if out.returncode == 0 and rows:
            return len(rows), max(1, int(round(int(rows[0]) / 1024)))
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            vram = int(round(
                torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            ))
            return max(1, n), vram
    except Exception:
        pass
    # 1 is the safe default and true of every box that predates this field.
    return 1, None


def _register(
    *,
    api_base: str,
    api_key: str,
    box_id: str,
    name: str,
    endpoint: str,
    gpu_model: str,
    gpu_count: int = 1,
    vram_gb: int | None = None,
    warmup_policy: str | None,
) -> None:
    """One dashboard-registration handshake. Raises SystemExit with an
    actionable message on failure — serving an unregistered box that the
    user *asked* to register would just hide the problem."""
    url = f"{api_base.rstrip('/')}/api/v1/compute/boxes/register"
    payload = {
        "box_id": box_id,
        "name": name,
        "endpoint": endpoint,
        "gpu_model": gpu_model,
        "gpu_count": gpu_count,
        "vram_gb": vram_gb,
        "gpu_id": "custom",
        "provider": "byo",
        "warmup_policy": warmup_policy or None,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"x-api-key": api_key, "content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15.0) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", "replace").strip()
        except Exception:
            pass
        raise SystemExit(
            f"Box registration failed (HTTP {e.code}): {detail or '<no body>'}\n"
            "Check INTERLATENT_API_KEY (an ilat_... key from the dashboard) "
            "and --api-base."
        )
    except Exception as e:
        raise SystemExit(
            f"Box registration failed: {e}\n"
            f"Could not reach {url} — check the network and --api-base."
        )
    log.info(
        "Registered self-hosted box %r (%s) — status %s on the Compute page",
        body.get("name"), box_id, body.get("status"),
    )


def _serve_argv(args) -> list[str]:
    """The argv handed to :func:`serve_gpu.main`.

    ``interlatent-serve`` delegates in-process rather than exec'ing, so the
    forwarding is a literal argv rebuild. Both call sites go through here —
    when they didn't, ``--warmup-image-keys`` was easy to add to one and
    forget in the other.
    """
    argv = [sys.argv[0], "--host", args.host, "--port", str(args.port)]
    if args.warmup_policy:
        argv += ["--warmup-policy", args.warmup_policy]
    if args.warmup_image_keys:
        argv += ["--warmup-image-keys", args.warmup_image_keys]
    return argv


def main() -> None:
    p = argparse.ArgumentParser(
        prog="interlatent-serve",
        description="Self-hosted Interlatent DRTC policy server: register "
        "this GPU machine with the dashboard, then serve.",
    )
    p.add_argument(
        "--api-key",
        default=os.environ.get("INTERLATENT_API_KEY", ""),
        help="Your ilat_ API key (env INTERLATENT_API_KEY). The box "
        "registers, reports status, and gates its gRPC port with it.",
    )
    p.add_argument(
        "--api-base",
        default=os.environ.get("INTERLATENT_API_BASE", "") or DEFAULT_API_BASE,
        help=f"Backend base URL (default {DEFAULT_API_BASE}).",
    )
    p.add_argument(
        "--advertise-address",
        default=os.environ.get("INTERLATENT_ADVERTISE_ADDRESS", ""),
        help="host[:port] your robot nodes can reach this machine at "
        "(public IP, LAN IP, or VPN address). Required to register — "
        "the dashboard hands this address to nodes.",
    )
    p.add_argument("--name", default=socket.gethostname(),
                   help="Display name on the Compute page (default: hostname).")
    p.add_argument("--box-id", default=os.environ.get("INTERLATENT_BOX_ID", ""),
                   help="Override the persisted box UUID (advanced).")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=50051)
    p.add_argument(
        "--warmup-policy",
        default=os.environ.get("DRTC_WARMUP_POLICY", ""),
        help="Optional HF repo / local path to pre-warm when no env is "
        "attached in the dashboard yet. Once an env IS attached, the "
        "backend's warmup target wins.",
    )
    p.add_argument(
        "--warmup-image-keys",
        default=os.environ.get("DRTC_WARMUP_IMAGE_KEYS", ""),
        help="Comma-separated camera names to pre-warm --warmup-policy "
        "with (e.g. 'top,wrist'). Required for MolmoAct2. Must match the "
        "node's --camera names — see serve_gpu for why a mismatch is worse "
        "than no warm at all.",
    )
    p.add_argument(
        "--insecure", action="store_true",
        help="Serve without the owner-key RPC check (air-gapped LANs only).",
    )
    p.add_argument(
        "--no-register", action="store_true",
        help="Skip dashboard registration (pure-local smoke test).",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from interlatent_server import box_status, serve_gpu

    if args.no_register:
        os.environ.pop("INTERLATENT_BOX_ID", None)
        sys.argv = _serve_argv(args) + ["--insecure"]
        serve_gpu.main()
        return

    if not args.api_key.strip():
        raise SystemExit(
            "An API key is required to register with the dashboard: pass "
            "--api-key or set INTERLATENT_API_KEY (create one on the "
            "dashboard). Use --no-register for a purely local run."
        )
    if not args.advertise_address.strip():
        raise SystemExit(
            "--advertise-address (or INTERLATENT_ADVERTISE_ADDRESS) is "
            "required: it is the address your robot nodes dial, and the "
            "dashboard hands it to them verbatim. Use this machine's "
            "public/LAN/VPN IP, e.g. --advertise-address 203.0.113.7"
        )
    endpoint = args.advertise_address.strip()
    if ":" not in endpoint:
        endpoint = f"{endpoint}:{args.port}"

    box_id = _resolve_box_id(args.box_id)
    gpu_model = _detect_gpu_model()
    gpu_count, vram_gb = _detect_gpu_capacity()
    log.info("GPU: %s x%d (%s GB each)", gpu_model, gpu_count, vram_gb or "?")

    _register(
        api_base=args.api_base,
        api_key=args.api_key.strip(),
        box_id=box_id,
        name=args.name,
        endpoint=endpoint,
        gpu_model=gpu_model,
        gpu_count=gpu_count,
        vram_gb=vram_gb,
        warmup_policy=args.warmup_policy or None,
    )

    # Hand the owner-key identity to serve_gpu / box_status / the recorder
    # via the environment — credentials.resolve() picks it up everywhere.
    os.environ["INTERLATENT_BOX_ID"] = box_id
    os.environ["INTERLATENT_API_BASE"] = args.api_base
    os.environ["INTERLATENT_API_KEY"] = args.api_key.strip()
    os.environ["INTERLATENT_ADVERTISE_ADDRESS"] = endpoint
    # Never inherit a stray system secret into the BYO identity.
    os.environ.pop("INTERLATENT_ADMIN_KEY", None)

    sys.argv = _serve_argv(args)
    if args.insecure:
        sys.argv += ["--insecure"]
    try:
        serve_gpu.main()
    except KeyboardInterrupt:
        pass
    finally:
        # Graceful goodbye so the dashboard doesn't show a ghost "ready"
        # box (wait=True — a daemon thread would die with the process). On
        # a hard kill the row simply goes stale until the next re-register.
        box_status.report_status(
            "stopped", detail="interlatent-serve exited", wait=True
        )


if __name__ == "__main__":
    main()
