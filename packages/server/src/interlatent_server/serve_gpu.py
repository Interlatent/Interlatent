"""Persistent-GPU DRTC inference server (native gRPC).

This is the production launcher. The reference deployment is a RunPod
pod reached at its public IP:port; any Linux host with a CUDA GPU
works the same way. The server is a long-running asyncio process so
the SmolVLA-class policies' multi-minute torch.compile happens
exactly once per process life — at first OpenSession, or up-front via
``--warmup-policy``.

Transport: native gRPC over HTTP/2. No HTTP proxy, no gRPC-Web
shim, no per-call container churn. A multi-minute OpenSession is
just a long-lived RPC the client waits on over a persistent stream.

Networking:
  The node reaches the box at ``<host>:<port>`` — the box's public
  IP:port (managed provisioning maps this automatically) or any address
  routable from the node (LAN, VPN, firewalled WAN). The node is always
  the connection initiator, so the box needs no inbound route back to
  it. A ``host:port`` address (no http/https scheme) makes the client
  select native gRPC automatically — see ``DRTCConfig.use_grpc_web``
  inference in ``connect_drtc``.

Auth:
  A self-hosted box (owner ``ilat_`` key identity) guards its gRPC port
  by default: every RPC's ``x-api-key`` metadata is validated against
  the backend's per-box authorization probe, so only the box owner's
  key can open sessions or record (``--insecure`` opts out for
  air-gapped LANs — then treat the network as the trust boundary). A
  dashboard-provisioned box keeps the historical unauthenticated port
  (its network posture is managed by the provisioner).

Run on the GPU box (prefer ``interlatent-serve``, which registers the
box with the dashboard first — see :mod:`interlatent_server.cli`):

    # Python >= 3.12 + torch 2.7 (cu128). MolmoAct2 is only on lerobot
    # main (post-0.5.1), so install from the pinned git ref until a
    # release ships it.
    pip install 'interlatent-server[lerobot]'
    python -m interlatent_server.serve_gpu --port 50051 \\
        --warmup-policy rn-1/boxlift_twocam_better

``--warmup-policy`` is optional: when given, the policy is loaded and
compiled at startup so the first robot session is fast. Otherwise the
first OpenSession pays the one-time torch.compile.
"""

from __future__ import annotations

import argparse
import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .server.sinks import DatasetSink
import logging
import os

log = logging.getLogger("serve_gpu")


def _resolve_core_split() -> tuple[int, int, int]:
    """Split CPU cores between inference preprocessing and recording I/O.

    Returns ``(n_inference, n_recording, total)``. Adaptive from
    ``os.cpu_count()`` and overridable with ``DRTC_INFERENCE_THREADS``.

    A VLA's CPU preprocessing (e.g. MolmoAct2 image tiling + tokenization,
    ~1.3 s/step) is multi-threaded across BLAS/OMP and, unconstrained,
    grabs every core. Once a session is also recording, the 30 Hz full-res
    RecordTick stream (gRPC ingest + disk writes) competes for the same
    cores, so the preprocessing wall-clock balloons. Reserving a recording
    slice keeps the two workloads off each other's cores. Inference gets
    the bulk (it's the latency-critical path); recording gets a reserved
    minority — min(4, total//4), floor 1.
    """
    total = os.cpu_count() or 4
    env = os.environ.get("DRTC_INFERENCE_THREADS", "").strip()
    if env:
        try:
            n_inf = max(1, min(total, int(env)))
        except ValueError:
            n_inf = max(1, total - min(4, max(1, total // 4)))
    else:
        reserved = min(4, max(1, total // 4))
        n_inf = max(1, total - reserved)
    n_rec = max(1, total - n_inf)
    return n_inf, n_rec, total


def _pin_affinity(cores: list[int]) -> None:
    """Best-effort: pin the calling worker thread to ``cores`` (Linux only).

    torch/BLAS intraop threads spawned from a pinned thread inherit its
    affinity, so pinning the single inference worker confines the policy's
    CPU preprocessing to the inference cores while recording workers stay
    on the reserved cores. No-op where ``sched_setaffinity`` is missing
    (non-Linux), the core set is empty, or the call isn't permitted.
    """
    try:
        if cores:
            os.sched_setaffinity(0, set(cores))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass


def _has_box_identity() -> bool:
    """True when this box carries an identity to talk to the backend —
    dashboard-provisioned (admin key) or self-hosted (owner ilat_ key,
    registered via ``interlatent-serve``). Gates whether the warmup-target
    fetch is worth attempting at all.
    """
    from interlatent_server import credentials

    return credentials.resolve() is not None


def _is_provisioned_box() -> bool:
    """True only for a DASHBOARD-PROVISIONED box (system/admin key).

    Distinct from :func:`_has_box_identity` on purpose: a provisioned box's
    warmup config comes solely from the backend, while a self-hosted box's
    operator may pass ``--warmup-policy`` directly. See :func:`_warmup`.
    """
    from interlatent_server import credentials

    creds = credentials.resolve()
    return creds is not None and creds.is_system


def _fetch_warmup_target_from_backend() -> dict | None:
    """Ask the backend what to warm up with based on the env attached to
    this box in the dashboard. Returns the parsed JSON body or None if
    the box has no system identity / the call fails / no env is
    attached.

    The contract: the backend's
    ``GET /api/v1/compute/boxes/{box_id}/warmup-target`` returns
    ``{"policy_uri", "image_keys", "num_inference_steps",
       "inference_action_mode"}`` — all derived from
    ``Environment.{base_model, camera_names, ...}`` populated via the
    dashboard. Single source of truth for what the box should warm
    with, so the operator never has to hand-set per-tenant env vars on
    the GPU.
    """
    import json
    import urllib.request
    import urllib.error

    from interlatent_server import credentials

    creds = credentials.resolve()
    # Log presence (never the secret value) so a silent skip is debuggable.
    log.info(
        "Warmup identity: %s",
        "<missing>" if creds is None else (
            f"box_id={creds.box_id} api_base={creds.api_base} "
            f"key=<set:{'system' if creds.is_system else 'owner'}>"
        ),
    )
    if creds is None:
        log.warning(
            "Warmup-target fetch skipped — missing box identity. The box "
            "can't ask the backend which env/cameras to warm with, so it "
            "will fall back to DRTC_WARMUP_POLICY (no image_keys). Provision "
            "through the dashboard, or run `interlatent-serve` with your "
            "INTERLATENT_API_KEY.",
        )
        return None
    box_id = creds.box_id

    url = f"{creds.api_root}/api/v1/compute/boxes/{box_id}/warmup-target"
    log.info("Fetching warmup target from %s", url)
    req = urllib.request.Request(url, headers={"x-api-key": creds.api_key})
    try:
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        log.info(
            "Warmup target: policy_uri=%r image_keys=%r num_inference_steps=%r "
            "inference_action_mode=%r",
            body.get("policy_uri"), body.get("image_keys"),
            body.get("num_inference_steps"), body.get("inference_action_mode"),
        )
        return body
    except urllib.error.HTTPError as e:
        # Surface the backend's error body — it carries the actionable
        # detail (e.g. a 409 camera-mismatch message, or a 500 trace
        # summary) that a bare status code hides. Best-effort read; never
        # let the diagnostics themselves raise.
        body = ""
        try:
            body = e.read().decode("utf-8", "replace").strip()
        except Exception:
            pass
        # 404 means the box has no env attached yet — warmup is a no-op.
        if e.code == 404:
            log.info(
                "Backend reports no env attached to box %s; skipping warmup",
                box_id,
            )
        else:
            log.warning(
                "Warmup-target fetch returned HTTP %d; skipping warmup. "
                "Backend said: %s", e.code, body or "<empty body>",
            )
        return None
    except Exception:
        log.warning(
            "Warmup-target fetch failed; skipping warmup", exc_info=True,
        )
        return None


_IMAGE_KEY_PREFIX = "observation.images."


def _normalize_image_keys(raw: str) -> list[str]:
    """Parse ``--warmup-image-keys`` into full LeRobot feature keys.

    Accepts either form in one comma-separated list: bare camera names
    (``top,wrist``) or already-qualified keys
    (``observation.images.top``). Bare names get the prefix, because that
    is the form the node builds from its ``--camera <name>=<device>``
    flags and the only one a policy's feature dict will match — a bare
    ``top`` would warm a runtime whose cameras can never bind.
    """
    keys: list[str] = []
    for part in str(raw or "").split(","):
        name = part.strip()
        if not name:
            continue
        keys.append(
            name if name.startswith("observation.")
            else f"{_IMAGE_KEY_PREFIX}{name}"
        )
    return keys


def _warmup(policy_uri_override: str, image_keys_override: str = "") -> str | None:
    """Load + compile a policy at startup so its torch.compile artifacts
    land in the on-disk inductor cache before the first real session.

    Returns a non-fatal warning string when pre-warm was ATTEMPTED but
    failed (e.g. a transient weight-download truncation). Pre-warm is
    best-effort: a failure does NOT make the box unusable — the first
    real session loads/compiles on demand and may well succeed — so the
    box still goes ``ready``. The returned warning is surfaced as the
    box's ``status_detail`` so the operator can see startup was degraded.
    Returns None when warmup succeeded or was deliberately skipped (no
    policy / unsatisfiable half-config), neither of which is degraded.

    Config precedence, in order:

    1. **The backend warmup target**, whenever the fetch returns one. It
       carries policy_uri AND image_keys AND the inference knobs together,
       all derived from the env attached in the dashboard, so warm and
       first-session configs agree by construction. A target can be
       *partial* though — the backend answers with the policy the box
       registered even when no env is attached, and then image_keys is
       empty. ``image_keys_override`` fills exactly that hole (it never
       overrides keys the backend did supply), because otherwise a
       MolmoAct2 pre-warm is unreachable: the guard below skips for want
       of cameras and there is no env to go configure.
    2. **A dashboard-provisioned box** (system identity: ``INTERLATENT_BOX_ID``
       + ``INTERLATENT_API_BASE`` + ``INTERLATENT_ADMIN_KEY``) with no target
       skips pre-warm. The backend is that box's only configuration source —
       falling back to a static policy would hide the real failure behind a
       half-configured warm.
    3. **Everyone else** — a self-hosted (owner-key) box, or no identity at
       all — falls back to ``policy_uri_override`` /
       ``image_keys_override`` (``--warmup-policy`` /
       ``--warmup-image-keys``, or ``DRTC_WARMUP_POLICY`` /
       ``DRTC_WARMUP_IMAGE_KEYS``). An operator who typed the flag is
       standing right there; refusing it because a box happens to be
       registered is not the "silent fallback" case rule 2 guards against.

    Rule 3 previously keyed on ``_has_box_identity()``, which is true for an
    owner key too — so ``interlatent-serve --warmup-policy ...`` silently
    ignored the flag the moment the box registered, which is always.

    Caveat on rule 3: ``PolicyRuntime`` caches on ``(backend, policy_uri)``
    and ignores per-session metadata on reuse, so image_keys that DON'T
    match the node's cameras don't just waste the warm — the first real
    session inherits the wrong-camera runtime. Rule 1 can't hit this (same
    source feeds both); a hand-typed list can. Match the node's
    ``--camera <name>=<device>`` names exactly.
    """
    from interlatent_server.server.policy_runtime import PolicyRuntime

    target = _fetch_warmup_target_from_backend()
    if target is not None:
        policy_uri = (target.get("policy_uri") or "").strip()
        if not policy_uri:
            log.info(
                "Backend warmup target has no policy_uri; skipping warmup"
            )
            return
        meta: dict[str, str] = {}
        image_keys = target.get("image_keys") or []
        override_keys = _normalize_image_keys(image_keys_override)
        if image_keys:
            meta["image_keys"] = ",".join(str(k) for k in image_keys)
            log.info("Resolved %d camera image_keys: %s", len(image_keys), image_keys)
            if override_keys and override_keys != [str(k) for k in image_keys]:
                # Never silently inert: the operator passed a flag and the
                # backend outranks it. Say which one is in effect and why.
                log.warning(
                    "Ignoring --warmup-image-keys %s — the backend's warmup "
                    "target supplied %s, and it wins because it feeds the "
                    "node's cameras and this warm from the same env, so the "
                    "two cannot disagree. Change the env's camera_names to "
                    "override.", override_keys, image_keys,
                )
        elif override_keys:
            # A target with a policy but no cameras. Filling that hole from
            # the operator's flag is not overriding the backend — the backend
            # supplied nothing here, and without it a MolmoAct2 pre-warm is
            # unreachable (its guard below skips, and there is no env to fix).
            meta["image_keys"] = ",".join(override_keys)
            log.info(
                "Backend warmup target carries no image_keys (no env attached, "
                "or the env has no camera_names); using --warmup-image-keys %s. "
                "These must match the node's --camera names — the runtime cache "
                "is keyed on (backend, policy_uri), so the first real session "
                "inherits this warm.", override_keys,
            )
        else:
            log.warning(
                "Warmup target for policy %s returned an EMPTY image_keys list "
                "— the attached env has no camera_names configured. Set camera "
                "names in the env's Configuration tab, or pass "
                "--warmup-image-keys.", policy_uri,
            )
        nis = target.get("num_inference_steps")
        if nis is not None:
            meta["num_inference_steps"] = str(nis)
        iam = (target.get("inference_action_mode") or "").strip()
        if iam:
            meta["inference_action_mode"] = iam
    elif _is_provisioned_box():
        # Dashboard-provisioned box that couldn't fetch its target (see the
        # warning above for which step failed). The backend is the single
        # source of truth here — do NOT fall back to a static policy, which
        # would hide the failure behind a half-configured warm. Skip; first
        # session compiles.
        log.warning(
            "Box has system (dashboard-provisioned) identity but the "
            "warmup-target fetch did not return a target — skipping pre-warm "
            "(the first real session will compile instead). Not falling back "
            "to a static policy: the backend is this box's only config source.",
        )
        return None
    else:
        # Self-hosted box, or none at all: the operator's own
        # --warmup-policy / --warmup-image-keys is the config source.
        policy_uri = (policy_uri_override or "").strip()
        if not policy_uri:
            return None
        meta = {}
        keys = _normalize_image_keys(image_keys_override)
        if keys:
            meta["image_keys"] = ",".join(keys)
        log.info(
            "No warmup target from the backend — using the operator-supplied "
            "--warmup-policy %s (image_keys=%s). Attach this box to an "
            "environment in the dashboard to have the target supplied "
            "automatically.",
            policy_uri, meta.get("image_keys") or "none",
        )

    # MolmoAct2's released checkpoint can't load without image_keys.
    # If we somehow got here without them (no env attached + a manual
    # DRTC_WARMUP_POLICY pointing at MolmoAct2), skip rather than fail
    # silently inside the cache lookup.
    if "molmoact" in policy_uri.lower() and "image_keys" not in meta:
        log.warning(
            "Pre-warm skipped for MolmoAct2 policy %s — no image_keys "
            "available, and its checkpoint cannot build its feature dict "
            "without them. Either attach this box to an env (with cameras "
            "configured) in the dashboard, or pass --warmup-image-keys "
            "<name>[,<name>...] naming the SAME cameras the node runs with.",
            policy_uri,
        )
        return None

    log.info(
        "Pre-warming policy %s (session_metadata keys=%s) ...",
        policy_uri, sorted(meta.keys()) or "none",
    )
    try:
        # Route released MolmoAct2 checkpoints to their dedicated backend,
        # matching the transport's OpenSession routing so the pre-warm and
        # the first real session share the same (backend, uri) cache key.
        from interlatent_server.server.molmoact2_backend import resolve_backend

        PolicyRuntime.load(
            backend=resolve_backend("lerobot", policy_uri),
            policy_uri=policy_uri,
            session_metadata=meta or None,
        )
        log.info("Pre-warm complete — torch.compile cache is now populated")
        return None
    except Exception as exc:
        log.warning(
            "Pre-warm failed — the first real session will load on demand "
            "(startup may be slower)",
            exc_info=True,
        )
        # Best-effort: the box stays launchable. Surface the degradation as
        # a non-fatal status_detail note rather than blocking readiness.
        return (
            f"pre-warm failed ({type(exc).__name__}: {exc}); first session "
            "will load on demand, startup may be slower"
        )


async def _serve(
    host: str,
    port: int,
    *,
    warmup_warning: str | None = None,
    guard_rpcs: bool = False,
    dataset_sink: "DatasetSink | None" = None,
) -> None:
    import grpc

    from interlatent_server.protocol import messages_pb2_grpc as pb_grpc
    from interlatent_server.server.transport import InferenceServicer

    # InferenceServicer defaults to an in-process InMemoryChunkBuffer,
    # which is what we want: a single long-running process never has
    # to ship session state to a second worker, so no external KV
    # store is required.
    #
    # Keepalive options: the client sends HTTP/2 pings every 10s of
    # idle to keep cloud TCP proxies from half-closing long-lived
    # streams (see DRTCClient.open for rationale). gRPC's server-side
    # defaults reject pings faster than every 5 minutes with a GOAWAY,
    # so we have to relax that here or the keepalive pings get the
    # client kicked off. ``max_pings_without_data=0`` allows unlimited
    # pings during idle; ``min_ping_interval_without_data_ms=5000``
    # accepts the client's 10s ping cadence without complaint.
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

    # CPU isolation between inference and recording (see _resolve_core_split):
    #   - inference_executor: one worker, pinned to the inference cores. The
    #     blocking policy.forward() runs here instead of on the event loop,
    #     so a ~1.3 s VLA preprocess no longer stalls 30 Hz RecordTick
    #     ingest. A single worker preserves the in-painting buffer ordering.
    #   - recording_executor: the loop's DEFAULT executor, pinned to the
    #     reserved cores. The recorder's disk writes + the upload rebuild
    #     (run_in_executor(None, ...)) land here, off the inference cores.
    from concurrent.futures import ThreadPoolExecutor

    n_inf, n_rec, total_cores = _resolve_core_split()
    try:
        import torch
        torch.set_num_threads(n_inf)
    except Exception:
        log.debug("torch.set_num_threads skipped", exc_info=True)
    inf_cores = list(range(0, n_inf))
    rec_cores = list(range(n_inf, total_cores)) or inf_cores
    inference_executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="drtc-infer",
        initializer=_pin_affinity,
        initargs=(inf_cores,),
    )
    recording_executor = ThreadPoolExecutor(
        max_workers=max(1, n_rec),
        thread_name_prefix="drtc-record",
        initializer=_pin_affinity,
        initargs=(rec_cores,),
    )
    asyncio.get_running_loop().set_default_executor(recording_executor)
    log.info(
        "CPU isolation active: inference cores=%s recording cores=%s "
        "(record pool workers=%d)",
        inf_cores, rec_cores, max(1, n_rec),
    )

    servicer = InferenceServicer(
        inference_executor=inference_executor,
        warmup_warning=warmup_warning,
        dataset_sink=dataset_sink,
    )
    if guard_rpcs:
        # Self-hosted default: gate every RPC on "does this x-api-key belong
        # to this box's owner?" via the backend's per-box authz probe. A
        # public-IP box without this is an open GPU.
        from interlatent_server import credentials
        from interlatent_server.server.auth import (
            build_box_key_validator,
            wrap_servicer_with_auth,
        )

        creds = credentials.resolve()
        if creds is None:
            log.warning(
                "RPC auth requested but the box has no identity — serving "
                "UNGUARDED (pass --insecure to silence this, or supply "
                "INTERLATENT_API_KEY so the owner check can run)."
            )
        else:
            servicer = wrap_servicer_with_auth(
                servicer,
                check_token=build_box_key_validator(
                    box_id=creds.box_id, api_base=creds.api_base
                ),
            )
            log.info("RPC auth active: only this box's owner key is accepted")
    pb_grpc.add_InferenceServiceServicer_to_server(servicer, server)

    server.add_insecure_port(f"{host}:{port}")
    await server.start()
    log.info("DRTC server listening on %s:%d (native gRPC)", host, port)

    # The gRPC port is now bound AND _warmup() (run synchronously before
    # _serve) has finished compiling, so the box is genuinely ready to
    # serve. Self-report "ready" to the backend — this is what surfaces
    # the box as launchable in the dashboard (replacing the old
    # Vercel-side TCP probe, which couldn't reliably reach the box).
    from interlatent_server.box_status import report_status as _report_box_status

    # The idle-GC reconcile becomes the sole status writer once a session
    # opens, but it doesn't run until the first RPC — so report the initial
    # "ready" here to surface the box as launchable. Carry warmup_warning as
    # status_detail so a degraded (but still usable) pre-warm is visible;
    # the first successful session clears it.
    # The address nodes should dial, when the box knows it (self-hosted
    # `--advertise-address`, or a provisioner that exports it). host or
    # host:port; bare hosts get this server's port appended.
    _box_endpoint = os.environ.get("INTERLATENT_ADVERTISE_ADDRESS", "").strip()
    if _box_endpoint and ":" not in _box_endpoint:
        _box_endpoint = f"{_box_endpoint}:{port}"
    _report_box_status(
        "ready",
        endpoint=_box_endpoint or None,
        detail=warmup_warning,
    )

    # Teleop is QUIC-only: the browser runs IK and streams joint targets to
    # the node over the WebTransport/QUIC relay (deployed separately). The pod
    # is not on the teleop control path, so no relay runs here. (The pod-side
    # WS relay + retarget stage were removed; a mid-rollout intervention relay
    # will be rebuilt separately if/when live takeover ships.)

    await server.wait_for_termination()


def main() -> None:
    p = argparse.ArgumentParser(prog="interlatent-drtc-server")
    p.add_argument(
        "--output-dir", default=os.environ.get("DRTC_OUTPUT_DIR", ""),
        help="Publish finished datasets into this directory, merge-on-stop "
             "into one flat LeRobot dataset. No account needed.",
    )
    p.add_argument(
        "--s3-uri", default=os.environ.get("DRTC_S3_URI", ""),
        help="Publish to s3://bucket/prefix (merge-on-stop against a local "
             "mirror, then upload). Needs the [s3] extra.",
    )
    p.add_argument("--s3-endpoint-url", default=os.environ.get("DRTC_S3_ENDPOINT_URL", ""),
                   help="For S3-compatible stores (MinIO, Cloudflare R2, ...).")
    p.add_argument("--s3-access-key", default=os.environ.get("DRTC_S3_ACCESS_KEY", ""))
    p.add_argument("--s3-secret-key", default=os.environ.get("DRTC_S3_SECRET_KEY", ""))
    p.add_argument("--s3-region", default=os.environ.get("DRTC_S3_REGION", ""))
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=50051)
    p.add_argument(
        "--warmup-policy",
        default=os.environ.get("DRTC_WARMUP_POLICY", ""),
        help="Optional HF repo / local path to load + compile at startup "
        "so the first robot session is fast. Used when the backend returns "
        "no warmup target; a dashboard-provisioned box ignores it.",
    )
    p.add_argument(
        "--warmup-image-keys",
        default=os.environ.get("DRTC_WARMUP_IMAGE_KEYS", ""),
        help="Comma-separated camera names (or full observation.images.* "
        "keys) to pre-warm --warmup-policy with. Required for MolmoAct2, "
        "which cannot load without them. MUST match the node's --camera "
        "names: the runtime cache is keyed on (backend, policy_uri), so a "
        "mismatched warm is inherited by the first real session.",
    )
    p.add_argument(
        "--insecure",
        action="store_true",
        default=os.environ.get("INTERLATENT_INSECURE", "").strip() == "1",
        help="Serve the gRPC port without the owner-key check. Only for "
        "air-gapped/private networks — a public-IP box without auth is an "
        "open GPU.",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Persist torch.compile artifacts across restarts. torch's default
    # cache lives under /tmp, which is wiped on a box reboot — so
    # every restart re-pays the full multi-minute compile. Pin the
    # inductor + triton caches under $HOME (persists for the
    # instance) unless the operator already set them. Must happen
    # before torch is imported / any compile runs.
    _cache = os.path.expanduser("~/.cache")
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(_cache, "torchinductor"))
    os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_cache, "triton"))
    os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
    log.info(
        "torch.compile cache -> %s (persists across restarts)",
        os.environ["TORCHINDUCTOR_CACHE_DIR"],
    )

    # Parallelize inductor's kernel compilation across CPU cores. The
    # default worker-pool size is often capped by cgroups inside a
    # container (sometimes to 1), which makes max-autotune sequential
    # over hundreds of kernels and stretches first-load to nearly an
    # hour. Must be set before torch is imported.
    #
    # Worker start method MUST be `spawn`, not `fork`. By the time inductor
    # spawns these workers (during the pre-warm's synthetic forward) the
    # policy has already been moved to CUDA, so the main process holds a
    # live CUDA context. Forking after CUDA init and then touching the GPU
    # in the workers (inductor autotune benchmarks kernels on-device)
    # deadlocks — the box hangs in pre-warm and never reaches _serve to
    # self-report "ready". This only bites compiling policies (SmolVLA /
    # Pi0 / ACT); MolmoAct2 leaves torch.compile off so it never forked and
    # never hit this. `spawn` re-imports torch per worker (slightly slower
    # startup) but is CUDA-safe.
    _cpu = os.cpu_count() or 8
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", str(_cpu))
    os.environ.setdefault("TORCHINDUCTOR_WORKER_START_METHOD", "spawn")
    log.info(
        "torch.compile workers -> %s (%s start)",
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"],
        os.environ["TORCHINDUCTOR_WORKER_START_METHOD"],
    )

    # Runtime intraop CPU budget — distinct from the compile-thread pool
    # above (those cores are used only transiently during torch.compile).
    # At inference time the policy's CPU preprocessing is multi-threaded
    # via BLAS/OMP and would otherwise grab every core, starving the gRPC
    # ingest + disk-writer threads that absorb the 30 Hz full-res
    # RecordTick stream (and vice versa) — which is what inflated `pre` to
    # ~1.3 s once recording turned on. Cap intraop parallelism to the
    # inference core budget so recording keeps dedicated cores. Must be
    # set before torch / numpy import so the BLAS backends honor it.
    _n_inf, _n_rec, _total_cores = _resolve_core_split()
    for _var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(_var, str(_n_inf))
    log.info(
        "CPU split: %d cores total -> %d inference intraop / %d recording+IO "
        "(override with DRTC_INFERENCE_THREADS)",
        _total_cores, _n_inf, _n_rec,
    )

    # Importing the server package registers all backends (echo,
    # tiny_torch, lerobot, molmoact2).
    import interlatent_server.server  # noqa: F401

    # Warm up when this is a dashboard box (system identity -> backend fetch
    # is the source of truth) OR when an operator explicitly passed
    # --warmup-policy for a manual/local run. A dashboard box no longer needs
    # DRTC_WARMUP_POLICY to trigger warmup.
    warmup_warning: str | None = None
    if _has_box_identity() or args.warmup_policy:
        warmup_warning = _warmup(args.warmup_policy, args.warmup_image_keys)

    # Owner-checked RPC auth is the default on a self-hosted box (owner-key
    # identity). Provisioned boxes (system identity) keep the historical
    # unauthenticated port; --insecure opts a self-hosted box out.
    from interlatent_server import credentials as _credentials

    _creds = _credentials.resolve()
    guard_rpcs = (
        not args.insecure and _creds is not None and not _creds.is_system
    )
    if not guard_rpcs and _creds is not None and not _creds.is_system:
        log.warning(
            "Serving with --insecure: the gRPC port accepts ANY caller. "
            "Keep it off the public internet."
        )

    # A destination configured here is this box's own fallback; a coordinator
    # that stamps one onto its sessions overrides it per session (ADR 0002).
    dataset_sink = None
    if args.output_dir:
        from .server.sinks import LocalDirSink
        dataset_sink = LocalDirSink(args.output_dir)
        log.info("Recording destination: local dir %s", args.output_dir)
    elif args.s3_uri:
        from .server.sinks import S3Sink
        dataset_sink = S3Sink.from_uri(
            args.s3_uri,
            endpoint_url=args.s3_endpoint_url or None,
            access_key=args.s3_access_key or None,
            secret_key=args.s3_secret_key or None,
            region=args.s3_region or None,
        )
        log.info("Recording destination: %s", args.s3_uri)

    asyncio.run(_serve(
        args.host,
        args.port,
        warmup_warning=warmup_warning,
        guard_rpcs=guard_rpcs,
        dataset_sink=dataset_sink,
    ))


if __name__ == "__main__":
    main()
