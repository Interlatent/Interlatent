"""Self-register a bring-your-own (BYO) GPU box with the hosted dashboard.

A box you run yourself (``interlatent-serve`` on your own hardware) is
invisible to the hosted dashboard — the dashboard only knows about boxes
it provisioned. When the operator supplies their own API key, this module
makes the box dial *out* and announce itself, so it shows up as a
launchable box, the same way a provisioned box does (cf. the closed-source
``interlatent.cloud.box_status``).

Two-phase, mirroring the dashboard contract:

1. **register** — a one-time handshake (``POST /api/v1/compute/boxes/register``)
   carrying the box's identity, reachable DRTC endpoint, GPU, and warmup
   policy. Authenticated with the operator's own ``x-api-key``. Idempotent:
   keyed by a UUID the box mints once and persists, so a restart re-attaches
   to the same dashboard box instead of orphaning a new one.
2. **status** — lightweight activity transitions
   (``POST /api/v1/compute/boxes/{box_id}/status``) reusing the same key:
   ``ready`` when idle, ``running`` while serving, ``uploading`` while
   flushing, and a best-effort ``stopped`` on clean shutdown.

Everything here is best-effort and stdlib-only (``urllib``): a reporting
failure must never block the gRPC loop or fail a session. The whole module
is a no-op when no API key is configured.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Activity states a box may self-report. ``ready``/``running``/``uploading``
# mirror the closed-source box; ``stopped`` is the graceful-shutdown signal
# the backend accepts only from the owning user (not the admin key).
_REPORTABLE = {"ready", "running", "uploading", "stopped"}

# Where the minted box id is persisted so it survives restarts. Mount this
# directory on a volume in containers, or the box re-registers fresh each
# boot (a new dashboard row every restart).
_BOX_ID_PATH = Path.home() / ".interlatent" / "box-id"


def _api_v1_root(api_base: str) -> str:
    """Normalize an API base so the result ends in exactly one ``/api/v1``."""
    base = api_base.rstrip("/")
    if not base.endswith("/api/v1"):
        base = f"{base}/api/v1"
    return base


def box_id() -> str:
    """Stable identifier for this box.

    Precedence: ``INTERLATENT_BOX_ID`` env (for ephemeral-disk containers or
    a dashboard-pinned id) → a value persisted under ``~/.interlatent`` →
    a freshly minted UUID, which is then persisted for next time.
    """
    env = os.environ.get("INTERLATENT_BOX_ID", "").strip()
    if env:
        return env
    try:
        existing = _BOX_ID_PATH.read_text().strip()
        if existing:
            return existing
    except OSError:
        pass
    import uuid

    minted = str(uuid.uuid4())
    try:
        _BOX_ID_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _BOX_ID_PATH.with_suffix(".tmp")
        tmp.write_text(minted)
        tmp.replace(_BOX_ID_PATH)
    except OSError:
        log.warning(
            "Could not persist box id to %s; this box will get a new "
            "dashboard identity on restart.",
            _BOX_ID_PATH,
            exc_info=True,
        )
    return minted


def detect_gpu() -> str:
    """Best-effort human label for the local GPU, e.g. ``NVIDIA A100 · 80GB``.

    Returns ``"unknown"`` without torch/CUDA (CPU test backends, etc.).
    Never raises — GPU discovery must not break startup.
    """
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            gb = round(props.total_memory / (1024 ** 3))
            return f"{name} · {gb}GB"
    except Exception:
        pass
    return "unknown"


class BoxReporter:
    """Dials out to the hosted dashboard to register + report box status."""

    def __init__(
        self,
        *,
        api_base: str,
        api_key: str,
        box_id: str,
        name: str,
        endpoint: str,
        gpu_model: str,
        warmup_policy: str = "",
        gpu_id: str = "custom",
    ) -> None:
        self._root = _api_v1_root(api_base)
        self._api_key = api_key
        self._box_id = box_id
        self._name = name
        self._endpoint = endpoint
        self._gpu_model = gpu_model
        self._warmup_policy = warmup_policy
        self._gpu_id = gpu_id

    @property
    def box_id(self) -> str:
        return self._box_id

    def _request(self, path: str, payload: dict) -> None:
        req = urllib.request.Request(
            self._root + path,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={"x-api-key": self._api_key, "content-type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10.0):
            pass

    def _register_blocking(self) -> bool:
        payload = {
            "box_id": self._box_id,
            "name": self._name,
            "endpoint": self._endpoint,
            "gpu_model": self._gpu_model,
            "gpu_id": self._gpu_id,
            "provider": "byo",
            "warmup_policy": self._warmup_policy or None,
        }
        try:
            self._request("/compute/boxes/register", payload)
            log.info(
                "Registered box %s with dashboard at %s (endpoint=%s, gpu=%s)",
                self._box_id, self._root, self._endpoint, self._gpu_model,
            )
            return True
        except urllib.error.HTTPError as e:
            log.warning(
                "Box register returned HTTP %d — box will not appear in the "
                "dashboard. Check the API key/base.",
                e.code,
            )
        except Exception:
            log.warning("Box register failed; box not discoverable", exc_info=True)
        return False

    async def register(self) -> bool:
        """Run the (blocking) register handshake off the event loop."""
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._register_blocking)

    def _status_blocking(self, status: str) -> None:
        try:
            self._request(
                f"/compute/boxes/{self._box_id}/status",
                {"status": status, "endpoint": self._endpoint},
            )
            log.info("Reported box status=%s", status)
        except urllib.error.HTTPError as e:
            log.warning("Box status report (%s) returned HTTP %d", status, e.code)
        except Exception:
            log.warning("Box status report (%s) failed", status, exc_info=True)

    def report_status(self, status: str, *, block: bool = False) -> None:
        """Report an activity transition.

        Fire-and-forget on a daemon thread by default (never stalls the
        caller). ``block=True`` posts synchronously — used on shutdown, where
        a daemon thread would be killed before the request leaves.
        """
        if status not in _REPORTABLE:
            log.warning("Ignoring non-reportable box status %r", status)
            return
        if block:
            self._status_blocking(status)
            return
        threading.Thread(
            target=self._status_blocking,
            args=(status,),
            name=f"box-status[{status}]",
            daemon=True,
        ).start()


def build_reporter(
    *,
    api_base: str,
    api_key: str,
    endpoint: str,
    warmup_policy: str = "",
    name: str = "",
) -> Optional[BoxReporter]:
    """Construct a reporter, or ``None`` when reporting is disabled.

    Disabled (returns ``None``) when no API key is configured — so every call
    site is a no-op by default and a keyless ``interlatent-serve`` behaves
    exactly as before.
    """
    if not api_key.strip():
        return None
    if not name:
        import socket

        name = socket.gethostname() or "interlatent-byo"
    return BoxReporter(
        api_base=api_base,
        api_key=api_key.strip(),
        box_id=box_id(),
        name=name,
        endpoint=endpoint,
        gpu_model=detect_gpu(),
        warmup_policy=warmup_policy,
    )
