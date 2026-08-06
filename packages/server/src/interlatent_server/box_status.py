"""Box self-reporting of its DRTC activity status to the backend.

A GPU box owns the *activity* slice of its own status (``ready`` |
``running`` | ``uploading``) — the backend can't see it (a serverless
backend can't dial into the box). So the box dials *out* and POSTs each
transition to ``POST /api/v1/compute/boxes/{box_id}/status`` using its
box identity (see :mod:`interlatent_server.credentials`): the shared
admin key on a dashboard-provisioned box, or the owner's ``ilat_`` key
on a self-hosted box. A self-hosted box's owner may additionally report
``stopped`` for a graceful shutdown.

This is a no-op when the box has no identity (local dev / smoke tests).
Reporting is fire-and-forget on a daemon thread: a status ping must
never block the gRPC event loop or fail a session, so all errors are
swallowed with a warning.
"""
from __future__ import annotations

import json
import logging
import threading
import urllib.error
import urllib.request

from interlatent_server import credentials

log = logging.getLogger(__name__)

# The backend-owned states (warming_up / error) are NOT reportable here —
# the backend rejects them. "stopped" is accepted only from a box
# authenticating as its owner (self-hosted graceful shutdown).
_REPORTABLE = {"ready", "running", "uploading", "stopped"}


def has_box_identity() -> bool:
    """True when this box carries an identity to talk to the backend."""
    return credentials.resolve() is not None


def _post(status: str, endpoint: str | None, detail: str | None) -> None:
    creds = credentials.resolve()
    if creds is None:
        return
    url = f"{creds.api_root}/api/v1/compute/boxes/{creds.box_id}/status"
    payload: dict[str, str] = {"status": status}
    if endpoint:
        payload["endpoint"] = endpoint
    # ``status_detail`` is a non-fatal human-facing note (e.g. a degraded
    # pre-warm). Always send the field so the backend can CLEAR a stale
    # note: detail=None serializes to status_detail=null, which the
    # backend writes back as "no note".
    payload["status_detail"] = detail  # type: ignore[assignment]
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"x-api-key": creds.api_key, "content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10.0):
            pass
        log.info("Reported box status=%s to backend", status)
    except urllib.error.HTTPError as e:
        log.warning("Box status report (%s) returned HTTP %d", status, e.code)
    except Exception:
        log.warning("Box status report (%s) failed", status, exc_info=True)


def report_status(
    status: str,
    endpoint: str | None = None,
    detail: str | None = None,
    *,
    wait: bool = False,
) -> None:
    """Fire-and-forget report of a box activity-state transition.

    ``detail`` is an optional non-fatal note persisted as the box's
    ``status_detail`` (e.g. a degraded pre-warm warning). Pass None to
    clear any existing note.

    No-op without box identity. Runs the blocking POST on a daemon thread
    so it never stalls the caller's event loop. ``wait=True`` runs it
    inline instead — for process-exit reports (a daemon thread dies with
    the process before the POST lands).
    """
    if status not in _REPORTABLE:
        log.warning("Ignoring non-reportable box status %r", status)
        return
    creds = credentials.resolve()
    if creds is None:
        return
    if status == "stopped" and creds.is_system:
        # A provisioned box's terminal state is backend-driven via the
        # provider; the backend rejects a system-key "stopped".
        log.warning("Ignoring 'stopped' self-report on a provisioned box")
        return
    if wait:
        _post(status, endpoint, detail)
        return
    threading.Thread(
        target=_post,
        args=(status, endpoint, detail),
        name=f"box-status[{status}]",
        daemon=True,
    ).start()
