"""Build the node's teleop channel.

Teleop runs over QUIC/WebTransport only: the browser owns IK and streams
``mode="targets"`` datagrams, while the node serves the kinematic_spec and tees
a live preview back. The channel exposes ``start``/``stop``/``latest_frame``/
``send_state``/``connected``, so the control loop is transport-agnostic.

A one-shot node-role token mint reveals the deployment's ``transport`` +
``webtransport_url``. ``aioquic`` is never imported in this process — the quic
channel's child process uses it — so availability is probed via ``find_spec``
before building the channel.

The probe distinguishes *definitive* failures from *transient* ones. A
deployment that answers "not QUIC" or "forbidden" disables teleop for the
session (``None``; the daemon re-runs the factory on the next assignment). But
a 409 while the session row isn't ``active`` yet, a 5xx, or a network blip is
just a race at session start — the channel is built optimistically, because
its child process re-mints with its own retry/backoff loop and never sees this
probe's result. Returning ``None`` on those used to disable teleop for the
whole session over a startup race.
"""
from __future__ import annotations

import logging
from typing import Optional

from ._mint import mint_teleop_token

_LOG = logging.getLogger(__name__)

#: HTTP statuses that mean "teleop will not become available for this
#: session" — everything else is treated as transient.
_DEFINITIVE_STATUSES = frozenset({401, 403, 404})


def make_teleop_channel(
    *,
    session_id: str,
    api_base: str,
    api_key: str,
    token_path: Optional[str] = None,
    robot_kind: Optional[str] = None,
):
    """Return a QUIC teleop channel, or ``None`` when teleop is unavailable."""
    probe_path = (
        token_path or f"/api/v1/inference/sessions/{session_id}/teleop-token"
    )
    probe_inconclusive = False
    try:
        data = mint_teleop_token(
            api_base=api_base,
            token_path=probe_path,
            api_key=api_key,
            role="node",
        )
        transport = str(data.get("transport") or "")
        webtransport_url = data.get("webtransport_url")
    except Exception as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in _DEFINITIVE_STATUSES:
            _LOG.info(
                "teleop transport probe refused (HTTP %s); teleop disabled",
                status,
            )
            return None
        # Transient: the session row may not be `active` yet (409 at session
        # start), the backend may be briefly unavailable, or the network
        # blipped. Build the channel anyway — its child process mints its own
        # token with retry/backoff and never reads this probe's result.
        probe_inconclusive = True
        transport, webtransport_url = "", None
        _LOG.info(
            "teleop transport probe inconclusive (%s); building the QUIC "
            "channel optimistically — the channel retries the mint itself",
            exc,
        )

    if not probe_inconclusive and (transport != "quic" or not webtransport_url):
        _LOG.info(
            "teleop unavailable: deployment is not QUIC-configured "
            "(transport=%r); teleop disabled",
            transport,
        )
        return None

    # The parent process never imports aioquic (the connection lives in the
    # QuicTeleopChannel child process, which uses the same interpreter/venv) —
    # so probe availability explicitly here.
    import importlib.util

    if importlib.util.find_spec("aioquic") is None:
        _LOG.warning(
            "QUIC teleop unavailable (aioquic not installed — "
            "pip install 'interlatent[teleop-quic]'); teleop disabled"
        )
        return None
    try:
        from .quic_channel import QuicTeleopChannel
    except Exception as exc:
        _LOG.warning("QUIC teleop unavailable (%s); teleop disabled", exc)
        return None

    _LOG.info("teleop transport=quic session=%s", session_id)
    # robot_kind is quic-only: the browser owns IK and builds its solver from
    # the node-served kinematic_spec.
    return QuicTeleopChannel(
        session_id=session_id,
        api_base=api_base,
        api_key=api_key,
        token_path=token_path,
        robot_kind=robot_kind,
    )
