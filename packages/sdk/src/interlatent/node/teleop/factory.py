"""Build the node's teleop channel.

Teleop runs over QUIC/WebTransport only: the browser owns IK and streams
``mode="targets"`` datagrams, while the node serves the kinematic_spec and tees
a live preview back. The channel exposes ``start``/``stop``/``latest_frame``/
``send_state``/``connected``, so the control loop is transport-agnostic.

A one-shot node-role token mint reveals the deployment's ``transport`` +
``webtransport_url``. ``aioquic`` is never imported in this process — the quic
channel's child process uses it — so availability is probed via ``find_spec``
before building the channel. When the deployment isn't QUIC-configured, aioquic
is missing, or the probe fails, teleop is unavailable and this returns ``None``;
the daemon already treats ``None`` as "teleop disabled" and re-runs the factory
on the next session assignment.
"""
from __future__ import annotations

import logging
from typing import Optional

from ._mint import mint_teleop_token

_LOG = logging.getLogger(__name__)


def make_teleop_channel(
    *,
    session_id: str,
    api_base: str,
    api_key: str,
    token_path: Optional[str] = None,
    bypass_key: Optional[str] = None,
    robot_kind: Optional[str] = None,
):
    """Return a QUIC teleop channel, or ``None`` when teleop is unavailable."""
    probe_path = (
        token_path or f"/api/v1/inference/sessions/{session_id}/teleop-token"
    )
    try:
        data = mint_teleop_token(
            api_base=api_base,
            token_path=probe_path,
            api_key=api_key,
            bypass_key=bypass_key,
            role="node",
        )
        transport = str(data.get("transport") or "")
        webtransport_url = data.get("webtransport_url")
    except Exception as exc:
        # Session may not be active yet, or teleop disabled — teleop is
        # unavailable for now; the daemon re-runs the factory on the next
        # session assignment.
        _LOG.info("teleop transport probe failed (%s); teleop disabled", exc)
        return None

    if transport != "quic" or not webtransport_url:
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
        bypass_key=bypass_key,
        robot_kind=robot_kind,
    )
