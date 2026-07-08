"""Pick the teleop channel transport from the backend's ``transport`` flag.

Both channels share the same surface (``start``/``stop``/``latest_frame``/
``send_state``/``connected``), so the daemon builds one via this factory and
the control loop is transport-agnostic.

Selection: a one-shot node-role token mint reveals the deployment's
``transport`` + ``webtransport_url``. ``quic`` → :class:`QuicTeleopChannel`;
anything else → the WS :class:`TeleopChannel`. The probe is best-effort — the
token always carries a working ``ws_url`` even in quic mode, so a failed or
absent quic signal degrades cleanly to the WS path (correct behaviour for the
parallel rollout, where both relays run). ``aioquic`` is only imported when the
quic path is actually chosen.
"""
from __future__ import annotations

import logging
from typing import Optional

from ._mint import mint_teleop_token
from .channel import TeleopChannel

_LOG = logging.getLogger(__name__)


def make_teleop_channel(
    *,
    session_id: str,
    api_base: str,
    api_key: str,
    token_path: Optional[str] = None,
    bypass_key: Optional[str] = None,
):
    probe_path = (
        token_path or f"/api/v1/inference/sessions/{session_id}/teleop-token"
    )
    transport, webtransport_url = "ws", None
    try:
        data = mint_teleop_token(
            api_base=api_base,
            token_path=probe_path,
            api_key=api_key,
            bypass_key=bypass_key,
            role="node",
        )
        transport = str(data.get("transport") or "ws")
        webtransport_url = data.get("webtransport_url")
    except Exception as exc:
        # Session may not be active yet, or teleop disabled — either way the
        # WS channel's own retry loop handles it. Default to ws.
        _LOG.info("teleop transport probe failed (%s); using ws", exc)

    common = dict(
        session_id=session_id,
        api_base=api_base,
        api_key=api_key,
        token_path=token_path,
        bypass_key=bypass_key,
    )
    if transport == "quic" and webtransport_url:
        try:
            from .quic_channel import QuicTeleopChannel
        except Exception as exc:  # aioquic missing on this node
            _LOG.warning("QUIC teleop unavailable (%s); falling back to ws", exc)
        else:
            _LOG.info("teleop transport=quic session=%s", session_id)
            return QuicTeleopChannel(**common)
    return TeleopChannel(**common)
