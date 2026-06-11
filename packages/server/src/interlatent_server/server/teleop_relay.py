"""WebSocket relay for browser-driven DAgger teleop.

Runs alongside the DRTC gRPC server on the persistent GPU box. Two
WebSocket paths::

    /teleop/browser/{session_id}    # dashboard connects here
    /teleop/node/{session_id}       # interlatent-node connects here

Each side authenticates with an HMAC token in the ``?token=...`` query
string. Tokens are minted by the Vercel backend (see
``site/app/services/teleop_token.py``) and verified here with the
SAME secret. Vercel never relays teleop traffic — its only role is
issuing the token.

Pairing is in-memory and per-session: when both sides are connected,
frames from the browser are forwarded to the node. The node side does
not currently push back (acks can be added trivially), and the relay
makes no assumption about frame contents — it just pumps text frames.

Lifecycle: each browser connection is treated as a single engagement.
When the browser disconnects, the relay sends a synthetic
``{"engaged": false}`` frame to the node so the control loop falls
back to policy mode cleanly even if the browser closes mid-engage.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Optional

import websockets
# websockets 12+ uses the asyncio.ServerConnection API: a 1-arg
# handler that pulls request metadata from ``connection.request``.
from websockets.asyncio.server import ServerConnection, serve as ws_serve

log = logging.getLogger("teleop_relay")


# ----------------------------------------------------------------------
# Token verification (mirror of site/app/services/teleop_token.py)
# ----------------------------------------------------------------------


def _b64u_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def _verify_token(
    token: str,
    *,
    secret: str,
    expected_session_id: str,
    expected_role: str,
    now: float | None = None,
) -> tuple[bool, str]:
    """Return ``(ok, reason)``. ``reason`` is a short human-readable
    string when ``ok`` is False; empty otherwise.

    Mirrors ``site/app/services/teleop_token.py`` exactly — keep the
    two in sync (one mints, the other verifies; same algorithm).
    """
    if not secret:
        return False, "secret_unset"
    if not token or token.count(".") != 1:
        return False, "malformed"
    payload_b64, sig_b64 = token.split(".", 1)
    expected_sig = hmac.new(
        secret.encode("utf-8"), payload_b64.encode("ascii"), hashlib.sha256,
    ).digest()
    try:
        given_sig = _b64u_decode(sig_b64)
    except Exception:
        return False, "bad_sig_encoding"
    if not hmac.compare_digest(expected_sig, given_sig):
        return False, "bad_sig"
    try:
        payload = json.loads(_b64u_decode(payload_b64).decode("utf-8"))
    except Exception:
        return False, "bad_payload"
    if payload.get("v") != 1:
        return False, "bad_version"
    if payload.get("session_id") != expected_session_id:
        return False, "session_mismatch"
    if payload.get("role") != expected_role:
        return False, "role_mismatch"
    exp = int(payload.get("exp") or 0)
    if exp <= int(now if now is not None else time.time()):
        return False, "expired"
    return True, ""


# ----------------------------------------------------------------------
# Per-session pairing
# ----------------------------------------------------------------------


@dataclass
class _Pair:
    browser: Optional[ServerConnection] = None
    node: Optional[ServerConnection] = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class TeleopRelay:
    """Single-process registry of active teleop pairs.

    All state is in-memory; one GPU box hosts one set of pairs. Each
    session id may have at most one browser and one node socket at a
    time — re-opening a side replaces the prior one (the older socket
    is closed). The relay is fully asyncio-native.
    """

    def __init__(self, *, secret: str) -> None:
        self._secret = secret
        self._pairs: dict[str, _Pair] = {}
        self._registry_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public entry point (passed to websockets.serve)
    # ------------------------------------------------------------------

    async def handle(self, ws: ServerConnection) -> None:
        """Dispatch one new connection by request path.

        Path shape: ``/teleop/<side>/<session_id>?token=...`` where
        ``<side>`` is ``browser`` or ``node``.
        """
        raw_path = ws.request.path
        path_only = raw_path.split("?", 1)[0]
        parts = [p for p in path_only.split("/") if p]
        if len(parts) != 3 or parts[0] != "teleop" or parts[1] not in ("browser", "node"):
            await ws.close(code=4404, reason="not_found")
            return
        side = parts[1]
        session_id = parts[2]

        token = _token_from_query(raw_path)
        ok, reason = _verify_token(
            token,
            secret=self._secret,
            expected_session_id=session_id,
            expected_role=side,
        )
        if not ok:
            log.warning(
                "teleop %s auth rejected (%s) session=%s", side, reason, session_id,
            )
            await ws.close(code=4401, reason=f"auth: {reason}")
            return

        await self._run_side(ws, session_id, side=side)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _attach(
        self, session_id: str, side: str, ws: ServerConnection
    ) -> _Pair:
        async with self._registry_lock:
            pair = self._pairs.get(session_id)
            if pair is None:
                pair = _Pair()
                self._pairs[session_id] = pair
            old = getattr(pair, side)
            setattr(pair, side, ws)
        if old is not None and old is not ws:
            # Replace the prior socket — close it after dropping the
            # registry lock so close() doesn't deadlock with concurrent
            # _detach calls.
            try:
                await old.close(code=4000, reason="superseded")
            except Exception:
                pass
        return pair

    async def _detach(
        self, session_id: str, side: str, ws: ServerConnection
    ) -> Optional[ServerConnection]:
        """Remove ``ws`` from the pair. Returns the peer socket (if any)
        so the caller can notify it that the engagement ended."""
        peer: Optional[ServerConnection] = None
        async with self._registry_lock:
            pair = self._pairs.get(session_id)
            if pair is None:
                return None
            if getattr(pair, side) is ws:
                setattr(pair, side, None)
            peer = pair.node if side == "browser" else pair.browser
            if pair.browser is None and pair.node is None:
                self._pairs.pop(session_id, None)
        return peer

    async def _run_side(
        self, ws: ServerConnection, session_id: str, *, side: str
    ) -> None:
        pair = await self._attach(session_id, side, ws)
        log.info("teleop %s connected session=%s", side, session_id)
        try:
            async for msg in ws:
                # Only forward in the browser→node direction. The node
                # side is the consumer; it does not send teleop frames.
                # Anything the node does send is dropped silently.
                if side != "browser":
                    continue
                peer = pair.node
                if peer is None:
                    continue
                try:
                    await peer.send(msg)
                except Exception:
                    # Peer is gone — drop the frame; the WS close
                    # handler on the other side will clean up the pair.
                    pass
        except websockets.ConnectionClosed:
            pass
        finally:
            peer = await self._detach(session_id, side, ws)
            # On browser disconnect, tell the node the engagement ended
            # so the control loop falls back to policy even if the
            # browser dies mid-engage without sending {"engaged": false}.
            if side == "browser" and peer is not None:
                try:
                    await peer.send(json.dumps({"engaged": False, "reason": "browser_closed"}))
                except Exception:
                    pass
            log.info("teleop %s disconnected session=%s", side, session_id)


def _token_from_query(path: str) -> str:
    """Pull ``?token=...`` from a WebSocket request path."""
    if "?" not in path:
        return ""
    _, qs = path.split("?", 1)
    params = urllib.parse.parse_qs(qs)
    return (params.get("token") or [""])[0]


# ----------------------------------------------------------------------
# Server bootstrap
# ----------------------------------------------------------------------


async def serve(
    *,
    host: str,
    port: int,
    secret: str,
):
    """Start the teleop WS server. Returns the running server handle.

    Path routing happens inside :meth:`TeleopRelay.handle` based on
    ``connection.request.path`` (the websockets 12+ asyncio API).
    """
    relay = TeleopRelay(secret=secret)
    server = await ws_serve(relay.handle, host, port, ping_interval=20)
    log.info("teleop WS relay listening on %s:%d", host, port)
    return server


__all__ = ["TeleopRelay", "serve"]
