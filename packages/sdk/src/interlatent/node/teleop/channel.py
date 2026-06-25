"""Node-side teleop channel.

Opens a WebSocket from the node daemon to the GPU box's teleop relay
(see ``interlatent.inference.server.teleop_relay``), runs a receive
loop on a background thread, and exposes the most recent browser frame
to the control loop via a thread-safe :meth:`latest_frame`.

We open the WS as soon as a session goes active — *not* only when the
browser engages. The cost is one idle TCP connection per active
inference session, and the alternative ("connect on demand") adds
~500ms of latency before the user's first keystroke takes effect,
which is enough to feel like the dashboard is broken.

Connection lifecycle:

  start() -> spawn thread -> POST teleop-token -> open WS -> receive loop
                                  ^                              |
                                  |  reconnect on transient error
                                  +------------------------------+

  stop() -> set stop flag -> close WS -> join thread

Frames are short JSON dicts emitted by the dashboard's
:file:`TeleopOverlay` component. The relay also injects a synthetic
``{"engaged": false, "reason": "browser_closed"}`` frame on browser
disconnect — the control loop sees that just like any other frame and
falls back to policy mode.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from .frame import TeleopFrame

_LOG = logging.getLogger(__name__)

# How stale a held-key set may be before we drop it. At 30 Hz the
# dashboard fires every ~33ms; 250ms covers half a dozen frames of
# jitter without letting a stuck key keep the arm moving after the
# browser has actually gone silent.
_FRAME_STALE_MS = 250

# Reconnect backoff between dropped connections. The teleop channel
# can fail for boring reasons (GPU box bounced, NAT timeout) and the
# user might engage again at any moment, so we keep retrying.
_RECONNECT_INITIAL_S = 1.0
_RECONNECT_MAX_S = 15.0


class TeleopChannel:
    """Background WS client that surfaces the latest browser teleop frame.

    Thread-safe: callers may read :meth:`latest_frame` from any thread.
    The receive loop runs in a single background daemon thread and is
    fully owned by this object.
    """

    def __init__(
        self,
        *,
        session_id: str,
        api_base: str,
        api_key: str,
    ) -> None:
        self._session_id = session_id
        self._api_base = api_base.rstrip("/")
        self._api_key = api_key

        self._lock = threading.Lock()
        self._latest: Optional[TeleopFrame] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._connected = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name=f"teleop-channel[{self._session_id[:8]}]",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    # ------------------------------------------------------------------
    # Read API (called from the control loop)
    # ------------------------------------------------------------------

    def latest_frame(self) -> Optional[TeleopFrame]:
        """Return the most recent non-stale frame, or None.

        Stale frames (>250ms old) are treated as absent so a frozen
        browser can't keep the arm moving. Returning ``None`` makes
        the control loop fall back to policy mode automatically.
        """
        with self._lock:
            frame = self._latest
        if frame is None:
            return None
        age_ms = (time.monotonic_ns() - frame.received_at_ns) / 1_000_000
        if age_ms > _FRAME_STALE_MS:
            return None
        return frame

    @property
    def connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Token-mint + connect + receive, with reconnect-on-drop."""
        backoff = _RECONNECT_INITIAL_S
        while not self._stop.is_set():
            try:
                ws_url, token = self._mint_token()
            except Exception as exc:
                # Token mint failed (often because the deployment doesn't
                # have INTERLATENT_TELEOP_SECRET set — that's a "feature
                # disabled" signal, not a real error). Back off and try
                # again periodically so the user can enable it without
                # restarting the node.
                _LOG.info("teleop token-mint failed: %s (retry in %.0fs)", exc, backoff)
                if self._stop.wait(backoff):
                    return
                backoff = min(backoff * 2, _RECONNECT_MAX_S)
                continue

            try:
                self._run_session(ws_url, token)
                backoff = _RECONNECT_INITIAL_S  # clean disconnect — reset
            except Exception:
                _LOG.warning("teleop WS errored; reconnecting", exc_info=True)
                if self._stop.wait(backoff):
                    return
                backoff = min(backoff * 2, _RECONNECT_MAX_S)

    def _mint_token(self) -> tuple[str, str]:
        """Synchronously POST to Vercel for a node-role join token.

        Returns ``(ws_url, token)``. Raises on any non-2xx response.
        """
        import httpx

        url = f"{self._api_base}/api/v1/inference/sessions/{self._session_id}/teleop-token"
        with httpx.Client(timeout=10.0) as client:
            r = client.post(
                url,
                params={"role": "node"},
                headers={"x-api-key": self._api_key},
            )
            r.raise_for_status()
            data = r.json()
        return str(data["ws_url"]), str(data["token"])

    def _run_session(self, ws_url: str, token: str) -> None:
        """Open the WS, set _connected, drain frames until close."""
        # ``websockets.sync`` is the synchronous client API — fits our
        # background-thread model without dragging asyncio into the
        # control loop.
        from websockets.sync.client import connect

        full_url = f"{ws_url}?token={token}"
        _LOG.info("teleop channel connecting: %s", _redact_token(full_url))
        with connect(full_url, open_timeout=10, close_timeout=5) as ws:
            self._connected = True
            _LOG.info("teleop channel connected session=%s", self._session_id)
            try:
                while not self._stop.is_set():
                    try:
                        raw = ws.recv(timeout=0.5)
                    except TimeoutError:
                        continue
                    if raw is None:
                        break
                    if isinstance(raw, bytes):
                        try:
                            raw = raw.decode("utf-8")
                        except UnicodeDecodeError:
                            continue
                    frame = TeleopFrame.from_json(raw)
                    if frame is None:
                        continue
                    with self._lock:
                        self._latest = frame
            finally:
                self._connected = False
                # Drop the last known frame on disconnect so a stale
                # "engaged" doesn't keep driving the arm. The control
                # loop sees latest_frame() == None and falls back to
                # policy mode immediately.
                with self._lock:
                    self._latest = None


def _redact_token(url: str) -> str:
    if "token=" not in url:
        return url
    before, _, _ = url.partition("token=")
    return f"{before}token=<redacted>"


__all__ = ["TeleopChannel", "TeleopFrame"]
