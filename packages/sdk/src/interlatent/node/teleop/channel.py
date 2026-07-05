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

# Node→pod state heartbeat rate. The pod-side retarget stage refuses to
# solve against a robot state older than 0.5s (STALE_OBS_S); RecordTick
# state rides the recorder's batched JPEG uplink and can arrive seconds
# apart on a slow link, so the control loop pushes its joint vector over
# this WS instead — tiny frames, RTT-bound. 15 Hz keeps the gate
# comfortably fed at ~1/7th its threshold.
_STATE_SEND_PERIOD_S = 1.0 / 15.0

# Period of the rolling frame-arrival latency summary (INFO). Matches the
# relay's 5s browser-frame summaries so pod and node logs line up.
_STATS_LOG_PERIOD_S = 5.0


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
        token_path: Optional[str] = None,
        bypass_key: Optional[str] = None,
    ) -> None:
        self._session_id = session_id
        self._api_base = api_base.rstrip("/")
        self._api_key = api_key
        self._bypass_key = bypass_key
        # Token-mint route. Defaults to the inference-session route; teleop
        # recordings pass their own (/api/v1/teleop-recordings/{id}/teleop-token).
        self._token_path = (
            token_path
            or f"/api/v1/inference/sessions/{session_id}/teleop-token"
        )

        self._lock = threading.Lock()
        self._latest: Optional[TeleopFrame] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._connected = False
        # Live WS handle for send_state() (websockets' sync connection is
        # thread-safe: the control loop sends while the receive loop recvs).
        self._ws = None
        self._last_state_sent_at = 0.0
        # Frame-arrival latency window (receive-loop thread only): counts
        # + inter-arrival gap stats since the last 5s summary. The gap is
        # the node-observable half of teleop latency — it captures
        # pod→node network jitter and retarget-stage stalls, which is
        # what actually makes the arm feel laggy.
        self._arr_count = 0
        self._arr_last_ns = 0
        self._arr_gap_sum_ms = 0.0
        self._arr_gap_max_ms = 0.0
        self._arr_seq_first = 0
        self._arr_seq_last = 0
        self._arr_window_started = time.monotonic()

    def _note_arrival(self, frame: TeleopFrame) -> None:
        """Track inter-arrival gaps of producer frames; log a 5s summary.

        Runs on the receive-loop thread only (no locking needed). Logged
        rate below the producer's send rate, or a large max gap, means
        frames are stalling somewhere between the pod and this node.
        """
        now_ns = frame.received_at_ns
        if self._arr_count == 0:
            self._arr_seq_first = frame.seq
        else:
            gap_ms = (now_ns - self._arr_last_ns) / 1e6
            self._arr_gap_sum_ms += gap_ms
            if gap_ms > self._arr_gap_max_ms:
                self._arr_gap_max_ms = gap_ms
        self._arr_last_ns = now_ns
        self._arr_seq_last = frame.seq
        self._arr_count += 1

        now = time.monotonic()
        elapsed = now - self._arr_window_started
        if elapsed < _STATS_LOG_PERIOD_S:
            return
        n = self._arr_count
        gaps = n - 1
        # seq span vs frames received ≈ producer frames the relay/retarget
        # collapsed or dropped en route (latest-wins everywhere, so >0 is
        # normal under load — it's the trend that matters).
        seq_span = self._arr_seq_last - self._arr_seq_first
        _LOG.info(
            "teleop WS frames (%.0fs): n=%d rate=%.1fHz gap mean/max=%.0f/%.0fms "
            "seq_span=%d session=%s",
            elapsed, n, n / elapsed if elapsed > 0 else 0.0,
            (self._arr_gap_sum_ms / gaps) if gaps > 0 else 0.0,
            self._arr_gap_max_ms, seq_span, self._session_id,
        )
        self._arr_count = 0
        self._arr_gap_sum_ms = 0.0
        self._arr_gap_max_ms = 0.0
        self._arr_window_started = now

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
    # Write API (called from the control loop)
    # ------------------------------------------------------------------

    def send_state(self, qpos) -> None:
        """Best-effort push of the robot's current joint vector (action
        order, robot-native units — same convention as RecordTick's
        ``observation_state``) to the pod's teleop relay.

        Rate-limited to ~15 Hz internally, so callers can just invoke it
        every control tick. Never raises and never blocks meaningfully —
        connection failures are owned by the reconnect loop; a send racing
        a disconnect is simply dropped.
        """
        ws = self._ws
        if ws is None or qpos is None:
            return
        now = time.monotonic()
        if now - self._last_state_sent_at < _STATE_SEND_PERIOD_S:
            return
        self._last_state_sent_at = now
        try:
            import json

            ws.send(json.dumps({
                "type": "state",
                "qpos": [float(x) for x in qpos],
            }))
        except Exception:
            # Socket mid-close / reconnecting — the receive loop notices
            # and re-establishes; state resumes on the next connection.
            pass

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

        url = f"{self._api_base}{self._token_path}"
        headers = {"x-api-key": self._api_key}
        if (self._bypass_key or "").strip():
            # Protected test domains (Vercel preview deployments) challenge
            # un-bypassed requests with a 302 to Vercel's SSO gate rather
            # than reaching the app at all; carry the automation bypass
            # secret same as the daemon's shared heartbeat/poll client.
            headers["x-vercel-protection-bypass"] = self._bypass_key.strip()
        with httpx.Client(timeout=10.0) as client:
            r = client.post(
                url,
                params={"role": "node"},
                headers=headers,
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
            self._ws = ws
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
                    self._note_arrival(frame)
                    with self._lock:
                        self._latest = frame
            finally:
                self._connected = False
                self._ws = None
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
