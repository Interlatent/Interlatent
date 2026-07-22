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

from .. import _env
from ._frame_store import LatestFrameStore
from ._mint import mint_teleop_token
from ._telemetry import ArrivalTracker
from .frame import TeleopFrame, frame_with_header

_LOG = logging.getLogger(__name__)

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

# Live-preview push rate (node→pod, small downscaled JPEGs over this WS).
# The in-headset video quad is fed from these instead of the batched
# full-resolution recording uplink, which over a real link runs seconds
# behind. The cadence is the dominant term in perceived video latency
# (mean staleness ≈ half the period), so it's tunable per node via
# INTERLATENT_PREVIEW_HZ; the cost is uplink bandwidth (~10-20 KB × cams
# × Hz — 10 Hz on a 2-cam rig is ~2 Mbit/s, 20 Hz ~4 Mbit/s). Clamped to
# [1, 30]; the control loop can't produce more than its tick rate anyway.
def _preview_period_s() -> float:
    return 1.0 / _env.env_float("INTERLATENT_PREVIEW_HZ", 10.0, 1.0, 30.0)


_PREVIEW_SEND_PERIOD_S = _preview_period_s()

# A viewer is "present" while browser frames keep arriving on this WS
# (the overlay sends keepalives even when disengaged). No frames for this
# long ⇒ nobody is watching ⇒ stop burning uplink on previews.
_VIEWER_PRESENCE_S = 5.0


class TeleopChannel(LatestFrameStore):
    """Background WS client that surfaces the latest browser teleop frame.

    Thread-safe: callers may read :meth:`latest_frame` from any thread.
    The receive loop runs in a single background daemon thread and is
    fully owned by this object. The latest-frame + sticky-estop store
    (:class:`LatestFrameStore`) is shared with the QUIC channel (ADR 0016).
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
        super().__init__()
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

        # _lock / _latest / _estop_seen live in LatestFrameStore (ADR 0016).
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._connected = False
        # Live WS handle for send_state() (websockets' sync connection is
        # thread-safe: the control loop sends while the receive loop recvs;
        # concurrent senders are serialized by the connection's own lock).
        self._ws = None
        self._last_state_sent_at = 0.0
        # Live-preview pipeline: the control loop stages the latest
        # downscaled JPEGs into a 1-slot buffer; a dedicated sender thread
        # ships them so a saturated uplink can never block the control
        # loop on a socket write. Latest-wins by construction.
        self._preview_lock = threading.Lock()
        self._preview_slot: Optional[tuple[dict[str, bytes], int]] = None
        self._preview_event = threading.Event()
        self._preview_thread: Optional[threading.Thread] = None
        self._last_preview_staged_at = 0.0
        # One-shot diagnostic flags: the preview path fails silently by
        # design (best-effort), so log the FIRST success and FIRST
        # failure to make a dead preview stream observable in the field.
        self._pv_sent_logged = False
        self._pv_err_logged = False
        # Cumulative preview counters, surfaced in the 5s WS summary. The
        # one-shot logs above catch a preview that fails; these catch a
        # preview that is never ATTEMPTED (preview_due() never opening,
        # control loop never staging) — the case where nothing else logs.
        # Written from the control-loop/sender threads, read from the
        # receive loop; plain ints under the GIL are fine for telemetry.
        self._pv_staged_total = 0
        self._pv_sent_total = 0
        # monotonic time of the last message received from the relay —
        # browser keepalives included — used as the viewer-presence gate.
        self._last_rx_at = 0.0
        # Frame-arrival latency window (receive-loop thread only): the gap is
        # the node-observable half of teleop latency — pod→node jitter and
        # retarget-stage stalls, what actually makes the arm feel laggy. This
        # channel appends its preview counters (pv_staged / pv_sent).
        self._arrivals = ArrivalTracker(_STATS_LOG_PERIOD_S)

    def _note_arrival(self, frame: TeleopFrame) -> None:
        """Track inter-arrival gaps of producer frames; log a 5s summary.
        Receive-loop thread only. A logged rate below the producer's send rate,
        or a large max gap, means frames are stalling between the pod and this
        node. ``seq_span`` vs ``n`` ≈ frames the relay/retarget collapsed or
        dropped (latest-wins everywhere, so >0 is normal — the trend matters).
        """
        summary = self._arrivals.note(frame)
        if summary is None:
            return
        _LOG.info(
            "teleop WS frames (%.0fs): n=%d rate=%.1fHz gap mean/max=%.0f/%.0fms "
            "seq_span=%d pv_staged=%d pv_sent=%d session=%s",
            summary["elapsed"], summary["n"], summary["rate_hz"],
            summary["gap_mean_ms"], summary["gap_max_ms"], summary["seq_span"],
            self._pv_staged_total, self._pv_sent_total, self._session_id,
        )

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
        self._preview_thread = threading.Thread(
            target=self._preview_run,
            name=f"teleop-preview[{self._session_id[:8]}]",
            daemon=True,
        )
        self._preview_thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._preview_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        if self._preview_thread is not None:
            self._preview_thread.join(timeout=3.0)
            self._preview_thread = None

    # ------------------------------------------------------------------
    # Read API (called from the control loop)
    # ------------------------------------------------------------------

    # latest_frame() and consume_estop() are inherited from LatestFrameStore
    # (identical staleness + sticky-estop semantics across transports, ADR 0016).

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

    def preview_due(self) -> bool:
        """Should the control loop encode + stage a preview this tick?

        True only when the WS is up, a viewer has sent a frame recently
        (the overlay keeps sending even while disengaged), the ~10 Hz
        rate window has elapsed, and the sender has consumed the prior
        slot. Callers check this BEFORE encoding so idle sessions pay
        zero encode cost.
        """
        if self._ws is None:
            return False
        now = time.monotonic()
        if now - self._last_rx_at > _VIEWER_PRESENCE_S:
            return False
        if now - self._last_preview_staged_at < _PREVIEW_SEND_PERIOD_S:
            return False
        with self._preview_lock:
            return self._preview_slot is None

    def send_preview(self, jpegs: dict[str, bytes], ts_ns: int) -> None:
        """Stage one set of downscaled camera JPEGs for the sender thread.

        Non-blocking and latest-wins: overwrites any unsent slot. ``ts_ns``
        is the node's monotonic capture time — the same clock RecordTick's
        control_timestamp uses, which is how the pod keeps whichever frame
        (preview vs recording) was captured later.
        """
        if not jpegs:
            return
        self._last_preview_staged_at = time.monotonic()
        self._pv_staged_total += 1
        with self._preview_lock:
            self._preview_slot = (jpegs, int(ts_ns))
        self._preview_event.set()

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _preview_run(self) -> None:
        """Drain the 1-slot preview buffer onto the WS.

        Binary frame layout (mirrors the relay's pod→browser video frame)::

            [0..2)   uint16 big-endian: header length H
            [2..2+H) UTF-8 JSON: {"type":"preview","cam":...,"ts_ns":...}
            [2+H..)  raw JPEG bytes

        One frame per camera per slot. Send errors are dropped — the
        receive loop owns reconnection, and the next slot rides the new
        connection. An old relay simply ignores binary node frames.
        """
        while not self._stop.is_set():
            if not self._preview_event.wait(timeout=0.5):
                continue
            self._preview_event.clear()
            with self._preview_lock:
                slot = self._preview_slot
                self._preview_slot = None
            if slot is None:
                continue
            ws = self._ws
            if ws is None:
                continue
            jpegs, ts_ns = slot
            try:
                for cam, data in jpegs.items():
                    if not data:
                        continue
                    ws.send(frame_with_header(
                        {"type": "preview", "cam": cam, "ts_ns": ts_ns}, data,
                    ))
                self._pv_sent_total += 1
                if not self._pv_sent_logged:
                    self._pv_sent_logged = True
                    _LOG.info(
                        "teleop preview tee active: sent %s",
                        ", ".join(
                            f"{c}={len(d)}B" for c, d in jpegs.items() if d
                        ),
                    )
            except Exception as exc:
                # Socket mid-close / reconnecting — drop this slot; the
                # next preview_due() cycle stages a fresh one.
                if not self._pv_err_logged:
                    self._pv_err_logged = True
                    _LOG.warning(
                        "teleop preview send failed (%s: %s); "
                        "further preview errors suppressed",
                        type(exc).__name__, exc,
                    )

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
            except (OSError, TimeoutError) as exc:
                # Expected transport churn: connection refused (relay/pod
                # stopped or not up yet), reset, host unreachable, open
                # timeout. The reconnect loop IS the handler — one quiet
                # line, no traceback.
                _LOG.warning(
                    "teleop WS unreachable (%s: %s); reconnecting in %.0fs",
                    type(exc).__name__, exc, backoff,
                )
                if self._stop.wait(backoff):
                    return
                backoff = min(backoff * 2, _RECONNECT_MAX_S)
            except Exception:
                _LOG.warning("teleop WS errored; reconnecting", exc_info=True)
                if self._stop.wait(backoff):
                    return
                backoff = min(backoff * 2, _RECONNECT_MAX_S)

    def _mint_token(self) -> tuple[str, str]:
        """Synchronously POST to Vercel for a node-role join token.

        Returns ``(ws_url, token)``. Raises on any non-2xx response.
        """
        data = mint_teleop_token(
            api_base=self._api_base,
            token_path=self._token_path,
            api_key=self._api_key,
            bypass_key=self._bypass_key,
            role="node",
        )
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
                    # Any relay→node message counts as viewer presence —
                    # the preview pipeline only spends uplink while a
                    # browser is actually attached and sending.
                    self._last_rx_at = time.monotonic()
                    if isinstance(raw, bytes):
                        try:
                            raw = raw.decode("utf-8")
                        except UnicodeDecodeError:
                            continue
                    frame = TeleopFrame.from_json(raw)
                    if frame is None:
                        continue
                    self._note_arrival(frame)
                    self._store_frame(frame)
                    if frame.estop:
                        self._latch_estop()
            finally:
                self._connected = False
                self._ws = None
                # Drop the last known frame on disconnect so a stale
                # "engaged" doesn't keep driving the arm. The control
                # loop sees latest_frame() == None and falls back to
                # policy mode immediately.
                self._drop_frame()


def _redact_token(url: str) -> str:
    if "token=" not in url:
        return url
    before, _, _ = url.partition("token=")
    return f"{before}token=<redacted>"


__all__ = ["TeleopChannel", "TeleopFrame"]
