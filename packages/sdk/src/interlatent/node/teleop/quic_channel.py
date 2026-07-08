"""Node-side QUIC/WebTransport teleop channel.

The low-latency counterpart to :class:`~.channel.TeleopChannel`: instead of a
WebSocket to the pod relay, it opens a WebTransport session to the co-located
QUIC relay and consumes **unreliable datagrams** carrying ``mode="targets"``
frames the browser already IK-solved. Exposes the exact surface the control
loop uses (``latest_frame`` / ``send_state`` / ``connected`` / ``start`` /
``stop``), so ``daemon._start_loop`` swaps it in with zero control-loop change.

Differences from the WS channel, by design:
  * targets arrive as duplicated datagrams; we dedupe by ``seq`` (latest-wins)
    so a late duplicate can't clobber a newer frame (drop-don't-buffer).
  * ``send_state`` streams the robot's live joint vector back **to the browser**
    (not the pod) — the browser FK's it to close the clutch loop + reconcile.
    Duplicated for loss tolerance, same ~15 Hz cadence.
  * no preview/video tee: WebTransport datagrams are ~1200 B, too small for
    JPEGs. Live video on the QUIC path is a separate pipeline (see the ADR);
    ``preview_due`` returns False so the getattr-guarded preview tee no-ops.

Split: the pure codec/dedup helpers below are unit-tested; the aioquic
client wire glue (``_quic_client``) is the one part validated live (Phase-0
gate: aioquic on the arm64 Pi + a reachable relay).

``aioquic`` is imported lazily so a node on the WS path never needs it.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from typing import Optional

from ._mint import mint_teleop_token
from .frame import TeleopFrame

_LOG = logging.getLogger(__name__)

# Match the WS channel so behavior is identical downstream.
_FRAME_STALE_MS = 250
_STATE_SEND_PERIOD_S = 1.0 / 15.0
_RECONNECT_INITIAL_S = 1.0
_RECONNECT_MAX_S = 15.0
# Duplicate each outbound state datagram this many times (loss tolerance).
_STATE_DUP = 2
# A seq that jumps this far *backward* is a browser reconnect/reset, not a
# reordered duplicate — accept it and re-anchor.
_SEQ_RESET_GAP = 1000


class LatestSeqBuffer:
    """Seq-dedupe gate: accept a frame only if it is newer than the last
    accepted one (or a reset). Prevents a late duplicate from overwriting a
    newer target. Pure + unit-tested."""

    def __init__(self) -> None:
        self._last_seq = -1

    def accept(self, seq: int) -> bool:
        if seq > self._last_seq or seq < self._last_seq - _SEQ_RESET_GAP:
            self._last_seq = seq
            return True
        return False

    def reset(self) -> None:
        self._last_seq = -1


def encode_state_datagram(qpos, seq: int) -> bytes:
    """Node→browser joint-state datagram (JSON). ``qpos`` is action-order,
    robot-native units — same convention as RecordTick's observation_state."""
    return json.dumps({
        "type": "state",
        "seq": int(seq),
        "qpos": [float(x) for x in qpos],
    }).encode("utf-8")


def decode_target_datagram(data: bytes) -> Optional[TeleopFrame]:
    """Browser→node datagram → TeleopFrame (stamps received_at_ns, honest
    deadman handling via TeleopFrame.from_json). None on garbage."""
    try:
        return TeleopFrame.from_json(data.decode("utf-8"))
    except Exception:
        return None


class QuicTeleopChannel:
    """WebTransport/QUIC teleop channel with the TeleopChannel surface."""

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
        self._token_path = (
            token_path or f"/api/v1/inference/sessions/{session_id}/teleop-token"
        )

        self._lock = threading.Lock()
        self._latest: Optional[TeleopFrame] = None
        self._dedup = LatestSeqBuffer()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._connected = False

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._send_dg = None  # set to the WT session's send_datagram when up
        self._out_seq = 0
        self._last_state_sent_at = 0.0

    # -- lifecycle --
    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=lambda: asyncio.run(self._run()),
            name=f"teleop-quic[{self._session_id[:8]}]",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        loop = self._loop
        if loop is not None:
            try:
                loop.call_soon_threadsafe(lambda: None)  # wake the loop
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    # -- read API (control loop) --
    def latest_frame(self) -> Optional[TeleopFrame]:
        with self._lock:
            frame = self._latest
        if frame is None:
            return None
        if (time.monotonic_ns() - frame.received_at_ns) / 1e6 > _FRAME_STALE_MS:
            return None
        return frame

    @property
    def connected(self) -> bool:
        return self._connected

    # -- write API (control loop) --
    def send_state(self, qpos) -> None:
        """Push the robot's live joint vector back to the browser (duplicated,
        ~15 Hz). Non-blocking; drops silently while reconnecting."""
        if qpos is None:
            return
        now = time.monotonic()
        if now - self._last_state_sent_at < _STATE_SEND_PERIOD_S:
            return
        self._last_state_sent_at = now
        send = self._send_dg
        loop = self._loop
        if send is None or loop is None:
            return
        self._out_seq += 1
        data = encode_state_datagram(qpos, self._out_seq)
        for _ in range(_STATE_DUP):
            try:
                loop.call_soon_threadsafe(send, data)
            except Exception:
                pass

    def preview_due(self) -> bool:
        # No video over the QUIC datagram control channel (see module docs).
        return False

    # -- background asyncio --
    async def _run(self) -> None:
        self._loop = asyncio.get_running_loop()
        backoff = _RECONNECT_INITIAL_S
        while not self._stop.is_set():
            try:
                wt_url, token = await self._loop.run_in_executor(None, self._mint)
            except Exception as exc:
                _LOG.info("teleop(quic) token-mint failed: %s (retry %.0fs)", exc, backoff)
                if await self._sleep_or_stop(backoff):
                    return
                backoff = min(backoff * 2, _RECONNECT_MAX_S)
                continue
            try:
                await self._session(wt_url, token)
                backoff = _RECONNECT_INITIAL_S
            except Exception as exc:
                _LOG.warning(
                    "teleop(quic) session ended (%s: %s); reconnect in %.0fs",
                    type(exc).__name__, exc, backoff,
                )
                if await self._sleep_or_stop(backoff):
                    return
                backoff = min(backoff * 2, _RECONNECT_MAX_S)

    async def _sleep_or_stop(self, secs: float) -> bool:
        try:
            await asyncio.wait_for(
                asyncio.get_running_loop().run_in_executor(None, self._stop.wait, secs),
                timeout=secs + 1.0,
            )
        except Exception:
            pass
        return self._stop.is_set()

    def _mint(self) -> tuple[str, str]:
        data = mint_teleop_token(
            api_base=self._api_base,
            token_path=self._token_path,
            api_key=self._api_key,
            bypass_key=self._bypass_key,
            role="node",
        )
        wt = data.get("webtransport_url")
        if not wt:
            raise RuntimeError("token response has no webtransport_url (transport != quic?)")
        return str(wt), str(data["token"])

    async def _session(self, wt_url: str, token: str) -> None:
        # aioquic WebTransport client — the live-gated wire glue.
        from ._quic_client import connect_webtransport

        self._dedup.reset()
        async with connect_webtransport(wt_url, token) as wt:
            self._connected = True
            self._send_dg = wt.send_datagram
            _LOG.info("teleop(quic) connected session=%s", self._session_id)
            try:
                async for data in wt.datagrams():
                    if self._stop.is_set():
                        break
                    frame = decode_target_datagram(data)
                    if frame is None or not self._dedup.accept(frame.seq):
                        continue
                    with self._lock:
                        self._latest = frame
            finally:
                self._connected = False
                self._send_dg = None
                with self._lock:
                    self._latest = None


__all__ = [
    "QuicTeleopChannel",
    "LatestSeqBuffer",
    "encode_state_datagram",
    "decode_target_datagram",
]
