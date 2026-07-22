"""aioquic WebTransport client glue for the node QUIC teleop channel.

Isolated here (lazy-imported) so the aioquic dependency is only pulled in on
nodes actually using the QUIC path, and so the wire-specific code — the one
part not exercisable offline — is in one clearly-marked place. Validated live
on the Phase-0 gate (aioquic on the arm64 Pi + a reachable relay).

``connect_webtransport(url, token)`` is an async context manager yielding a
:class:`WebTransportSession` with ``send_datagram(bytes)`` and an async
``datagrams()`` iterator. Establishes an HTTP/3 extended-CONNECT WebTransport
session to ``<url>?token=<token>`` and exposes its unreliable datagram flow.
"""
from __future__ import annotations

import asyncio
import logging
import os
import ssl
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional
from urllib.parse import urlsplit

from aioquic.asyncio.client import connect
from aioquic.asyncio.protocol import QuicConnectionProtocol
from aioquic.h3.connection import H3Connection
from aioquic.h3.events import DatagramReceived, HeadersReceived
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.events import ConnectionTerminated, ProtocolNegotiated, QuicEvent

_LOG = logging.getLogger(__name__)

# Drop-don't-queue bound for outbound datagrams. aioquic parks datagrams in an
# unbounded pending deque with no expiry, so during a congestion-window stall
# a backlog of state heartbeats would drain seconds late — the browser then
# FKs stale joint state and its applied_seq RTT measure reports queue depth,
# not network latency. Outbound datagrams are latest-wins (each ~15 Hz state
# snapshot supersedes the previous), so beyond this backlog we drop at the
# source. Sized to a few send_state calls (2 dups each) of headroom.
_DATAGRAM_PENDING_MAX = 8


class _WTClientProtocol(QuicConnectionProtocol):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._http: Optional[H3Connection] = None
        self._session_stream_id: Optional[int] = None
        self.datagrams_dropped = 0  # drop-don't-queue counter (stats line)
        # NOT `_connected`: the aioquic base class uses `self._connected` as an
        # internal handshake boolean, and shadowing it with a (truthy) Future
        # makes wait_connected() return instantly — the client then aborts the
        # handshake ~1 ms in with a bare ConnectionTerminated. That collision
        # was the real cause of the "handshake never completes on the node"
        # failure originally blamed on GIL starvation.
        self._wt_connected: "asyncio.Future[bool]" = (
            asyncio.get_event_loop().create_future()
        )
        self._datagrams: "asyncio.Queue[bytes]" = asyncio.Queue()

    def open_session(self, authority: str, path: str) -> None:
        """Send the extended CONNECT that opens the WebTransport session."""
        if self._http is None:
            # No H3 layer means ProtocolNegotiated never fired with an 'h3'
            # ALPN — i.e. the QUIC/TLS handshake did not complete ALPN before
            # this point (TLS trust failure or the handshake flight was dropped,
            # e.g. MTU/QUIC-hostile network). Fail loudly instead of a bare
            # AssertionError so the node log says *why*.
            raise RuntimeError(
                "WebTransport handshake failed: no HTTP/3 ALPN negotiated "
                "(QUIC/TLS handshake did not complete — check cert trust or a "
                "QUIC-hostile network/MTU on this host)"
            )
        self._session_stream_id = self._quic.get_next_available_stream_id()
        self._http.send_headers(
            self._session_stream_id,
            [
                (b":method", b"CONNECT"),
                (b":protocol", b"webtransport"),
                (b":scheme", b"https"),
                (b":authority", authority.encode()),
                (b":path", path.encode()),
            ],
        )
        self.transmit()

    def send_datagram(self, data: bytes) -> None:
        if self._http is None or self._session_stream_id is None:
            return
        try:
            # Drop-don't-queue (see _DATAGRAM_PENDING_MAX): a standing
            # backlog means the congestion window is stalled — queueing more
            # latest-wins state on top only delivers it stale. Private attr,
            # same pinned-aioquic caveat as uni_stream_finished; on attr
            # error degrade to always-send.
            try:
                if len(self._quic._datagrams_pending) >= _DATAGRAM_PENDING_MAX:
                    self.datagrams_dropped += 1
                    return
            except AttributeError:
                pass
            self._http.send_datagram(self._session_stream_id, data)
            self.transmit()
        except Exception:
            pass

    # -- unidirectional streams (video tee: one short-lived stream per frame) --
    def open_uni_stream(self, payload: bytes) -> Optional[int]:
        """Open a WebTransport uni stream, write ``payload``, FIN. Returns the
        QUIC stream id (for completion/reset tracking) or None if the session
        is down/mid-close. If the relay's uni-stream credit is momentarily
        exhausted aioquic parks the stream until MAX_STREAMS arrives — the
        caller's in-flight cap keeps that from ever piling up."""
        if self._http is None or self._session_stream_id is None:
            return None
        try:
            sid = self._http.create_webtransport_stream(
                self._session_stream_id, is_unidirectional=True
            )
            self._quic.send_stream_data(sid, payload, end_stream=True)
            self.transmit()
            return sid
        except Exception:
            return None

    def reset_uni_stream(self, sid: int) -> None:
        """Abandon an in-flight uni stream (RESET_STREAM) — drops any unacked
        retransmission so a stale frame stops competing with control."""
        try:
            self._quic.reset_stream(sid, 0)
            self.transmit()
        except Exception:
            pass

    def uni_stream_finished(self, sid: int) -> bool:
        """True once the stream's send side is fully acked+FIN.

        Checks ``sender.is_finished`` directly (private attrs, hence the
        pinned aioquic range): a send-only uni stream is NEVER popped from
        ``_quic._streams`` — aioquic's ``QuicStreamReceiver.__init__`` ignores
        its ``readable`` flag, so ``QuicStream.is_finished`` (receiver AND
        sender) stays False forever and the discard sweep never collects it.
        The naive ``sid not in _streams`` check therefore reported every
        frame as unfinished, and the governor TTL-reset streams that had long
        been delivered. On any attr error we degrade to 'always finished',
        leaving the TTL as the only shedding signal."""
        try:
            stream = self._quic._streams.get(sid)
            if stream is None:
                return True
            return bool(stream.sender.is_finished)
        except Exception:
            return True

    def discard_uni_stream(self, sid: int) -> bool:
        """Drop a send-only uni stream from aioquic's per-connection
        bookkeeping once its send side is fully acked (FIN or RESET
        delivered). Returns True when the stream is gone (or already was),
        False if the send side is still in flight — retry later.

        aioquic never collects these streams itself: its discard sweep
        requires ``QuicStream.is_finished`` (receiver AND sender), and
        ``QuicStreamReceiver.__init__`` hardcodes ``is_finished = False``
        even for a send-only stream (contradicting its own docstring), so
        every frame stream stays in ``_quic._streams`` and
        ``_streams_queue`` forever. Both are iterated on EVERY packet build
        (``_write_application``), so at ~72 frame-streams/s the per-packet
        cost grows linearly with session age — measured on the Jetson as
        per-frame completion time creeping ~40 → ~100 ms over a minute
        (delivered fps ~23 → ~9 against a pinned 24 Hz offer), resetting
        only on reconnect. This replicates exactly what aioquic's own
        sweep does when it does fire: pop from ``_streams``, record the id
        in ``_streams_finished`` (so a late peer frame is treated as
        already-handled, not a protocol error), drop from
        ``_streams_queue``. Private attrs — same pinned-aioquic caveat as
        ``uni_stream_finished``; on attr error report True so callers
        don't retry forever."""
        try:
            stream = self._quic._streams.get(sid)
            if stream is None:
                return True
            if not stream.sender.is_finished:
                return False
            self._quic._streams.pop(sid, None)
            self._quic._streams_finished.add(sid)
            try:
                self._quic._streams_queue.remove(stream)
            except ValueError:
                pass
            return True
        except Exception:
            return True

    def quic_event_received(self, event: QuicEvent) -> None:
        if isinstance(event, ConnectionTerminated):
            _LOG.warning(
                "teleop(quic) terminated: error_code=%s frame_type=%s reason=%r",
                getattr(event, "error_code", None),
                getattr(event, "frame_type", None),
                getattr(event, "reason_phrase", None),
            )
        if isinstance(event, ProtocolNegotiated):
            _LOG.info("teleop(quic) ALPN negotiated: %r", event.alpn_protocol)
            self._http = H3Connection(self._quic, enable_webtransport=True)
        if self._http is None:
            return
        for h3_event in self._http.handle_event(event):
            if isinstance(h3_event, HeadersReceived) and (
                h3_event.stream_id == self._session_stream_id
            ):
                status = dict(h3_event.headers).get(b":status")
                if self._wt_connected.done():
                    continue
                if status == b"200":
                    self._wt_connected.set_result(True)
                else:
                    self._wt_connected.set_exception(
                        RuntimeError(f"WebTransport CONNECT rejected: {status!r}")
                    )
            elif isinstance(h3_event, DatagramReceived) and (
                h3_event.stream_id == self._session_stream_id
            ):
                self._datagrams.put_nowait(h3_event.data)


class WebTransportSession:
    def __init__(self, proto: _WTClientProtocol) -> None:
        self._proto = proto

    def send_datagram(self, data: bytes) -> None:
        self._proto.send_datagram(data)

    async def datagrams(self) -> AsyncIterator[bytes]:
        while True:
            yield await self._proto._datagrams.get()

    def open_uni_stream(self, payload: bytes) -> Optional[int]:
        return self._proto.open_uni_stream(payload)

    def reset_uni_stream(self, sid: int) -> None:
        self._proto.reset_uni_stream(sid)

    def uni_stream_finished(self, sid: int) -> bool:
        return self._proto.uni_stream_finished(sid)

    def discard_uni_stream(self, sid: int) -> bool:
        return self._proto.discard_uni_stream(sid)

    def datagrams_dropped(self) -> int:
        """Cumulative outbound datagrams shed by drop-don't-queue."""
        return self._proto.datagrams_dropped


@asynccontextmanager
async def connect_webtransport(url: str, token: str):
    """Open a WebTransport session to ``url?token=token``; yield a session."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    port = parts.port or 443
    path = parts.path or "/"
    if parts.query:
        path = f"{path}?{parts.query}&token={token}"
    else:
        path = f"{path}?token={token}"

    config = QuicConfiguration(
        alpn_protocols=["h3"],
        is_client=True,
        max_datagram_frame_size=65536,
    )
    # Dev escape hatch for a self-signed relay cert (serverCertificateHashes is
    # a browser-only feature; the node verifies normally in production).
    if os.environ.get("INTERLATENT_TELEOP_INSECURE") == "1":
        config.verify_mode = ssl.CERT_NONE

    async with connect(
        host, port, configuration=config, create_protocol=_WTClientProtocol
    ) as proto:
        assert isinstance(proto, _WTClientProtocol)
        await proto.wait_connected()
        proto.open_session(host, path)
        await asyncio.wait_for(proto._wt_connected, timeout=10.0)
        yield WebTransportSession(proto)


__all__ = ["connect_webtransport", "WebTransportSession"]
