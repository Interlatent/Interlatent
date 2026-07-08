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
from aioquic.quic.events import ProtocolNegotiated, QuicEvent


class _WTClientProtocol(QuicConnectionProtocol):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._http: Optional[H3Connection] = None
        self._session_stream_id: Optional[int] = None
        self._connected: "asyncio.Future[bool]" = asyncio.get_event_loop().create_future()
        self._datagrams: "asyncio.Queue[bytes]" = asyncio.Queue()

    def open_session(self, authority: str, path: str) -> None:
        """Send the extended CONNECT that opens the WebTransport session."""
        assert self._http is not None
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
            self._http.send_datagram(self._session_stream_id, data)
            self.transmit()
        except Exception:
            pass

    def quic_event_received(self, event: QuicEvent) -> None:
        if isinstance(event, ProtocolNegotiated):
            self._http = H3Connection(self._quic, enable_webtransport=True)
        if self._http is None:
            return
        for h3_event in self._http.handle_event(event):
            if isinstance(h3_event, HeadersReceived) and (
                h3_event.stream_id == self._session_stream_id
            ):
                status = dict(h3_event.headers).get(b":status")
                if self._connected.done():
                    continue
                if status == b"200":
                    self._connected.set_result(True)
                else:
                    self._connected.set_exception(
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
        await asyncio.wait_for(proto._connected, timeout=10.0)
        yield WebTransportSession(proto)


__all__ = ["connect_webtransport", "WebTransportSession"]
