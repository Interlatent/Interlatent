"""Interlatent teleop QUIC/WebTransport datagram relay.

A thin, co-located relay for the low-latency in-browser teleop path. The
browser (running IK) and the robot node each open a WebTransport session to
``https://<this-host>/teleop/<role>/<session_id>?token=<hmac>``; the relay
verifies the HMAC token (same secret Vercel mints with), pairs the two by
session id, and forwards **unreliable datagrams** between them verbatim. No
buffering, no retransmit, no dedup — the endpoints duplicate + dedup by seq
(drop-don't-buffer). This replaces the WS relay + Fly proxy hop for QUIC
sessions; the control path never touches the compute pod.

Video rides the same sessions on WebTransport **unidirectional streams** (one
short-lived stream per JPEG frame, node→browser): each incoming uni stream is
copied chunk-by-chunk to a lazily-opened uni stream on the paired connection,
FIN/RESET propagated, payload never parsed. Bounds (stream count, per-stream
bytes) live in ``relay_core.StreamForwardTable``; when the peer is absent the
stream is discarded (drop-don't-buffer, same as datagrams).

Deliberate split:
  * ``hmac_token.py`` + ``relay_core.py`` — pure, unit-tested logic.
  * this file — the aioquic H3/WebTransport wire glue (validated live on the
    Fly UDP deploy + on-device Quest, the Phase-0 gate; aioquic's exact
    WebTransport event surface is the one thing not exercisable offline).

Vendored into the SDK so a self-hosted coordinator can serve teleop itself:
"managed by the CLI alone" is not true if putting on a headset still requires
a hosted service. The upstream deployment stays the reference implementation.

Two things differ from upstream, and both follow from the coordinator being
the token authority (ADR 0038):

* Tokens are verified against the **coordinator's own** issued set rather than
  a shared HMAC secret minted elsewhere.
* The certificate is minted and rotated locally, and its SHA-256 is handed to
  the browser as ``serverCertificateHashes`` — there is no public CA for
  ``http://192.168.1.20:8900``.
"""
from __future__ import annotations

import logging
from typing import Callable
from urllib.parse import parse_qs, urlparse

from aioquic.asyncio import serve
from aioquic.asyncio.protocol import QuicConnectionProtocol
from aioquic.h3.connection import H3Connection
from aioquic.h3.events import (
    DatagramReceived,
    H3Event,
    HeadersReceived,
    WebTransportStreamDataReceived,
)
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.events import (
    ConnectionTerminated,
    ProtocolNegotiated,
    QuicEvent,
    StreamReset,
)

from .relay_core import (
    BROWSER_CLOSED_FRAME,
    Registry,
    StreamForwardTable,
    parse_teleop_path,
)

_LOG = logging.getLogger("interlatent.coordinator.relay")

#: Set by :func:`serve_relay`. Verifies (token, session_id, role)
#: against whatever the coordinator issued.
_VERIFY: "Callable[[str, str, str], tuple[bool, str]] | None" = None
REGISTRY: Registry["Peer"] = Registry()


class Peer:
    """A forwardable handle to one WebTransport session (its CONNECT stream)."""

    def __init__(self, proto: "RelayProtocol", session_stream_id: int) -> None:
        self.proto = proto
        self.session_stream_id = session_stream_id

    def send(self, data: bytes) -> None:
        try:
            self.proto._http.send_datagram(self.session_stream_id, data)
            self.proto.transmit()
        except Exception:
            pass  # peer mid-close; the sender's next datagram supersedes
        # Piggyback the uni-stream discard sweep on datagram cadence so a
        # browser connection keeps getting swept even while video pauses
        # (no-op when nothing is pending).
        self.proto.sweep_uni_discards()

    # -- uni-stream forwarding (video tee) --
    def open_uni_stream(self) -> int:
        """Open an outgoing uni stream on this peer's connection. Raises if
        the connection is mid-close — the caller drops the incoming stream."""
        return self.proto._http.create_webtransport_stream(
            self.session_stream_id, is_unidirectional=True
        )

    def send_stream(self, stream_id: int, data: bytes, end: bool) -> None:
        try:
            self.proto._quic.send_stream_data(stream_id, data, end_stream=end)
            self.proto.transmit()
        except Exception:
            pass  # connection mid-close; teardown reaps the mapping
        if end:
            self.proto.note_uni_done(stream_id)

    def reset_stream(self, stream_id: int) -> None:
        try:
            self.proto._quic.reset_stream(stream_id, 0)
            self.proto.transmit()
        except Exception:
            pass
        self.proto.note_uni_done(stream_id)


def _query_token(path: str) -> str:
    return (parse_qs(urlparse(path).query).get("token") or [""])[0]


class RelayProtocol(QuicConnectionProtocol):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._http: H3Connection | None = None
        # session-stream-id -> (role, session_id, Peer)
        self._sessions: dict[int, tuple[str, str, Peer]] = {}
        # incoming uni stream -> (dest Peer, outgoing uni stream) (video tee)
        self._fwd: StreamForwardTable[tuple[Peer, int]] = StreamForwardTable()
        # Outgoing uni streams fully written (FIN/RESET sent), awaiting the
        # peer's ack so they can be discarded from aioquic's bookkeeping —
        # see sweep_uni_discards.
        self._uni_discard_pending: list[int] = []

    # -- outgoing uni-stream discard (the send-only-stream leak) --
    #
    # aioquic never collects a locally-opened send-only uni stream: its
    # discard sweep requires QuicStream.is_finished (receiver AND sender),
    # and QuicStreamReceiver.__init__ hardcodes is_finished=False even for
    # a send-only stream. Every forwarded video frame therefore leaves a
    # dead QuicStream in the BROWSER-facing connection's ``_streams`` and
    # ``_streams_queue`` — both iterated on EVERY packet build — so the
    # relay's event loop slows linearly with connection age (~72 streams/s
    # at 24 Hz x 3 cams). Observed end-to-end as per-frame delivery time
    # growing ~40 -> ~100+ ms over a minute (node-side fps decaying
    # ~23 -> ~9 against a constant offer) and resetting on reconnect.
    # (Incoming node-side streams are receive-only — their sender half is
    # born finished, so aioquic collects those fine.)
    #
    # A stream can only be discarded once its send side is fully acked
    # (FIN or RESET delivered) — earlier, and the pending frame would
    # never be (re)sent. So completed sids park in _uni_discard_pending
    # and are swept at frame cadence by the SOURCE connection's handler.

    def note_uni_done(self, sid: int) -> None:
        self._uni_discard_pending.append(sid)
        del self._uni_discard_pending[:-512]  # runaway guard on a dying conn

    def sweep_uni_discards(self) -> None:
        """Discard acked send-only uni streams, replicating exactly what
        aioquic's own sweep does when it fires: pop from ``_streams``,
        record in ``_streams_finished`` (late peer frames treated as
        already-handled), drop from ``_streams_queue`` — plus the H3-layer
        entry. Private attrs; pinned by requirements (aioquic >=1.3,<2)."""
        if not self._uni_discard_pending:
            return
        pending, self._uni_discard_pending = self._uni_discard_pending, []
        for sid in pending:
            try:
                stream = self._quic._streams.get(sid)
                if stream is None:
                    continue
                if not stream.sender.is_finished:
                    self._uni_discard_pending.append(sid)  # ack not in yet
                    continue
                self._quic._streams.pop(sid, None)
                self._quic._streams_finished.add(sid)
                try:
                    self._quic._streams_queue.remove(stream)
                except ValueError:
                    pass
                try:
                    self._http._stream.pop(sid, None)
                except Exception:
                    pass
            except Exception:
                pass

    def quic_event_received(self, event: QuicEvent) -> None:
        if isinstance(event, ProtocolNegotiated) and event.alpn_protocol.startswith("h3"):
            self._http = H3Connection(self._quic, enable_webtransport=True)
        elif isinstance(event, ConnectionTerminated):
            self._teardown_all()
        elif isinstance(event, StreamReset):
            # H3 never surfaces resets — propagate to the forwarded half here.
            ref = self._fwd.remove(event.stream_id)
            if ref is not None:
                dest, out_sid = ref
                dest.reset_stream(out_sid)
            self._fwd.forget_dropped(event.stream_id)
            self._prune_h3_stream(event.stream_id)
        if self._http is not None:
            for h3_event in self._http.handle_event(event):
                self._on_h3(h3_event)

    def _on_h3(self, event: H3Event) -> None:
        if isinstance(event, HeadersReceived):
            self._on_headers(event)
        elif isinstance(event, DatagramReceived):
            self._on_datagram(event)
        elif isinstance(event, WebTransportStreamDataReceived):
            try:
                self._on_wt_stream(event)
            except Exception:
                # A malformed/raced stream event must never kill the
                # connection handler (control datagrams share it).
                _LOG.exception("uni-stream forward failed (stream dropped)")

    def _on_headers(self, event: HeadersReceived) -> None:
        h = {k: v for k, v in event.headers}
        if h.get(b":method") != b"CONNECT" or h.get(b":protocol") != b"webtransport":
            self._respond(event.stream_id, b"400")
            return
        path = h.get(b":path", b"").decode("utf-8", "replace")
        parsed = parse_teleop_path(path)
        if parsed is None:
            self._respond(event.stream_id, b"404")
            return
        role, session_id = parsed
        # The browser percent-encodes the token and the node does not
        # (_quic_client.py builds the query string by hand); parse_qs decodes,
        # so both spellings land on the same value.
        ok, reason = _VERIFY(_query_token(path), session_id, role)
        if not ok:
            _LOG.info("reject %s/%s: %s", role, session_id, reason)
            self._respond(event.stream_id, b"401")
            return

        # Accept the WebTransport session and register the peer.
        self._http.send_headers(
            event.stream_id,
            [(b":status", b"200"), (b"sec-webtransport-http3-draft", b"draft02")],
        )
        peer = Peer(self, event.stream_id)
        REGISTRY.attach(session_id, role, peer)  # supersede handled by GC on close
        self._sessions[event.stream_id] = (role, session_id, peer)
        self.transmit()
        _LOG.info("attach %s/%s (sessions=%d)", role, session_id, REGISTRY.session_count())

    def _on_datagram(self, event: DatagramReceived) -> None:
        info = self._sessions.get(event.stream_id)
        if info is None:
            return
        role, session_id, _ = info
        peer = REGISTRY.peer_of(session_id, role)
        if peer is not None:
            peer.send(event.data)

    def _on_wt_stream(self, event: WebTransportStreamDataReceived) -> None:
        """Copy one incoming uni stream (a video frame) to the paired peer.

        Chunk-at-a-time — aioquic emits one event per QUIC data chunk with
        ``stream_ended`` on the last. The outgoing stream is opened lazily on
        the first chunk; no peer / over caps → the stream is refused
        (STOP_SENDING) and its remaining chunks discarded via the dropped set.
        """
        in_sid = event.stream_id
        if self._fwd.is_dropped(in_sid):
            if event.stream_ended:
                self._fwd.forget_dropped(in_sid)
                self._prune_h3_stream(in_sid)
            return
        ref = self._fwd.get(in_sid)
        if ref is None:
            info = self._sessions.get(event.session_id)
            dest = REGISTRY.peer_of(info[1], info[0]) if info is not None else None
            if dest is None or not self._fwd.can_open():
                self._refuse_incoming(in_sid, event.stream_ended)
                return
            try:
                out_sid = dest.open_uni_stream()
            except Exception:
                self._refuse_incoming(in_sid, event.stream_ended)
                return
            ref = (dest, out_sid)
            self._fwd.open(in_sid, ref)
        dest, out_sid = ref
        # Frame-cadence sweep of the destination's completed uni streams —
        # the previous frame's FIN is normally acked by the time the next
        # frame arrives, so the pending list stays a handful of entries.
        dest.proto.sweep_uni_discards()
        if not self._fwd.note_bytes(in_sid, len(event.data)):
            # Runaway sender: kill both halves and discard the rest.
            self._fwd.remove(in_sid)
            dest.reset_stream(out_sid)
            self._refuse_incoming(in_sid, event.stream_ended)
            return
        dest.send_stream(out_sid, event.data, event.stream_ended)
        if event.stream_ended:
            self._fwd.remove(in_sid)
            self._prune_h3_stream(in_sid)

    def _refuse_incoming(self, in_sid: int, ended: bool) -> None:
        """Refuse/abandon an incoming uni stream: STOP_SENDING toward the
        sender and remember the sid so later chunks are discarded."""
        if ended:
            self._prune_h3_stream(in_sid)
            return
        try:
            self._quic.stop_stream(in_sid, 0)
        except Exception:
            pass
        self._fwd.drop(in_sid)

    def _prune_h3_stream(self, in_sid: int) -> None:
        """aioquic never removes finished streams from ``H3Connection._stream``
        (~200 B each) — at ~20 frame-streams/s that's a real leak on a
        long-lived relay process, so prune explicitly. Private attr; pinned
        by requirements (aioquic >=1.3,<2)."""
        try:
            self._http._stream.pop(in_sid, None)
        except Exception:
            pass

    def _respond(self, stream_id: int, status: bytes) -> None:
        try:
            self._http.send_headers(stream_id, [(b":status", status)], end_stream=True)
            self.transmit()
        except Exception:
            pass

    def _teardown_all(self) -> None:
        # This connection is dying — its streams die with it.
        self._uni_discard_pending.clear()
        # Half-forwarded outgoing streams get RESET rather than dangling on
        # the surviving connection.
        for dest, out_sid in self._fwd.drain():
            dest.reset_stream(out_sid)
        for role, session_id, peer in list(self._sessions.values()):
            survivor = REGISTRY.detach(session_id, role, peer)
            # Browser drop → tell the node to fall back to policy/hold, exactly
            # like the WS relay's synthetic close frame.
            if role == "browser" and survivor is not None:
                survivor.send(BROWSER_CLOSED_FRAME)
        self._sessions.clear()


async def serve_relay(
    *,
    host: str,
    port: int,
    cert_file: str,
    key_file: str,
    verify,
):
    """Serve the relay. ``verify(token, session_id, role) -> (ok, reason)``.

    Returns the aioquic server object; the caller owns its lifetime. Runs on
    the coordinator's own asyncio loop — the relay is pure I/O and the
    coordinator's HTTP surface is on threads, so they do not contend.
    """
    global _VERIFY
    _VERIFY = verify

    config = QuicConfiguration(
        alpn_protocols=["h3"],
        is_client=False,
        max_datagram_frame_size=65536,  # enable QUIC datagrams (WebTransport)
    )
    config.load_cert_chain(cert_file, key_file)

    _LOG.info(
        "teleop QUIC relay listening on %s:%d (h3/webtransport)", host, port
    )
    return await serve(host, port, configuration=config,
                       create_protocol=RelayProtocol)
