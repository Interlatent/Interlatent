"""Dumb-pipe QUIC child process for the node teleop channel.

Run as ``python -m interlatent.node.teleop._quic_proc`` by
:class:`~.quic_channel.QuicTeleopChannel`. Owns ONLY the aioquic WebTransport
connection — connect, handshake, reconnect-with-backoff — and pumps raw
datagrams verbatim between the relay and the parent's loopback UDP socket
(framing in ``_quic_ipc``). No codec, no dedupe, no pacing: all protocol
logic stays in the parent — with ONE child-owned policy: the video tee's
load shedding (:class:`_VideoGovernor` — in-flight stream cap + TTL resets).
It must live here because only the child can observe when a unidirectional
stream finishes (acked+FIN) or needs RESET_STREAM; the parent just hands
over framed TYPE_VIDEO payloads and the child ships each on its own uni
stream (still never inspecting the wire bytes).

Why a process: the robot drivers (e.g. i2rt's ~270 Hz gravity-comp/CAN
threads) monopolize the GIL and starve an in-process asyncio loop, so the
timing-sensitive QUIC handshake never completes. A child process has its own
GIL. See the ADR 0017 amendment.

Lifecycle: exits when the parent closes its stdin pipe (EOF — covers parent
crash and clean stop alike). Mints its own relay tokens on every reconnect
from creds passed via env. Relay flakiness stays inside the reconnect loop
here; the parent only respawns this process if it dies.
"""
from __future__ import annotations

import asyncio
import logging
import os
import socket
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from . import _quic_ipc
from ._mint import mint_teleop_token

_LOG = logging.getLogger("interlatent.node.teleop._quic_proc")

_RECONNECT_INITIAL_S = 1.0
_RECONNECT_MAX_S = 15.0
_HELLO_PERIOD_S = 1.0
_SHUTDOWN_GRACE_S = 3.0
# One-line pump summary cadence (matches the parent's telemetry period).
_STATS_LOG_PERIOD_S = 5.0

# Video tee load shedding (drop-don't-buffer): at most this many unfinished
# frame-streams per camera / overall before new frames are dropped at the
# source, and a stream still unfinished after the TTL is RESET so stale
# bytes stop competing with control datagrams in the congestion window.
_VIDEO_INFLIGHT_PER_CAM = 2
_VIDEO_INFLIGHT_GLOBAL = 6
_VIDEO_STREAM_TTL_S = 1.0


class _VideoGovernor:
    """In-flight cap + TTL for per-frame video streams. Pure policy —
    ``now``/``is_finished``/``reset`` are injected so unit tests never touch
    aioquic. Counters feed the 5s stats line."""

    def __init__(
        self,
        *,
        now: Callable[[], float],
        is_finished: Callable[[int], bool],
        reset: Callable[[int], None],
    ) -> None:
        self._now = now
        self._is_finished = is_finished
        self._reset = reset
        self._inflight: "dict[int, tuple[str, float]]" = {}
        self.opened = 0
        self.finished = 0
        self.dropped_cap = 0
        self.reset_ttl = 0

    def _sweep(self) -> None:
        now = self._now()
        for sid in list(self._inflight):
            cam, opened_at = self._inflight[sid]
            if self._is_finished(sid):
                del self._inflight[sid]
                self.finished += 1
            elif now - opened_at > _VIDEO_STREAM_TTL_S:
                try:
                    self._reset(sid)
                except Exception:
                    pass
                del self._inflight[sid]
                self.reset_ttl += 1

    def admit(self, cam: str) -> bool:
        self._sweep()
        if len(self._inflight) >= _VIDEO_INFLIGHT_GLOBAL or (
            sum(1 for c, _ in self._inflight.values() if c == cam)
            >= _VIDEO_INFLIGHT_PER_CAM
        ):
            self.dropped_cap += 1
            return False
        return True

    def note_open(self, sid: int, cam: str) -> None:
        self.opened += 1
        self._inflight[sid] = (cam, self._now())

    def reset_all(self) -> None:
        # Session teardown: the streams die with the connection; just forget.
        self._inflight.clear()


@dataclass
class _Cfg:
    parent_port: int
    cookie: str
    api_base: str
    api_key: str
    session_id: str
    token_path: str
    bypass_key: Optional[str]


def _load_cfg() -> _Cfg:
    def required(name: str) -> str:
        val = os.environ.get(name, "").strip()
        if not val:
            print(f"quic-proc: missing required env var {name}", file=sys.stderr)
            sys.exit(2)
        return val

    return _Cfg(
        parent_port=int(required(_quic_ipc.ENV_PARENT_PORT)),
        cookie=required(_quic_ipc.ENV_COOKIE),
        api_base=required(_quic_ipc.ENV_API_BASE),
        api_key=required(_quic_ipc.ENV_API_KEY),
        session_id=required(_quic_ipc.ENV_SESSION_ID),
        token_path=required(_quic_ipc.ENV_TOKEN_PATH),
        bypass_key=os.environ.get(_quic_ipc.ENV_BYPASS_KEY) or None,
    )


class _ParentLink(asyncio.DatagramProtocol):
    """Loopback endpoint toward the parent. Inbound TYPE_DATA goes straight
    to the live WT session's send_datagram (we're already on the event loop —
    aioquic's send is a synchronous enqueue+transmit, no queue needed);
    dropped while no session is up."""

    def __init__(self) -> None:
        self._transport: Optional[asyncio.DatagramTransport] = None
        self._relay_send: Optional[Callable[[bytes], None]] = None
        self._video_send: Optional[Callable[[str, bytes], None]] = None
        self.video_governor: Optional[_VideoGovernor] = None  # stats only
        self.rx_from_parent = 0  # DATA datagrams parent→relay
        self.rx_video_from_parent = 0  # VIDEO frames parent→relay
        self.tx_to_parent = 0  # DATA datagrams relay→parent

    def connection_made(self, transport) -> None:  # type: ignore[override]
        self._transport = transport
        sock = transport.get_extra_info("socket")
        if sock is not None:
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, _quic_ipc.SOCK_BUF_BYTES)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, _quic_ipc.SOCK_BUF_BYTES)
            except OSError:
                pass

    def datagram_received(self, data: bytes, addr) -> None:  # type: ignore[override]
        parsed = _quic_ipc.parse(data)
        if parsed is None:
            return
        kind, payload = parsed
        if kind == _quic_ipc.TYPE_DATA:
            send = self._relay_send
            if send is not None:
                self.rx_from_parent += 1
                try:
                    send(payload)
                except Exception:
                    pass
            return
        if kind == _quic_ipc.TYPE_VIDEO:
            vsend = self._video_send
            if vsend is None:
                return  # no session up — drop, the next frame supersedes
            parsed_video = _quic_ipc.parse_video(payload)
            if parsed_video is None:
                return
            self.rx_video_from_parent += 1
            try:
                vsend(*parsed_video)
            except Exception:
                pass

    def set_relay_sender(self, fn: Optional[Callable[[bytes], None]]) -> None:
        self._relay_send = fn

    def set_video_sender(self, fn: Optional[Callable[[str, bytes], None]]) -> None:
        self._video_send = fn

    def send_data(self, payload: bytes) -> None:
        if self._transport is not None:
            self.tx_to_parent += 1
            self._transport.sendto(_quic_ipc.encode_data(payload))

    def send_control(self, obj: dict) -> None:
        if self._transport is not None:
            self._transport.sendto(_quic_ipc.encode_ctrl(obj))


async def _hello_loop(link: _ParentLink, cfg: _Cfg) -> None:
    """1s hello heartbeat: makes a lost first hello a non-event and proves to
    the parent that the child imported and is running (its backoff reset)."""
    hello = {"t": "hello", "cookie": cfg.cookie, "pid": os.getpid()}
    while True:
        link.send_control(hello)
        await asyncio.sleep(_HELLO_PERIOD_S)


def _mint(cfg: _Cfg) -> "tuple[str, str]":
    data = mint_teleop_token(
        api_base=cfg.api_base,
        token_path=cfg.token_path,
        api_key=cfg.api_key,
        bypass_key=cfg.bypass_key,
        role="node",
    )
    wt = data.get("webtransport_url")
    if not wt:
        raise RuntimeError("token response has no webtransport_url (transport != quic?)")
    return str(wt), str(data["token"])


async def _sleep_or_stop(stop: asyncio.Event, secs: float) -> bool:
    try:
        await asyncio.wait_for(stop.wait(), timeout=secs)
    except asyncio.TimeoutError:
        pass
    return stop.is_set()


async def _session_loop(cfg: _Cfg, link: _ParentLink, stop: asyncio.Event) -> None:
    """Mint → connect → pump relay datagrams to the parent, forever. The
    reconnect backoff lives here so relay flakiness never exits the process."""
    # aioquic import deferred to keep startup (and the first hello) fast.
    from ._quic_client import connect_webtransport

    backoff = _RECONNECT_INITIAL_S
    while not stop.is_set():
        try:
            wt_url, token = await asyncio.to_thread(_mint, cfg)
        except Exception as exc:
            _LOG.info("quic-proc token-mint failed: %s (retry %.0fs)", exc, backoff)
            if await _sleep_or_stop(stop, backoff):
                return
            backoff = min(backoff * 2, _RECONNECT_MAX_S)
            continue
        reason = "closed"
        try:
            async with connect_webtransport(wt_url, token) as wt:
                governor = _VideoGovernor(
                    now=time.monotonic,
                    is_finished=wt.uni_stream_finished,
                    reset=wt.reset_uni_stream,
                )

                def _send_video(cam: str, wire: bytes) -> None:
                    # One uni stream per frame; the governor sheds load when
                    # the uplink can't keep up (cap) or a frame goes stale
                    # (TTL reset). Wire bytes stay opaque.
                    if governor.admit(cam):
                        sid = wt.open_uni_stream(wire)
                        if sid is not None:
                            governor.note_open(sid, cam)

                link.video_governor = governor
                link.set_video_sender(_send_video)
                link.set_relay_sender(wt.send_datagram)
                link.send_control({"t": "connected"})
                _LOG.info("quic-proc connected session=%s", cfg.session_id)
                backoff = _RECONNECT_INITIAL_S
                async for data in wt.datagrams():
                    link.send_data(data)
        except asyncio.CancelledError:
            reason = "shutdown"
            raise
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            _LOG.warning(
                "quic-proc session ended (%s); reconnect in %.0fs", reason, backoff
            )
        finally:
            link.set_relay_sender(None)
            link.set_video_sender(None)
            gov = link.video_governor
            if gov is not None:
                try:
                    gov.reset_all()
                except Exception:
                    pass
            link.send_control({"t": "disconnected", "reason": reason})
        if await _sleep_or_stop(stop, backoff):
            return
        backoff = min(backoff * 2, _RECONNECT_MAX_S)


async def _stats_loop(link: _ParentLink) -> None:
    """One line per 5s so a dead pump is observable in the node log."""
    last_rx = last_tx = 0
    last_video = (0, 0, 0, 0)
    last_gov: Optional[_VideoGovernor] = None
    while True:
        await asyncio.sleep(_STATS_LOG_PERIOD_S)
        rx, tx = link.rx_from_parent, link.tx_to_parent
        gov = link.video_governor
        if gov is not last_gov:  # new session → counters restarted
            last_gov = gov
            last_video = (0, 0, 0, 0)
        video = (
            (gov.opened, gov.finished, gov.dropped_cap, gov.reset_ttl)
            if gov is not None
            else (0, 0, 0, 0)
        )
        if rx != last_rx or tx != last_tx or video != last_video:
            dv = tuple(a - b for a, b in zip(video, last_video))
            _LOG.info(
                "quic-proc pumped (%.0fs): parent->relay=%d relay->parent=%d "
                "video: open=%d fin=%d drop_cap=%d reset_ttl=%d",
                _STATS_LOG_PERIOD_S, rx - last_rx, tx - last_tx,
                dv[0], dv[1], dv[2], dv[3],
            )
        last_rx, last_tx = rx, tx
        last_video = video


def _watch_stdin(loop: asyncio.AbstractEventLoop, stop: asyncio.Event) -> None:
    """Blocks until the parent closes our stdin pipe (or dies), then stops the
    loop. A thread rather than loop.add_reader — reader callbacks on pipes are
    platform-dependent; a blocking read is portable."""
    try:
        sys.stdin.buffer.read()
    except Exception:
        pass
    try:
        loop.call_soon_threadsafe(stop.set)
    except RuntimeError:
        pass  # loop already closed


async def _amain(cfg: _Cfg) -> None:
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    threading.Thread(
        target=_watch_stdin, args=(loop, stop), name="quic-proc-stdin", daemon=True
    ).start()

    transport, link = await loop.create_datagram_endpoint(
        _ParentLink,
        local_addr=("127.0.0.1", 0),
        remote_addr=("127.0.0.1", cfg.parent_port),
    )
    tasks = [
        asyncio.create_task(_hello_loop(link, cfg)),
        asyncio.create_task(_session_loop(cfg, link, stop)),
        asyncio.create_task(_stats_loop(link)),
    ]
    try:
        await stop.wait()
    finally:
        # Shutdown ordering: cancelling the session task unwinds the
        # `async with connect_webtransport`, sending QUIC CONNECTION_CLOSE.
        for t in tasks:
            t.cancel()
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True), _SHUTDOWN_GRACE_S
            )
        except asyncio.TimeoutError:
            pass
        transport.close()
    _LOG.info("quic-proc exiting session=%s", cfg.session_id)


def main() -> None:
    logging.basicConfig(
        stream=sys.stderr,
        level=os.environ.get("INTERLATENT_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(_amain(_load_cfg()))


if __name__ == "__main__":
    main()
