"""Dumb-pipe QUIC child process for the node teleop channel.

Run as ``python -m interlatent.node.teleop._quic_proc`` by
:class:`~.quic_channel.QuicTeleopChannel`. Owns the aioquic WebTransport
connection — connect, handshake, reconnect-with-backoff — and pumps raw
datagrams verbatim between the relay and the parent's loopback UDP socket
(framing in ``_quic_ipc``). No codec, no dedupe: DATA payloads are opaque here.

The video preview path is deliberately **bicameral**: the child owns the
low-level *mechanism* — the in-flight stream cap + TTL resets
(:class:`_VideoGovernor`), which must live here because only the child can
observe when a unidirectional stream finishes (acked+FIN) or needs RESET —
while the parent owns the *rate control* (its ``PreviewBackoff`` reacts to the
child's ``reset_ttl`` counter, shipped via ``vstats``). The child still never
inspects the video wire bytes; it just ships each framed TYPE_VIDEO payload on
its own uni stream. aioquic's own send-only-uni-stream leak GC lives one layer
down in ``_quic_client._UniStreamGC`` (ADR 0020).

Why a process: the robot drivers (e.g. i2rt's ~270 Hz gravity-comp/CAN
threads) monopolize the GIL and starve an in-process asyncio loop, so the
timing-sensitive QUIC handshake never completes. A child process has its own
GIL. See ADR 0021.

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

from .. import _env
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
#
# The per-camera cap bounds the delivered preview rate on a long-RTT
# relay path: fps/cam ~= cap / stream_completion_time (cap 2 at ~200ms
# RTT tops out ~10 fps). INTERLATENT_QUIC_VIDEO_INFLIGHT raises it for
# operators who want more headset fps — but note the latency cost: more
# unfinished streams is a DEEPER queue on a bufferbloated uplink, and a
# preview frame's glass-to-eye age grows to fill it. The global cap stays
# 3x per-cam (the 3-camera rig ratio).
#
# The TTL is the freshness ceiling, and it is the load-bearing knob for a
# LIVE preview: a frame still unsent after the TTL is RESET (dropped), not
# delivered late, because a stale preview frame is worse than a gap. The
# 1.0 s legacy default let glass-to-eye latency climb to ~1 s under
# congestion (a standing queue draining slower than it fills — measured
# browser-side as video_lag growing while offered fps fell); 350 ms bounds
# the felt lag without ever touching a healthy link (which delivers frames
# in tens of ms). Lower it for a tighter latency cap at the cost of more
# dropped frames, raise it to tolerate a slow link. Env-tunable in ms.

_DEFAULT_INFLIGHT_PER_CAM = 2
_VIDEO_INFLIGHT_PER_CAM = _env.env_int(
    "INTERLATENT_QUIC_VIDEO_INFLIGHT", _DEFAULT_INFLIGHT_PER_CAM, 1, 16
)
_VIDEO_INFLIGHT_GLOBAL = 3 * _VIDEO_INFLIGHT_PER_CAM
_VIDEO_STREAM_TTL_S = _env.env_int(
    "INTERLATENT_QUIC_VIDEO_TTL_MS", 350, 50, 5000
) / 1000.0


class _VideoGovernor:
    """In-flight cap + TTL for per-frame video streams. Pure policy —
    ``now``/``is_finished``/``reset`` are injected so unit tests never touch
    aioquic. Counters feed the 5s stats line.

    Load shedding only: it caps how many uni streams are in flight per camera /
    overall, and RESETs a stream that goes stale (unfinished past the TTL) so
    its bytes stop competing with control datagrams. It does NOT touch aioquic's
    per-connection stream bookkeeping — that leak GC (ADR 0020) lives in
    ``_WTClientProtocol`` (``_UniStreamGC``), which discards every uni stream
    once its send side acks, independent of this governor.
    """

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
        self._spec_send: Optional[Callable[[bytes], None]] = None
        self.video_governor: Optional[_VideoGovernor] = None  # stats only
        self.wt_session = None  # live session (datagram-drop counter), stats only
        self.rx_from_parent = 0  # DATA datagrams parent→relay
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
            try:
                vsend(*parsed_video)
            except Exception:
                pass
            return
        if kind == _quic_ipc.TYPE_SPEC:
            # The framed kinematic_spec — ship it on its own uni stream,
            # ungoverned (one-shot and load-bearing, unlike per-frame video).
            ssend = self._spec_send
            if ssend is None:
                return  # no session up — the browser retries its request
            try:
                ssend(payload)
            except Exception:
                pass

    def set_relay_sender(self, fn: Optional[Callable[[bytes], None]]) -> None:
        self._relay_send = fn

    def set_video_sender(self, fn: Optional[Callable[[str, bytes], None]]) -> None:
        self._video_send = fn

    def set_spec_sender(self, fn: Optional[Callable[[bytes], None]]) -> None:
        self._spec_send = fn

    def send_data(self, payload: bytes) -> None:
        if self._transport is not None:
            self.tx_to_parent += 1
            self._transport.sendto(_quic_ipc.encode_data(payload))

    def send_control(self, obj: dict) -> None:
        if self._transport is not None:
            self._transport.sendto(_quic_ipc.encode_ctrl(obj))


def _vstats_payload(gov: Optional[_VideoGovernor]) -> Optional[dict]:
    """The video-path counters the parent's preview backoff actually consumes:
    ``drop_cap`` (cap-pacing, diagnostic only) and ``reset_ttl`` (the sole
    backoff signal). None while no video governor exists (no relay session).

    CUMULATIVE, not per-window deltas: a lost loopback datagram then costs
    nothing — the parent diffs against its last sample (and clamps negative
    deltas after a reconnect restarts these counters). The child only OBSERVES;
    the rate policy stays in the parent. (The ``qs`` leak gauge and ``dg_drop``
    ride the 5s stats log line, not this IPC — the parent has no use for them.)
    """
    if gov is None:
        return None
    return {
        "t": "vstats",
        "drop_cap": gov.dropped_cap,
        "reset_ttl": gov.reset_ttl,
    }


async def _hello_loop(link: _ParentLink, cfg: _Cfg) -> None:
    """1s hello heartbeat: makes a lost first hello a non-event and proves to
    the parent that the child imported and is running (its backoff reset).
    Rides a vstats message alongside each hello while video is flowing —
    the parent's preview backoff consumes it; an old parent ignores it."""
    hello = {"t": "hello", "cookie": cfg.cookie, "pid": os.getpid()}
    while True:
        link.send_control(hello)
        vstats = _vstats_payload(link.video_governor)
        if vstats is not None:
            link.send_control(vstats)
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

                def _send_spec(wire: bytes) -> None:
                    # One uni stream, ungoverned: the kinematic_spec is one-shot
                    # and load-bearing (the browser can't build its solver
                    # without it), so it must never be cap-shed like video.
                    wt.open_uni_stream(wire)

                link.video_governor = governor
                link.wt_session = wt
                link.set_video_sender(_send_video)
                link.set_spec_sender(_send_spec)
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
            link.set_spec_sender(None)
            link.wt_session = None
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
    last_dg_drop = 0
    last_gov: Optional[_VideoGovernor] = None
    while True:
        await asyncio.sleep(_STATS_LOG_PERIOD_S)
        rx, tx = link.rx_from_parent, link.tx_to_parent
        gov = link.video_governor
        if gov is not last_gov:  # new session → counters restarted
            last_gov = gov
            last_video = (0, 0, 0, 0)
            last_dg_drop = 0
        video = (
            (gov.opened, gov.finished, gov.dropped_cap, gov.reset_ttl)
            if gov is not None
            else (0, 0, 0, 0)
        )
        wt = link.wt_session
        try:
            dg_drop = wt.datagrams_dropped() if wt is not None else last_dg_drop
        except Exception:
            dg_drop = last_dg_drop
        try:
            qs = wt.quic_stream_count() if wt is not None else 0
        except Exception:
            qs = -1
        if (
            rx != last_rx or tx != last_tx or video != last_video
            or dg_drop != last_dg_drop
        ):
            dv = tuple(a - b for a, b in zip(video, last_video))
            _LOG.info(
                "quic-proc pumped (%.0fs): parent->relay=%d relay->parent=%d "
                "video: open=%d fin=%d drop_cap=%d reset_ttl=%d dg_drop=%d qs=%d",
                _STATS_LOG_PERIOD_S, rx - last_rx, tx - last_tx,
                dv[0], dv[1], dv[2], dv[3], dg_drop - last_dg_drop, qs,
            )
        last_rx, last_tx = rx, tx
        last_video = video
        last_dg_drop = dg_drop


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
    overrides = _env.overrides()
    if overrides:
        _LOG.info(
            "teleop knob overrides: %s",
            ", ".join(f"{k}={v}" for k, v in sorted(overrides.items())),
        )
    asyncio.run(_amain(_load_cfg()))


if __name__ == "__main__":
    main()
