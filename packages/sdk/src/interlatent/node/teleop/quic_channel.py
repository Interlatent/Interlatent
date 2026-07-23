"""Node-side QUIC/WebTransport teleop channel.

The node's teleop transport: it opens a WebTransport session to the co-located
QUIC relay and consumes **unreliable datagrams** carrying ``mode="targets"``
frames the browser already IK-solved. Exposes the surface the control loop uses
(``latest_frame`` / ``send_state`` / ``connected`` / ``start`` / ``stop``), so
the daemon drives it without any transport-specific control-loop code.

Design notes:
  * targets arrive as duplicated datagrams; we dedupe by ``seq`` (latest-wins)
    so a late duplicate can't clobber a newer frame (drop-don't-buffer).
  * ``send_state`` streams the robot's live joint vector back **to the browser**
    (not the pod) — the browser FK's it to close the clutch loop + reconcile.
    Duplicated for loss tolerance, same ~15 Hz cadence.
  * the preview/video tee rides the SAME WebTransport session as control, but
    on **unidirectional streams** (one short-lived stream per JPEG frame, FIN
    after the last byte) — datagrams are ~1200 B, too small for JPEGs, while
    per-frame streams are reliable within a frame yet independent across
    frames (a lost packet delays only its own frame). The parent frames each
    preview JPEG with the browser-facing ``video`` header and hands it to the
    child (``TYPE_VIDEO``); the child owns stream opening plus the in-flight
    cap/TTL load shedding (only it can see stream completion). Kill switch:
    ``INTERLATENT_QUIC_VIDEO=0`` reverts to control-only.

The aioquic connection itself runs in a dumb-pipe **child process**
(``_quic_proc``, spawned ``python -m ...``): QUIC's handshake/loss-recovery
timers live in userspace Python, and robot-driver threads (e.g. i2rt's ~270 Hz
gravity comp) share the GIL — isolation makes timer starvation structurally
impossible, while WS can stay in-process because TCP retransmission is
kernel-side. The child owns only connect/handshake/reconnect and pumps raw
datagrams verbatim over a loopback UDP socket (framing in ``_quic_ipc``); ALL
protocol logic — codec, dedupe, staleness, pacing, applied-seq echo, arrival
telemetry — stays here in the parent. Child exits on stdin EOF (no orphans);
this class respawns a crashed child with 1→15 s backoff. See ADR 0021 (which
also records the ``_connected`` attribute-shadowing bug in ``_quic_client``
that was the actual cause of the v1 handshake failure).

This process never imports aioquic; only the child does.
"""
from __future__ import annotations

import json
import logging
import os
import secrets
import socket
import subprocess
import sys
import threading
import time
from typing import Dict, Optional

from .. import _env
from . import _quic_ipc
from ._frame_store import LatestFrameStore
from ._telemetry import ArrivalTracker
from .frame import TeleopFrame, frame_with_header

_LOG = logging.getLogger(__name__)

# Reconnect/child-respawn backoff between dropped connections. Teleop can fail
# for boring reasons (GPU box bounced, NAT timeout) and the operator might
# engage again at any moment, so we keep retrying.
_RECONNECT_INITIAL_S = 1.0
_RECONNECT_MAX_S = 15.0

# Node→browser state heartbeat rate. Tiny frames, RTT-bound; 15 Hz keeps the
# browser's IK seed fresh without competing with control/preview traffic.
_STATE_SEND_PERIOD_S = 1.0 / 15.0

# Period of the rolling frame-arrival latency summary (INFO). Matches the
# relay's 5s browser-frame summaries so the logs line up.
_STATS_LOG_PERIOD_S = 5.0


# Live-preview push rate (node→browser, small downscaled JPEGs). The cadence is
# the dominant term in perceived video latency (mean staleness ≈ half the
# period), so it's tunable per node via INTERLATENT_PREVIEW_HZ. Clamped to
# [1, 30]; the control loop can't produce more than its tick rate anyway.
def _preview_period_s() -> float:
    return 1.0 / _env.env_float("INTERLATENT_PREVIEW_HZ", 10.0, 1.0, 30.0)


# A viewer is "present" while browser frames keep arriving (the overlay sends
# keepalives even when disengaged). No frames for this long ⇒ nobody is
# watching ⇒ stop burning uplink on previews.
_VIEWER_PRESENCE_S = 5.0

# Duplicate each outbound state datagram this many times (loss tolerance).
_STATE_DUP = 2
# A seq that jumps this far *backward* is a browser reconnect/reset, not a
# reordered duplicate — accept it and re-anchor.
_SEQ_RESET_GAP = 1000
# Supervisor read cadence: recvfrom timeout doubles as the supervision tick.
_SUPERVISE_TICK_S = 0.5
# Rate-limit for answering browser ``request_spec`` datagrams: the browser
# retries the request until it sees the spec, so without a floor a burst of
# retries would open a burst of uni streams. One every this often is plenty.
_SPEC_SEND_MIN_INTERVAL_S = 0.2


def frame_spec_wire(spec_obj: dict, robot_kind: str) -> bytes:
    """Frame a kinematic_spec for the browser using the shared
    :func:`~.frame.frame_with_header` envelope (header ``type:"spec"``
    distinguishes it from a video frame), so the browser's inbound-uni-stream
    reader is one parser. Pure + unit-tested."""
    return frame_with_header(
        {"type": "spec", "robot_kind": robot_kind},
        json.dumps(spec_obj).encode("utf-8"),
    )


def is_request_spec(payload: bytes) -> bool:
    """True if a browser→node datagram is a ``request_spec`` control message
    (not a target frame). Cheap substring guard first so the ~60 Hz target
    hot path never pays a JSON parse. Pure + unit-tested."""
    if b"request_spec" not in payload:
        return False
    try:
        obj = json.loads(payload.decode("utf-8"))
    except Exception:
        return False
    return isinstance(obj, dict) and obj.get("type") == "request_spec"


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


def _advance_preview_deadline(next_due: float, now: float, period_s: float) -> float:
    """Next preview deadline after a send at ``now``. Pure + unit-tested.

    Credit-based: advance one period from where the deadline WAS, not from
    ``now`` — the deadline is only sampled once per control tick, so
    anchoring on ``now`` rounds every gap up to a whole number of ticks and
    the delivered rate quantizes below the configured one (46 ms ticks vs a
    50 ms period → every other tick → half rate). Advancing the deadline
    itself lets a slightly-late send borrow from the next interval, so the
    rate averages out to exactly the configured one. The ``now`` floor keeps
    a stale deadline (viewer-absent gap, or ticks slower than the period)
    from scheduling sends in the past — no catch-up burst; the rate simply
    tops out at the tick rate.
    """
    return max(next_due + period_s, now)


class PreviewBackoff:
    """Multiplicative preview-rate backoff driven by the child's vstats.

    The parent cannot observe video stream completion (the hand-off to
    the child is a fire-and-forget loopback sendto), so the child
    reports cumulative counters (``vstats``) and this class turns
    per-window deltas into a period multiplier. AIMD-shaped asymmetry:
    double the period the moment a window shows sustained loss, recover
    slowly (/1.25 per good window, ~10 windows from the floor back to
    the configured rate) so a marginal link re-probes instead of
    flapping.

    The backoff signal is **TTL resets only** (frames that exceeded
    INTERLATENT_QUIC_VIDEO_TTL_MS while in flight), never the
    governor's admission denials (``drop_cap``). Admission denials are
    the pacing mechanism, not congestion: with the in-flight cap the
    timer intentionally over-offers and the governor discards, for
    free, every frame that arrives while a slot is busy — delivered
    fps then equals the link's completion rate. Counting those
    denials as loss made the backoff strangle its own pacing loop:
    any window where the timer outran completion tripped a halving,
    recovery never fired (a busy link always shows collisions), and
    the multiplier ratcheted the timer *below* the completion rate —
    delivered fps fell while the link had headroom and every drop was
    already free. A TTL reset, by contrast, means a frame sat in
    flight past the freshness budget — the one unambiguous "link
    cannot keep up" signal, already time-normalized by the TTL, so an
    absolute threshold is correct: >= 2 stale frames in a window
    halves the rate, a fully fresh window recovers a step, exactly 1
    holds (dead band for a stray WiFi burp).

    The floor rate is 1 Hz (period ceiling 1 s): the operator's quad
    stays alive at negligible bandwidth while the configured
    INTERLATENT_PREVIEW_HZ becomes a ceiling, not a promise. Pure +
    unit-tested.
    """

    _BACKOFF_STALE = 2
    _BACKOFF_FACTOR = 2.0
    _RECOVER_FACTOR = 1.25
    _MAX_PERIOD_S = 1.0

    def __init__(self) -> None:
        self._mult = 1.0

    @property
    def mult(self) -> float:
        return self._mult

    def on_window(self, stale: int, base_period_s: float) -> None:
        """Feed one ~1s window's TTL-reset delta.

        The multiplier is capped exactly where the effective period hits
        the 1 s ceiling for this base period — overshooting past the
        floor would only lengthen recovery.
        """
        cap = max(1.0, self._MAX_PERIOD_S / max(base_period_s, 1e-6))
        if stale >= self._BACKOFF_STALE:
            self._mult = min(self._mult * self._BACKOFF_FACTOR, cap)
        elif stale == 0:
            self._mult = max(self._mult / self._RECOVER_FACTOR, 1.0)

    def period(self, base_period_s: float) -> float:
        # The 1 s ceiling (and the 1.0 floor) already live in on_window's
        # clamp of _mult, so a plain scale is exact here.
        return base_period_s * self._mult


def encode_state_datagram(qpos, seq: int, applied_seq: int = -1) -> bytes:
    """Node→browser joint-state datagram (JSON). ``qpos`` is action-order,
    robot-native units — same convention as RecordTick's observation_state.

    ``applied_seq`` echoes the target seq the control loop most recently
    executed, so the browser can compute command round-trip latency against
    its own clock (no cross-machine clock sync needed).

    ``ts_ms`` is the node's monotonic clock at send — the SAME clock domain
    as the preview tee's video-frame ``ts_ms`` header. State datagrams are
    tiny and drop-don't-queue (never parked behind a stalled congestion
    window), so the browser's min over recent ``(arrival - ts_ms)`` is a
    live clock-skew anchor: subtracting it from a video frame's skew gives
    ABSOLUTE glass-to-eye video age, honest even while video frames queue —
    the case the old above-fastest-frame lag metric absorbed into its
    baseline."""
    return json.dumps({
        "type": "state",
        "seq": int(seq),
        "applied_seq": int(applied_seq),
        "ts_ms": time.monotonic_ns() // 1_000_000,
        "qpos": [float(x) for x in qpos],
    }).encode("utf-8")


def decode_target_datagram(data: bytes) -> Optional[TeleopFrame]:
    """Browser→node datagram → TeleopFrame (stamps received_at_ns, honest
    deadman handling via TeleopFrame.from_json). None on garbage."""
    try:
        return TeleopFrame.from_json(data.decode("utf-8"))
    except Exception:
        return None


class QuicTeleopChannel(LatestFrameStore):
    """WebTransport/QUIC teleop channel.

    Inherits the thread-safe latest-frame + sticky-estop store
    (:class:`LatestFrameStore`) so ``latest_frame``/``consume_estop`` and the
    staleness/estop semantics live in one place (ADR 0016).
    """

    def __init__(
        self,
        *,
        session_id: str,
        api_base: str,
        api_key: str,
        token_path: Optional[str] = None,
        bypass_key: Optional[str] = None,
        robot_kind: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._session_id = session_id
        self._api_base = api_base.rstrip("/")
        self._api_key = api_key
        self._bypass_key = bypass_key
        self._robot_kind = robot_kind
        self._token_path = (
            token_path or f"/api/v1/inference/sessions/{session_id}/teleop-token"
        )

        # Node-sourced kinematic_spec: the framed bytes we ship to the browser
        # (on its own uni stream) when it sends a ``request_spec`` datagram, so
        # it can build its IK solver from THIS node's installed robot data
        # rather than the platform backend. Loaded best-effort at start(); None
        # when no local data exists (browser falls back to the HTTP endpoint).
        self._spec_wire: Optional[bytes] = None
        self._last_spec_sent_at = 0.0
        # One-shot per session so the spec handshake logs once, not per retry
        # (reset on each connect in _handle_datagram).
        self._spec_served = False
        self._spec_miss_warned = False

        # _lock / _latest / _estop_seen live in LatestFrameStore (ADR 0016).
        self._dedup = LatestSeqBuffer()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._connected = False

        # Child-process plumbing (the aioquic connection lives in the child).
        self._sock: Optional[socket.socket] = None
        self._proc: Optional[subprocess.Popen] = None
        self._child_addr = None  # pinned from the first cookie-valid hello
        self._cookie = ""
        self._spawn_backoff = _RECONNECT_INITIAL_S
        self._next_spawn_at = 0.0
        self._child_exits = 0
        self._hello_ever = False  # a valid hello proves the child runs
        self._spawn_warned = False  # one-shot install/interpreter hint

        # Video tee (see module docs). On by default in QUIC mode;
        # INTERLATENT_QUIC_VIDEO=0 is the kill switch (control unaffected).
        self._video_enabled = _env.env_bool("INTERLATENT_QUIC_VIDEO", True)
        self._preview_period_s = _preview_period_s()
        # Congestion-adaptive preview rate (INTERLATENT_PREVIEW_HZ is the
        # ceiling): the child's 1s vstats messages feed PreviewBackoff.
        # INTERLATENT_PREVIEW_ADAPTIVE=0 pins today's fixed behavior.
        self._preview_adaptive = _env.env_bool("INTERLATENT_PREVIEW_ADAPTIVE", True)
        self._preview_backoff = PreviewBackoff()
        self._vstats_last: Optional[tuple] = None  # (drop_cap, reset_ttl)
        self._last_rx_at = 0.0  # viewer presence: any decoded target frame
        self._next_preview_due = 0.0  # credit deadline; see preview_due()
        self._preview_seq = 0
        self._pv_window = 0  # frames handed to the child this stats window
        self._pv_logged_once = False

        self._out_seq = 0
        self._last_state_sent_at = 0.0
        # Seq of the target the control loop most recently executed — echoed
        # to the browser for its RTT measurement. -1 = nothing applied yet.
        self._last_applied_seq = -1

        # Arrival telemetry (supervisor thread only): inter-arrival gap of
        # target datagrams — the node-observable half of teleop latency
        # (relay→node jitter). Shared 5s rolling summary; this channel appends
        # its preview counters (pv / pv_hz) to the line.
        self._arrivals = ArrivalTracker(_STATS_LOG_PERIOD_S)

    def _load_spec_wire(self) -> Optional[bytes]:
        """Frame this node's installed kinematic_spec, or None when there is no
        local robot data — the node is the only source of the spec, so None means
        QUIC teleop will not start. Best-effort — never raises into start()."""
        kind = (self._robot_kind or "").strip()
        if not kind:
            return None
        try:
            from interlatent import robots
            spec = robots.load_kinematic_spec(kind)
        except Exception as exc:
            # Robot data ships in the SDK wheel for every kind, so this is not a
            # missing-extra problem: the kind is unknown to this SDK version (or
            # the node reports a robot_kind no robots dir matches).
            _LOG.warning(
                "teleop(quic) no local kinematic_spec for robot_kind=%r (%s) — "
                "QUIC teleop will not start for this node (no fallback source). "
                "This SDK ships no data for that kind; check --robot, or upgrade "
                "interlatent if the kind is newer than this install.", kind, exc,
            )
            return None
        try:
            wire = frame_spec_wire(spec, kind)
        except Exception as exc:
            _LOG.warning(
                "teleop(quic) failed to frame kinematic_spec for %r: %s", kind, exc
            )
            return None
        _LOG.info(
            "teleop(quic) serving local kinematic_spec for robot_kind=%r (%d bytes)",
            kind, len(wire),
        )
        return wire

    # -- lifecycle --
    def start(self) -> None:
        if self._thread is not None:
            return
        self._spec_wire = self._load_spec_wire()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", 0))
        sock.settimeout(_SUPERVISE_TICK_S)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, _quic_ipc.SOCK_BUF_BYTES)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, _quic_ipc.SOCK_BUF_BYTES)
        except OSError:
            pass
        self._sock = sock
        self._cookie = secrets.token_hex(16)
        try:
            self._spawn_child()
        except OSError as exc:
            # Supervisor retries with backoff; don't fail session start.
            _LOG.warning("teleop(quic) child spawn failed: %s", exc)
            self._next_spawn_at = time.monotonic() + self._spawn_backoff
        self._thread = threading.Thread(
            target=self._supervise,
            name=f"teleop-quic[{self._session_id[:8]}]",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._shutdown_child(self._proc)
        self._proc = None
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        # The supervisor can spawn between our shutdown above and _stop being
        # observed — after the join, sweep any straggler.
        self._shutdown_child(self._proc)
        self._proc = None
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        self._connected = False

    @staticmethod
    def _child_argv() -> list:
        # Seam for tests (monkeypatch to a fake child).
        return [sys.executable, "-m", "interlatent.node.teleop._quic_proc"]

    def _spawn_child(self) -> None:
        if self._stop.is_set() or self._sock is None:
            return
        env = {
            **os.environ,
            _quic_ipc.ENV_PARENT_PORT: str(self._sock.getsockname()[1]),
            _quic_ipc.ENV_COOKIE: self._cookie,
            _quic_ipc.ENV_API_BASE: self._api_base,
            _quic_ipc.ENV_API_KEY: self._api_key,
            _quic_ipc.ENV_SESSION_ID: self._session_id,
            _quic_ipc.ENV_TOKEN_PATH: self._token_path,
        }
        if self._bypass_key:
            env[_quic_ipc.ENV_BYPASS_KEY] = self._bypass_key
        # stdin PIPE is the lifetime tether (child exits on EOF); stderr is
        # inherited so child logs land in the node's logs.
        self._proc = subprocess.Popen(
            self._child_argv(),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=None,
            env=env,
        )
        _LOG.info(
            "teleop(quic) child spawned pid=%s session=%s",
            self._proc.pid, self._session_id,
        )

    @staticmethod
    def _shutdown_child(proc: Optional[subprocess.Popen]) -> None:
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.close()  # EOF → child exits
        except Exception:
            pass
        try:
            proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                proc.kill()
        except Exception:
            pass

    # -- supervisor (reader thread) --
    def _supervise(self) -> None:
        while not self._stop.is_set():
            now = time.monotonic()
            proc = self._proc
            if proc is not None and proc.poll() is not None:
                _LOG.warning(
                    "teleop(quic) child exited rc=%s; respawn in %.0fs session=%s",
                    proc.returncode, self._spawn_backoff, self._session_id,
                )
                self._child_exits += 1
                if (
                    not self._hello_ever
                    and self._child_exits >= 3
                    and not self._spawn_warned
                ):
                    self._spawn_warned = True
                    _LOG.warning(
                        "teleop(quic) child has never come up — check that %s "
                        "can import interlatent and that aioquic is installed "
                        "(pip install 'interlatent[teleop-quic]')",
                        sys.executable,
                    )
                self._proc = None
                self._child_addr = None
                self._connected = False
                self._drop_frame()
                self._next_spawn_at = now + self._spawn_backoff
                self._spawn_backoff = min(self._spawn_backoff * 2, _RECONNECT_MAX_S)
            if self._proc is None and now >= self._next_spawn_at:
                try:
                    self._spawn_child()
                except OSError as exc:
                    _LOG.warning(
                        "teleop(quic) child spawn failed: %s (retry %.0fs)",
                        exc, self._spawn_backoff,
                    )
                    self._next_spawn_at = time.monotonic() + self._spawn_backoff
                    self._spawn_backoff = min(
                        self._spawn_backoff * 2, _RECONNECT_MAX_S
                    )
            sock = self._sock
            if sock is None:
                return
            try:
                data, addr = sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                return  # socket closed under us during stop()
            self._handle_datagram(data, addr)

    def _handle_datagram(self, data: bytes, addr) -> None:
        parsed = _quic_ipc.parse(data)
        if parsed is None:
            return
        kind, payload = parsed
        if kind == _quic_ipc.TYPE_CTRL:
            msg = _quic_ipc.parse_ctrl(payload)
            if msg is None:
                return
            t = msg.get("t")
            if t == "hello":
                if msg.get("cookie") != self._cookie:
                    return  # stray local sender — ignore
                self._child_addr = addr
                self._hello_ever = True
                self._spawn_backoff = _RECONNECT_INITIAL_S
                return
            if addr != self._child_addr:
                return
            if t == "connected":
                self._dedup.reset()
                self._connected = True
                # New session (or reconnect) → let the spec handshake log once
                # more, so each browser attach is observable in the node log.
                self._spec_served = False
                self._spec_miss_warned = False
                # A fresh connection means a fresh child governor: its
                # cumulative vstats counters restarted; re-baseline.
                self._vstats_last = None
                _LOG.info("teleop(quic) connected session=%s", self._session_id)
            elif t == "disconnected":
                # Drop the latest frame so a stale engaged target can't keep
                # driving the arm across a reconnect.
                self._connected = False
                self._vstats_last = None
                self._drop_frame()
            elif t == "vstats":
                self._on_vstats(msg)
            return
        if kind != _quic_ipc.TYPE_DATA or addr != self._child_addr:
            return
        # A browser control datagram, not a target frame: answer its request
        # for our kinematic_spec on a uni stream (via the child). Checked before
        # frame decode; the substring guard keeps the target hot path parse-free.
        if is_request_spec(payload):
            self._maybe_send_spec(addr)
            return
        frame = decode_target_datagram(payload)
        if frame is None:
            return
        # Presence is stamped before dedupe: the overlay's disengaged
        # keepalives arrive duplicated, and dupes must still count as a
        # viewer being connected (matches the WS channel's semantics).
        self._last_rx_at = time.monotonic()
        # E-stop latches BEFORE dedupe: a duplicated/late estop datagram must
        # still stop the robot even when its seq loses the latest-wins race.
        if frame.estop:
            self._latch_estop()
        if not self._dedup.accept(frame.seq):
            return
        self._note_arrival(frame)
        self._store_frame(frame)

    def _maybe_send_spec(self, addr) -> None:
        """Hand the framed kinematic_spec to the child (TYPE_SPEC) to ship on a
        uni stream to the browser, answering its ``request_spec``. Throttled: the
        browser retries until it sees the spec, so we cap how often a retry burst
        can open streams. Logs the handshake once per session (both the serve and
        the no-local-spec case), so on-hardware you can see whether the browser
        built its solver from this node or fell back to the platform."""
        if not self._connected:
            return
        wire = self._spec_wire
        if wire is None:
            if not self._spec_miss_warned:
                self._spec_miss_warned = True
                _LOG.warning(
                    "teleop(quic) browser requested kinematic_spec but this node "
                    "has none for robot_kind=%r — QUIC teleop WILL NOT START "
                    "(there is no fallback source). Install this robot's data: "
                    "pip install 'interlatent[%s]'  session=%s",
                    self._robot_kind, self._robot_kind or "<kind>", self._session_id,
                )
            return
        now = time.monotonic()
        if now - self._last_spec_sent_at < _SPEC_SEND_MIN_INTERVAL_S:
            return
        sock = self._sock
        if sock is None:
            return
        self._last_spec_sent_at = now
        try:
            sock.sendto(_quic_ipc.encode_spec(wire), addr)
        except OSError:
            return
        if not self._spec_served:
            self._spec_served = True
            _LOG.info(
                "teleop(quic) served kinematic_spec/ik_hints to browser "
                "(%d bytes) robot_kind=%r session=%s",
                len(wire), self._robot_kind, self._session_id,
            )

    # -- read API (control loop) --
    # latest_frame() and consume_estop() are inherited from LatestFrameStore
    # (identical staleness + sticky-estop semantics across transports, ADR 0016).

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
        sock = self._sock
        addr = self._child_addr
        if sock is None or addr is None or not self._connected:
            return
        self._out_seq += 1
        data = _quic_ipc.encode_data(
            encode_state_datagram(qpos, self._out_seq, self._last_applied_seq)
        )
        # Loopback sendto is effectively non-blocking; concurrent recvfrom on
        # the same socket from the supervisor thread is safe.
        for _ in range(_STATE_DUP):
            try:
                sock.sendto(data, addr)
            except OSError:
                pass

    def note_applied(self, seq: int) -> None:
        """Record the target seq the control loop just executed (for the
        browser's RTT echo). Called from the control-loop thread; a plain int
        write is fine under the GIL."""
        self._last_applied_seq = int(seq)

    def _on_vstats(self, msg: dict) -> None:
        """One ~1s vstats window from the child → preview backoff policy.

        Counters are cumulative per child QUIC connection; diff against
        the last sample and clamp negative deltas (a reconnect restarts
        the child's governor; connected/disconnected also re-baseline).
        Runs on the supervisor thread; the multiplier is a plain float
        read by the control-loop thread under the GIL (same precedent
        as _last_applied_seq). Backoff steps log at INFO; recovery logs
        once on reaching the configured rate; decay steps are silent.
        """
        if not self._preview_adaptive or not self._video_enabled:
            return
        try:
            drop_cap = int(msg.get("drop_cap", 0))
            reset_ttl = int(msg.get("reset_ttl", 0))
        except (TypeError, ValueError):
            return
        last = self._vstats_last
        self._vstats_last = (drop_cap, reset_ttl)
        if last is None:
            return  # first sample of this connection — baseline only
        d_cap = max(0, drop_cap - last[0])
        # Only TTL resets (frames gone stale in flight) drive backoff;
        # d_cap is the in-flight cap pacing the timer — free, expected,
        # not congestion (see PreviewBackoff).
        stale = max(0, reset_ttl - last[1])
        before = self._preview_backoff.mult
        self._preview_backoff.on_window(stale, self._preview_period_s)
        after = self._preview_backoff.mult
        if after > before:
            _LOG.info(
                "teleop(quic) preview backing off to %.1f Hz "
                "(stale frames %d/1s, cap-paced %d/1s) session=%s",
                1.0 / self._effective_preview_period_s(), stale, d_cap,
                self._session_id,
            )
        elif after < before and after == 1.0:
            _LOG.info(
                "teleop(quic) preview recovered to %.1f Hz session=%s",
                1.0 / self._preview_period_s, self._session_id,
            )

    def _effective_preview_period_s(self) -> float:
        """The configured period scaled by the congestion backoff. When
        adaptive is off, ``_on_vstats`` never runs so the multiplier stays 1.0
        and this is just the base period — no separate branch needed."""
        return self._preview_backoff.period(self._preview_period_s)

    def preview_due(self) -> bool:
        """True when the control loop should encode + hand over a preview set.

        Gates (cheapest first): kill switch, link up, viewer present within
        the last few seconds (so idle sessions pay zero encode cost), preview
        deadline reached. Unlike the WS channel there is no send-slot check —
        the hand-off to the child is a non-blocking loopback sendto and load
        shedding (in-flight cap/TTL) lives in the child, which is the only
        side that can see stream completion.

        The deadline is credit-based (see _advance_preview_deadline): this
        method is only sampled once per control tick, and a naive
        "period elapsed since last send" check beats against the tick rate —
        46 ms ticks vs a 50 ms period quantizes to every-other-tick and caps
        delivered fps at half the configured rate."""
        if not self._video_enabled:
            return False
        if not self._connected or self._child_addr is None:
            return False
        now = time.monotonic()
        if now - self._last_rx_at > _VIEWER_PRESENCE_S:
            return False
        return now >= self._next_preview_due

    def send_preview(self, jpegs: Dict[str, bytes], ts_ns: int) -> None:
        """Hand one preview set (cam → JPEG bytes) to the child, one
        TYPE_VIDEO loopback datagram per camera. Each frame is already framed
        here with the browser-facing header — ``uint16-BE header length +
        JSON {"type","cam","seq","ts_ms"} + JPEG`` — byte-identical to the WS
        pod relay's video framing, so the overlay's parser is shared. The
        child ships each frame on its own unidirectional stream (no
        duplication: streams are reliable). Non-blocking, best-effort."""
        if not jpegs:
            return
        sock = self._sock
        addr = self._child_addr
        if sock is None or addr is None or not self._connected:
            return
        self._next_preview_due = _advance_preview_deadline(
            self._next_preview_due, time.monotonic(),
            self._effective_preview_period_s(),
        )
        self._preview_seq += 1
        ts_ms = int(ts_ns) // 1_000_000
        for cam, jpeg in jpegs.items():
            if not jpeg:
                continue
            wire = frame_with_header(
                {"type": "video", "cam": cam,
                 "seq": self._preview_seq, "ts_ms": ts_ms},
                jpeg,
            )
            try:
                sock.sendto(_quic_ipc.encode_video(cam, wire), addr)
            except OSError:
                continue
            self._pv_window += 1
        if not self._pv_logged_once and self._pv_window > 0:
            self._pv_logged_once = True
            _LOG.info(
                "teleop(quic) preview tee active: %d cam(s), period=%.0fms "
                "session=%s",
                len(jpegs), self._preview_period_s * 1000.0, self._session_id,
            )

    def _note_arrival(self, frame: TeleopFrame) -> None:
        """Track inter-arrival gaps of target datagrams; log a 5s summary.
        Supervisor thread only. A low rate or a large max gap means targets
        are stalling between the browser and this node (relay/network jitter).
        ``seq_span`` vs ``n`` ≈ dedupe drops + datagrams lost en route (dup×3,
        so span > n is normal)."""
        summary = self._arrivals.note(frame)
        if summary is None:
            return
        _LOG.info(
            "teleop(quic) target datagrams (%.0fs): n=%d rate=%.1fHz "
            "gap mean/max=%.0f/%.0fms seq_span=%d pv=%d pv_hz=%.1f "
            "session=%s",
            summary["elapsed"], summary["n"], summary["rate_hz"],
            summary["gap_mean_ms"], summary["gap_max_ms"], summary["seq_span"],
            self._pv_window, 1.0 / self._effective_preview_period_s(),
            self._session_id,
        )
        self._pv_window = 0


__all__ = [
    "QuicTeleopChannel",
    "LatestSeqBuffer",
    "PreviewBackoff",
    "encode_state_datagram",
    "decode_target_datagram",
    "frame_spec_wire",
    "is_request_spec",
]
