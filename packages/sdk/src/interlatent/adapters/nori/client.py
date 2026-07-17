"""Nori session client: the TCP NDJSON link to the on-Pi daemon.

Owns the socket plus two background threads:

- **reader** — parses inbound frames (telemetry/error/action_status) into
  latest-wins caches, latches fatal errors, and drives reconnect with backoff
  when the link drops.
- **keep-alive pump** — the daemon has no heartbeat message; the control-frame
  stream IS its watchdog feed, and silence beyond ``t_stop_ms`` safe-stops the
  robot. The pump sends motion-free ``{"type":"control","seq":N}`` frames at
  ``pump_hz``, but ONLY while the control loop proves liveness by calling
  ``note_liveness()`` (via ``get_observation``) within roughly the daemon's
  ``t_warn_ms``. A stalled/wedged loop stops the pump and the daemon safe-stops
  exactly as its watchdog intends — an unconditional pump would defeat that
  safety feature and is forbidden. See ADR 0015 and CONTEXT.md
  "Keep-alive pump (Nori)".

All motion safety is enforced daemon-side (clamping, e-stop latch, watchdog);
this client discloses daemon state, it never re-enforces it.
"""
from __future__ import annotations

import logging
import socket
import threading
import time
from typing import Any, Dict, Optional

from . import protocol as _p
from .config import NoriAdapterConfig, resolve_token

_logger = logging.getLogger(__name__)

# Ceiling on the pump's liveness window: even if the daemon discloses a huge
# t_warn_ms, a loop silent this long is not "alive with jitter" any more.
_LIVENESS_CAP_S = 0.5
# Socket write timeout — a wedged send must not hang estop/pump threads.
_SEND_TIMEOUT_S = 1.0
# Reader recv granularity; doubles as the stop-flag poll interval.
_RECV_TIMEOUT_S = 0.5


class NoriSessionClient:
    """One handshaked control session to the Nori daemon (single-client slot)."""

    def __init__(self, cfg: NoriAdapterConfig, profile: Any) -> None:
        self._cfg = cfg
        self._profile = profile
        self._sock: Optional[socket.socket] = None
        self._send_lock = threading.Lock()
        self._seq_lock = threading.Lock()
        self._stop = threading.Event()
        self._reader_thread: Optional[threading.Thread] = None
        self._pump_thread: Optional[threading.Thread] = None

        self._seq = 0  # shared monotonic control-frame counter
        self._connected = False
        self._session_dead = False
        self._dead_reason = ""
        self._disconnected_at: Optional[float] = None

        self._last_liveness = 0.0  # monotonic ts of the loop's last obs call
        self._last_sent_at = 0.0  # monotonic ts of the last outbound control frame

        self._state_lock = threading.Lock()
        self._state: Dict[str, float] = {}
        self._state_at = 0.0  # monotonic arrival ts of the newest telemetry
        self._status: Optional[Dict[str, Any]] = None
        self._fatal: Optional[_p.ErrorFrame] = None

        self._ack: Optional[_p.Ack] = None

    # ------------------------------------------------------------------ #
    # Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def connect(self) -> _p.Ack:
        """Dial, handshake, fail-closed validate, then start reader + pump."""
        ack = self._handshake()
        self._reader_thread = threading.Thread(
            target=self._reader_run, name="nori-reader", daemon=True
        )
        self._pump_thread = threading.Thread(
            target=self._pump_run, name="nori-pump", daemon=True
        )
        self._reader_thread.start()
        self._pump_thread.start()
        return ack

    def _handshake(self) -> _p.Ack:
        sock = socket.create_connection(
            (self._cfg.host, self._cfg.port), timeout=self._cfg.connect_timeout_s
        )
        sock.settimeout(self._cfg.connect_timeout_s)
        self._recv_buf = b""  # drop any leftovers from a dead socket
        try:
            hello = _p.make_hello(
                token=resolve_token(self._cfg), bus_choice=self._cfg.bus_choice
            )
            sock.sendall(_p.encode_frame(hello))
            try:
                first = self._read_line(sock)
            except TimeoutError:
                # TCP connected but the hello was never answered: the kernel
                # parked us in the listen backlog while the daemon serves its
                # one client. Name the real problem instead of "timed out".
                raise _p.NoriHandshakeError(
                    "TCP connected but the daemon sent no ack within "
                    f"{self._cfg.connect_timeout_s:.0f}s — it serves ONE client "
                    "at a time and the slot is likely held by another client "
                    "(teleop bridge webrtc_robot.py? a stale session?). Check "
                    "`ss -tnp | grep 7777` on the robot for the ESTABLISHED "
                    "peer, and stop it before starting an interlatent session."
                ) from None
            ack = _p.parse_line(first) if first is not None else None
            if not isinstance(ack, _p.Ack):
                raise _p.NoriHandshakeError(
                    f"first daemon frame was not an ack: {first!r}"
                )
            if not ack.accepted:
                raise _p.NoriHandshakeError(
                    f"daemon rejected hello: {ack.error or '<no error text>'}"
                )
            problems = _p.validate_ack(self._profile, ack)
            if problems:
                raise _p.NoriHandshakeError(
                    "descriptor/profile mismatch (fail closed; "
                    f"{len(problems)} problem(s)):\n  - " + "\n  - ".join(problems)
                )
            if not ack.joints:
                _logger.warning(
                    "Nori ack carried no descriptor block — topology validated "
                    "against initial_state and ranges pinned by norm_mode=%s; "
                    "cameras CANNOT be discovered until the daemon discloses "
                    "descriptor.cameras (state-only observations).",
                    ack.norm_mode,
                )
        except Exception:
            try:
                sock.close()
            finally:
                raise

        sock.settimeout(_RECV_TIMEOUT_S)
        with self._state_lock:
            # Seed from initial_state so the first observation predates the
            # first telemetry frame.
            self._state.update(ack.initial_state)
            self._state_at = time.monotonic()
        self._ack = ack
        self._sock = sock
        self._connected = True
        self._disconnected_at = None
        _logger.info(
            "Nori session up (%s:%d): %d joints%s, cameras=%s, "
            "watchdog warn/stop=%s/%s ms",
            self._cfg.host, self._cfg.port, len(self._profile.joint_names),
            "" if ack.joints else " (descriptorless ack)", list(ack.cameras),
            ack.watchdog.t_warn_ms if ack.watchdog else "?",
            ack.watchdog.t_stop_ms if ack.watchdog else "?",
        )
        return ack

    def close(self) -> None:
        """Best-effort bye, stop both threads, close the socket. Idempotent.
        The pump cannot outlive this call (joined below)."""
        if self._stop.is_set():
            return
        self._stop.set()
        try:
            self._send_frame(_p.make_bye(), count_as_control=False)
        except Exception:  # noqa: BLE001 - teardown is best-effort
            pass
        for t in (self._pump_thread, self._reader_thread):
            if t is not None and t.is_alive():
                t.join(timeout=2.0)
        self._teardown_socket()
        self._connected = False

    # ------------------------------------------------------------------ #
    # Outbound                                                            #
    # ------------------------------------------------------------------ #

    def send_action(self, action: Dict[str, float]) -> None:
        """One absolute-target control frame. Non-blocking beyond the socket
        write; latest-wins daemon-side (the daemon latches the newest action)."""
        with self._seq_lock:
            self._seq += 1
            frame = _p.make_control_action(self._seq, action)
        self._send_frame(frame)

    def send_estop(self) -> None:
        self._send_frame(_p.make_estop(), count_as_control=False)

    def send_reset_latch(self, token: str) -> None:
        self._send_frame(_p.make_reset_latch(token), count_as_control=False)

    def _send_frame(self, frame: dict, *, count_as_control: bool = True) -> None:
        sock = self._sock
        if sock is None or not self._connected:
            raise _p.NoriProtocolError(
                f"cannot send {frame.get('type')!r}: session not connected"
            )
        data = _p.encode_frame(frame)
        with self._send_lock:
            try:
                sock.settimeout(_SEND_TIMEOUT_S)
                sock.sendall(data)
                sock.settimeout(_RECV_TIMEOUT_S)
            except OSError as exc:
                self._mark_disconnected(f"send failed: {exc}")
                raise _p.NoriProtocolError(f"send failed: {exc}") from exc
        if count_as_control and frame.get("type") == "control":
            self._last_sent_at = time.monotonic()

    # ------------------------------------------------------------------ #
    # Liveness-tied keep-alive pump (ADR 0015)                            #
    # ------------------------------------------------------------------ #

    def note_liveness(self) -> None:
        """The control loop's proof of life; called by ``get_observation()``."""
        self._last_liveness = time.monotonic()

    def _liveness_window_s(self) -> float:
        t_warn_s = 0.15
        if self._ack is not None and self._ack.watchdog is not None:
            t_warn_s = self._ack.watchdog.t_warn_ms / 1000.0
        # At least two pump periods so a tiny disclosed t_warn can't make the
        # pump flap against its own cadence; capped so a generous t_warn never
        # keeps a wedged loop's robot armed.
        floor = 2.0 / max(self._cfg.pump_hz, 1.0)
        return max(min(t_warn_s, _LIVENESS_CAP_S), floor)

    def _pump_run(self) -> None:
        period = 1.0 / max(self._cfg.pump_hz, 1.0)
        while not self._stop.wait(period):
            if not self._connected:
                continue  # never pump while disconnected/reconnecting
            now = time.monotonic()
            if now - self._last_liveness > self._liveness_window_s():
                # Loop stalled: stop feeding the daemon watchdog so it
                # safe-stops. This silence is the safety feature, not a bug.
                continue
            if now - self._last_sent_at < period:
                continue  # real control frames are already the heartbeat
            with self._seq_lock:
                self._seq += 1
                frame = _p.make_keepalive(self._seq)
            try:
                self._send_frame(frame)
            except _p.NoriProtocolError:
                continue  # reader owns reconnect; pump just goes quiet

    # ------------------------------------------------------------------ #
    # Inbound caches                                                      #
    # ------------------------------------------------------------------ #

    def latest_state(self) -> tuple[Dict[str, float], float]:
        """(joint-state dict, age of the newest telemetry in ms)."""
        with self._state_lock:
            age_ms = (time.monotonic() - self._state_at) * 1000.0
            return dict(self._state), age_ms

    def latest_status(self) -> Optional[Dict[str, Any]]:
        with self._state_lock:
            return dict(self._status) if self._status is not None else None

    def take_fatal(self) -> Optional[_p.ErrorFrame]:
        return self._fatal

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def session_dead(self) -> bool:
        return self._session_dead

    @property
    def dead_reason(self) -> str:
        return self._dead_reason

    @property
    def watchdog(self) -> Optional[_p.WatchdogProfile]:
        return self._ack.watchdog if self._ack is not None else None

    @property
    def descriptor_cameras(self) -> tuple[str, ...]:
        return self._ack.cameras if self._ack is not None else ()

    # ------------------------------------------------------------------ #
    # Reader + reconnect                                                  #
    # ------------------------------------------------------------------ #

    def _read_line(self, sock: socket.socket) -> Optional[bytes]:
        """Read one LF-terminated line, buffering across recv calls."""
        buf = getattr(self, "_recv_buf", b"")
        while b"\n" not in buf:
            chunk = sock.recv(4096)
            if not chunk:
                self._recv_buf = b""
                return None  # EOF
            buf += chunk
        line, _, rest = buf.partition(b"\n")
        self._recv_buf = rest
        return line

    def _reader_run(self) -> None:
        while not self._stop.is_set():
            sock = self._sock
            if sock is None or not self._connected:
                if not self._try_reconnect():
                    return  # session dead or stopping
                continue
            try:
                line = self._read_line(sock)
            except socket.timeout:
                continue
            except OSError as exc:
                if self._stop.is_set():
                    return  # clean teardown: our own close raced the recv
                self._mark_disconnected(f"recv failed: {exc}")
                continue
            if line is None:
                if self._stop.is_set():
                    return  # clean teardown: EOF after our bye is expected
                self._mark_disconnected("daemon closed the connection")
                continue
            self._handle_frame(_p.parse_line(line))

    def _handle_frame(self, frame: Optional[_p.InboundFrame]) -> None:
        if isinstance(frame, _p.Telemetry):
            with self._state_lock:
                self._state.update(frame.state)
                self._state_at = time.monotonic()
                if frame.status is not None:
                    self._status = frame.status
        elif isinstance(frame, _p.ErrorFrame):
            if frame.fatal:
                self._fatal = frame
                self._mark_dead(f"fatal daemon error {frame.code}: {frame.msg}")
            else:
                _logger.warning("Nori daemon error %s: %s", frame.code, frame.msg)
        # Ack outside handshake and ActionStatus are informational; unknown -> None.

    def _mark_disconnected(self, reason: str) -> None:
        if self._connected:
            _logger.warning("Nori session link lost: %s", reason)
        self._connected = False
        if self._disconnected_at is None:
            self._disconnected_at = time.monotonic()
        self._teardown_socket()

    def _mark_dead(self, reason: str) -> None:
        self._session_dead = True
        self._dead_reason = reason
        self._connected = False
        _logger.error("Nori session dead: %s", reason)

    def _teardown_socket(self) -> None:
        sock, self._sock = self._sock, None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def _try_reconnect(self) -> bool:
        """Backoff + full re-handshake (incl. fail-closed validation).

        Returns False when the session is over (dead/stopping); True when the
        reader should continue (either reconnected or still inside the window).
        """
        if self._session_dead:
            return False
        started = self._disconnected_at or time.monotonic()
        backoff = self._cfg.reconnect_backoff_s
        while not self._stop.is_set():
            if time.monotonic() - started > self._cfg.reconnect_window_s:
                self._mark_dead(
                    f"link down longer than reconnect_window_s="
                    f"{self._cfg.reconnect_window_s}"
                )
                return False
            if self._stop.wait(backoff):
                return False
            backoff = min(backoff * 2, self._cfg.max_backoff_s)
            try:
                self._handshake()
                _logger.info("Nori session reconnected")
                return True
            except _p.NoriHandshakeError as exc:
                # The daemon is up but rejected/mismatched us — not transient.
                self._mark_dead(f"re-handshake rejected: {exc}")
                return False
            except OSError:
                continue  # daemon still down; keep backing off
        return False


__all__ = ["NoriSessionClient"]
