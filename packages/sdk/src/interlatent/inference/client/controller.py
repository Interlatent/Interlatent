"""DRTC client controller.

Owns the lifecycle of a single inference session against the
Modal-hosted server. Composes the parts:

    [robot loop] -- step()/observe() --> [controller]
                                              |
                                              | -- spans + obs --> ObservationSender --> gRPC
                                              |
                                              | <-- chunks ------- ActionReceiver  <-- gRPC
                                              |
                                              v
                                        ActionSchedule (LWW)
                                              |
                                              v
                                   pop_next() — one action per step

Usage (sketch):

    # ``model_id`` here is the DRTC protocol field — the SDK passes the
    # backend env slug through it (the wire contract with Modal is out
    # of scope for the SDK model_id retirement).
    cfg = DRTCConfig(server_address="https://...modal.run", model_id="smolvla-x")
    client = DRTCClient(cfg)
    client.open()
    while running:
        obs = capture()
        action = client.step(obs.tobytes())
        robot.apply(action)
    client.close()

`step()` returns the next due action vector (or None if the queue is
empty). The controller decides when to send a new observation based
on queue depth, cooldown, and the estimated execution horizon.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Union

import numpy as np

from ..protocol import messages_pb2 as pb
from ..protocol import messages_pb2_grpc as pb_grpc
from ..protocol.timestamps import ControlClock
from .cooldown import Cooldown
from .latency import JacobsonKarels
from .merge import ActionSchedule
from .receiver import ActionReceiver
from .sender import ObservationSender, PendingObservation
from .spool import TickSpool

log = logging.getLogger(__name__)


# The recorder queue carries WAKEUP TOKENS, not tick data (ADR 0023):
# ticks are journaled to the disk spool on capture and the sender reads
# batches from the spool, deleting only on the server's ack. Under an
# uplink deficit the backlog therefore grows on DISK (bounded by the
# spool cap, which hard-stops capture at the limit), not in RAM, and
# survives a process crash. Unbounded is fine for tokens.
_RECORDER_QUEUE_MAXSIZE = 0


def _rec_pace_bytes_per_s() -> float:
    """Recording-uplink budget from INTERLATENT_REC_MAX_KBPS (KiB/s).

    Defaults to 8000 KiB/s; set 0 to disable pacing. When set, the drain
    thread sleeps after each
    batch so recording never offers more than this to the uplink —
    protecting the teleop path (state heartbeat, pose targets, live
    previews) that shares the same physical link from bufferbloat behind
    recording bursts. If the budget is below the capture bitrate the
    backlog banks in the DISK spool (ADR 0023) and ships during the
    close drain, which runs unpaced — recording stays complete; disk
    and close-drain time are the cost.
    """
    import os

    try:
        kbps = float(os.environ.get("INTERLATENT_REC_MAX_KBPS", "") or 8000.0)
    except (TypeError, ValueError):
        kbps = 8000.0
    return max(0.0, kbps) * 1024.0


# Conservative floor for sizing the close-drain ceiling: just under the
# ~300 KiB/s session cap the low-bandwidth recipe recommends, so any
# link that sustained the session at all drains faster than this and the
# computed ceiling only ever over-budgets. The 12s ack-progress stall
# detector (not this ceiling) remains the dead-link escape.
_REC_DRAIN_ASSUMED_MIN_BPS = 250 * 1024


def _rec_drain_ceiling_s(pending_bytes: int) -> float:
    """Close-drain hard ceiling, scaled to the backlog actually banked.

    A paced session on a deficit uplink can bank GBs in the spool; a
    fixed ceiling would guillotine the drain mid-flight and retain a
    tail that — for a completed session — is never re-drained and is
    GC'd after retention. Scale the ceiling by the pending bytes at an
    assumed worst-case drain rate instead; ``max(base, ...)`` preserves
    the historical 600s floor. INTERLATENT_REC_DRAIN_CEILING_S (float
    seconds > 0) forces a fixed value. A 3 GB backlog yields a ~3.4h
    ceiling — long, but the alternative is deleting recorded data, and
    close is off the control-critical path.
    """
    import os

    try:
        forced = float(os.environ.get("INTERLATENT_REC_DRAIN_CEILING_S", "") or 0.0)
    except (TypeError, ValueError):
        forced = 0.0
    if forced > 0.0:
        return forced
    return max(
        _REC_DRAIN_CEILING_S,
        float(max(0, pending_bytes)) / _REC_DRAIN_ASSUMED_MIN_BPS,
    )


def _drain_ceiling_logged(pending_bytes: int) -> float:
    """The drain ceiling for ``pending_bytes``, announcing it when a banked
    backlog scales it past the 600s floor.

    Split out from ``_drain_recorder`` so the log decision is a pure function
    of the pending bytes, testable without racing the background sender.
    """
    ceiling_s = _rec_drain_ceiling_s(pending_bytes)
    if ceiling_s > _REC_DRAIN_CEILING_S:
        log.info(
            "DRTC recorder drain: %.0f MB banked in the spool — "
            "ceiling scaled to %.0fs (assumes >= %d KiB/s)",
            pending_bytes / 1e6, ceiling_s,
            _REC_DRAIN_ASSUMED_MIN_BPS // 1024,
        )
    return ceiling_s

# Cap the wire size of one batched RecordTicks RPC. Two constraints:
# the server's default gRPC receive limit is 4 MiB (hard ceiling), and —
# the binding one — each batch is a single head-of-line burst on the
# node's uplink. During teleop the same physical link carries the ~15 Hz
# state heartbeat, pose targets, and live video previews; a 2 MiB write
# monopolizes a 10-20 Mbit/s uplink for 1-2 s and bufferbloats all of
# them (the ready/stale flip-flop). 256 KiB keeps any single burst under
# ~200 ms while still amortizing the RTT well. A batch is flushed early
# once the accumulated JPEG bytes would cross this line, regardless of
# the tick-count cap.
_REC_BATCH_MAX_BYTES = 256 * 1024

# Per-RPC RecordTicks timeout. The server enqueue is non-blocking, so the
# call time is dominated by uploading the batch's JPEG bytes over the
# link; 30s comfortably covers a ~2 MiB batch even on a slow uplink.
_REC_BATCH_TIMEOUT_S = 30.0

# Close-path drain bounds. Recording is off the control-critical path at
# close (the robot has already disconnected), so we drain the whole backlog
# rather than dropping the tail. We only give up if the sender stops
# shipping for _REC_DRAIN_STALL_S (a dead link — set above the 10s per-RPC
# RecordTick timeout so a merely-slow link is not mistaken for a dead one),
# or a hard ceiling elapses as an ultimate backstop. The ceiling base
# covers ~2 GB at 3.5 MB/s; when the spool has banked more than that
# would drain, _rec_drain_ceiling_s scales it up with the pending bytes
# so a low-capped long session never has its tail guillotined.
_REC_DRAIN_STALL_S = 12.0
_REC_DRAIN_CEILING_S = 600.0


@dataclass
class DRTCConfig:
    server_address: str                   # "host:port" for plain gRPC, or full URL for gRPC-Web
    # DRTC wire protocol field — kept as ``model_id`` for backward
    # compatibility with the protobuf schema. The SDK passes the
    # backend environment slug through here. Out of scope for the
    # SDK model_id retirement.
    model_id: str
    api_key: str = ""                     # Interlatent API key (ilat_...); sent as Bearer auth
    policy_uri: str = ""
    policy_backend: str = ""              # server backend name; "" -> "echo"
    chunk_size: int = 50                  # SmolVLA's native chunk; bigger = more jitter headroom
    action_dim: int = 6                   # hint to backend
    min_execution_horizon: int = 12        # prefetch margin above the latency estimate (steps)
    control_period_s: float = 1 / 30       # default 30Hz
    cooldown_steps: int = 16
    payload_codec: str = "raw_f32"
    use_grpc_web: bool = False             # set True when talking to Modal asgi
    metadata: dict[str, str] = field(default_factory=dict)
    stats_interval_s: float = 5.0          # period of the DRTC telemetry log line; 0 disables
    rec_batch_max_ticks: int = 16          # coalesce up to N queued ticks per RecordTicks RPC
    synchronous: bool = False              # sequential (request-response) chunking: one fully-drained chunk per observation, no overlap



class DRTCClient:
    """Synchronous facade around the async DRTC machinery.

    The robot loop is almost always synchronous, so we run gRPC on a
    dedicated background event loop and bridge step() through queues.
    """

    def __init__(self, cfg: DRTCConfig) -> None:
        self.cfg = cfg
        self.clock = ControlClock()
        self.schedule = ActionSchedule()
        self.latency = JacobsonKarels()
        self.cooldown = Cooldown(epsilon=2)
        self._sent_at: dict[int, float] = {}
        self.session_id: Optional[str] = None
        self.action_dim: Optional[int] = None

        self._channel = None
        self._stub: Optional[pb_grpc.InferenceServiceStub] = None
        self._sender: Optional[ObservationSender] = None
        self._receiver: Optional[ActionReceiver] = None
        self._poller: Optional[threading.Thread] = None
        self._stop = threading.Event()
        # Sequential (--synchronous) chunking gate: set when a chunk request is
        # outstanding, cleared when the chunk — or a failed Infer — returns. Lets
        # step() fire exactly one observation per fully-drained schedule with no
        # overlap. Set by the control-loop thread, cleared by the sender thread;
        # an Event gives the cross-thread visibility a plain bool would not. Unused
        # in async mode.
        self._in_flight = threading.Event()
        # close() can be driven from two threads: the control-loop runner's
        # finally AND the node daemon's stop path (which force-closes when a
        # robot teardown wedges). Guard the body so exactly one caller runs
        # it — CloseSession/upload must fire once, not race.
        self._closed = False
        self._close_lock = threading.Lock()
        self._auth_metadata: tuple[tuple[str, str], ...] = (
            (("x-api-key", cfg.api_key),) if cfg.api_key else ()
        )

        # --- telemetry --------------------------------------------------
        # Per-window counters (reset each time stats() is sampled) plus a
        # couple of persistent values. Plain ints — under CPython's GIL
        # the increments are good enough for telemetry; no lock needed.
        self._stats_thread: Optional[threading.Thread] = None
        self._stat_t0 = time.monotonic()
        self._stat_steps = 0          # step() calls this window
        self._stat_none = 0           # step() calls that returned None (starved)
        self._stat_wait = 0           # step() None-returns that are expected sync-mode holds (not starvation)
        self._stat_infer = 0          # observations sent this window
        self._stat_chunk = 0          # action chunks received this window
        self._stat_qmin = 1 << 30     # min queue depth seen this window
        self._stat_jitter_sum = 0.0   # sum of |Δaction| between consecutive steps
        self._stat_jitter_n = 0
        self._last_action: Optional[np.ndarray] = None
        self._last_rtt_s = 0.0        # most recent measured Infer round-trip
        self._last_compute_s = 0.0    # server-reported compute time of that Infer
        self._stat_rec_sent = 0       # RecordTicks shipped this window
        self._stat_rec_bytes = 0      # JPEG bytes shipped this window

        # --- per-tick recorder pipeline (ADR 0023) ----------------------
        # Write-through spool: the hot path journals each tick to disk
        # (TickSpool, created at open() once the session id is known) and
        # drops a wakeup token into ``_rec_q``. The background sender
        # reads batches FROM THE SPOOL and deletes a tick only after the
        # server's honest accepted-prefix ack — delete-after-ack. The
        # queue therefore carries signals, not data: ``1`` = new tick
        # journaled, ``None`` = close sentinel.
        self._rec_q: "queue.Queue[Optional[int]]" = queue.Queue(
            maxsize=_RECORDER_QUEUE_MAXSIZE,
        )
        self._rec_thread: Optional[threading.Thread] = None
        self._spool: Optional[TickSpool] = None
        # Set by _drain_recorder when it gives up on a dead link: the
        # sender exits and the un-acked tail STAYS ON DISK (retained, not
        # lost — see _log_recording_summary / spool.gc_orphans).
        self._rec_abandon = threading.Event()
        # Consecutive-failure backoff for the sender (reset on success).
        self._rec_backoff_s = 0.0
        # Accounting:
        #   _rec_captured        — ticks journaled to the spool
        #   _rec_refused         — ticks refused at capture (spool full /
        #                          disk error) — the hard-stop counter
        #   _rec_sent            — ticks the server durably accepted
        #   _rec_unsent_retained — un-acked ticks left on disk at close
        # Invariant on a clean close:
        #   captured == sent + unsent_retained
        self._rec_captured = 0
        self._rec_refused = 0
        self._rec_unsent_retained = 0
        self._rec_sent = 0
        self._rec_refused_logged = False
        # Set once if the server 404s RecordTicks (an older node that only
        # speaks unary RecordTick); after that the drain ships tick-by-tick.
        self._rec_batch_unsupported = False
        # Optional recording-uplink budget (bytes/s; 0 = unpaced). See
        # _rec_pace_bytes_per_s. Read once — it's a deployment knob.
        self._rec_pace_bps = _rec_pace_bytes_per_s()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        import grpc

        if self.cfg.use_grpc_web:
            # gRPC-Web path goes through a sonora client; we keep the
            # import lazy so plain-gRPC users don't need it installed.
            from sonora.client import insecure_web_channel  # type: ignore
            self._channel = insecure_web_channel(self.cfg.server_address)
        else:
            # Native gRPC. The default channel has no keepalive, so
            # long-lived streams sitting behind a cloud TCP proxy
            # (RunPod's public port forwarder is the case that bit us)
            # get half-closed during quiet windows and both ends go
            # silent without surfacing an error. Send a HTTP/2 ping
            # every 10s of idle so the proxy always sees traffic, and
            # cap the reconnect backoff so a transient drop recovers
            # within seconds instead of gRPC's 2-minute default.
            self._channel = grpc.insecure_channel(
                self.cfg.server_address,
                options=[
                    ("grpc.keepalive_time_ms", 10000),
                    ("grpc.keepalive_timeout_ms", 5000),
                    ("grpc.keepalive_permit_without_calls", 1),
                    ("grpc.http2.min_time_between_pings_ms", 10000),
                    ("grpc.max_reconnect_backoff_ms", 5000),
                    ("grpc.min_reconnect_backoff_ms", 250),
                ],
            )
        self._stub = pb_grpc.InferenceServiceStub(self._channel)

        # Force the gRPC channel to fully establish (TCP + HTTP/2)
        # BEFORE any RPC we measure. Without this, the first Infer pays
        # the cold-connection cost — the TCP connect + HTTP/2 handshake
        # to a fresh peer can take seconds on a high-latency path — and
        # seeds the latency estimator at an outlier value that freezes
        # the cooldown counter for ages. ``channel_ready_future`` blocks until the
        # connection actually transitions to READY; we don't care about
        # the result, only the side effect. 30s ceiling so a genuinely
        # broken endpoint surfaces as a hard error instead of hanging.
        if not self.cfg.use_grpc_web:
            try:
                grpc.channel_ready_future(self._channel).result(timeout=30.0)
            except grpc.FutureTimeoutError:
                log.warning(
                    "gRPC channel not READY within 30s; proceeding anyway "
                    "(first Infer may be slow)",
                )

        resp: pb.OpenSessionResponse = self._stub.OpenSession(
            pb.OpenSessionRequest(
                model_id=self.cfg.model_id,
                policy_uri=self.cfg.policy_uri,
                policy_backend=self.cfg.policy_backend,
                chunk_size=self.cfg.chunk_size,
                action_dim=self.cfg.action_dim,
                min_execution_horizon=self.cfg.min_execution_horizon,
                payload_codec=self.cfg.payload_codec,
                metadata=self.cfg.metadata,
            ),
            metadata=self._auth_metadata,
        )
        self.session_id = resp.session_id
        self.action_dim = resp.action_dim
        log.info("DRTC session opened session_id=%s action_dim=%d",
                 self.session_id, self.action_dim)

        # Write-through tick spool (ADR 0023). Keyed by session id, so a
        # crashed process that reopens the SAME session resumes its
        # un-acked backlog from disk automatically.
        self._spool = TickSpool(
            self.session_id, server_address=self.cfg.server_address,
        )

        self._receiver = ActionReceiver(
            schedule=self.schedule,
            latency=self.latency,
            sent_at=self._sent_at,
            # Sequential (--synchronous) chunking: clear the in-flight gate when a
            # chunk lands so step() can fire the next observation once the schedule
            # drains. Runs on the sender thread; harmless no-op in async mode.
            on_chunk=lambda _chunk: self._in_flight.clear(),
        )

        def _send(msg: pb.Observation) -> None:
            # Record send wall-time keyed by control_timestamp so the
            # receiver can compute RTT. The cooldown is NOT touched here:
            # per the DRTC design it is a pure step counter owned by the
            # control loop, never re-armed by the receive path. If this
            # RPC is lost the cooldown elapses on its own and the
            # controller re-fires.
            self._sent_at[msg.control_timestamp] = time.monotonic()
            self._stat_infer += 1
            t0 = time.monotonic()
            try:
                chunk = self._stub.Infer(msg, metadata=self._auth_metadata)
            except Exception:
                log.exception("Infer RPC failed; cooldown will re-fire")
                self._sent_at.pop(msg.control_timestamp, None)
                # Sync mode has no cooldown fallback: release the in-flight gate so
                # the next tick retries instead of stalling the robot forever. No-op
                # in async mode (the gate is unused there).
                self._in_flight.clear()
                return
            self._last_rtt_s = time.monotonic() - t0
            self._last_compute_s = chunk.server_compute_ns / 1e9
            self._stat_chunk += 1
            self._receiver.on_chunk(chunk)

        self._sender = ObservationSender(
            session_id=self.session_id,
            schedule=self.schedule,
            clock=self.clock,
            send_fn=_send,
        )
        self._sender.start()

        # Telemetry logger — periodic DRTC health line so the operator
        # can tell transport problems (starvation) apart from policy
        # problems (jumpy actions on a healthy queue).
        if self.cfg.stats_interval_s > 0:
            self._stat_t0 = time.monotonic()
            self._stats_thread = threading.Thread(
                target=self._stats_loop, name="drtc-stats", daemon=True
            )
            self._stats_thread.start()

        # Background RecordTick sender. Runs on its own thread so the
        # control loop never blocks on gRPC even if the server pauses.
        self._rec_thread = threading.Thread(
            target=self._rec_loop, name="drtc-rec", daemon=True,
        )
        self._rec_thread.start()

    def close(self) -> None:
        # Idempotent + single-writer: whichever thread arrives first (runner
        # finally or daemon force-close) runs the teardown; the other returns.
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        self._stop.set()
        if self._stats_thread:
            self._stats_thread.join(timeout=2.0)
            self._stats_thread = None
        if self._sender:
            self._sender.stop()
        # Flush remaining RecordTicks BEFORE CloseSession so the server's
        # recorder receives them while the session is still alive. Drains the
        # whole backlog (the tail) rather than the old 15s guillotine — see
        # _drain_recorder.
        self._drain_recorder()
        if self._stub and self.session_id:
            try:
                self._stub.CloseSession(
                    pb.CloseSessionRequest(session_id=self.session_id),
                    metadata=self._auth_metadata,
                )
            except Exception:
                log.warning("CloseSession failed; server will GC eventually")
        if self._channel:
            try:
                self._channel.close()
            except Exception:
                pass
        # Authoritative, loud end-of-episode accounting — replaces the old
        # silent truncation.
        self._log_recording_summary()

    def _drain_recorder(self) -> None:
        """Drain the spool before CloseSession — this IS drain-done.

        Sends the close sentinel, then waits while the sender is still
        getting ticks acked. Bails only if the sender stops making
        progress for _REC_DRAIN_STALL_S (link down) or the hard ceiling
        elapses — and on a bail the un-acked tail is RETAINED ON DISK
        (``_rec_unsent_retained``), never lost: an orphaned spool is
        surfaced at the next daemon startup (spool.orphan_sessions). Only
        a fully-drained spool is disposed.
        """
        spool = self._spool

        def _pending() -> int:
            return spool.pending_count if spool is not None else 0

        thread = self._rec_thread
        if thread is None or not thread.is_alive():
            self._rec_thread = None
            self._rec_unsent_retained = _pending()
            if spool is not None and self._rec_unsent_retained == 0:
                # Sender already drained the spool and exited before we could
                # send the sentinel — nothing left to protect, so tidy up
                # rather than leave an empty dir behind as an orphan.
                spool.dispose()
            return
        # Close sentinel: the sender flushes the whole spool, then exits.
        self._rec_q.put(None)
        pending_bytes = spool.pending_bytes if spool is not None else 0
        ceiling_s = _drain_ceiling_logged(pending_bytes)
        deadline = time.monotonic() + ceiling_s
        last_sent = self._rec_sent
        last_progress = time.monotonic()
        while thread.is_alive():
            thread.join(timeout=0.5)
            now = time.monotonic()
            # Progress = the server ACKED another tick (not "an RPC was
            # attempted") — a dead link is correctly seen as no progress.
            if self._rec_sent > last_sent:
                last_sent = self._rec_sent
                last_progress = now
            if now - last_progress > _REC_DRAIN_STALL_S:
                log.warning(
                    "DRTC recorder drain stalled (%d ticks unsent — link "
                    "down?); retaining them on disk for later recovery",
                    _pending(),
                )
                self._rec_abandon.set()
                break
            if now > deadline:
                log.warning(
                    "DRTC recorder drain hit %.0fs ceiling (%d ticks "
                    "unsent); retaining them on disk for later recovery",
                    ceiling_s, _pending(),
                )
                self._rec_abandon.set()
                break
        thread.join(timeout=5.0)
        self._rec_thread = None
        self._rec_unsent_retained = _pending()
        if spool is not None and self._rec_unsent_retained == 0:
            # Fully drained and acked — nothing left to protect.
            spool.dispose()

    def _log_recording_summary(self) -> None:
        """One authoritative line on how complete the recording was."""
        captured = self._rec_captured
        if captured == 0 and self._rec_refused == 0:
            return
        retained = self._rec_unsent_retained
        if retained or self._rec_refused:
            if retained and self._spool is not None:
                detail = (
                    "un-acked ticks (%.0f MB) kept in the spool at %s — "
                    "they resume ONLY if this session is re-assigned; "
                    "otherwise spool GC deletes them after ~7 days"
                    % (self._spool.pending_bytes / 1e6, self._spool.dir)
                )
            elif retained:
                detail = "un-acked ticks kept in the spool for recovery"
            else:
                detail = "capture was hard-stopped part of the time"
            log.warning(
                "DRTC recording finished: captured=%d sent=%d "
                "retained_on_disk=%d refused_at_capture=%d — %s",
                captured, self._rec_sent, retained, self._rec_refused,
                detail,
            )
        else:
            log.info(
                "DRTC recording finished: captured=%d sent=%d (complete)",
                captured, self._rec_sent,
            )

    # ------------------------------------------------------------------
    # Recorder (per-tick capture → background RecordTick RPC)
    # ------------------------------------------------------------------

    @property
    def recording_blocked(self) -> bool:
        """True while the spool is hard-stopped (full disk backlog).

        The node loop/daemon must refuse to start new episodes while this
        is set; it clears automatically once the sender drains the spool
        below the resume threshold (hysteresis in TickSpool.blocked).
        """
        return self._spool is not None and self._spool.blocked

    def record_tick(
        self,
        *,
        step: int,
        observation_state: Optional[list[float]],
        action: list[float],
        jpegs: dict[str, bytes],
        control_timestamp_ns: int,
        control_source: Optional[str] = None,
    ) -> bool:
        """Journal one captured tick to the spool; wake the sender.

        Called from the control loop after each successful step(). The
        hot path pays one protobuf serialize + one small file write
        (write-through spool, ADR 0023); the background ``_rec_loop``
        thread ships spooled ticks and deletes them only on the server's
        ack. Returns False when the tick was REFUSED — spool hard-stopped
        (full) or disk error — so the caller knows this tick is NOT part
        of the episode. Never silently thinned.

        ``control_source`` is recorded as
        ``annotation.interlatent.control_source``. ``None`` means
        "policy" on the server side.
        """
        if self.session_id is None or self._stub is None or self._spool is None:
            return False
        if self._spool.blocked:
            self._rec_refused += 1
            if not self._rec_refused_logged:
                log.error(
                    "record_tick refused: tick spool is full — capture is "
                    "hard-stopped until the uplink drains the backlog "
                    "(spool=%d ticks / %.0f MB)",
                    self._spool.pending_count,
                    self._spool.pending_bytes / 1e6,
                )
                self._rec_refused_logged = True
            return False
        req = self._build_tick_req({
            "step": int(step),
            "observation_state": observation_state,
            "action": action,
            "jpegs": jpegs,
            "control_timestamp_ns": int(control_timestamp_ns),
            "control_source": control_source,
        })
        if self._spool.append(req.SerializeToString()) is None:
            self._rec_refused += 1
            return False
        self._rec_refused_logged = False
        self._rec_captured += 1
        self._rec_q.put_nowait(1)
        return True

    def _rec_loop(self) -> None:
        """Ship spooled ticks in coalesced batches; delete only on ack.

        The queue carries wakeup tokens, not data — the spool is the
        single source of truth for what remains to be sent, so a failed
        RPC needs no re-queueing: the un-acked ticks are simply still
        there on the next attempt. Batching amortizes the RTT (a unary
        RPC per tick can't keep up with a 30 Hz capture rate).

        This is the sole writer of ``_rec_sent`` and the ``_stat_rec_*``
        window counters, so those stay lock-free.
        """
        while True:
            try:
                tok = self._rec_q.get(timeout=0.25)
            except queue.Empty:
                # Stay alive while the spool still holds data, even after
                # _stop: close() sets _stop up to ~2s before _drain_recorder
                # enqueues the close sentinel, and that sentinel must reach a
                # LIVE sender for _flush_backlog (and the subsequent dispose)
                # to run. Returning here on _stop alone would strand a
                # perfectly drainable spool as an on-disk orphan.
                if self._stop.is_set() and (
                    self._spool is None or self._spool.pending_count == 0
                ):
                    return
                # Idle tick: retry any backlog a failed send left behind.
                if self._spool is not None and self._spool.pending_count:
                    self._ship_available()
                continue
            saw_pill = tok is None or self._drain_wake_tokens()
            self._ship_available()
            if saw_pill:
                self._flush_backlog()
                return

    def _drain_wake_tokens(self) -> bool:
        """Swallow queued wake tokens (they carry no data); True if the
        close sentinel was among them."""
        saw_pill = False
        while True:
            try:
                tok = self._rec_q.get_nowait()
            except queue.Empty:
                return saw_pill
            if tok is None:
                saw_pill = True

    def _ship_available(self) -> None:
        """Ship spooled batches until the spool is empty or a send fails
        (failure leaves the backlog on disk; backoff, then the outer loop
        retries)."""
        if self._spool is None:
            return
        while not self._rec_abandon.is_set():
            batch = self._spool.peek_batch(
                max(1, int(self.cfg.rec_batch_max_ticks)),
                _REC_BATCH_MAX_BYTES,
            )
            if not batch:
                return
            if self._send_batch(batch):
                self._rec_backoff_s = 0.0
                continue
            # Failure or partial accept: back off (interruptible), leave
            # the rest spooled. Progress made this attempt still counted.
            self._rec_backoff_s = min(5.0, (self._rec_backoff_s or 0.5) * 2)
            self._stop.wait(self._rec_backoff_s)
            return

    def _flush_backlog(self) -> None:
        """Close path: keep retrying until the spool drains or
        _drain_recorder abandons the link (tail stays on disk)."""
        if self._spool is None:
            return
        while self._spool.pending_count and not self._rec_abandon.is_set():
            before = self._rec_sent
            self._ship_available()
            if self._spool.pending_count and self._rec_sent == before:
                # No progress this round; brief pause, then retry until
                # _drain_recorder's stall detector calls it.
                time.sleep(0.5)

    def _build_tick_req(self, item: dict) -> "pb.RecordTickRequest":
        req = pb.RecordTickRequest(
            session_id=self.session_id,
            step=int(item["step"]),
            control_timestamp=int(item["control_timestamp_ns"]),
            action=[float(x) for x in item["action"]],
        )
        state = item.get("observation_state")
        if state:
            req.observation_state.extend(float(x) for x in state)
        jpegs = item.get("jpegs") or {}
        for cam, data in jpegs.items():
            if data:
                req.jpegs[cam] = data
        cs = item.get("control_source")
        if cs:
            req.control_source = str(cs)
        return req

    def _send_batch(self, batch: list[tuple[int, bytes]]) -> bool:
        """Ship one spooled batch via RecordTicks; ack the accepted prefix.

        ``batch`` is [(seq, serialized RecordTickRequest), ...] straight
        from the spool. On success the server's ``accepted`` is a PREFIX
        count (honest acks, ADR 0023): exactly that many ticks are
        deleted from the spool; the rest stay for retry. Returns True iff
        the whole batch was accepted. Recording failures must never break
        inference — every path logs and returns; the control loop keeps
        running regardless.
        """
        if not batch or self._stub is None or self.session_id is None:
            return False
        if self._rec_batch_unsupported:
            return self._send_unary(batch)
        import grpc

        batch_bytes = sum(len(data) for _, data in batch)
        _pace_t0 = time.monotonic()
        try:
            req = pb.RecordTicksRequest(
                ticks=[pb.RecordTickRequest.FromString(data) for _, data in batch],
            )
            resp = self._stub.RecordTicks(
                req, metadata=self._auth_metadata, timeout=_REC_BATCH_TIMEOUT_S,
            )
            accepted = max(0, min(int(getattr(resp, "accepted", 0)), len(batch)))
            if accepted and self._spool is not None:
                # Delete-after-ack: only the accepted prefix leaves disk.
                self._spool.ack(batch[accepted - 1][0])
                self._rec_sent += accepted
                # Window telemetry — successfully-shipped ticks/bytes only,
                # so rec_hz / rec_bytes_s measure the drain capacity.
                self._stat_rec_sent += accepted
                self._stat_rec_bytes += sum(
                    len(data) for _, data in batch[:accepted]
                )
            return accepted == len(batch)
        except grpc.RpcError as exc:
            if exc.code() == grpc.StatusCode.UNIMPLEMENTED:
                # Older server without RecordTicks; ship tick-by-tick and
                # stay on the unary path for the rest of the session.
                log.info(
                    "server has no RecordTicks; falling back to unary RecordTick"
                )
                self._rec_batch_unsupported = True
                return self._send_unary(batch)
            log.debug("RecordTicks failed (backlog stays spooled)", exc_info=True)
            return False
        except Exception:
            log.debug("RecordTicks failed (backlog stays spooled)", exc_info=True)
            return False
        finally:
            # Uplink pacing: hold the drain until this batch's bytes fit
            # the configured budget. The RPC's own upload time counts
            # toward the quota, so a slow link is never double-penalized.
            # Skipped during close (self._stop set) — the final drain
            # should finish as fast as the link allows, not as slow as
            # the pacer.
            if self._rec_pace_bps > 0 and not self._stop.is_set():
                quota_s = batch_bytes / self._rec_pace_bps
                spent_s = time.monotonic() - _pace_t0
                if quota_s > spent_s:
                    time.sleep(min(quota_s - spent_s, 5.0))

    def _send_unary(self, batch: list[tuple[int, bytes]]) -> bool:
        """Unary fallback: ship + ack tick-by-tick, stop at first failure
        (the rest stays spooled for retry). ``ok`` is honest on new
        servers; an old server that over-acks is no worse than before."""
        if self._stub is None or self.session_id is None:
            return False
        for seq, data in batch:
            try:
                req = pb.RecordTickRequest.FromString(data)
                resp = self._stub.RecordTick(
                    req, metadata=self._auth_metadata, timeout=10,
                )
                if not bool(getattr(resp, "ok", True)):
                    return False
                if self._spool is not None:
                    self._spool.ack(seq)
                self._rec_sent += 1
                self._stat_rec_sent += 1
                self._stat_rec_bytes += len(data)
            except Exception:
                # Recording failures must never break inference; the
                # un-acked remainder stays on disk.
                log.debug("RecordTick failed (backlog stays spooled)",
                          exc_info=True)
                return False
        return True

    # ------------------------------------------------------------------
    # Per-control-step entry point
    # ------------------------------------------------------------------

    def step(
        self,
        observation: Union[bytes, Callable[[], bytes]],
        *,
        codec: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """Advance one control step.

        Decisions, in order:
          1. If the queue has drained below the execution horizon and
             no request is in flight, prefetch a new chunk.
          2. Pop and return exactly the next action from the schedule.

        Exactly one action is consumed per call — the full action chunk
        is executed step by step, never bulk-popped and discarded. If
        the queue is starved (cold start, or a chunk arrived late) this
        returns None *without* advancing the cursor, so the next chunk
        resumes from the same step with no actions skipped.

        `observation` may be raw bytes or a zero-arg callable returning
        bytes. The callable form lets the caller skip an expensive
        encode on the (majority of) ticks where no observation is sent.
        """
        if self._sender is None:
            raise RuntimeError("DRTCClient.open() not called")

        self._stat_steps += 1
        depth = self.schedule.queue_depth()
        if depth < self._stat_qmin:
            self._stat_qmin = depth

        # DRTC cooldown O^c: decrement once per control step.
        self.cooldown.tick()

        # Trigger a new inference request. Two cadences:
        #  - ASYNC (default, overlapping/replace-mode chunking): fire when the
        #    schedule has drained below the execution horizon AND the cooldown
        #    O^c has elapsed. The horizon (latency estimate + min_execution_horizon)
        #    lands the chunk before the queue empties; the cooldown — armed to the
        #    estimated latency — prevents double-firing within one inference and
        #    re-fires automatically if a chunk is lost.
        #  - SYNCHRONOUS (sequential / request-response chunking, opt-in): fire ONLY
        #    when the schedule is fully drained (depth == 0) and no request is in
        #    flight. One chunk executes to completion before the next observation,
        #    so a fresh chunk never overwrites an unexecuted tail. The in-flight
        #    Event replaces the cooldown as the anti-double-fire gate — there is no
        #    cooldown fallback here, so the _send error path clears the gate to keep
        #    a dropped Infer from stalling the robot forever.
        delay = self.latency.estimate_steps(self.cfg.control_period_s)
        if self.cfg.synchronous:
            should_send = depth == 0 and not self._in_flight.is_set()
        else:
            horizon = delay + self.cfg.min_execution_horizon
            should_send = depth < horizon and self.cooldown.ready()
        if should_send:
            payload = observation() if callable(observation) else observation
            if self.cfg.synchronous:
                self._in_flight.set()
            else:
                # O^c <- (latency estimate, or s_min before one exists) + epsilon
                self.cooldown.arm(delay if delay > 0 else self.cfg.min_execution_horizon)
            self._sender.submit(
                PendingObservation(
                    payload=payload,
                    payload_codec=codec or self.cfg.payload_codec,
                    inference_delay=delay,
                )
            )

        action = self.schedule.pop_next()
        if action is None:
            # In sync mode the gap between chunks is an EXPECTED hold (the robot
            # holds its last pose while it waits for the next chunk), not transport
            # starvation — count it separately so starvation_pct stays meaningful.
            if self.cfg.synchronous and self._in_flight.is_set():
                self._stat_wait += 1
            else:
                self._stat_none += 1
            return None
        # Track step-to-step action change — large jumps on a HEALTHY
        # queue point at the policy, not the transport.
        if self._last_action is not None:
            self._stat_jitter_sum += float(
                np.linalg.norm(action.vector - self._last_action)
            )
            self._stat_jitter_n += 1
        self._last_action = action.vector
        return action.vector

    # ------------------------------------------------------------------
    # Introspection (handy for tests + integration)
    # ------------------------------------------------------------------

    @property
    def queue_depth(self) -> int:
        return self.schedule.queue_depth()

    @property
    def estimated_latency_s(self) -> float:
        return self.latency.estimate_s

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Snapshot DRTC health since the last call, and reset the window.

        Reading the numbers:
          - control_hz   — rate step() is being called (your loop rate)
          - action_hz    — rate step() actually returned an action
          - starvation   — % of steps that returned None; >0 mid-rollout
                            means the queue ran dry (transport too slow)
          - queue_min    — lowest queue depth this window; near 0 is bad
          - infer_ms     — measured Infer round-trip (server + network)
          - compute_ms   — server-reported policy compute (subset of infer_ms)
          - net_ms       — infer_ms - compute_ms; the network/serialization
                           portion. High net_ms -> the link is the problem;
                           high compute_ms -> the GPU/policy is.
          - action_delta — mean |Δaction| between consecutive steps
          - rec_hz       — RecordTick *drain* rate (ticks the server accepted
                           this window). Sits below control_hz when recording
                           is losing ticks. The number to watch.
          - rec_bytes_s  — JPEG bytes/s successfully shipped this window; the
                           recorder's drain bandwidth.
          - rec_qdepth   — current RecordTick backlog depth.
          - rec_dropped  — cumulative recording drops (queue-full + tail).

        DRTC healthy + still jittery  -> starvation ~0, queue_min well
        above 0: the jitter is the policy, not the transport.
        DRTC unhealthy                -> starvation >0 / queue_min ~0.

        Classifying a recording bottleneck (feeds the throughput fix):
          rec_hz ≈ 1000/net_ms and rec_bytes_s below the uplink -> per-RPC /
            latency-bound (serial 1-RTT sender) -> batch or pipeline.
          rec_bytes_s plateaued while rec_qdepth climbs -> bandwidth-bound ->
            reduce payload (JPEG quality/resolution or record fps).
        """
        now = time.monotonic()
        dt = max(now - self._stat_t0, 1e-6)
        steps, none = self._stat_steps, self._stat_none
        jitter = (
            self._stat_jitter_sum / self._stat_jitter_n
            if self._stat_jitter_n else 0.0
        )
        snap = {
            "control_hz": round(steps / dt, 1),
            "action_hz": round((steps - none) / dt, 1),
            "starvation_pct": round(100.0 * none / steps, 1) if steps else 0.0,
            "sync_wait_pct": round(100.0 * self._stat_wait / steps, 1) if steps else 0.0,
            "queue_depth": self.schedule.queue_depth(),
            "queue_min": 0 if self._stat_qmin == (1 << 30) else self._stat_qmin,
            "infer_ms": round(self._last_rtt_s * 1000.0, 1),
            "infer_est_ms": round(self.latency.estimate_s * 1000.0, 1),
            "compute_ms": round(self._last_compute_s * 1000.0, 1),
            "net_ms": round(max(self._last_rtt_s - self._last_compute_s, 0.0) * 1000.0, 1),
            "infer_sent": self._stat_infer,
            "chunks_recv": self._stat_chunk,
            "action_delta": round(jitter, 4),
            "rec_hz": round(self._stat_rec_sent / dt, 1),
            "rec_bytes_s": round(self._stat_rec_bytes / dt, 0),
            # Backlog now lives in the disk spool, not RAM (ADR 0023).
            "rec_qdepth": (self._spool.pending_count if self._spool else 0),
            "rec_spool_bytes": (self._spool.pending_bytes if self._spool else 0),
            "rec_refused": self._rec_refused,
        }
        self._stat_t0 = now
        self._stat_steps = self._stat_none = self._stat_wait = 0
        self._stat_infer = self._stat_chunk = 0
        self._stat_qmin = 1 << 30
        self._stat_jitter_sum = 0.0
        self._stat_jitter_n = 0
        self._stat_rec_sent = 0
        self._stat_rec_bytes = 0
        return snap

    def _stats_loop(self) -> None:
        while not self._stop.wait(self.cfg.stats_interval_s):
            s = self.stats()
            # Teleop-recording sessions never call step(), so control_hz stays
            # 0 while the recorder is hard at work — gate on recorder activity
            # too, or the one line showing rec_qdepth growth is muted in
            # exactly the mode where the recording uplink is the bottleneck.
            if (
                s["control_hz"] == 0.0
                and s["rec_hz"] == 0.0
                and s["rec_qdepth"] == 0
            ):
                continue  # nothing running yet — nothing to report
            log.info(
                "DRTC | %.1f Hz (actions %.1f Hz, starvation %.1f%%, "
                "sync-wait %.1f%%) | "
                "queue %d (min %d) | infer %.0fms = compute %.0fms + net %.0fms "
                "(est %.0fms) | sent %d recv %d | action Δ %.4f | "
                "rec %.1f Hz (%.0f KB/s, spool %d/%0.f MB, refused %d)",
                s["control_hz"], s["action_hz"], s["starvation_pct"],
                s["sync_wait_pct"],
                s["queue_depth"], s["queue_min"], s["infer_ms"],
                s["compute_ms"], s["net_ms"], s["infer_est_ms"],
                s["infer_sent"], s["chunks_recv"], s["action_delta"],
                s["rec_hz"], s["rec_bytes_s"] / 1024.0, s["rec_qdepth"],
                s["rec_spool_bytes"] / 1e6, s["rec_refused"],
            )
