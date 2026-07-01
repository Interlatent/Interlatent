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

log = logging.getLogger(__name__)


# Bound the per-tick recorder queue. At 30 Hz this is ~8.5s of capture
# buffer before drops kick in — enough to ride out a brief network blip
# or a server pause without blocking the control loop.
_RECORDER_QUEUE_MAXSIZE = 256

# Close-path drain bounds. Recording is off the control-critical path at
# close (the robot has already disconnected), so we drain the whole backlog
# rather than dropping the tail. We only give up if the sender stops
# shipping for _REC_DRAIN_STALL_S (a dead link — set above the 10s per-RPC
# RecordTick timeout so a merely-slow link is not mistaken for a dead one),
# or a hard ceiling elapses as an ultimate backstop.
_REC_DRAIN_STALL_S = 12.0
_REC_DRAIN_CEILING_S = 120.0


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

        # --- per-tick recorder pipeline ---------------------------------
        # The hot path (step()) only does a non-blocking put_nowait into
        # ``_rec_q``. A dedicated background thread drains the queue and
        # makes gRPC RecordTick calls so inference latency is unaffected.
        # On queue overflow the row is dropped (see _RECORDER_QUEUE_MAXSIZE).
        self._rec_q: "queue.Queue[Optional[dict]]" = queue.Queue(
            maxsize=_RECORDER_QUEUE_MAXSIZE,
        )
        self._rec_thread: Optional[threading.Thread] = None
        # Drop accounting is split by failure mode so telemetry can tell
        # distributed thinning apart from tail loss:
        #   _rec_dropped_full  — queue-full drops during the run (thinning)
        #   _rec_dropped_close — backlog abandoned at close (the tail)
        #   _rec_captured      — every record_tick() call (intended count)
        #   _rec_sent          — RecordTicks the server actually accepted
        # Invariant when the session ends cleanly:
        #   captured == sent + dropped_full + dropped_close
        self._rec_captured = 0
        self._rec_dropped_full = 0
        self._rec_dropped_close = 0
        self._rec_first_warn = False
        self._rec_sent = 0

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

        # Force the gRPC channel to fully establish (TCP + HTTP/2 +
        # whatever Tailscale needs to bring its tunnel up to this peer)
        # BEFORE any RPC we measure. Without this, the first Infer pays
        # the cold-tunnel cost — easily 10-20s when Tailscale falls back
        # to DERP relay on first contact — and seeds the latency
        # estimator at an outlier value that freezes the cooldown
        # counter for ages. ``channel_ready_future`` blocks until the
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

        self._receiver = ActionReceiver(
            schedule=self.schedule,
            latency=self.latency,
            sent_at=self._sent_at,
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
        """Flush the RecordTick backlog before CloseSession.

        Sends an ordered poison-pill, then waits while the sender is still
        shipping ticks. Bails only if the sender stops making progress for
        _REC_DRAIN_STALL_S (link down) or the hard ceiling elapses; whatever
        is still queued at that point is counted as tail loss
        (``_rec_dropped_close``).
        """
        thread = self._rec_thread
        if thread is None:
            self._rec_dropped_close = self._rec_q.qsize()
            return
        # Ordered sentinel. Blocking put guarantees it lands *after* the
        # current backlog (unlike the old put_nowait, which the poison-pill
        # silently lost whenever the queue was full — exactly the loaded
        # case). Bounded so a wedged sender can't hang close() forever.
        try:
            self._rec_q.put(None, timeout=_REC_DRAIN_CEILING_S)
        except queue.Full:
            pass
        deadline = time.monotonic() + _REC_DRAIN_CEILING_S
        last_sent = self._rec_sent
        last_progress = time.monotonic()
        while thread.is_alive():
            thread.join(timeout=0.5)
            now = time.monotonic()
            # Progress = the server accepted another tick. Using _rec_sent
            # (not qsize) means a dead link — which still *consumes* items as
            # each RPC fails — is correctly seen as "no progress".
            if self._rec_sent > last_sent:
                last_sent = self._rec_sent
                last_progress = now
            if now - last_progress > _REC_DRAIN_STALL_S:
                log.warning(
                    "DRTC recorder drain stalled (%d ticks queued, sender not "
                    "shipping — link down?); abandoning tail",
                    self._rec_q.qsize(),
                )
                break
            if now > deadline:
                log.warning(
                    "DRTC recorder drain hit %.0fs ceiling (%d ticks queued); "
                    "abandoning tail",
                    _REC_DRAIN_CEILING_S, self._rec_q.qsize(),
                )
                break
        self._rec_thread = None
        # Whatever is still queued is tail loss (± the sentinel if we bailed
        # before it was consumed — immaterial for accounting).
        self._rec_dropped_close = self._rec_q.qsize()

    def _log_recording_summary(self) -> None:
        """One authoritative line on how complete the recording was."""
        captured = self._rec_captured
        if captured == 0:
            return
        dropped = self._rec_dropped_full + self._rec_dropped_close
        if dropped:
            log.warning(
                "DRTC recording finished: captured=%d sent=%d dropped_full=%d "
                "dropped_close=%d (%.1f%% lost) — episode is lossy",
                captured, self._rec_sent, self._rec_dropped_full,
                self._rec_dropped_close, 100.0 * dropped / captured,
            )
        else:
            log.info(
                "DRTC recording finished: captured=%d sent=%d (complete)",
                captured, self._rec_sent,
            )

    # ------------------------------------------------------------------
    # Recorder (per-tick capture → background RecordTick RPC)
    # ------------------------------------------------------------------

    def record_tick(
        self,
        *,
        step: int,
        observation_state: Optional[list[float]],
        action: list[float],
        jpegs: dict[str, bytes],
        control_timestamp_ns: int,
        control_source: Optional[str] = None,
    ) -> None:
        """Non-blocking enqueue of one captured tick.

        Called from the control loop after each successful step(). The hot
        path pays only a ``queue.put_nowait`` (microseconds); JPEG bytes
        are passed by reference. The background ``_rec_loop`` thread does
        the actual gRPC call. On queue overflow the row is dropped and a
        single WARN is emitted.

        ``control_source`` is recorded as
        ``annotation.interlatent.control_source``. ``None`` means
        "policy" on the server side.
        """
        if self.session_id is None or self._stub is None:
            return
        # Count every captured tick (the node-side "intended" count) before
        # the queue can reject it, so captured == sent + dropped_* holds.
        self._rec_captured += 1
        item = {
            "step": int(step),
            "observation_state": observation_state,
            "action": action,
            "jpegs": jpegs,
            "control_timestamp_ns": int(control_timestamp_ns),
            "control_source": control_source,
        }
        try:
            self._rec_q.put_nowait(item)
        except queue.Full:
            self._rec_dropped_full += 1
            if not self._rec_first_warn:
                log.warning(
                    "DRTC recorder queue full at step %d — dropping tick "
                    "(distributed thinning; running total in the DRTC stats "
                    "line as rec_dropped)",
                    step,
                )
                self._rec_first_warn = True

    def _rec_loop(self) -> None:
        """Drain the recorder queue and ship each tick via RecordTick."""
        while True:
            try:
                item = self._rec_q.get(timeout=0.25)
            except queue.Empty:
                if self._stop.is_set():
                    return
                continue
            if item is None:
                # poison pill — drain remaining items then exit
                try:
                    while True:
                        leftover = self._rec_q.get_nowait()
                        if leftover is None:
                            continue
                        self._send_record_tick(leftover)
                except queue.Empty:
                    return
                return
            self._send_record_tick(item)

    def _send_record_tick(self, item: dict) -> None:
        if self._stub is None or self.session_id is None:
            return
        try:
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
            tick_bytes = 0
            for cam, data in jpegs.items():
                if data:
                    req.jpegs[cam] = data
                    tick_bytes += len(data)
            cs = item.get("control_source")
            if cs:
                req.control_source = str(cs)
            self._stub.RecordTick(req, metadata=self._auth_metadata, timeout=10)
            self._rec_sent += 1
            # Window telemetry — count only successfully-shipped ticks/bytes,
            # so rec_hz / rec_bytes_s measure the drain capacity (post-drop).
            self._stat_rec_sent += 1
            self._stat_rec_bytes += tick_bytes
        except Exception:
            # Recording failures must never break inference. Log once at
            # debug; the control loop keeps going.
            log.debug("RecordTick failed", exc_info=True)

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

        # DRTC trigger: fire an inference request when the schedule has
        # drained below the horizon AND the cooldown has elapsed
        # (O^c == 0). The horizon is the latency estimate plus a margin
        # (min_execution_horizon) so the chunk lands before the queue
        # hits zero; the cooldown — armed to the estimated latency —
        # both prevents double-firing within one inference and re-fires
        # automatically if a chunk is lost (O^c reaches 0 with the
        # schedule still below the horizon).
        delay = self.latency.estimate_steps(self.cfg.control_period_s)
        horizon = delay + self.cfg.min_execution_horizon
        if depth < horizon and self.cooldown.ready():
            payload = observation() if callable(observation) else observation
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
            "rec_qdepth": self._rec_q.qsize(),
            "rec_dropped": self._rec_dropped_full + self._rec_dropped_close,
        }
        self._stat_t0 = now
        self._stat_steps = self._stat_none = 0
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
            if s["control_hz"] == 0.0:
                continue  # loop not running yet — nothing to report
            log.info(
                "DRTC | %.1f Hz (actions %.1f Hz, starvation %.1f%%) | "
                "queue %d (min %d) | infer %.0fms = compute %.0fms + net %.0fms "
                "(est %.0fms) | sent %d recv %d | action Δ %.4f | "
                "rec %.1f Hz (%.0f KB/s, q %d, dropped %d)",
                s["control_hz"], s["action_hz"], s["starvation_pct"],
                s["queue_depth"], s["queue_min"], s["infer_ms"],
                s["compute_ms"], s["net_ms"], s["infer_est_ms"],
                s["infer_sent"], s["chunks_recv"], s["action_delta"],
                s["rec_hz"], s["rec_bytes_s"] / 1024.0, s["rec_qdepth"],
                s["rec_dropped"],
            )
