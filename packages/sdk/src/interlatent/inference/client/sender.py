"""Observation sender.

Pulls observations off a queue and pushes them to the server. Runs as
a daemon thread launched by the controller. The controller owns the
schedule and the in-flight gate; the sender only:

  1. Reads the next observation to send.
  2. Stamps it with a monotonic control_timestamp.
  3. Attaches the current scheduled_spans, next_action_step, and the
     estimated inference_delay (for server-side RTC in-painting).
  4. Hands it to the transport.

We deliberately don't await the response here — `receiver` handles
that side. This decoupling is what lets DRTC tolerate the response
being late, duplicated, or never arriving.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from typing import Callable, Optional

from ..protocol import messages_pb2 as pb
from ..protocol.timestamps import ControlClock
from .merge import ActionSchedule

log = logging.getLogger(__name__)


@dataclass
class PendingObservation:
    """What the controller hands to the sender."""

    payload: bytes
    payload_codec: str = "raw_f32"
    # Estimated round-trip inference latency in control steps, snapshot
    # by the controller at submit time. Sent to the server so RTC knows
    # how many leading actions to freeze during in-painting.
    inference_delay: int = 0


class ObservationSender:
    def __init__(
        self,
        session_id: str,
        schedule: ActionSchedule,
        clock: ControlClock,
        send_fn: Callable[[pb.Observation], None],
        outbox: Optional[queue.Queue[PendingObservation]] = None,
    ) -> None:
        self._session_id = session_id
        self._sched = schedule
        self._clock = clock
        self._send = send_fn
        self._outbox: queue.Queue[PendingObservation] = outbox or queue.Queue()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def submit(self, obs: PendingObservation) -> None:
        self._outbox.put(obs)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="drtc-sender", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._outbox.put(None)  # type: ignore[arg-type]
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            item = self._outbox.get()
            if item is None:
                break
            try:
                self._dispatch(item)
            except Exception:
                log.exception("sender dispatch failed; will retry on next tick")

    def _dispatch(self, item: PendingObservation) -> None:
        spans = self._sched.scheduled_spans()
        msg = pb.Observation(
            session_id=self._session_id,
            control_timestamp=self._clock.tick(),
            next_action_step=self._sched.next_action_step(),
            payload=item.payload,
            payload_codec=item.payload_codec,
            inference_delay=max(0, int(item.inference_delay)),
        )
        for lo, hi in spans:
            s = msg.scheduled_spans.add()
            s.start_step = lo
            s.end_step = hi
        self._send(msg)
