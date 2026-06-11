"""Action receiver.

Counterpart to sender: consumes ActionChunk messages from the
transport (either by reading a response stream, polling, or being
pushed into via a callback) and merges them into the LWW schedule.

Also feeds the latency estimator with the round-trip time observed
on each chunk so the execution horizon stays adaptive.
"""

from __future__ import annotations

import logging
import time
from typing import Callable

import numpy as np

from ..protocol import messages_pb2 as pb
from .latency import JacobsonKarels
from .merge import ActionSchedule, ScheduledAction

log = logging.getLogger(__name__)


class ActionReceiver:
    def __init__(
        self,
        schedule: ActionSchedule,
        latency: JacobsonKarels,
        sent_at: dict[int, float],
        on_chunk: Callable[[pb.ActionChunk], None] | None = None,
    ) -> None:
        self._sched = schedule
        self._lat = latency
        # Map control_timestamp -> wall-clock send time, populated by
        # the sender so we can compute RTT on response.
        self._sent_at = sent_at
        self._on_chunk = on_chunk

    def on_chunk(self, chunk: pb.ActionChunk) -> None:
        """Called by transport when an ActionChunk arrives."""
        # RTT for latency estimation.
        sent = self._sent_at.pop(chunk.control_timestamp, None)
        if sent is not None:
            self._lat.observe(time.monotonic() - sent)

        actions = [
            ScheduledAction(
                action_step=a.action_step,
                control_timestamp=a.control_timestamp,
                vector=np.asarray(a.vector, dtype=np.float32),
            )
            for a in chunk.actions
        ]
        installed = self._sched.merge(actions)
        log.debug(
            "merged chunk ts=%d installed=%d/%d depth=%d",
            chunk.control_timestamp, installed, len(actions),
            self._sched.queue_depth(),
        )

        if self._on_chunk is not None:
            self._on_chunk(chunk)
