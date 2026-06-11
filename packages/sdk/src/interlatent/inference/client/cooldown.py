"""Inference-request cooldown counter (DRTC ``O^c``).

Faithful to the DRTC design: the client gates re-triggering inference
with a step-counting cooldown.

    O^c(t+1) = s_min + e        until a latency estimate exists
             = l_delta + e      if a request was fired at t
             = max(O^c(t) - 1, 0)   otherwise

where ``l_delta`` is the estimated inference latency in control steps
and ``e`` is a small epsilon margin. A new request may be fired only
when ``O^c`` reaches 0.

In the normal case the action chunk arrives and refills the schedule
before ``O^c`` expires, so the schedule-depth condition gates the next
request. If an observation or action chunk is lost, ``O^c`` still
reaches 0 after ~``l_delta + e`` steps and the request is re-fired —
drop recovery with no explicit retry timer or acknowledgements.

The counter is owned by the control loop: ``tick()`` once per control
step, ``arm()`` when a request is fired. The receive path never
touches it.
"""

from __future__ import annotations

import threading


class Cooldown:
    def __init__(self, epsilon: int = 2) -> None:
        # epsilon: small margin added on top of the latency estimate so
        # the cooldown outlasts the actual round-trip in the normal case.
        self._n = 0
        self._epsilon = max(1, int(epsilon))
        self._lock = threading.Lock()

    def arm(self, latency_steps: int = 0) -> None:
        """Called when a request is fired. Sets ``O^c`` to the estimated
        inference latency (in control steps) plus epsilon."""
        with self._lock:
            self._n = max(1, int(latency_steps) + self._epsilon)

    def tick(self) -> None:
        """Decrement ``O^c`` once per control step."""
        with self._lock:
            if self._n > 0:
                self._n -= 1

    def ready(self) -> bool:
        """True when ``O^c`` has reached 0 — a new request may be fired."""
        with self._lock:
            return self._n == 0

    @property
    def remaining(self) -> int:
        with self._lock:
            return self._n
