"""Node-side video context ring for world-action model (WAM) policies.

A WAM (DreamZero and friends) does not consume a single observation — it
carries a rolling KV cache of recent video and jointly denoises the future
frames and the action chunk. It was trained on a *steady* low frame rate
(5 FPS for DreamZero, 33 frames = 6.6 s of context), and because the actions
are denoised jointly with the video, a stuttery or stale context degrades the
*actions* with no error surfaced anywhere.

Three properties fall out of that, and each one rules out an approach that
looks cheaper:

* **The ring is fed on every control tick, not on every ``Infer``.** ``Infer``
  fires irregularly (cooldown-gated, and only on a drained queue in
  synchronous mode — roughly every 3 s on current hardware). A ring filled
  from the ``Infer`` cadence would span ~100 s of wall time instead of 6.6 s.
* **It is fed independently of recording.** ``capture_fn`` in the control loop
  is gated on ``outcome.should_record``, which is False while the arm holds
  waiting for a chunk — precisely the window a WAM most needs to keep
  observing.
* **It rides in the ``Infer`` payload, not the RecordTick uplink.** RecordTick
  is deliberately never-drop: write-through disk spool, delete-after-ack,
  bandwidth-paced, drained as an ordered *oldest-first* prefix (ADR 0023). It
  optimises completeness at the cost of timeliness. Under congestion a
  RecordTick-fed context would serve the model the *most stale* frames
  available. See ADR 0037.

Frames are JPEG-encoded here, on the control thread, for the same reason
``_capture_tick`` does it: the buffer then holds compressed bytes rather than
raw RGB. Encoding is capability-adaptive (nvjpeg / turbojpeg / cv2 / PIL).

``offer`` and ``drain`` are called from the same thread — the control loop
calls ``offer`` directly, and the DRTC client evaluates the payload callable
inline before handing it to the sender thread — so this class holds no lock.
Nothing here may raise: a context failure must degrade the policy, never stop
the robot.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Optional

import numpy as np

from .jpeg import encode_jpeg as _encode_jpeg

_LOG = logging.getLogger(__name__)

# DreamZero's context window: 8 latent frames (4x2) == 33 raw frames at 5 FPS
# == 6.6 s. Both are per-model and are passed in by the caller; these are the
# defaults for the only family we serve today.
DEFAULT_CAPACITY = 33
DEFAULT_TARGET_FPS = 5.0

# Context frames exist to give the model temporal grounding, not detail — the
# sidecar downsamples hard anyway (DreamZero works at 320x176). Capping the
# long edge keeps a 33-frame burst inside a sane payload; at 3 cameras and
# q80 this lands around 250-400 KB per drain.
DEFAULT_MAX_DIM = 320
DEFAULT_QUALITY = 80

#: Tolerance when testing a tick against the sampling grid. A caller computing
#: its clock one way and the grid computing it another can disagree by an ULP
#: or two; without slack that lands as a dropped sample every few seconds.
#: Small enough to be meaningless against any real control period.
_GRID_EPS_S = 1e-6

_WARNED_NO_FRAMES = False


class ContextRing:
    """A fixed-capacity ring of recent, decimated, JPEG-encoded camera frames.

    The ring always holds the *most recent* ``capacity`` samples. When more
    frames are produced between two drains than the ring can hold — a long
    teleop intervention, say — the oldest are dropped and the drain returns
    the tail, which is the correct behaviour: the model wants the last 6.6 s,
    not the first. The monotonic sequence number lets the server see that a
    gap occurred rather than silently assuming contiguity.
    """

    def __init__(
        self,
        *,
        capacity: int = DEFAULT_CAPACITY,
        target_fps: float = DEFAULT_TARGET_FPS,
        quality: int = DEFAULT_QUALITY,
        max_dim: Optional[int] = DEFAULT_MAX_DIM,
    ) -> None:
        self.capacity = max(1, int(capacity))
        self.target_fps = float(target_fps) if target_fps and target_fps > 0 else 0.0
        self.quality = int(quality)
        self.max_dim = max_dim
        self._period = (1.0 / self.target_fps) if self.target_fps > 0 else 0.0

        self._frames: deque = deque(maxlen=self.capacity)
        self._seq = 0
        # Sampling grid, held as (origin, samples-taken) rather than a running
        # "next due" timestamp. Repeated ``+= period`` accumulates float error
        # — 0.2 added three times is 0.6000000000000001, which sorts *after* a
        # caller's exact 0.6 and silently drops that sample. Multiplying an
        # integer count against the origin keeps the grid exact.
        self._t0: Optional[float] = None
        self._taken = 0
        # Frames appended since the last drain. May exceed ``capacity``; the
        # drain reports the true count so a gap is visible server-side.
        self._since_drain = 0

    # -- lifecycle ------------------------------------------------------

    def reset(self) -> None:
        """Drop all context. Called when the episode's continuity breaks."""
        self._frames.clear()
        self._t0 = None
        self._taken = 0
        self._since_drain = 0
        # ``_seq`` deliberately keeps counting: a monotonic sequence across a
        # reset lets the server distinguish "context was cleared" from
        # "frames were lost", which are different recovery paths.

    # -- producer side (control loop, every tick) -----------------------

    def offer(self, obs: Any, *, now: float) -> None:
        """Sample ``obs`` into the ring if the decimation interval has elapsed.

        Called on every control tick regardless of what the arm is doing.
        Cheap on the ticks it skips, which is most of them (30 Hz loop, 5 Hz
        ring => ~5 of every 6 ticks return immediately).
        """
        try:
            if self._period > 0.0:
                if self._t0 is None:
                    self._t0 = now
                due = self._t0 + self._taken * self._period
                if now + _GRID_EPS_S < due:
                    return
                self._taken += 1
                # A loop that stalled for many periods should resume sampling,
                # not fire a catch-up burst of near-identical frames — re-base
                # the grid on the tick that resumed it.
                if now - due > self._period:
                    self._t0 = now
                    self._taken = 1

            frames = self._encode_cameras(obs)
            if not frames:
                return
            self._frames.append((self._seq, frames))
            self._seq += 1
            self._since_drain += 1
        except Exception:
            # A context miss degrades the policy; it must never stop the robot.
            _LOG.debug("context ring offer failed", exc_info=True)

    def _encode_cameras(self, obs: Any) -> dict:
        """JPEG-encode every camera in ``obs``, keyed by short camera name.

        Uses the same image-detection rule as ``_to_policy_schema`` and
        ``_capture_tick`` (uint8, ndim >= 2, coerce-then-detect) so the
        context, the served observation and the recording all agree on what
        counts as a camera frame.
        """
        global _WARNED_NO_FRAMES
        out: dict = {}
        saw_camera = False
        for key, value in obs.items():
            if key == "task":
                continue
            try:
                arr = value if isinstance(value, np.ndarray) else np.asarray(value)
            except Exception:
                continue
            if arr.dtype != np.uint8 or arr.ndim < 2:
                continue
            saw_camera = True
            data = _encode_jpeg(arr, quality=self.quality, max_dim=self.max_dim)
            if data:
                out[key.rsplit(".", 1)[-1]] = data
        if saw_camera and not out and not _WARNED_NO_FRAMES:
            _LOG.warning(
                "context ring: observation had camera arrays but none encoded — "
                "the world model will run without video context and its actions "
                "will degrade. Install PyTurboJPEG, opencv-python, or Pillow."
            )
            _WARNED_NO_FRAMES = True
        return out

    # -- consumer side (payload encode, on send ticks) ------------------

    def drain(self) -> Optional[dict]:
        """Return the **whole current context window**, or None if empty.

        Shape::

            {"first_seq": int, "produced": int, "fps": float,
             "frames": [{"<cam>": <jpeg bytes>}, ...]}

        Note this ships the full ring on every send, not the delta since the
        last one. That roughly doubles the payload (~1 MB per ``Infer`` at 3
        cameras, versus ~450 KB for a delta at a 3 s cadence) and buys three
        things back:

        * **No refill protocol.** The server dedupes against its own last-seen
          sequence and VAE-encodes only the frames it has not seen, so sending
          more than it needs costs bandwidth, never correctness.
        * **A dropped ``Infer`` self-heals.** The next one carries everything
          the lost one would have, with no round trip and no gap bookkeeping.
        * **Session resumption is free.** A superseded session (ADR 0024)
          rebuilds its context from the first observation it receives.

        ``produced`` is the number of frames sampled since the last drain. It
        exceeds ``len(frames)`` exactly when the ring overflowed between sends
        — i.e. the window is a *tail*, and some intermediate motion was never
        shown to the model. The server logs that rather than assuming
        contiguity.
        """
        try:
            if not self._frames:
                return None
            items = list(self._frames)
            produced = self._since_drain
            self._since_drain = 0
            return {
                "first_seq": int(items[0][0]),
                "produced": int(produced),
                "fps": self.target_fps,
                "frames": [f for _, f in items],
            }
        except Exception:
            _LOG.debug("context ring drain failed", exc_info=True)
            return None

    # -- introspection --------------------------------------------------

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def seq(self) -> int:
        return self._seq
