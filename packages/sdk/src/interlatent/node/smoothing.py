"""Client-side action smoothing: a low-frequency Butterworth low-pass filter.

The policy streams overlapping action chunks that DRTC LWW-merges into the
``ActionSchedule``; ``pop_next()`` returns one joint-target vector per control
tick. Chunk boundaries, model jitter, and the merge can make that per-tick stream
*volatile* — small high-frequency wobble on top of the intended motion. Running it
through a low-pass filter on the node, just before ``send_action``, attenuates that
wobble while preserving the low-frequency trajectory the policy actually intends.

This is a 2nd-order (biquad) Butterworth low-pass, designed by the bilinear
transform with frequency pre-warping, evaluated per joint with the Direct-Form-I
difference equation::

    y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]

Butterworth is chosen for its maximally-flat passband (no ripple on the slow motion
the policy intends) and a gentle, well-damped step response (no overshoot that would
fling a joint past its target). A 3 Hz cutoff at a 30 Hz control rate keeps deliberate
arm motion intact while killing per-tick jitter; the ~one-tick group delay it adds is
negligible next to inference + DRTC latency.

numpy-only — no scipy. The module must stay importable on a barebones Pi (it sits on
the ``node.control`` import path), so it computes its own biquad coefficients rather
than pulling in ``scipy.signal``.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

__all__ = ["ButterworthLowPass"]


def _butter2_lowpass_coeffs(cutoff_hz: float, sample_hz: float) -> tuple[
    tuple[float, float, float], tuple[float, float]
]:
    """2nd-order Butterworth low-pass biquad coefficients ``(b0,b1,b2),(a1,a2)``.

    Bilinear transform with pre-warping so the analog −3 dB point lands exactly on
    ``cutoff_hz`` at the given sample rate. ``a0`` is normalized to 1 and dropped.
    """
    # Pre-warp the cutoff so bilinear mapping preserves it.
    k = math.tan(math.pi * cutoff_hz / sample_hz)
    k2 = k * k
    sqrt2 = math.sqrt(2.0)
    a0 = 1.0 + sqrt2 * k + k2
    b0 = k2 / a0
    b1 = 2.0 * k2 / a0
    b2 = k2 / a0
    a1 = (2.0 * (k2 - 1.0)) / a0
    a2 = (1.0 - sqrt2 * k + k2) / a0
    return (b0, b1, b2), (a1, a2)


class ButterworthLowPass:
    """Vectorized 2nd-order Butterworth low-pass over a fixed-width joint vector.

    One filter instance smooths a whole action vector: each joint is filtered
    independently with the same coefficients but its own delay-line state. The
    cutoff and sample rate are fixed at construction (a Butterworth's coefficients
    are only valid for the rate they were designed at), so callers must feed it at
    a steady ``sample_hz`` — exactly the control loop's per-tick cadence.

    Warm-started: the first sample after construction or :meth:`reset` initializes
    the entire delay line to that sample, so the filter's output begins *at* the
    current pose instead of ramping up from zero. That matters because the output
    drives the arm — a cold (zero) start would command a slam toward the origin.
    """

    def __init__(self, cutoff_hz: float, sample_hz: float) -> None:
        if cutoff_hz <= 0.0 or sample_hz <= 0.0:
            raise ValueError(
                f"cutoff_hz and sample_hz must be positive, got "
                f"cutoff_hz={cutoff_hz}, sample_hz={sample_hz}"
            )
        # Nyquist guard: a digital low-pass cutoff must sit below half the sample
        # rate or the bilinear design is meaningless. Clamp just under Nyquist.
        nyq = 0.5 * sample_hz
        if cutoff_hz >= nyq:
            cutoff_hz = 0.99 * nyq
        self.cutoff_hz = cutoff_hz
        self.sample_hz = sample_hz
        (self._b0, self._b1, self._b2), (self._a1, self._a2) = _butter2_lowpass_coeffs(
            cutoff_hz, sample_hz
        )
        # Per-joint delay line (input x[n-1], x[n-2]; output y[n-1], y[n-2]).
        # Allocated lazily on the first sample once the width is known.
        self._x1: Optional[np.ndarray] = None
        self._x2: Optional[np.ndarray] = None
        self._y1: Optional[np.ndarray] = None
        self._y2: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Drop the delay line so the next sample warm-starts the filter again.

        Call when the action stream is discontinuous — e.g. after a teleop
        takeover flushes the policy buffer — so the filter doesn't carry stale
        pre-interruption state across the gap.
        """
        self._x1 = self._x2 = self._y1 = self._y2 = None

    def filter(self, x: np.ndarray) -> np.ndarray:
        """Filter one joint vector and return the smoothed vector (same shape).

        On the first call (or first after :meth:`reset`) the delay line is seeded
        from ``x`` and ``x`` is returned unchanged — a warm start, no transient.
        """
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if self._x1 is None or self._x1.shape != x.shape:
            # Warm start: seed the whole delay line with the current sample so the
            # steady-state output equals the input (no startup ramp toward zero).
            self._x1 = x.copy()
            self._x2 = x.copy()
            self._y1 = x.copy()
            self._y2 = x.copy()
            return x.copy()

        y = (
            self._b0 * x
            + self._b1 * self._x1
            + self._b2 * self._x2
            - self._a1 * self._y1
            - self._a2 * self._y2
        ).astype(np.float32)

        # Shift the delay line.
        self._x2 = self._x1
        self._x1 = x
        self._y2 = self._y1
        self._y1 = y
        return y
