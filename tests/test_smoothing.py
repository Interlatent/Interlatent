"""ButterworthLowPass action smoother (interlatent.node.smoothing).

Verifies the 2nd-order Butterworth low-pass damps high-frequency volatility while
passing low-frequency motion, warm-starts (no zero-ramp transient), and resets.
"""
from __future__ import annotations

import numpy as np
import pytest

from interlatent.node.smoothing import ButterworthLowPass


def test_warm_start_returns_first_sample_unchanged():
    f = ButterworthLowPass(cutoff_hz=3.0, sample_hz=30.0)
    x0 = np.array([10.0, -5.0, 50.0], dtype=np.float32)
    out = f.filter(x0)
    # No ramp from zero: the very first output equals the input pose.
    assert np.allclose(out, x0)


def test_dc_gain_is_unity():
    # A constant input must converge to that constant (low-pass passes DC).
    f = ButterworthLowPass(cutoff_hz=3.0, sample_hz=30.0)
    const = np.array([7.0, 7.0], dtype=np.float32)
    for _ in range(50):
        out = f.filter(const)
    assert np.allclose(out, const, atol=1e-4)


def test_attenuates_high_frequency_jitter():
    # 3 Hz cutoff at 30 Hz: a fast (~15 Hz, Nyquist) per-tick oscillation should be
    # strongly attenuated, while the signal's mean is preserved.
    f = ButterworthLowPass(cutoff_hz=3.0, sample_hz=30.0)
    base = 20.0
    raw, smoothed = [], []
    for n in range(400):
        jitter = 5.0 * ((-1.0) ** n)  # alternating ±5 each tick = Nyquist
        x = np.array([base + jitter], dtype=np.float32)
        raw.append(x[0])
        smoothed.append(f.filter(x)[0])
    # Compare amplitude on the settled tail (skip warm-up).
    raw_amp = np.std(raw[100:])
    smooth_amp = np.std(smoothed[100:])
    assert smooth_amp < 0.1 * raw_amp, (raw_amp, smooth_amp)
    # Mean (the DC trajectory) survives.
    assert np.mean(smoothed[100:]) == pytest.approx(base, abs=0.5)


def test_passes_slow_motion_with_modest_lag():
    # A slow 0.5 Hz ramp (well below the 3 Hz cutoff) should pass with its
    # amplitude essentially intact.
    fs = 30.0
    f = ButterworthLowPass(cutoff_hz=3.0, sample_hz=fs)
    smoothed = []
    raw = []
    for n in range(300):
        t = n / fs
        x = np.array([30.0 * np.sin(2 * np.pi * 0.5 * t)], dtype=np.float32)
        raw.append(x[0])
        smoothed.append(f.filter(x)[0])
    raw_amp = np.max(np.abs(raw[60:]))
    smooth_amp = np.max(np.abs(smoothed[60:]))
    assert smooth_amp > 0.85 * raw_amp, (raw_amp, smooth_amp)


def test_reset_drops_state_and_rewarmstarts():
    f = ButterworthLowPass(cutoff_hz=3.0, sample_hz=30.0)
    for _ in range(20):
        f.filter(np.array([0.0], dtype=np.float32))
    f.reset()
    # After reset, a jump to a new pose warm-starts there (no slow drift from 0).
    out = f.filter(np.array([99.0], dtype=np.float32))
    assert out[0] == pytest.approx(99.0)


def test_cutoff_clamped_below_nyquist():
    # Asking for a cutoff at/above Nyquist is clamped, not an error.
    f = ButterworthLowPass(cutoff_hz=30.0, sample_hz=30.0)
    assert f.cutoff_hz < 15.0


def test_rejects_nonpositive_params():
    with pytest.raises(ValueError):
        ButterworthLowPass(cutoff_hz=0.0, sample_hz=30.0)
    with pytest.raises(ValueError):
        ButterworthLowPass(cutoff_hz=3.0, sample_hz=0.0)
