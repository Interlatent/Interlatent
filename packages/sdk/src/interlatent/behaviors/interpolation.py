"""Interpolation profiles between keyframes, plus trajectory sampling.

Everything here is a pure function of a normalized phase ``tau in [0, 1]`` and the two
bounding waypoints — no robot, no time, no units. A profile is a scalar shape
function ``s(tau)`` with ``s(0) == 0`` and ``s(1) == 1``; a segment's position is
``p0 + (p1 - p0) * s(tau)``.

Three profiles:

- **min_jerk** (default) — the quintic ``10τ³ − 15τ⁴ + 6τ⁵``. Zero velocity *and* zero
  acceleration at both endpoints, so keyframes are joined without a jerk spike. This
  is the minimum-jerk boundary condition tested in ``tests/test_behaviors.py``.
- **linear** — constant velocity, discontinuous at keyframes.
- **trapezoidal** — a trapezoidal velocity profile: accelerate for the first
  ``TRAPEZOID_BLEND`` of the segment, cruise, then decelerate. Zero velocity at the
  endpoints (acceleration is discontinuous — that is the defining difference from
  min-jerk).

Each profile also exposes a **peak-velocity factor**: ``max |s'(tau)|`` over the
segment. The planner uses it to check ``factor · |Δ| / T ≤ velocity_cap`` *before*
any motion, so a too-fast behavior (or ``speed`` scaling) is rejected up front rather
than silently clamped by the adapter.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

# Fraction of a trapezoidal segment spent accelerating (and, symmetrically,
# decelerating). Must be in (0, 0.5]; 0.25 gives a 25% ramp / 50% cruise / 25% ramp.
TRAPEZOID_BLEND: float = 0.25


def _min_jerk_s(tau: np.ndarray) -> np.ndarray:
    # 10τ³ − 15τ⁴ + 6τ⁵ = τ³(10 − 15τ + 6τ²)
    return tau ** 3 * (10.0 - 15.0 * tau + 6.0 * tau ** 2)


def _linear_s(tau: np.ndarray) -> np.ndarray:
    return tau


def _trapezoid_s(tau: np.ndarray, r: float = TRAPEZOID_BLEND) -> np.ndarray:
    """Normalized position of a trapezoidal velocity profile with ramp fraction ``r``.

    Peak (cruise) velocity is ``v = 1 / (1 - r)`` so the area under the velocity
    profile is 1 (the segment travels unit normalized distance).
    """
    tau = np.asarray(tau, dtype=np.float64)
    v = 1.0 / (1.0 - r)
    s = np.empty_like(tau)
    accel = tau < r
    decel = tau > (1.0 - r)
    cruise = ~(accel | decel)
    # Accelerate: s = v · τ² / (2r)
    s[accel] = v * tau[accel] ** 2 / (2.0 * r)
    # Cruise: s = v·r/2 + v·(τ − r)
    s[cruise] = v * r / 2.0 + v * (tau[cruise] - r)
    # Decelerate: mirror of the accel ramp about the segment midpoint.
    s[decel] = 1.0 - v * (1.0 - tau[decel]) ** 2 / (2.0 * r)
    return s


_SHAPES: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "min_jerk": _min_jerk_s,
    "linear": _linear_s,
    "trapezoidal": _trapezoid_s,
}

# max |s'(τ)| over [0, 1] for each profile.
#   min_jerk:    s'(τ) = 30τ²(1−τ)², peak at τ=0.5 → 30·0.0625 = 1.875
#   linear:      s'(τ) = 1
#   trapezoidal: peak = cruise velocity = 1/(1−r)
_PEAK_VELOCITY_FACTOR: dict[str, float] = {
    "min_jerk": 1.875,
    "linear": 1.0,
    "trapezoidal": 1.0 / (1.0 - TRAPEZOID_BLEND),
}


def shape(interpolation: str) -> Callable[[np.ndarray], np.ndarray]:
    """Return the scalar shape function ``s(tau)`` for a profile name."""
    try:
        return _SHAPES[interpolation]
    except KeyError:
        raise ValueError(
            f"unknown interpolation {interpolation!r}; expected one of {list(_SHAPES)}"
        ) from None


def peak_velocity_factor(interpolation: str) -> float:
    """Return ``max |s'(tau)|`` for a profile — the segment peak-velocity multiplier."""
    try:
        return _PEAK_VELOCITY_FACTOR[interpolation]
    except KeyError:
        raise ValueError(
            f"unknown interpolation {interpolation!r}; expected one of {list(_SHAPES)}"
        ) from None


def smoothstep(x: float) -> float:
    """Hermite smoothstep ``3x² − 2x³`` on ``[0, 1]`` (clamped). Used for stop ramps."""
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


def build_samples(
    waypoints: Sequence[np.ndarray],
    seg_durations: Sequence[float],
    interpolation: str,
    dt: float,
) -> np.ndarray:
    """Sample a multi-segment trajectory at a fixed control step.

    ``waypoints`` are ``m + 1`` joint vectors; ``seg_durations`` are the ``m`` segment
    lengths in seconds. Each segment is sampled with ``ceil(T / dt)`` steps, so the
    effective per-tick duration never exceeds ``dt`` — which, combined with the
    planner's ``peak_velocity_factor`` check, guarantees every per-tick delta stays
    within ``velocity_cap · dt``. Every waypoint (including keyframes) is hit exactly,
    since the last sample of each segment lands on ``tau == 1``.

    Returns an ``(N, n_joints)`` array whose first row is ``waypoints[0]``.
    """
    s = shape(interpolation)
    wps = [np.asarray(w, dtype=np.float64) for w in waypoints]
    out: list[np.ndarray] = [wps[0].copy()]
    for k, T in enumerate(seg_durations):
        p0, p1 = wps[k], wps[k + 1]
        # ceil so the effective per-tick duration never exceeds dt, but shave a tiny
        # epsilon first so an integer number of ticks (e.g. 0.6s / (1/30)) isn't
        # bumped to n+1 by float round-off (0.6*30 == 18.0000006).
        n = max(1, int(np.ceil(T / dt - 1e-6))) if dt > 0 else 1
        taus = np.arange(1, n + 1, dtype=np.float64) / n
        svals = s(taus)  # (n,)
        seg = p0[None, :] + (p1 - p0)[None, :] * svals[:, None]  # (n, n_joints)
        out.extend(seg)
    return np.asarray(out, dtype=np.float64)


__all__ = [
    "TRAPEZOID_BLEND",
    "shape",
    "peak_velocity_factor",
    "smoothstep",
    "build_samples",
]
