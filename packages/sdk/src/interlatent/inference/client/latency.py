"""Jacobson-Karels latency estimator.

The DRTC paper notes that tracking max-latency-over-recent-samples
over-estimates after a transient spike and causes the execution
horizon to stay inflated for too long. The TCP RTT estimator
(Jacobson & Karels, 1988) instead tracks a smoothed mean and a
smoothed mean-deviation, so the estimate decays back toward the
underlying mean as variance subsides.

    srtt   = (1 - alpha) * srtt   + alpha * sample
    rttvar = (1 - beta)  * rttvar + beta  * |sample - srtt|
    estimate = srtt + K * rttvar

Defaults match the DRTC paper: alpha=1/8, beta=1/4, K=1.5. (TCP's
classic K is 4; DRTC tunes it down to 1.5 so the execution horizon
tracks latency more tightly.) We expose them so users can tune.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class JacobsonKarels:
    alpha: float = 1 / 8
    beta: float = 1 / 4
    k: float = 1.5
    _srtt: float = 0.0
    _rttvar: float = 0.0
    _initialized: bool = False

    # Cap for the very first RTT sample. The first Infer of a session
    # pays one-time costs the rest never do — gRPC HTTP/2 handshake over
    # a fresh Tailscale tunnel (userspace WireGuard + NAT discovery +
    # potential DERP-relay fallback) can take 10-20s on cold paths, and
    # MTU/path-MTU discovery on the first sizeable payload (JPEG frames)
    # adds more. Seeding ``srtt`` with that outlier inflates the cooldown
    # counter for seconds at a time while ``alpha=1/8`` slowly decays it,
    # which presents as the controller starving the action queue. The
    # cap is well above any plausible steady-state inference latency
    # (~150ms wire + ~50ms compute), so a real slow path still seeds
    # high enough to engage the cooldown — it just won't seed an
    # impossible 20s.
    SEED_CAP_S: float = 1.0

    def observe(self, sample_s: float) -> None:
        if not self._initialized:
            sample_s = min(sample_s, self.SEED_CAP_S)
            self._srtt = sample_s
            self._rttvar = sample_s / 2
            self._initialized = True
            return
        err = sample_s - self._srtt
        self._srtt += self.alpha * err
        self._rttvar += self.beta * (abs(err) - self._rttvar)

    @property
    def estimate_s(self) -> float:
        if not self._initialized:
            return 0.0
        return self._srtt + self.k * self._rttvar

    def estimate_steps(self, control_period_s: float) -> int:
        """How many control steps of latency we expect, rounded up.

        Used to set the DRTC execution horizon
        s = max(s_min, ceil(estimate / control_period))."""
        if control_period_s <= 0:
            return 0
        import math
        return max(0, math.ceil(self.estimate_s / control_period_s))
