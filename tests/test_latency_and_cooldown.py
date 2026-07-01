"""Latency estimator + inference cooldown counter."""
from interlatent.inference.client.cooldown import Cooldown
from interlatent.inference.client.latency import JacobsonKarels


def test_estimate_zero_before_first_sample():
    jk = JacobsonKarels()
    assert jk.estimate_s == 0.0


def test_first_sample_is_seed_capped():
    jk = JacobsonKarels()
    jk.observe(20.0)  # cold-connection path outlier
    assert jk.estimate_s <= jk.SEED_CAP_S * (1 + jk.k / 2) + 1e-9


def test_estimate_tracks_steady_state():
    jk = JacobsonKarels()
    for _ in range(200):
        jk.observe(0.1)
    assert abs(jk._srtt - 0.1) < 1e-3
    assert jk.estimate_s < 0.15


def test_estimate_decays_after_spike():
    jk = JacobsonKarels()
    for _ in range(50):
        jk.observe(0.1)
    jk.observe(2.0)  # one spike
    spiked = jk.estimate_s
    for _ in range(100):
        jk.observe(0.1)
    assert jk.estimate_s < spiked / 2  # decays back, unlike max-tracking


def test_estimate_steps_ceil():
    jk = JacobsonKarels()
    for _ in range(100):
        jk.observe(0.095)
    # ~0.095s estimate at 30 Hz (0.0333s period) -> ceil to >= 3 steps
    assert jk.estimate_steps(1 / 30) >= 3
    assert jk.estimate_steps(0) == 0


def test_cooldown_counts_down_then_ready():
    cd = Cooldown(epsilon=2)
    assert cd.ready()  # never armed
    cd.arm(latency_steps=3)
    assert cd.remaining == 5
    for _ in range(4):
        cd.tick()
        assert not cd.ready()
    cd.tick()
    assert cd.ready()


def test_cooldown_never_negative():
    cd = Cooldown()
    cd.tick()
    assert cd.remaining == 0
    assert cd.ready()


def test_cooldown_arm_is_at_least_one():
    cd = Cooldown(epsilon=1)
    cd.arm(latency_steps=0)
    assert cd.remaining >= 1
