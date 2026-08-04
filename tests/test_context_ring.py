"""World-model context ring + malformed-chunk rejection (ADR 0037).

Two node-side behaviours land together because they protect the same policy
family from opposite directions: the ring decides what a world-action model
*sees*, and the rejection guard decides what its output is allowed to *do*.

What is asserted here is the reasoning in ADR 0037, not the implementation's
current shape:

* The ring samples on a wall-clock grid, not per tick — a 30 Hz loop feeding a
  5 FPS window must decimate, or 33 frames span 1.1 s instead of 6.6 s.
* It keeps the **tail** when it overflows, and says so. A long teleop
  intervention produces far more frames than the window holds; the model wants
  the most recent 6.6 s, and the server needs to know contiguity was broken
  rather than assume it.
* A drain ships the **whole** window, not the delta. That is what removes the
  gap-refill protocol: an ``Infer`` that never arrives costs nothing.
* Context keys are namespaced away from observation images, because the
  server's ``_to_batch`` treats *any* uint8 array as a camera frame.
* A non-finite or wrong-width action is refused before ``send_action``. The
  delta clamp cannot do this: ``abs(delta) > max_step`` is False for NaN.
"""
from __future__ import annotations

import numpy as np
import pytest

from interlatent.node.context_ring import ContextRing


def _obs(seed: int = 0) -> dict:
    """A two-camera observation in the shape lerobot hands back."""
    return {
        "shoulder_pan.pos": 1.0 + seed,
        "gripper.pos": 0.5,
        "front": np.full((48, 64, 3), seed % 256, dtype=np.uint8),
        "wrist": np.full((48, 64, 3), (seed * 7) % 256, dtype=np.uint8),
    }


def _has_encoder() -> bool:
    from interlatent.node.jpeg import encode_jpeg

    return encode_jpeg(np.zeros((8, 8, 3), dtype=np.uint8)) is not None


needs_jpeg = pytest.mark.skipif(
    not _has_encoder(), reason="no JPEG encoder backend available"
)


# --- sampling cadence -------------------------------------------------


@needs_jpeg
def test_decimates_to_target_fps_not_per_tick():
    """A 30 Hz loop must produce ~5 samples/s, else the window spans 1/6 the time."""
    ring = ContextRing(capacity=33, target_fps=5.0)
    # Two seconds of a 30 Hz control loop.
    for i in range(60):
        ring.offer(_obs(i), now=i / 30.0)
    # 2s at 5 FPS -> ~10 frames (grid start is inclusive, so 10 or 11).
    assert 9 <= len(ring) <= 11, f"expected ~10 samples in 2s, got {len(ring)}"


@needs_jpeg
def test_sampling_grid_does_not_drift():
    """Fixed-grid advance, so jitter doesn't accumulate into a slow drift."""
    ring = ContextRing(capacity=100, target_fps=5.0)
    for i in range(300):  # 10 s at 30 Hz
        ring.offer(_obs(i), now=i / 30.0)
    assert 49 <= len(ring) <= 51, f"10s at 5 FPS should be ~50, got {len(ring)}"


@needs_jpeg
def test_stalled_loop_rebases_instead_of_bursting():
    """After a long stall the ring resumes sampling; it does not fire a catch-up burst."""
    ring = ContextRing(capacity=33, target_fps=5.0)
    ring.offer(_obs(0), now=0.0)
    # A 10-second gap (a wedged robot read, a blocked adapter).
    ring.offer(_obs(1), now=10.0)
    ring.offer(_obs(2), now=10.05)  # inside one period -> skipped
    assert len(ring) == 2, "a stall must not backfill the missed grid points"


# --- overflow / tail semantics ----------------------------------------


@needs_jpeg
def test_overflow_keeps_the_tail_and_reports_the_gap():
    """A long intervention: keep the most recent window, and say frames were lost."""
    ring = ContextRing(capacity=5, target_fps=5.0)
    for i in range(20):
        ring.offer(_obs(i), now=i / 5.0)

    out = ring.drain()
    assert out is not None
    assert len(out["frames"]) == 5, "ring must cap at capacity"
    assert out["produced"] == 20, "produced counts everything sampled since last drain"
    assert out["produced"] > len(out["frames"]), "the gap must be visible to the server"
    # The retained window is the most recent one.
    assert out["first_seq"] == 15


@needs_jpeg
def test_drain_ships_whole_window_not_delta():
    """Full-window sends are what make a dropped Infer self-healing."""
    ring = ContextRing(capacity=10, target_fps=5.0)
    for i in range(6):
        ring.offer(_obs(i), now=i / 5.0)
    first = ring.drain()
    assert first is not None and len(first["frames"]) == 6

    # One more sample, then drain again: the window still carries everything,
    # so a consumer that missed the first drain loses nothing.
    ring.offer(_obs(6), now=6 / 5.0)
    second = ring.drain()
    assert second is not None
    assert len(second["frames"]) == 7, "drain is a window, not a delta"
    assert second["produced"] == 1, "produced is per-drain, unlike the window"


@needs_jpeg
def test_reset_clears_frames_but_keeps_sequence_monotonic():
    """A reset must be distinguishable from frame loss, so seq keeps counting."""
    ring = ContextRing(capacity=10, target_fps=5.0)
    for i in range(4):
        ring.offer(_obs(i), now=i / 5.0)
    seq_before = ring.seq
    ring.reset()
    assert len(ring) == 0
    assert ring.drain() is None
    ring.offer(_obs(99), now=100.0)
    assert ring.seq == seq_before + 1, "sequence must not restart at 0 on reset"


# --- robustness -------------------------------------------------------


@needs_jpeg
def test_offer_never_raises_on_a_bad_observation():
    """A context miss degrades the policy; it must never stop the robot."""
    ring = ContextRing(capacity=5, target_fps=0.0)  # no decimation

    class Hostile:
        def __array__(self, *a, **k):
            raise RuntimeError("boom")

    ring.offer({"cam": Hostile(), "j.pos": 1.0}, now=0.0)  # must not raise
    ring.offer({"task": "pick up the cube"}, now=0.1)
    assert len(ring) == 0, "nothing encodable -> nothing stored, but no crash"


@needs_jpeg
def test_task_string_is_not_treated_as_a_camera():
    ring = ContextRing(capacity=5, target_fps=0.0)
    obs = _obs(1)
    obs["task"] = "fold the towel"
    ring.offer(obs, now=0.0)
    out = ring.drain()
    assert out is not None
    assert set(out["frames"][0]) == {"front", "wrist"}


@needs_jpeg
def test_state_only_observation_yields_no_frames():
    ring = ContextRing(capacity=5, target_fps=0.0)
    ring.offer({"shoulder_pan.pos": 1.0, "gripper.pos": 0.0}, now=0.0)
    assert ring.drain() is None


# --- npz packing ------------------------------------------------------


@needs_jpeg
def test_context_packs_under_a_namespaced_prefix():
    """Context frames must not look like observation images to the server.

    ``_to_batch`` on the server treats *any* uint8 array as a camera frame, so
    an un-namespaced context frame would be fed to the policy as an extra
    input.
    """
    import io

    from interlatent.node.control import _encode_npz, _to_policy_schema

    ring = ContextRing(capacity=4, target_fps=0.0)
    for i in range(3):
        ring.offer(_obs(i), now=float(i))

    raw = _encode_npz(_to_policy_schema(_obs(0)), context=ring.drain())
    packed = np.load(io.BytesIO(raw))
    keys = set(packed.files)

    assert "ctx.meta" in keys
    meta = packed["ctx.meta"]
    assert meta.tolist() == [0, 3, 3, 0]  # first_seq, n_frames, produced, fps_milli

    ctx_keys = {k for k in keys if k.startswith("ctx.")}
    obs_keys = keys - ctx_keys
    # Every real observation key is untouched and no context frame leaked into it.
    assert "observation.state" in obs_keys
    assert any(k.startswith("observation.images.") for k in obs_keys)
    assert not any(k.startswith("observation.") for k in ctx_keys)
    # 3 frames x 2 cameras, plus the meta header.
    assert len(ctx_keys) == 3 * 2 + 1


def test_encode_npz_without_context_is_unchanged():
    """The context kwarg must be inert for every existing policy family."""
    import io

    from interlatent.node.control import _encode_npz

    obs = {"observation.state": np.array([1.0, 2.0], dtype=np.float32)}
    a = np.load(io.BytesIO(_encode_npz(dict(obs))))
    b = np.load(io.BytesIO(_encode_npz(dict(obs), context=None)))
    assert set(a.files) == set(b.files) == {"observation.state"}


# --- malformed-chunk rejection ---------------------------------------


def _bus(action_keys):
    """A CommandBus with only what _policy_action_ok touches."""
    from interlatent.node.movement import CommandBus

    return CommandBus(
        teleop_channel=None,
        teleop_gate=None,
        teleop_profile=None,
        policy_enabled=True,
        action_keys=action_keys,
    )


@pytest.mark.parametrize(
    "vector",
    [
        [1.0, float("nan"), 3.0],
        [1.0, float("inf"), 3.0],
        [float("-inf"), 2.0, 3.0],
    ],
)
def test_non_finite_action_is_rejected(vector):
    """The delta clamp cannot catch this: NaN compares False against any limit."""
    bus = _bus(["a.pos", "b.pos", "c.pos"])
    arr = np.asarray(vector, dtype=np.float32)
    assert bus._policy_action_ok(arr, step=0) is False


def test_wrong_width_action_is_rejected():
    """Otherwise the action is silently truncated and drives the wrong joints."""
    bus = _bus(["a.pos", "b.pos", "c.pos"])
    assert bus._policy_action_ok(np.zeros(2, dtype=np.float32), step=0) is False
    assert bus._policy_action_ok(np.zeros(4, dtype=np.float32), step=0) is False


def test_well_formed_action_passes():
    bus = _bus(["a.pos", "b.pos", "c.pos"])
    assert bus._policy_action_ok(np.zeros(3, dtype=np.float32), step=0) is True


def test_arity_check_is_skipped_when_action_keys_are_unknown():
    """A loop with no declared action keys must not start refusing everything."""
    bus = _bus([])
    assert bus._policy_action_ok(np.zeros(7, dtype=np.float32), step=0) is True
    # Finiteness still applies — it needs no schema.
    assert bus._policy_action_ok(
        np.asarray([float("nan")], dtype=np.float32), step=0
    ) is False


def test_rejection_logging_is_rate_limited():
    """Loud for the first few, then every hundredth: 30 Hz would flood the journal."""
    from interlatent.node.movement import _should_log

    assert [n for n in range(1, 8) if _should_log(n)] == [1, 2, 3]
    assert _should_log(100) and _should_log(200)
    assert not _should_log(101)
