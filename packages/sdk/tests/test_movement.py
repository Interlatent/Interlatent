"""Tests for the node movement arbiter (node/movement.py).

The Phase-1 CommandBus/Arbiter must reproduce the *exact* control-loop
decision it replaced. This exhaustively enumerates the inputs and checks the
arbitrated source against the original boolean cascade:

    engaged = frame and frame.engaged and frame.deadman
    teleop_ok = engaged and gate is not None and action_keys \\
                and len(action_keys) == len(profile.joint_names)
    -> TELEOP if teleop_ok
    -> HOLD   if (not teleop_ok) and (not policy_enabled)
    -> POLICY otherwise

Runs standalone (``python tests/test_movement.py``) or under pytest.
"""
from __future__ import annotations

import importlib.util
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path

# Load movement.py in isolation — it is stdlib-only by design, so we avoid
# importing the whole `interlatent.node` package (heavy deps, not Pi-safe here).
_MOD_PATH = (
    Path(__file__).resolve().parents[1]
    / "src" / "interlatent" / "node" / "movement.py"
)
_spec = importlib.util.spec_from_file_location("_il_movement", _MOD_PATH)
movement = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = movement  # so dataclass annotation resolution works
_spec.loader.exec_module(movement)  # type: ignore[union-attr]

CommandBus = movement.CommandBus
MovementSource = movement.MovementSource


@dataclass
class _FakeFrame:
    engaged: bool
    deadman: bool


@dataclass
class _FakeProfile:
    joint_names: tuple


class _FakeChannel:
    def __init__(self, frame):
        self._frame = frame

    def latest_frame(self):
        return self._frame


def _reference_decision(*, frame, gate, profile, action_keys, policy_enabled):
    """The original control-loop cascade, verbatim."""
    engaged = bool(frame and frame.engaged and frame.deadman)
    teleop_ok = (
        engaged
        and gate is not None
        and action_keys
        and len(action_keys) == len(profile.joint_names)
    )
    if teleop_ok:
        return MovementSource.TELEOP
    if not policy_enabled:
        return MovementSource.HOLD
    return MovementSource.POLICY


def test_arbiter_matches_original_cascade_exhaustively():
    profile = _FakeProfile(joint_names=("a", "b", "c"))  # arity 3
    gate = object()  # any non-None gate

    frame_opts = [
        None,
        _FakeFrame(engaged=True, deadman=True),
        _FakeFrame(engaged=True, deadman=False),
        _FakeFrame(engaged=False, deadman=True),
        _FakeFrame(engaged=False, deadman=False),
    ]
    gate_opts = [gate, None]
    profile_opts = [profile, None]
    action_key_opts = [["a", "b", "c"], ["a", "b"], []]  # match / mismatch / empty
    policy_opts = [True, False]

    checked = 0
    for frame, g, prof, akeys, policy_enabled in itertools.product(
        frame_opts, gate_opts, profile_opts, action_key_opts, policy_opts
    ):
        # The gate only exists when a profile exists (loop invariant); skip the
        # impossible gate-without-profile combos so the reference — which may
        # dereference profile.joint_names — matches the real world.
        if g is not None and prof is None:
            continue

        bus = CommandBus(
            teleop_channel=_FakeChannel(frame),
            teleop_gate=g,
            teleop_profile=prof,
            policy_enabled=policy_enabled,
        )
        got = bus.arbitrate(bus.sample_teleop(), akeys)
        want = _reference_decision(
            frame=frame, gate=g, profile=prof,
            action_keys=akeys, policy_enabled=policy_enabled,
        )
        assert got is want, (
            f"mismatch: frame={frame} gate={'set' if g else None} "
            f"profile={'set' if prof else None} action_keys={akeys} "
            f"policy_enabled={policy_enabled}: got {got} want {want}"
        )
        checked += 1
    assert checked > 0


def test_no_teleop_channel_is_policy_or_hold():
    bus = CommandBus(
        teleop_channel=None, teleop_gate=None,
        teleop_profile=None, policy_enabled=True,
    )
    assert bus.sample_teleop() is None
    assert bus.arbitrate(None, ["a"]) is MovementSource.POLICY

    bus_hold = CommandBus(
        teleop_channel=None, teleop_gate=None,
        teleop_profile=None, policy_enabled=False,
    )
    assert bus_hold.arbitrate(None, ["a"]) is MovementSource.HOLD


def test_source_values_match_legacy_labels():
    # The recorded control_source strings must not change.
    assert MovementSource.TELEOP.value == "teleop"
    assert MovementSource.HOLD.value == "hold"
    assert MovementSource.POLICY.value == "policy"


def _run_all() -> None:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  [PASS] {t.__name__}")
    print(f"All {len(tests)} tests passed.")


if __name__ == "__main__":
    _run_all()
