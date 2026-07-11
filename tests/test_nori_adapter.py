"""Nori adapter (interlatent.adapters.nori) — no daemon, no sockets, no zmq.

Covers what must hold before the on-Pi daemon is ever dialed: the
profile/adapter joint-name contract `base.py` enforces, the pure
`_motor_targets` action seam (delta clamp, gripper exemption), the fail-closed
accumulated ack validation, and config/token parsing. Socket behavior lives in
test_nori_client.py; wire shapes in test_nori_protocol_conformance.py.
"""
from __future__ import annotations

import pytest

from interlatent.adapters.nori import protocol as _p
from interlatent.adapters.nori.config import build_adapter_config, resolve_token
from interlatent.adapters.nori.robot import NoriNativeRobot
from interlatent.node.teleop.robot_profile import get_profile

PROFILE = get_profile("nori")


def _adapter(**extra) -> NoriNativeRobot:
    return NoriNativeRobot(build_adapter_config(extra, None))


class _StubClient:
    """In-memory stand-in for NoriSessionClient (no sockets)."""

    def __init__(self, state=None, status=None):
        self._state = dict(state or {})
        self._status = status
        self.connected = True
        self.session_dead = False
        self.dead_reason = ""
        self.liveness_calls = 0
        self.sent: list[dict] = []

    def note_liveness(self):
        self.liveness_calls += 1

    def latest_state(self):
        return dict(self._state), 10.0

    def latest_status(self):
        return self._status

    def send_action(self, action):
        self.sent.append(dict(action))


# --------------------------------------------------------------------------- #
# Topology + the profile/adapter joint-name contract                           #
# --------------------------------------------------------------------------- #


def test_action_features_match_profile_order():
    robot = _adapter()
    assert len(robot.action_features) == 12
    # The base.py:166-171 invariant: adapter feature order == profile order.
    assert [f"{n}.pos" for n in PROFILE.joint_names] == robot.action_features
    assert robot.action_features[0] == "left_arm_shoulder_pan.pos"
    assert robot.action_features[5] == "left_arm_gripper.pos"
    assert robot.action_features[-1] == "right_arm_gripper.pos"


def test_joint_specs_split_position_and_gripper():
    robot = _adapter()
    specs = robot.joint_specs
    assert len(specs) == len(robot.action_features)
    for spec, key in zip(specs, robot.action_features):
        assert f"{spec.name}.pos" == key
        if key.endswith("_gripper.pos"):
            assert spec.control_mode == "gripper"
        else:
            assert spec.control_mode == "position"
            assert spec.settle_tolerance == pytest.approx(2.0)


def test_profile_limits_are_normalized_units():
    for name, (lo, hi) in zip(PROFILE.joint_names, PROFILE.joint_limits):
        if name.endswith("gripper"):
            assert (lo, hi) == (0.0, 100.0)
        else:
            assert (lo, hi) == (-100.0, 100.0)


# --------------------------------------------------------------------------- #
# _motor_targets: the pure action-writing seam                                 #
# --------------------------------------------------------------------------- #


def _full_action(value: float = 0.0, **overrides) -> dict:
    action = {k: value for k in _adapter().action_features}
    action.update(overrides)
    return action


def test_delta_clamp_arm_clamped_gripper_exempt():
    robot = _adapter(max_step="3.0")
    robot._motor_targets(_full_action(0.0))  # anchor last-accepted at zeros
    out = robot._motor_targets(
        _full_action(
            0.0,
            **{"left_arm_elbow_flex.pos": 50.0, "right_arm_gripper.pos": 100.0},
        )
    )
    # Arm joint advances by at most one max_step toward the target...
    assert out["left_arm_elbow_flex.pos"] == pytest.approx(3.0)
    # ...while the gripper passes through unclamped (mirrors YAM).
    assert out["right_arm_gripper.pos"] == pytest.approx(100.0)


def test_delta_clamp_measures_from_last_accepted_not_target():
    robot = _adapter(max_step="3.0")
    robot._motor_targets(_full_action(0.0))
    for expected in (3.0, 6.0, 9.0):
        out = robot._motor_targets(
            _full_action(0.0, **{"left_arm_shoulder_pan.pos": 50.0})
        )
        assert out["left_arm_shoulder_pan.pos"] == pytest.approx(expected)


def test_delta_clamp_disabled_with_inf():
    robot = _adapter(max_step="inf")
    robot._motor_targets(_full_action(0.0))
    out = robot._motor_targets(_full_action(0.0, **{"left_arm_wrist_flex.pos": 90.0}))
    assert out["left_arm_wrist_flex.pos"] == pytest.approx(90.0)


def test_send_action_puts_targets_on_the_wire():
    robot = _adapter()
    stub = _StubClient()
    robot._client = stub
    robot.send_action(_full_action(1.25))
    assert len(stub.sent) == 1
    assert set(stub.sent[0]) == set(robot.action_features)
    assert stub.sent[0]["right_arm_wrist_roll.pos"] == pytest.approx(1.25)


def test_send_action_missing_joint_raises():
    robot = _adapter()
    robot._client = _StubClient()
    action = _full_action(0.0)
    del action["left_arm_gripper.pos"]
    with pytest.raises(KeyError):
        robot.send_action(action)


# --------------------------------------------------------------------------- #
# Observation mapping + liveness proof (stub client, no sockets)               #
# --------------------------------------------------------------------------- #


def test_get_observation_maps_state_and_notes_liveness():
    robot = _adapter()
    stub = _StubClient(
        state={"left_arm_shoulder_pan.pos": 12.5, "right_arm_gripper.pos": 40.0},
        status={"safety": "ok", "watchdog": "ok", "link": "lan"},
    )
    robot._client = stub
    obs = robot.get_observation()
    assert stub.liveness_calls == 1, "get_observation must feed the pump gate"
    assert obs["left_arm_shoulder_pan.pos"] == pytest.approx(12.5)
    assert obs["right_arm_gripper.pos"] == pytest.approx(40.0)
    # Unreported joints default to 0.0; the dict carries floats only.
    assert obs["left_arm_elbow_flex.pos"] == 0.0
    assert set(obs) == set(robot.action_features)
    # Safety status stays OUT of the observation dict (capture-safe)...
    assert "safety" not in obs
    # ...and is disclosed via the property instead.
    assert robot.last_status["safety"] == "ok"


class _WarmupCam:
    """Camera stub: fails `fail_reads` times, then serves frames."""

    def __init__(self, fail_reads: int, host="127.0.0.1", port=5555):
        self._fails = fail_reads
        from interlatent.adapters.nori.cameras import NoriCameraSpec

        self.spec = NoriCameraSpec(
            obs_key="cam", daemon_name="cam", index=0, host=host, port=port
        )

    def read(self):
        if self._fails > 0:
            self._fails -= 1
            raise RuntimeError("no frame yet")
        return object()

    def disconnect(self):
        pass


def test_camera_warmup_tolerates_slow_publisher():
    robot = _adapter(camera_warmup_s="5.0")
    robot._cameras = {"cam": _WarmupCam(fail_reads=3)}
    robot._warm_cameras()  # must not raise: publisher came up on the 4th poll


def test_camera_warmup_fails_actionably_on_dead_publisher():
    robot = _adapter(camera_warmup_s="0.5")
    robot._cameras = {"cam": _WarmupCam(fail_reads=10_000)}
    with pytest.raises(RuntimeError, match="capture bridge.*5555|5555.*capture bridge"):
        robot._warm_cameras()


def test_camera_warmup_disabled_by_zero():
    robot = _adapter(camera_warmup_s="0")
    robot._cameras = {"cam": _WarmupCam(fail_reads=10_000)}
    robot._warm_cameras()  # disabled: returns immediately


# --------------------------------------------------------------------------- #
# validate_ack accumulation (pure — the socketed twin lives in test_nori_client)
# --------------------------------------------------------------------------- #


def test_validate_ack_clean():
    ack = _p.Ack(
        accepted=True,
        protocol_version=1,
        norm_mode="range_m100_100",
        joints=tuple(f"{n}.pos" for n in PROFILE.joint_names),
        ranges={
            f"{n}.pos": (float(lo), float(hi))
            for n, (lo, hi) in zip(PROFILE.joint_names, PROFILE.joint_limits)
        },
    )
    assert _p.validate_ack(PROFILE, ack) == []


def test_validate_ack_descriptorless_fallback_clean():
    # Replays the shape a real daemon build sent on 2026-07-10: full
    # initial_state + norm_mode + watchdog, but NO descriptor block. Topology
    # comes from initial_state; ranges are pinned by the units contract.
    ack = _p.Ack(
        accepted=True,
        protocol_version=1,
        norm_mode="range_m100_100",
        watchdog=_p.WatchdogProfile(t_warn_ms=150.0, t_stop_ms=500.0),
        initial_state={f"{n}.pos": 0.5 for n in PROFILE.joint_names}
        | {"x.vel": 0.0, "theta.vel": 0.0},  # base keys must not confuse it
    )
    assert _p.validate_ack(PROFILE, ack) == []


def test_validate_ack_descriptorless_still_fails_closed_on_topology():
    state = {f"{n}.pos": 0.0 for n in PROFILE.joint_names if n != "left_arm_gripper"}
    state["tail_motor.pos"] = 1.0
    ack = _p.Ack(
        accepted=True, protocol_version=1, norm_mode="range_m100_100",
        initial_state=state,
    )
    problems = _p.validate_ack(PROFILE, ack)
    joined = "\n".join(problems)
    assert len(problems) == 2, joined
    assert "left_arm_gripper.pos" in joined and "tail_motor.pos" in joined
    assert "initial_state (descriptor absent)" in joined


def test_validate_ack_no_disclosure_at_all_fails():
    ack = _p.Ack(accepted=True, protocol_version=1, norm_mode="range_m100_100")
    problems = _p.validate_ack(PROFILE, ack)
    assert len(problems) == 1 and "nothing to validate topology" in problems[0]


def test_validate_ack_descriptorless_wrong_norm_mode_fails():
    ack = _p.Ack(
        accepted=True, protocol_version=1, norm_mode="degrees",
        initial_state={f"{n}.pos": 0.0 for n in PROFILE.joint_names},
    )
    problems = _p.validate_ack(PROFILE, ack)
    assert any("norm_mode" in p for p in problems)


def test_validate_ack_accumulates_every_problem():
    ack = _p.Ack(
        accepted=True,
        protocol_version=2,  # wrong version
        norm_mode="degrees",  # wrong units
        joints=tuple(
            f"{n}.pos" for n in PROFILE.joint_names if n != "left_arm_gripper"
        )
        + ("tail_motor.pos",),  # one missing + one alien
        ranges={
            f"{n}.pos": (float(lo), float(hi))
            for n, (lo, hi) in zip(PROFILE.joint_names, PROFILE.joint_limits)
            if n not in ("left_arm_gripper", "right_arm_wrist_roll")
        }
        | {"right_arm_gripper.pos": (0.0, 42.0)},  # one wrong range
    )
    problems = _p.validate_ack(PROFILE, ack)
    joined = "\n".join(problems)
    assert len(problems) == 6, joined
    assert "protocol_version" in joined and "norm_mode" in joined
    assert "left_arm_gripper.pos" in joined and "tail_motor.pos" in joined
    assert "range mismatch for right_arm_gripper.pos" in joined
    assert "range undisclosed by daemon for right_arm_wrist_roll.pos" in joined


# --------------------------------------------------------------------------- #
# Config / token parsing                                                       #
# --------------------------------------------------------------------------- #


def test_build_adapter_config_defaults_and_overrides(caplog):
    cfg = build_adapter_config(
        {"host": "10.0.0.5", "port": "7000", "max_step": "1.5", "bogus_key": "x"},
        {"top": "front"},
    )
    assert cfg.host == "10.0.0.5" and cfg.port == 7000
    assert cfg.max_step == pytest.approx(1.5)
    assert cfg.cameras == {"top": "front"}
    assert cfg.camera_host == "10.0.0.5"  # cam_host defaults to host
    assert any("bogus_key" in r.message for r in caplog.records)


def test_resolve_token_precedence(tmp_path):
    token_file = tmp_path / "agent.token"
    token_file.write_text("file-token\n")
    explicit = build_adapter_config(
        {"token": "explicit", "token_path": str(token_file)}, None
    )
    from_file = build_adapter_config({"token_path": str(token_file)}, None)
    absent = build_adapter_config(
        {"token_path": str(tmp_path / "missing")}, None
    )
    assert resolve_token(explicit) == "explicit"
    assert resolve_token(from_file) == "file-token"
    assert resolve_token(absent) is None


def test_reset_latch_requires_some_token(tmp_path):
    robot = NoriNativeRobot(
        build_adapter_config({"token_path": str(tmp_path / "missing")}, None)
    )
    robot._client = _StubClient()
    with pytest.raises(RuntimeError, match="token"):
        robot.reset_latch()
