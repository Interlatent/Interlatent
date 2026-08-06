"""Unit tests for the dimos adapter's kind declarations, config, and profile.

No dimos installed: everything here must import without the ``[dimos]`` extra
(the config/kinds modules are deliberately dimos-import-free).
"""
from __future__ import annotations

import pytest

from interlatent.adapters.dimos.config import build_adapter_config
from interlatent.adapters.dimos.kinds import (
    A1Z,
    XARM7,
    DimosKind,
    feature_key_for,
    get_kind,
)
from interlatent.node.teleop.robot_profile import get_profile


# ---------------------------------------------------------------------------
# kinds
# ---------------------------------------------------------------------------


def test_xarm7_kind_feature_keys_order_and_shape():
    assert XARM7.dimos_arm_joints == tuple(f"arm/joint{i}" for i in range(1, 8))
    assert XARM7.dimos_gripper_joint == "arm/gripper"
    assert XARM7.feature_keys == tuple(
        f"arm_joint{i}.pos" for i in range(1, 8)
    ) + ("arm_gripper.pos",)
    # Gripper is LAST (yam/nori precedent) — policies bind to order.
    assert XARM7.feature_keys[-1] == "arm_gripper.pos"


def test_xarm6_kind_feature_keys_order_and_shape():
    xarm6 = get_kind("xarm6")
    assert xarm6.dimos_arm_joints == tuple(f"arm/joint{i}" for i in range(1, 7))
    assert xarm6.dimos_gripper_joint == "arm/gripper"
    assert xarm6.feature_keys == tuple(
        f"arm_joint{i}.pos" for i in range(1, 7)
    ) + ("arm_gripper.pos",)
    assert xarm6.feature_keys[-1] == "arm_gripper.pos"
    assert xarm6.profile_name == "dimos_xarm6"


def test_a1z_kind_feature_keys_order_and_shape():
    assert A1Z.dimos_arm_joints == tuple(f"arm/joint{i}" for i in range(1, 7))
    assert A1Z.dimos_gripper_joint == "arm/gripper"
    assert A1Z.feature_keys == tuple(
        f"arm_joint{i}.pos" for i in range(1, 7)
    ) + ("arm_gripper.pos",)
    assert A1Z.feature_keys[-1] == "arm_gripper.pos"


def test_name_map_is_bijective_and_inverts():
    for kind in (XARM7, get_kind("xarm6"), A1Z):
        for dimos_name, feature in zip(kind.dimos_joint_names, kind.feature_keys):
            assert feature_key_for(dimos_name) == feature
            assert kind.dimos_name_for(feature) == dimos_name


def test_dimos_name_for_unknown_feature_raises():
    with pytest.raises(KeyError, match="no joint for feature"):
        XARM7.dimos_name_for("elbow_flex.pos")


def test_colliding_joint_names_rejected():
    # 'arm/joint1' and 'arm_joint1' both map to 'arm_joint1.pos' — the
    # '/'->'_' map must stay bijective within a kind.
    with pytest.raises(ValueError, match="bijective"):
        DimosKind(
            name="bad",
            profile_name="bad",
            dimos_arm_joints=("arm/joint1", "arm_joint1"),
        )


def test_get_kind_unknown_lists_known():
    with pytest.raises(ValueError, match="known kinds.*xarm7"):
        get_kind("go1")


def test_get_kind_resolves_a1z():
    assert get_kind("a1z") is A1Z


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def test_kind_is_required():
    with pytest.raises(ValueError, match="--robot-arg kind="):
        build_adapter_config({}, {})


def test_defaults():
    cfg = build_adapter_config({"kind": "xarm7"}, None)
    assert cfg.kind is XARM7
    assert cfg.transport is None
    assert cfg.joint_state_topic == "/coordinator_joint_state"
    assert cfg.joint_command_topic == "/joint_command"
    assert cfg.episode_topic == "/interlatent/episode"
    assert cfg.staleness_ms == 200.0
    assert cfg.max_step_rad == 0.05
    assert cfg.cameras == {}


def test_defaults_a1z():
    cfg = build_adapter_config({"kind": "a1z"}, None)
    assert cfg.kind is A1Z


def test_defaults_xarm6():
    cfg = build_adapter_config({"kind": "xarm6"}, None)
    assert cfg.kind is get_kind("xarm6")


def test_knobs_parse_and_cameras_map_to_topics():
    cfg = build_adapter_config(
        {
            "kind": "xarm7",
            "transport": "zenoh",
            "max_step_rad": "0.1",
            "staleness_ms": "100",
        },
        {"wrist": "/camera/wrist/color", "front": "/color_image"},
    )
    assert cfg.transport == "zenoh"
    assert cfg.max_step_rad == 0.1
    assert cfg.staleness_ms == 100.0
    assert cfg.cameras == {"wrist": "/camera/wrist/color", "front": "/color_image"}


def test_unknown_robot_arg_warns_but_is_ignored(caplog):
    with caplog.at_level("WARNING"):
        cfg = build_adapter_config({"kind": "xarm7", "bogus": "1"}, {})
    assert cfg.kind is XARM7
    assert "bogus" in caplog.text


def test_invalid_transport_rejected():
    with pytest.raises(ValueError, match="transport"):
        build_adapter_config({"kind": "xarm7", "transport": "ros2"}, {})


def test_no_verify_escape_hatch():
    # verify=false must NOT be a recognized knob (fail-closed by design);
    # it lands in the ignored-keys warning path, never on the config.
    cfg = build_adapter_config({"kind": "xarm7", "verify": "false"}, {})
    assert not hasattr(cfg, "verify")


# ---------------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------------


def test_dimos_xarm7_profile_registered_and_aligned_with_kind():
    profile = get_profile("dimos_xarm7")
    assert profile is not None
    # Profile order must equal the kind's feature order minus '.pos' —
    # the same check base.py enforces at action() time.
    assert list(profile.joint_names) == [
        k.removesuffix(".pos") for k in XARM7.feature_keys
    ]


def test_dimos_xarm7_limits_sanity():
    profile = get_profile("dimos_xarm7")
    assert profile is not None
    # J2/J4/J6 are NOT the +-2pi placeholder dimos ships.
    lims = dict(zip(profile.joint_names, profile.joint_limits))
    assert lims["arm_joint2"] == (-2.059, 2.0944)
    assert lims["arm_joint4"] == (-0.19198, 3.927)
    assert lims["arm_joint6"] == (-1.69297, 3.14159)
    # Rest pose inside limits (J6 rest is -0.7, within its range).
    for (lo, hi), rest in zip(profile.joint_limits, profile.rest_pose):
        assert lo <= rest <= hi
    # Velocity caps well below the 3.14 rad/s motor max for arm joints.
    assert all(v <= 1.5 for v in profile.max_velocity[:7])


def test_dimos_xarm6_profile_registered_and_aligned_with_kind():
    profile = get_profile("dimos_xarm6")
    assert profile is not None
    assert list(profile.joint_names) == [
        k.removesuffix(".pos") for k in get_kind("xarm6").feature_keys
    ]


def test_dimos_xarm6_limits_sanity():
    profile = get_profile("dimos_xarm6")
    assert profile is not None
    lims = dict(zip(profile.joint_names, profile.joint_limits))
    # Transcribed from dimos's bundled xarm6.urdf <limit> tags — J2/J3/J5 are
    # NOT the +-2pi placeholder dimos's XArmAdapter.get_limits() reports.
    assert lims["arm_joint2"] == (-2.059, 2.0944)
    assert lims["arm_joint3"] == (-3.927, 0.19198)
    assert lims["arm_joint5"] == (-1.69297, 3.141592653589793)
    # Gripper: same UFACTORY gripper as xarm7 — dimos "meters", not A1Z's
    # normalized fraction.
    assert lims["arm_gripper"] == (0.0, 0.85)
    for (lo, hi), rest in zip(profile.joint_limits, profile.rest_pose):
        assert lo <= rest <= hi
    # Velocity caps well below the URDF's 3.14 rad/s motor max on all 6 joints.
    assert all(v <= 1.5 for v in profile.max_velocity[:6])


def test_dimos_xarm6_is_not_a_truncated_xarm7():
    """xArm6 is a different wrist, not xArm7 minus a joint.

    The transcription trap this pins: xarm6's J3 is ``[-3.927, 0.19198]``, the
    exact NEGATIVE MIRROR of xarm7's J4 ``[-0.19198, 3.927]``. Copying the
    xarm7 profile and deleting a row (or keeping xarm7's sign) puts the whole
    usable elbow range outside the clamp — the SafetyGate would pin every
    command to ~0.19 rad and the arm could never leave its -0.87 rad rest
    pose, which reads on hardware as "the arm is dead", not as a limit bug.
    """
    six = get_profile("dimos_xarm6")
    seven = get_profile("dimos_xarm7")
    assert six is not None and seven is not None

    six_lims = dict(zip(six.joint_names, six.joint_limits))
    seven_lims = dict(zip(seven.joint_names, seven.joint_limits))

    lo, hi = six_lims["arm_joint3"]
    assert (lo, hi) == (-3.927, 0.19198)
    assert (lo, hi) != seven_lims["arm_joint4"], "J3 copied with xarm7's sign"
    assert hi < 0.2, "xarm6 J3 opens downward (elbow-up); a positive-major range is flipped"
    # And the rest pose actually exercises that negative range, so a flipped
    # sign cannot pass by sitting harmlessly at zero.
    rest = dict(zip(six.joint_names, six.rest_pose))
    assert rest["arm_joint3"] < 0
    assert lo <= rest["arm_joint3"] <= hi

    assert len(six.joint_names) == 7 and len(seven.joint_names) == 8


def test_dimos_a1z_profile_registered_and_aligned_with_kind():
    profile = get_profile("dimos_a1z")
    assert profile is not None
    assert list(profile.joint_names) == [
        k.removesuffix(".pos") for k in A1Z.feature_keys
    ]


def test_dimos_a1z_limits_sanity():
    profile = get_profile("dimos_a1z")
    assert profile is not None
    lims = dict(zip(profile.joint_names, profile.joint_limits))
    # Transcribed verbatim from dimos's A1Z URDF <limit> tags.
    assert lims["arm_joint1"] == (-2.094, 2.094)
    assert lims["arm_joint2"] == (0.0, 3.142)
    assert lims["arm_joint3"] == (-3.142, 0.0)
    assert lims["arm_joint4"] == (-1.309, 1.309)
    assert lims["arm_joint5"] == (-1.484, 1.484)
    assert lims["arm_joint6"] == (-2.007, 2.007)
    # Gripper is a normalized fraction (0 closed .. 1 open), not xarm7's meters.
    assert lims["arm_gripper"] == (0.0, 1.0)
    # Rest pose inside limits (J2/J3 rest exactly on their own mechanical stop).
    for (lo, hi), rest in zip(profile.joint_limits, profile.rest_pose):
        assert lo <= rest <= hi
    # Velocity caps well below both the URDF's velocity_limit (min 3.77 rad/s)
    # and the vendor adapter's own cap (min 7.0 rad/s) for the 6 arm joints.
    assert all(v <= 1.5 for v in profile.max_velocity[:6])
