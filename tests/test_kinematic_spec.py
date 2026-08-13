"""Kinematic-spec exporter + ik_config parsing tests.

Covers the maintainer tooling under ``packaging/`` that turns a robot bundle's
URDF + ``ik_config.json`` into the ``kinematic_spec.json`` the in-browser
solver walks. Runs against synthetic URDF fixtures written to tmp dirs — no
external robot assets — and is skipped wholesale when mujoco isn't installed,
which the exporter needs to drive the kinematic model. CI does not install it,
so this file is a local/maintainer check by design.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest

pytest.importorskip("mujoco")

# The exporter is a maintainer tool, not a package: it lives beside
# verify_urdf.py under packaging/ and is run as `python packaging/
# kinematic_spec.py <bundle>`, which puts that directory on sys.path itself.
# Importing it here has to do the same thing by hand.
_PACKAGING = Path(__file__).resolve().parent.parent / "packaging"
if str(_PACKAGING) not in sys.path:
    sys.path.insert(0, str(_PACKAGING))

# Imports intentionally follow importorskip + the path insert so the module is
# collectable without mujoco; noqa the resulting E402.
from ik_config import (  # noqa: E402
    IkConfig,
    parse_ik_config,
    parse_ik_config_bundle,
)
from kinematic_spec import (  # noqa: E402
    export_kinematic_spec,
    export_kinematic_spec_bundle,
    write_kinematic_spec,
)


def _link(name: str) -> str:
    return f"""
  <link name="{name}">
    <inertial>
      <mass value="0.2"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="1e-3" iyy="1e-3" izz="1e-3" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>"""


def _joint(name, parent, child, axis, xyz, lo, hi) -> str:
    return f"""
  <joint name="{name}" type="revolute">
    <parent link="{parent}"/>
    <child link="{child}"/>
    <origin xyz="{xyz}" rpy="0 0 0"/>
    <axis xyz="{axis}"/>
    <limit lower="{lo}" upper="{hi}" effort="10" velocity="10"/>
  </joint>"""


# 5-DOF SO101-like: yaw + three pitch + wrist roll, plus a gripper joint
# that is deliberately NOT an IK joint (exercises name-based indexing).
URDF_5DOF = f"""<?xml version="1.0"?>
<robot name="test5dof">
{_link("base_link")}{_link("l1")}{_link("l2")}{_link("l3")}{_link("l4")}{_link("l5")}{_link("jaw")}
{_joint("shoulder_pan", "base_link", "l1", "0 0 1", "0 0 0.05", -3.1, 3.1)}
{_joint("shoulder_lift", "l1", "l2", "0 1 0", "0 0 0.05", -1.9, 1.9)}
{_joint("elbow_flex", "l2", "l3", "0 1 0", "0.20 0 0", -1.9, 1.9)}
{_joint("wrist_flex", "l3", "l4", "0 1 0", "0.15 0 0", -1.7, 1.7)}
{_joint("wrist_roll", "l4", "l5", "1 0 0", "0.04 0 0", -3.1, 3.1)}
{_joint("gripper", "l5", "jaw", "0 1 0", "0.05 0 0", -1.0, 1.0)}
</robot>
"""

# A 6-DOF (decoupled) config document — used only for parse-time validation
# (its wrist-damping keys are known only to the 6-DoF solver). No URDF needed:
# parse_ik_config validates the document structure, it does not load the model.
CFG_6DOF = {
    "solver_type": "decoupled_6dof",
    "urdf": "robot.urdf",
    "urdf_joint_names": ["j1", "j2", "j3", "j4", "j5", "j6"],
    "ee_body": "l6",
    "tool0_offset_xyz": [0.10, 0.0, 0.0],
    "tool0_offset_rpy": [0.0, 0.0, 0.0],
    "anchor_body": "l3",
    "anchor_offset_xyz": [0.15, 0.0, 0.0],
    "q_rest": [0.0, 0.5, 0.8, 0.0, 0.0, 0.0],
    "ik_joint_action_indices": [0, 1, 2, 3, 4, 5],
    "gripper_action_index": None,
}

CFG_5DOF = {
    "solver_type": "so101_5dof",
    "urdf": "robot.urdf",
    "urdf_joint_names": [
        "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll",
    ],
    "ee_body": "l5",
    "tool0_offset_xyz": [0.08, 0.0, 0.0],
    "tool0_offset_rpy": [0.0, 0.0, 0.0],
    "q_rest": [0.0, 0.4, 0.6, 0.2, 0.0],
    "ik_joint_action_indices": [0, 1, 2, 3, 4],
    "gripper_action_index": 5,
    # SO101 speaks degrees on the wire; gripper 0..100 (100 open).
    "robot_units_to_rad": [{"scale": 0.017453292519943295, "offset": 0.0}] * 5,
    "gripper_range": [100.0, 0.0],
}


def _write_bundle(tmp_path, urdf: str, cfg: dict):
    (tmp_path / "robot.urdf").write_text(urdf)
    (tmp_path / "ik_config.json").write_text(json.dumps(cfg))
    return tmp_path


class TestDampingKeyValidation:
    """A damping key no solver reads is a typo, not a knob.

    The live ``nori`` bundle shipped ``{"default": 0.1}`` on both chains and
    ran stock damping while looking tuned, because every reader silently
    ignored it. Strict at curation, fail-open at runtime.
    """

    def test_unknown_key_raises_when_strict(self):
        with pytest.raises(ValueError, match="default"):
            parse_ik_config({**CFG_5DOF, "damping": {"default": 0.1}}, strict=True)

    def test_unknown_key_only_warns_by_default(self, caplog):
        # A bundle that already shipped must still export: refusing it here
        # means no teleop at all, over a cosmetic typo.
        with caplog.at_level(logging.WARNING):
            cfg = parse_ik_config({**CFG_5DOF, "damping": {"default": 0.1}})
        assert cfg.damping == {"default": 0.1}
        assert "default" in caplog.text

    def test_known_keys_accepted_strict(self):
        cfg = parse_ik_config(
            {**CFG_5DOF, "damping": {"lam_pos": 0.1, "w_rot": 0.3}}, strict=True
        )
        assert cfg.damping["lam_pos"] == 0.1

    def test_wrist_keys_are_known_only_to_the_decoupled_solver(self):
        # yam's live bundles tune lam0_rot/w0_rot; they're real for the 6-DoF
        # decoupled solver and meaningless for the 5-DoF one.
        parse_ik_config({**CFG_6DOF, "damping": {"lam0_rot": 0.4}}, strict=True)
        with pytest.raises(ValueError, match="lam0_rot"):
            parse_ik_config({**CFG_5DOF, "damping": {"lam0_rot": 0.4}}, strict=True)

    def test_weighted_dls_reads_the_same_knobs_as_so101_5dof(self):
        # xarm6/xarm7 name the browser's generic solver by what it is. Same
        # solver, so the same knob set — and the same rejection of the
        # decoupled solver's wrist knobs (which both bundles do carry, copied
        # from yam; see interlatent_robots/README.md).
        cfg = parse_ik_config(
            {**CFG_5DOF, "solver_type": "weighted_dls", "damping": {"w_rot": 0.3}},
            strict=True,
        )
        assert cfg.solver_type == "weighted_dls"
        with pytest.raises(ValueError, match="w0_rot"):
            parse_ik_config(
                {**CFG_5DOF, "solver_type": "weighted_dls", "damping": {"w0_rot": 0.25}},
                strict=True,
            )


class TestParseIkConfigBundle:
    def test_config_validation(self):
        with pytest.raises(ValueError):
            parse_ik_config({**CFG_6DOF, "solver_type": "nope"})
        with pytest.raises(ValueError):
            parse_ik_config({**CFG_6DOF, "urdf_joint_names": ["a", "b"]})
        with pytest.raises(ValueError):
            parse_ik_config({**CFG_6DOF, "anchor_body": None})

    def test_flat_document_unchanged(self):
        # No "chains" key -> defers to parse_ik_config exactly as before.
        cfg = parse_ik_config_bundle(CFG_5DOF)
        assert isinstance(cfg, IkConfig)
        assert cfg.solver_type == "so101_5dof"

    def test_chains_document_returns_dict(self):
        left = {**CFG_5DOF, "ik_joint_action_indices": [0, 1, 2, 3, 4], "gripper_action_index": 5}
        right = {**CFG_5DOF, "ik_joint_action_indices": [6, 7, 8, 9, 10], "gripper_action_index": 11}
        out = parse_ik_config_bundle({"chains": {"left": left, "right": right}})
        assert set(out) == {"left", "right"}
        assert isinstance(out["left"], IkConfig) and isinstance(out["right"], IkConfig)
        assert out["left"].gripper_action_index == 5
        assert out["right"].gripper_action_index == 11

    def test_one_armed_chains_document_is_valid(self):
        # The wrapper does not mean bimanual: a1z, so101, xarm6 and xarm7 all
        # ship {"chains": {"right": ...}}, and the browser reads them that way.
        out = parse_ik_config_bundle({"chains": {"right": CFG_5DOF}})
        assert set(out) == {"right"}
        assert isinstance(out["right"], IkConfig)

    def test_chains_document_rejects_junk_sides(self):
        with pytest.raises(ValueError):
            parse_ik_config_bundle({"chains": {}})
        with pytest.raises(ValueError):
            parse_ik_config_bundle({"chains": {"left": CFG_5DOF, "middle": CFG_5DOF}})


# ---------------------------------------------------------------------------
# Kinematic-spec export: bundle-aware (flat vs chains)
# ---------------------------------------------------------------------------


class TestExportKinematicSpecBundle:
    """The browser-facing spec exporter must mirror parse_ik_config_bundle:
    flat bundles export the flat spec byte-for-byte as before; a wrapped
    bundle exports one complete flat spec per populated side under
    ``chains``."""

    @staticmethod
    def _chain_cfgs() -> tuple[dict, dict]:
        left = {**CFG_5DOF, "ik_joint_action_indices": [0, 1, 2, 3, 4], "gripper_action_index": 5}
        right = {**CFG_5DOF, "ik_joint_action_indices": [6, 7, 8, 9, 10], "gripper_action_index": 11}
        return left, right

    def test_flat_bundle_exports_flat_spec(self, tmp_path):
        bundle = _write_bundle(tmp_path, URDF_5DOF, CFG_5DOF)
        spec = export_kinematic_spec_bundle(bundle)
        assert "chains" not in spec
        assert spec == export_kinematic_spec(bundle)
        assert spec["n_ik_joints"] == 5

    def test_chains_bundle_exports_per_arm_specs(self, tmp_path):
        left, right = self._chain_cfgs()
        bundle = _write_bundle(
            tmp_path, URDF_5DOF, {"chains": {"left": left, "right": right}},
        )
        spec = export_kinematic_spec_bundle(bundle)
        assert set(spec["chains"]) == {"left", "right"}
        # Each chain is a complete flat spec, identical to exporting its
        # config alone against the same URDF.
        for side, raw in (("left", left), ("right", right)):
            chain = spec["chains"][side]
            assert chain == export_kinematic_spec(bundle, parse_ik_config(raw))
            assert chain["n_ik_joints"] == 5
        # The action-vector layout seam survives per chain: disjoint blocks
        # of the shared 12-wide vector.
        assert [j["action_index"] for j in spec["chains"]["left"]["joints"]] == [0, 1, 2, 3, 4]
        assert [j["action_index"] for j in spec["chains"]["right"]["joints"]] == [6, 7, 8, 9, 10]
        assert spec["chains"]["left"]["gripper"]["action_index"] == 5
        assert spec["chains"]["right"]["gripper"]["action_index"] == 11

    def test_one_armed_bundle_exports_one_chain(self, tmp_path):
        bundle = _write_bundle(tmp_path, URDF_5DOF, {"chains": {"right": CFG_5DOF}})
        spec = export_kinematic_spec_bundle(bundle)
        assert set(spec["chains"]) == {"right"}
        assert spec["chains"]["right"] == export_kinematic_spec(
            bundle, parse_ik_config(CFG_5DOF)
        )

    def test_write_kinematic_spec_persists_chains_shape(self, tmp_path):
        left, right = self._chain_cfgs()
        bundle = _write_bundle(
            tmp_path, URDF_5DOF, {"chains": {"left": left, "right": right}},
        )
        out = write_kinematic_spec(bundle)
        data = json.loads(out.read_text())
        assert set(data["chains"]) == {"left", "right"}
        assert data["chains"]["left"]["joints"][0]["name"] == "shoulder_pan"
