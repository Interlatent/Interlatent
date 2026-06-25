"""interlatent-act one-shot manual-move CLI (interlatent.node.act_cli)."""
from __future__ import annotations

import numpy as np
import pytest

from interlatent.node import act_cli
from interlatent.adapters.base import JointSpec, ManualActionInterface

_FEATURES = [
    "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
    "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
]


class _FakeAdapter(ManualActionInterface):
    robot_kind = "so101"

    def __init__(self, *_a, **_k):
        self._pos = np.zeros(len(_FEATURES), dtype=np.float32)
        self.connected = False

    @property
    def action_features(self):
        return list(_FEATURES)

    @property
    def joint_specs(self):
        return [
            JointSpec(name=f.rsplit(".", 1)[0],
                      control_mode="gripper" if "gripper" in f else "position",
                      settle_tolerance=2.0)
            for f in _FEATURES
        ]

    def connect(self):
        self.connected = True

    def disconnect(self):
        self.connected = False

    def get_observation(self):
        return {f: float(self._pos[i]) for i, f in enumerate(_FEATURES)}

    def send_action(self, action):
        self._pos = np.array([action[f] for f in _FEATURES], dtype=np.float32)
        return action


@pytest.fixture
def patched(monkeypatch):
    """Patch the lazily-imported LeRobotAdapter to the in-memory fake."""
    import interlatent.adapters.lerobot.robot as robot_mod
    monkeypatch.setattr(robot_mod, "LeRobotAdapter", _FakeAdapter)


_BASE = ["--robot", "so101", "--port", "/dev/null", "--rate-hz", "500", "--timeout", "3"]


def test_kv_parsers():
    assert act_cli._joint_kv("shoulder_pan=30") == ("shoulder_pan", 30.0)
    with pytest.raises(Exception):
        act_cli._joint_kv("shoulder_pan")
    with pytest.raises(Exception):
        act_cli._joint_kv("shoulder_pan=abc")


def test_no_joints_without_show_errors():
    # Returns before touching hardware/adapter.
    assert act_cli.main(["--robot", "so101", "--port", "/dev/null"]) == 2


def test_show_reads_pose(patched, capsys):
    assert act_cli.main(_BASE + ["--show"]) == 0
    out = capsys.readouterr().out
    assert "shoulder_pan=0.0" in out


def test_move_settles(patched, capsys):
    rc = act_cli.main(_BASE + ["shoulder_pan=20", "--hold-missing"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "settled" in out and "shoulder_pan=20" in out


def test_unknown_joint_exits_2(patched):
    assert act_cli.main(_BASE + ["elbow=10", "--hold-missing"]) == 2


def test_missing_joint_without_hold_exits_2(patched):
    assert act_cli.main(_BASE + ["shoulder_pan=10"]) == 2


def test_out_of_range_exits_2(patched):
    assert act_cli.main(_BASE + ["shoulder_pan=999", "--hold-missing"]) == 2


def test_help_does_not_import_lerobot(capsys):
    with pytest.raises(SystemExit) as e:
        act_cli.main(["--help"])
    assert e.value.code == 0
    assert "interlatent-act" in capsys.readouterr().out
