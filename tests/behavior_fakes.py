"""A hardware-free fake SO-101 adapter for the behaviors tests.

Mirrors the fakes in ``test_action_interface.py`` / ``test_act_cli.py`` but adds the
knobs the behavior executor exercises: a settable tracking mode and a recorded command
stream. Joint order matches the SO-101 ``RobotProfile`` exactly.
"""
from __future__ import annotations

import numpy as np

from interlatent.adapters.base import JointSpec, ManualActionInterface

FEATURES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


class FakeAdapter(ManualActionInterface):
    """In-memory SO-101-shaped adapter.

    ``track``:
      - ``"full"``  — the commanded pose is reached exactly each tick.
      - ``"none"``  — commands are ignored (joints never move) → stall/timeout tests.
    """

    robot_kind = "so101"

    def __init__(self, track: str = "full", start: float = 0.0) -> None:
        self._pos = np.full(len(FEATURES), float(start), dtype=np.float32)
        self._track = track
        self.commands: list[np.ndarray] = []
        self.connected = False

    @property
    def action_features(self) -> list[str]:
        return list(FEATURES)

    @property
    def joint_specs(self) -> list[JointSpec]:
        specs = []
        for f in FEATURES:
            name = f.rsplit(".", 1)[0]
            if name == "gripper":
                specs.append(JointSpec(name=name, control_mode="gripper"))
            else:
                specs.append(JointSpec(name=name, control_mode="position", settle_tolerance=2.0))
        return specs

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def get_observation(self):
        return {f: float(self._pos[i]) for i, f in enumerate(FEATURES)}

    def send_action(self, action):
        vec = np.array([action[f] for f in FEATURES], dtype=np.float32)
        self.commands.append(vec.copy())
        if self._track == "full":
            self._pos = vec
        # "none": ignore — joints never move.
        return action

    @property
    def command_array(self) -> np.ndarray:
        """The recorded command stream as an ``(N, 6)`` array."""
        return np.asarray(self.commands, dtype=np.float64)
