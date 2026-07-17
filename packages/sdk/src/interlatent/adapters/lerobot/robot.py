"""``LeRobotAdapter`` — a LeRobot robot behind the formal adapter interface.

Wraps a LeRobot ``Robot`` (built via the same ``_make_lerobot_robot`` the engine
loop uses) so so101/koch expose the manual block-then-settle ``action()`` from
:class:`~interlatent.adapters.base.ManualActionInterface`. This is the **manual**
path only — the engine control loop (``lerobot_control_loop``) is unchanged and keeps
its own per-tick dispatch + policy-frame calibration.

Frame note: manual ``action()`` targets are in the **robot's own frame** (the caller
is commanding the physical arm, in degrees for SO-101), so ``send_action`` is a raw
passthrough to LeRobot. The OLD<->NEW policy-calibration affine
(``_coerce_action_for_robot``) is an *engine*-path concern and is deliberately not
applied here.

Manual ``action()`` needs a :class:`RobotProfile` for the kind (limits + velocity);
so101 has a hardware-verified one. koch carries only a placeholder profile and is
not supported — do not drive a Koch arm through this adapter.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from ..base import JointSpec, ManualActionInterface
from ...node.control import _joint_name

_logger = logging.getLogger(__name__)


class LeRobotAdapter(ManualActionInterface):
    """Manual-path adapter wrapping a LeRobot follower robot."""

    def __init__(
        self,
        robot_kind: str,
        *,
        port: Optional[str] = None,
        extra: Optional[Dict[str, str]] = None,
        cameras: Optional[Dict[str, str]] = None,
    ) -> None:
        self.robot_kind = robot_kind
        self._port = port
        self._extra = dict(extra or {})
        self._cameras = dict(cameras or {})
        self._robot: Any = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        # Reuse the engine path's instantiation so manual + engine drive an
        # identically-configured robot. Imported lazily — lerobot is heavy.
        from ...node.control import _make_lerobot_robot

        self._robot = _make_lerobot_robot(
            self.robot_kind, port=self._port, extra=self._extra, cameras=self._cameras
        )
        self._robot.connect()
        _logger.info(
            "LeRobotAdapter %r connected; action_features=%s",
            self.robot_kind, self.action_features,
        )

    def disconnect(self) -> None:
        if self._robot is not None:
            self._robot.disconnect()

    # ------------------------------------------------------------------
    # Observe / act
    # ------------------------------------------------------------------

    def get_observation(self) -> Dict[str, Any]:
        return self._robot.get_observation()

    def send_action(self, action: Dict[str, Any]) -> Any:
        # Robot-frame passthrough (no policy calibration on the manual path).
        return self._robot.send_action(action)

    @property
    def action_features(self) -> list[str]:
        return list(getattr(self._robot, "action_features", None) or [])

    @property
    def joint_specs(self) -> list[JointSpec]:
        """Settle metadata aligned with :attr:`action_features`.

        Grippers settle on "command issued"; arm joints settle by a position
        tolerance in degrees (LeRobot SO-101 reports joints in degrees). Ranges come
        from the :class:`RobotProfile`, not from here.
        """
        specs: list[JointSpec] = []
        for feature in self.action_features:
            name = _joint_name(feature)
            if "gripper" in name:
                specs.append(JointSpec(name=name, control_mode="gripper"))
            else:
                specs.append(JointSpec(name=name, control_mode="position", settle_tolerance=2.0))
        return specs


__all__ = ["LeRobotAdapter"]
