"""Node-side teleop: the thin client receiver stub for the hosted DAgger path.

The teleop *engine* (keyboard integration, pose IK, retargeting) runs on the
Interlatent platform; the node keeps only a receiver stub plus the last-hop
safety clamp. The node connects to the GPU-box relay
(`interlatent.inference.server.teleop_relay`), receives ``mode="targets"`` frames
(absolute joint vectors the platform already computed), and drives the robot
through the single `SafetyGate`-gated path in `control.py`.

- `channel`        — WS client to the relay; surfaces the latest frame.
- `frame`/`TeleopFrame` — the authoritative wire-frame parser (keys/pose/targets).
- `safety`         — `SafetyGate`: the node's authoritative velocity/workspace clamp.
- `robot_profile`  — static per-robot limits/velocity/rest-pose (lerobot can't supply these).

See docs/adr/0012-teleop-receiver-stub-open-core-boundary.md. The keyboard/pose
modality compute (`keyboard`, `kinematics`, `retarget`) lives on the platform and
is intentionally absent here.
"""
from __future__ import annotations

from .channel import TeleopChannel
from .frame import TeleopFrame
from .robot_profile import RobotProfile, SO101_PROFILE, get_profile
from .safety import SafetyConfig, SafetyGate, TargetSample

__all__ = [
    "TeleopChannel",
    "TeleopFrame",
    "RobotProfile",
    "SO101_PROFILE",
    "get_profile",
    "SafetyConfig",
    "SafetyGate",
    "TargetSample",
]
