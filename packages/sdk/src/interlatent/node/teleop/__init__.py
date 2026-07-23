"""Node-side teleop: a thin client that receives absolute joint targets and
drives the robot through the single `SafetyGate`-gated path in `control.py`.
The node never solves IK itself — teleop runs over QUIC/WebTransport, and the
IK runs in the **browser**:

- **QUIC/WebTransport** (`quic_channel.QuicTeleopChannel`) — the low-latency
  path (VR/WebXR). The browser solves IK against a kinematic_spec the node
  serves, and streams ``mode="targets"`` datagrams; the node also tees a live
  JPEG preview back to the browser. The aioquic connection is isolated in a
  child process so a busy robot-driver GIL can't starve its timers (ADR 0021).

`factory.make_teleop_channel` builds the channel (or returns ``None`` when the
deployment isn't QUIC-configured) behind the surface the control loop uses
(`start`/`stop`/`latest_frame`/`send_state`/`connected`). The safety-critical
latest-frame + sticky-estop store (`_frame_store`, ADR 0016) and the wire-frame
parser (`frame`) live in one place.

- `factory`        — builds the channel; the control loop sees one interface.
- `quic_channel`   — the QUIC/WebTransport transport.
- `frame`/`TeleopFrame` — the authoritative wire-frame parser (keys/pose/targets).
- `safety`         — `SafetyGate`: the node's authoritative velocity/workspace clamp.
- `robot_profile`  — static per-robot limits/velocity/rest-pose (lerobot can't supply these).

See ADR 0012 (receiver-stub / open-core boundary) and ADR 0021 (QUIC child
process). The pose modality compute lives upstream (in the browser), not here.
"""
from __future__ import annotations

from .factory import make_teleop_channel
from .frame import TeleopFrame
from .robot_profile import RobotProfile, SO101_PROFILE, get_profile
from .safety import SafetyConfig, SafetyGate, TargetSample

__all__ = [
    "make_teleop_channel",
    "TeleopFrame",
    "RobotProfile",
    "SO101_PROFILE",
    "get_profile",
    "SafetyConfig",
    "SafetyGate",
    "TargetSample",
]
