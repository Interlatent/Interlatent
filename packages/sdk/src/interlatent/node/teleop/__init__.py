"""Node-side teleop: a thin client that receives absolute joint targets and
drives the robot through the single `SafetyGate`-gated path in `control.py`.
The node never solves IK itself — but *where* the IK runs depends on the
transport, and there are two, chosen at session start by `factory`:

- **WS** (`channel.TeleopChannel`) — the node opens a WebSocket to the GPU-box
  relay (`interlatent.inference.server.teleop_relay`) and receives
  ``mode="targets"`` frames the **pod's** retarget stage already solved. The
  original path; keyboard-overlay steering.
- **QUIC/WebTransport** (`quic_channel.QuicTeleopChannel`) — the low-latency
  path (VR/WebXR). The **browser** solves IK against a kinematic_spec the node
  serves, and streams ``mode="targets"`` datagrams; the node also tees a live
  JPEG preview back to the browser. The aioquic connection is isolated in a
  child process so a busy robot-driver GIL can't starve its timers (ADR 0021).

`factory.make_teleop_channel` probes the deployment's transport and returns the
right one behind one shared surface (`start`/`stop`/`latest_frame`/`send_state`/
`connected`), so the control loop is transport-agnostic. Both share the
safety-critical latest-frame + sticky-estop store (`_frame_store`, ADR 0016)
and the wire-frame parser (`frame`).

- `factory`        — picks the transport; the control loop sees one interface.
- `channel` / `quic_channel` — the two transports (WS / QUIC).
- `frame`/`TeleopFrame` — the authoritative wire-frame parser (keys/pose/targets).
- `safety`         — `SafetyGate`: the node's authoritative velocity/workspace clamp.
- `robot_profile`  — static per-robot limits/velocity/rest-pose (lerobot can't supply these).

See ADR 0012 (receiver-stub / open-core boundary) and ADR 0021 (QUIC child
process). The keyboard/pose modality compute lives upstream (pod or browser),
not here.
"""
from __future__ import annotations

from .channel import TeleopChannel
from .factory import make_teleop_channel
from .frame import TeleopFrame
from .robot_profile import RobotProfile, SO101_PROFILE, get_profile
from .safety import SafetyConfig, SafetyGate, TargetSample

__all__ = [
    "TeleopChannel",
    "make_teleop_channel",
    "TeleopFrame",
    "RobotProfile",
    "SO101_PROFILE",
    "get_profile",
    "SafetyConfig",
    "SafetyGate",
    "TargetSample",
]
