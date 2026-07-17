# Nori robot configuration

Robot-specific arguments for `--robot nori` (the Nori robot, driven through its
on-Pi daemon — `NoriCoreAgent` — over the Nori-Protocol v1 wire contract:
newline-delimited JSON on TCP, `localhost:7777` by default). Parsed by
[`config.py`](config.py) `build_adapter_config`. The node runs on the robot's
Pi (LAN/on-Pi only in v1); v1 is arms-only.

Arguments are passed two ways, identical across `interlatent-node` and
`interlatent-act`:

- `--robot-arg key=value` — repeatable; one knob each (table below).
- `--camera name=value` — repeatable; maps a daemon camera onto a policy
  observation key.

Unrecognized `--robot-arg` keys are warned about and ignored. `--port` (the
serial-port flag) is **not** used by Nori — it speaks TCP to the daemon; use
`--robot-arg host=`/`port=`.

## `--robot-arg` keys

| Key | Default | Description |
|---|---|---|
| `host` | `127.0.0.1` | Daemon TCP host. |
| `port` | `7777` | Daemon TCP port. |
| `token` | *(unset)* | Daemon agent token. Explicit value beats `token_path`. |
| `token_path` | `/etc/nori/agent.token` | Daemon token file to read when `token` is unset (same trust domain: the node runs on the Pi next to the daemon). |
| `bus_choice` | `3` | Bus selection passed through in the daemon session-init frame. Leave at the default unless your daemon setup requires otherwise. |
| `max_step` | `3.0` | Execution-safety per-send delta clamp on arm joints, in daemon-normalized units (grippers exempt). `inf` disables. |
| `pump_hz` | `50.0` | Keep-alive pump rate (see below). |
| `cam_host` | same as `host` | Host for the daemon's ZeroMQ MJPEG camera channel. |
| `cam_base_port` | `5555` | Base port for the camera channel. |
| `connect_timeout_s` | `5.0` | TCP connect + daemon-ack timeout. |
| `camera_warmup_s` | `10.0` | `connect()` blocks until every configured camera has delivered one frame, up to this long (the Pi's capture bridge publishes lazily). `0` disables the wait. |
| `reconnect_window_s` | `10.0` | TCP down longer than this ⇒ the session is dead (the loop ends the episode). |
| `staleness_ms` | `250.0` | Telemetry older than this ⇒ the observation is not fresh (the loop holds rather than acting on old joints). |

## `--camera` declarations

`--camera <obs_key>=<daemon_camera_name>` maps a daemon camera (as named in the
daemon's ack descriptor) onto a policy observation key — `<obs_key>` **must
match the policy's training camera keys**. There are no camera-SDK settings
here: frames arrive on the daemon's companion ZeroMQ MJPEG channel. With no
`--camera` flags, every descriptor camera is subscribed under its native name.

## Safety model

Nori keeps all safety **enforcement** robot-side (range clamping, e-stop hard
latch, watchdog safe-stop); the adapter discloses that state and never
re-enforces it. On `connect()` the adapter fail-closes if the daemon's live ack
descriptor disagrees with the static `nori`
[`RobotProfile`](../../node/teleop/robot_profile.py) — every mismatch is
accumulated into one raise.

### Keep-alive pump

The daemon has no heartbeat message — the control-frame stream *is* the
watchdog heartbeat, and silence safe-stops the robot. The adapter therefore
runs an internal ~`pump_hz` pump of motion-free control frames, **but only
while the control loop proves liveness**: if the loop stalls, the pump stops
and the daemon safe-stops as designed. Don't raise `pump_hz` to "fix" watchdog
trips — a stall is exactly what the watchdog is for. See
[ADR 0015](../../../../../../docs/adr/0015-nori-liveness-tied-keepalive.md).

### E-stop and `--reset-latch`

A teleop e-stop (see [the teleop guide](../../../../../../docs/teleop.md#e-stop))
latches the node's `SafetyGate` on every robot; on Nori the loop additionally
sends the daemon's `estop` command, which **hard-latches robot-side**. A
daemon-reported latch/safe-stop is a hard episode boundary: the loop ends the
session, freeing the daemon's single control-client slot for recovery.

Recovery is a human act, never automatic:

```bash
interlatent-act --robot nori --reset-latch
```

This sends the daemon's token-gated `reset_latch` and then clears the gate
latch — daemon first, gate second. The token resolves from `--token`, else
`--robot-arg token=…`, else the daemon's token file. It never moves the robot
and takes no joint targets. See
[ADR 0016](../../../../../../docs/adr/0016-teleop-estop-ingress-human-only-reset.md).

### Teleop

While the node holds the daemon's single control-client slot, Nori's **own**
browser/VR teleop cannot connect — during an interlatent session, teleop rides
the interlatent relay (VR) like any other robot. The two teleop stacks
are separate systems; Nori's is displaced, not reused.

## Joint names & units

12 joints, left arm block then right: `left_arm_shoulder_pan`,
`left_arm_shoulder_lift`, `left_arm_elbow_flex`, `left_arm_wrist_flex`,
`left_arm_wrist_roll`, `left_arm_gripper`, then the `right_arm_*` block (wire
keys carry a `.pos` suffix; `action()` / `interlatent-act` use the bare names).
All values are in the daemon-normalized `range_m100_100` scale. Limits come
from the `nori` `RobotProfile`.

## Examples

```bash
# Read the current pose, no motion
interlatent-act --robot nori --show

# Move one left-arm joint, hold the rest
interlatent-act --robot nori left_arm_shoulder_pan=10 --hold-missing

# Clear the daemon's e-stop hard latch (token from /etc/nori/agent.token)
interlatent-act --robot nori --reset-latch

# Run the node against a daemon on a non-default port, mapping one camera
interlatent-node run --robot nori --robot-arg port=7878 \
  --camera observation.images.head=head
```

**Host requirements:** `pip install 'interlatent[nori]'` (brings `zmq`/`cv2`
for the camera channel; the base adapter import needs neither), running on the
robot's Pi alongside the daemon.
