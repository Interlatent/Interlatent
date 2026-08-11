# SO-101 / LeRobot robot configuration

Robot-specific arguments for `--robot so101`, the LeRobot-backed serial arms.
Built by [`_make_lerobot_robot`](../../node/control.py); extra keys are
forwarded to the matching LeRobot `*FollowerConfig` dataclass.

SO-101 is the reference platform for this project. Koch v1.1 shares the code
path and is still wired up (`--robot koch`, a placeholder `RobotProfile`), but
is **not supported** — it has never been verified on hardware. Do not use it.

Arguments are the same across `interlatent-node` and `interlatent-act`:

- `--port` — **required** serial port (e.g. `/dev/ttyACM0`, `/dev/ttyUSB0`).
- `--robot-arg key=value` — repeatable (table below).
- `--camera name=device` — repeatable; opens an OpenCV camera (`name` becomes
  the `observation.images.<name>` key the policy sees).

## Host requirements

```bash
pip install 'interlatent[lerobot]'
```

The SO-101's Feetech servos need the Feetech servo SDK for the serial bus.
LeRobot pulls it in on supported platforms; if `interlatent-node run --robot
so101` fails to open the bus, install it explicitly:

```bash
pip install feetech-servo-sdk
```

You also need read/write access to the serial device — on Linux add yourself to
the `dialout` group (`sudo usermod -aG dialout $USER`, then re-login) rather
than running as root.

## `--robot-arg` keys

Two keys are read by the node's control loop itself:

| Key | Default | Description |
|---|---|---|
| `max_step` | *(unset ⇒ clamp disabled)* | Per-tick delta clamp on the policy/teleop action, in motor-norm units (degrees for SO-101 under MolmoAct2). Non-numeric or non-positive ⇒ disabled. `interlatent-node run` only. |
| `action_filter_hz` | `3.0` | Butterworth low-pass cutoff on the policy action stream. `0` / `none` / `off` disables it. `interlatent-node run` only. |

Every other key is forwarded to LeRobot's `SO101FollowerConfig` (SO-101) or
`KochFollowerConfig` (Koch); keys those dataclasses do not declare are silently
dropped, so the authoritative list is whatever your installed LeRobot declares.
The one you'll usually set:

| Key | Description |
|---|---|
| `id` | Calibration id LeRobot uses to locate the saved calibration file for this arm. Set this if you keep multiple arms calibrated on one host. |

Print the field names your installed LeRobot accepts (handles both the new
`so_follower` and the older `so101_follower` module layouts):

```bash
python -c "
try:
    from lerobot.robots.so_follower import SO101FollowerConfig
except ImportError:
    from lerobot.robots.so101_follower import SO101FollowerConfig
import dataclasses
print([f.name for f in dataclasses.fields(SO101FollowerConfig)])
"
```

## `--camera` declarations

`--camera <name>=<device>`

- `<name>` — observation key; **must match the policy's training camera keys**
  (e.g. `front`, `wrist`).
- `<device>` — an OpenCV device path or index (e.g. `/dev/video0` or `0`).

Cameras open at 640×480 @ 30 with the MJPG fourcc, so several USB cameras on a
Pi stay within USB 2.0 limits. Cameras are optional for a manual
`interlatent-act` joint move.

## Joint names & units

Six keys, in LeRobot SO-101 order:

`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`

Arm joints are in **degrees**; the gripper is `0` (closed) to `100` (open),
matching the LeRobot SO-101 convention. Limits and per-joint velocity come from
the SO-101 [`RobotProfile`](../../node/teleop/robot_profile.py).

## Calibration migration (`so101_pre777`)

LeRobot PR #777 changed the SO-100/SO-101 joint zero convention. Policies
trained on pre-#777 data expect the old frame, so the node auto-enables the
`so101_pre777` affine remap when the session's `policy_uri` contains
`molmoact`. Override with `INTERLATENT_CALIB_PRESET`:

- `INTERLATENT_CALIB_PRESET=so101_pre777` — force the migration on.
- `INTERLATENT_CALIB_PRESET=none` — turn it off.

See [`docs/robots-and-policies.md`](../../../../../../docs/robots-and-policies.md)
and the note in [`control.py`](../../node/control.py).

## Examples

```bash
# Read the current pose, no motion
interlatent-act --robot so101 --port /dev/ttyACM0 --show

# Move two joints to absolute angles (degrees), hold the rest where they are
interlatent-act --robot so101 --port /dev/ttyACM0 \
  shoulder_pan=30 gripper=80 --hold-missing

# Run a policy session with one camera and the delta clamp enabled
interlatent-node run --robot so101 --port /dev/ttyACM0 \
  --camera front=/dev/video0 --robot-arg max_step=5
```
