# YAM robot configuration

Robot-specific arguments for `--robot yam` (I2RT YAM bimanual arms over the
`i2rt` CAN driver). Parsed by [`config.py`](config.py) `build_adapter_config`.

Arguments are the same across `interlatent-node` and `interlatent-act`:

- `--robot-arg key=value` — repeatable (table below).
- `--camera name=<device>[,key=val…]` — repeatable; declares an RGB camera.

Unrecognized `--robot-arg` keys are warned about and ignored. `--port` is
**not** used by YAM (it speaks CAN, not serial).

## `--robot-arg` keys

| Key | Default | Values | Description |
|---|---|---|---|
| `arms` | `both` | `both` \| `left` \| `right` | Which follower arms are active. Sets the action space (14-DOF both / 7-DOF single) and the matching profile (`yam` / `yam_left` / `yam_right`). |
| `left_channel` | `can_follower_l` | SocketCAN iface name | CAN bus for the left follower. |
| `right_channel` | `can_follower_r` | SocketCAN iface name | CAN bus for the right follower. |
| `max_step_rad` | `0.05` | float (rad), or `inf` | Adapter-side per-step delta clamp on arm joints — the arm advances at most this far per tick toward a commanded target. The gripper is not clamped; `inf` disables. |
| `auto_home` | `true`¹ | bool | On `connect()`, smooth-move every active arm to the rest pose (`FOLLOWER_HOME_POS` = 6 zeros + gripper open) over ~5 s. **Moves hardware the instant you connect.** |
| `gripper_mode` | `continuous` | `continuous` \| `bangbang` | `continuous` passes the gripper value through; `bangbang` snaps to open/closed at `gripper_threshold`. |
| `gripper_threshold` | `0.5` | float `[0,1]` | Snap point for `bangbang` gripper mode. |

Two more keys are read by the control loop rather than by
`build_adapter_config`, so `interlatent-node run` logs them as "unrecognized"
and still honors them:

| Key | Default | Description |
|---|---|---|
| `max_step` | *(unset ⇒ disabled)* | Loop-level delta clamp (radians here), applied before the adapter's own `max_step_rad`. |
| `action_filter_hz` | `3.0` | Butterworth low-pass cutoff on the policy action stream. `0` / `none` / `off` disables it. |

¹ `auto_home` defaults to `true` under `interlatent-node run`, but to `false`
for `interlatent-act` and named behaviors, so a one-shot move (or `--show`)
never surprise-homes the arm. Pass `--robot-arg auto_home=true` to re-enable it
there. `--robot yam_left` / `yam_right` also default `arms` to that side.

Bool values accept `1/true/yes/on` (case-insensitive); anything else is false.

## `--camera` declarations

`--camera <name>=<device>[,key=val…]` — `<name>` is the observation key and
**must match the policy's training camera keys**. `<device>` selects the
backend three ways:

| `<device>` form | Backend | Example |
|---|---|---|
| `realsense[:<serial>]` | Intel RealSense (`pyrealsense2`) | `--camera wrist=realsense:1234` (omit serial → first found: `realsense`) |
| `zed[:<serial>]` | Stereolabs ZED (`pyzed`, host-installed) | `--camera overhead=zed:5678` |
| `/dev/videoN`, a bare index, or `uvc:<path-or-index>` | Generic UVC/V4L2 webcam via OpenCV (`opencv-python-headless`) | `--camera front=/dev/video2`, `--camera front=2`, `--camera front=uvc:/dev/video2` |

Any standard USB webcam works through the OpenCV backend. The
`uvc`/`opencv`/`v4l2`/`webcam` prefixes are interchangeable; a bare
`/dev/video*` path or numeric index is treated as a UVC webcam automatically.
Find your device with `v4l2-ctl --list-devices`.

Capture settings ride as comma-separated options on the device value:

| Option | Default | Applies to |
|---|---|---|
| `width` / `height` | `640` / `480` | RealSense, UVC (ZED ignores them and uses its SDK-default resolution) |
| `fps` | `30` | all backends |
| `pixel_format` | `mjpg` (`mjpg` \| `yuyv` \| `default`) | UVC only |

```bash
--camera front=/dev/video2,width=1280,height=720,fps=15,pixel_format=yuyv
```

MJPG is the UVC default because uncompressed 640×480@30 YUYV reserves
~147 Mbit/s of USB isochronous bandwidth *per camera*, which starves a shared
USB 2.0 bus. A rejected format falls back to the driver default with a warning.

Capture is **RGB only**. Cameras are optional — a manual `interlatent-act`
joint move needs none.

## CAN bus setup

Each follower is one SocketCAN interface, so a bimanual set needs **two** buses.
On `connect()` the adapter checks each channel with `ip link show`; if one is
present but down it makes a single non-interactive attempt
(`sudo -n ip link set <iface> up type can bitrate 1000000`) and raises only if
the interface is still not up. `sudo -n` never prompts, so on a host without
passwordless sudo you must bring the buses up yourself.

**1. Bring up the buses.** A USB-CAN adapter enumerates under kernel-default
names (`can0`, `can1`) until renamed. The bitrate **must** be `1000000`
(1 Mbit/s) — both ends of a CAN bus must agree:

```bash
sudo ip link set can0 up type can bitrate 1000000
sudo ip link set can1 up type can bitrate 1000000
ip -details link show type can      # confirm each is state UP / UNKNOWN
```

If `ip link set …` reports **`Cannot find device "can_follower_l"`**, the
renamed interfaces don't exist yet — the buses are still `can0`/`can1` (see step
3). If they don't appear under any name, check the adapter is enumerated
(`lsusb`, `dmesg | grep -i can`) and the kernel module is loaded (e.g.
`sudo modprobe gs_usb`).

**2. Point the SDK at the right names.** The adapter defaults to
`can_follower_l` / `can_follower_r`. If you brought up `can0`/`can1`, override
the channels — but `can0`/`can1` are assigned by USB enumeration order and **can
swap on reboot or replug**, so verify which physical arm each drives:

```bash
# moves whichever arm is on can0 — watch which one physically moves:
interlatent-act --robot yam --robot-arg arms=left --robot-arg left_channel=can0 \
  left_joint_0=0.2 --hold-missing
```

Then map left/right accordingly, e.g. if `can0` turned out to be the **right**
arm:

```bash
interlatent-node run --robot yam --robot-arg arms=both \
  --robot-arg left_channel=can1 --robot-arg right_channel=can0
```

**3. Lock in stable names (recommended).** Pin each adapter to the SDK's default
names by USB serial so you never pass `*_channel` flags again. Get the serials
(`udevadm info -a -p $(udevadm info -q path -n can0) | grep -m1 serial`), then
add `/etc/udev/rules.d/90-yam-can.rules`:

```
SUBSYSTEM=="net", ACTION=="add", ATTRS{serial}=="<left-arm-serial>",  NAME="can_follower_l"
SUBSYSTEM=="net", ACTION=="add", ATTRS{serial}=="<right-arm-serial>", NAME="can_follower_r"
```

```bash
sudo udevadm control --reload-rules && sudo udevadm trigger   # then replug the adapters
```

After that `ip link show type can` lists `can_follower_l` / `can_follower_r` and
plain `interlatent-node run --robot yam --robot-arg arms=both` works with no
channel overrides. i2rt/raiden's `rd reset_can` installs equivalent naming.

> **Note on shutdown noise.** On a clean exit, i2rt's background motor-control
> thread can log one round of `Bad file descriptor [9]` / `file descriptor
> cannot be a negative integer (-1)` / `Failed to communicate with motor …
> Retrying` *after* `YAMNativeRobot disconnected`. That is a teardown race
> inside the i2rt driver; torque is already zeroed, so it is cosmetic.

## Joint names & units

`left_joint_0 … left_joint_5`, `left_gripper`, then the `right_*` block (left
arm before right; wire keys carry a `.pos` suffix, `action()` /
`interlatent-act` use the bare names). Arm joints are **radians**, gripper is
`[0, 1]` (0 closed, 1 open). Limits come from the YAM
[`RobotProfile`](../../node/teleop/robot_profile.py) — arm limits transcribed
from the i2rt YAM URDF; velocity cap and gripper range are conservative
placeholders, so verify on hardware.

## Examples

```bash
# Read the left arm's pose, no motion
interlatent-act --robot yam --robot-arg arms=left --show

# Move the left base joint to 0.2 rad, hold the rest, close the gripper
interlatent-act --robot yam --robot-arg arms=left \
  left_joint_0=0.2 left_gripper=0.0 --hold-missing

# Run a bimanual policy session with two cameras and a looser step clamp
# (the default is 0.05 rad/tick; raise it if the policy needs faster moves)
interlatent-node run --robot yam \
  --robot-arg arms=both --robot-arg max_step_rad=0.3 \
  --camera overhead=zed:41234567 --camera wrist=realsense:1122
```

**Host requirements:** `pip install 'interlatent[yam]'`, Linux + SocketCAN, CAN
buses up (see [CAN bus setup](#can-bus-setup)). The ZED SDK / `pyzed` is
host-installed (not on PyPI) and needed only for ZED cameras.
