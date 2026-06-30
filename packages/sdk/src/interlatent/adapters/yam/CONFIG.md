# YAM robot configuration

Robot-specific arguments for `--robot yam` (I2RT YAM bimanual arms, driven through the
`i2rt` CAN driver). Parsed by [`config.py`](config.py) `build_adapter_config`.

Arguments are passed two ways, identical across `interlatent-node` and `interlatent-act`:

- `--robot-arg key=value` — repeatable; one knob each (table below).
- `--camera name=<type>:<serial>` — repeatable; declares an RGB camera.

Unrecognized `--robot-arg` keys are warned about and ignored. `--port` is **not** used
by YAM (it speaks CAN, not serial).

## `--robot-arg` keys

| Key | Default | Values | Description |
|---|---|---|---|
| `arms` | `both` | `both` \| `left` \| `right` | Which follower arms are active. Sets the action space (14-DOF both / 7-DOF single) and the matching profile (`yam` / `yam_left` / `yam_right`). |
| `left_channel` | `can_follower_l` | SocketCAN iface name | CAN bus for the left follower. |
| `right_channel` | `can_follower_r` | SocketCAN iface name | CAN bus for the right follower. |
| `max_step_rad` | `0.5` | float (rad), or `inf` | Execution-safety per-step delta clamp on arm joints — the arm advances at most this far per tick toward a commanded target (guards against a model glitch / bad chunk). The gripper is not clamped. |
| `auto_home` | `true`¹ | bool | On `connect()`, smooth-move every active arm to the rest pose (`FOLLOWER_HOME_POS` = 6 zeros + gripper open). **Moves hardware the instant you connect.** |
| `gripper_mode` | `continuous` | `continuous` \| `bangbang` | `continuous` passes the gripper value through; `bangbang` snaps to open/closed at `gripper_threshold`. |
| `gripper_threshold` | `0.5` | float `[0,1]` | Snap point for `bangbang` gripper mode. |

¹ `auto_home` defaults to `true`, **except** `interlatent-act` forces it to `false` so a
one-shot CLI move (or `--show`) never surprise-homes the arm. Pass
`--robot-arg auto_home=true` to re-enable it there.

Bool values accept `1/true/yes/on` (case-insensitive); anything else is false.

## `--camera` declarations

`--camera <name>=<type>:<serial>`

- `<name>` — observation key; **must match the policy's training camera keys**.
- `<type>` — `realsense` (Intel RealSense, `pyrealsense2`) or `zed` (Stereolabs, `pyzed`).
- `<serial>` — vendor serial number; optional (omit → first available of that type, e.g.
  `--camera wrist=realsense`).

Capture is **RGB only** (the learned-depth backends stay in raiden). RealSense captures
at 640×480 @ 30; ZED captures at its SDK-default resolution and 30 fps. Resolution/fps
are not yet CLI-configurable for YAM. Cameras are optional — a manual `interlatent-act`
joint move needs none.

## CAN bus setup

The arms speak CAN, not serial — each follower is one SocketCAN interface, so a bimanual
set needs **two** buses up before `connect()`. The adapter does **not** configure CAN for
you (that needs root and could disturb a bus another process owns); on `connect()` it only
runs `ip link show <channel>` per arm and refuses to start unless each reports `state UP`
(or `UNKNOWN`). Bringing them up is a host responsibility.

**1. Bring up the buses.** A USB-CAN adapter enumerates under kernel-default names
(`can0`, `can1`) until renamed. The bitrate **must** be `1000000` (1 Mbit/s) — both ends
of a CAN bus must agree:

```bash
sudo ip link set can0 up type can bitrate 1000000
sudo ip link set can1 up type can bitrate 1000000
ip -details link show type can      # confirm each is state UP / UNKNOWN
```

If `ip link set …` reports **`Cannot find device "can_follower_l"`**, the renamed
interfaces don't exist yet — the buses are still `can0`/`can1` (see step 3). If they don't
appear under any name, check the adapter is enumerated (`lsusb`, `dmesg | grep -i can`) and
the kernel module is loaded (e.g. `sudo modprobe gs_usb`).

**2. Point the SDK at the right names.** The adapter defaults to `can_follower_l` /
`can_follower_r`. If you brought up `can0`/`can1`, override the channels — but note
`can0`/`can1` are assigned by USB enumeration order and **can swap on reboot or replug**,
so verify which physical arm each drives before trusting the mapping:

```bash
# moves whichever arm is on can0 — watch which one physically moves:
interlatent-act --robot yam --robot-arg arms=left --robot-arg left_channel=can0 \
  left_joint_0=0.2 --hold-missing
```

Then map left/right accordingly, e.g. if `can0` turned out to be the **right** arm:

```bash
interlatent-node run --robot yam --robot-arg arms=both \
  --robot-arg left_channel=can1 --robot-arg right_channel=can0
```

**3. Lock in stable names (recommended).** Pin each adapter to the SDK's default names by
USB serial so left is always left and you never pass `*_channel` flags again. Get the
serials (`udevadm info -a -p $(udevadm info -q path -n can0) | grep -m1 serial`), then add
`/etc/udev/rules.d/90-yam-can.rules`:

```
SUBSYSTEM=="net", ACTION=="add", ATTRS{serial}=="<left-arm-serial>",  NAME="can_follower_l"
SUBSYSTEM=="net", ACTION=="add", ATTRS{serial}=="<right-arm-serial>", NAME="can_follower_r"
```

```bash
sudo udevadm control --reload-rules && sudo udevadm trigger   # then replug the adapters
```

After that `ip link show type can` lists `can_follower_l` / `can_follower_r` and plain
`interlatent-node run --robot yam --robot-arg arms=both` works with no channel overrides.
i2rt/raiden's `rd reset_can` installs equivalent naming if you have raiden.

> **Note on shutdown noise.** On a clean exit, i2rt's background motor-control thread can
> log one round of `Bad file descriptor [9]` / `file descriptor cannot be a negative
> integer (-1)` / `Failed to communicate with motor … Retrying` *after*
> `YAMNativeRobot disconnected`. This is a teardown race inside the i2rt driver (the
> control thread outliving the closed CAN socket); torque is already zeroed by then, so it
> is cosmetic and safe to ignore.

## Joint names & units

`left_joint_0 … left_joint_5`, `left_gripper`, then the `right_*` block (left arm before
right). Arm joints are **radians**, gripper is **[0, 1]**. Limits come from the
[YAM `RobotProfile`](../../node/teleop/robot_profile.py) (arm limits transcribed from the
i2rt YAM URDF; velocity cap and gripper range are conservative — verify on hardware).

## Examples

```bash
# Read the left arm's pose, no motion
interlatent-act --robot yam --robot-arg arms=left --show

# Move the left base joint to 0.2 rad, hold the rest, open the gripper
interlatent-act --robot yam --robot-arg arms=left \
  left_joint_0=0.2 left_gripper=0.0 --hold-missing

# Run a bimanual policy session with two cameras and a tighter step clamp
interlatent-node run --robot yam \
  --robot-arg arms=both --robot-arg max_step_rad=0.3 \
  --camera overhead=zed:41234567 --camera wrist=realsense:1122
```

**Host requirements:** `pip install 'interlatent[yam]'`, Linux + SocketCAN, CAN buses up
(see [CAN bus setup](#can-bus-setup) above). The ZED SDK / `pyzed` is host-installed (not
on PyPI) and needed only for ZED cameras.
