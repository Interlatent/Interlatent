# Axol robot configuration

Robot-specific arguments for `--robot axol` (Almond Axol dual-arm robot, driven through
the native async `almond_axol` SDK). Parsed by [`config.py`](config.py)
`build_adapter_config`.

Arguments are passed two ways:

- `--robot-arg key=value` — repeatable; one knob each (table below).
- `--camera name=<serial>` — repeatable; opens an onboard ZED **by serial number**.

Unrecognized `--robot-arg` keys are warned about and ignored. `--port` is **not** used by
Axol (it speaks CAN, not serial). At least one `--camera` is **required**.

## `--robot-arg` keys

| Key | Default | Values | Description |
|---|---|---|---|
| `left_channel` | `almond_axol` `CAN_LEFT` | SocketCAN iface name | CAN bus for the left arm. |
| `right_channel` | `almond_axol` `CAN_RIGHT` | SocketCAN iface name | CAN bus for the right arm. |
| `config_path` | _(none)_ | path | A full native `AxolConfig` (deep per-joint gains) loaded via draccus; flat overlays below are applied on top. |
| `left_stiffness` | _(AxolConfig)_ | scalar or comma 7-vector (ARM_JOINTS order) | Impedance stiffness override. **Must match data collection.** |
| `right_stiffness` | _(AxolConfig)_ | scalar or comma 7-vector | Impedance stiffness override. **Must match data collection.** |
| `max_step_rad` | _AxolConfig (≈`0.5`)_ | float (rad) | Execution-safety per-step delta clamp on arm joints (the gripper, index 7, is not clamped). |
| `telemetry_hz` | `120.0` | float | CAN telemetry streaming rate for cached positions. |
| `observe_torques` | `false` | bool | Include joint torques in telemetry. |
| `gripper_mode` | `continuous` | `continuous` \| `bangbang` | `continuous` passes the gripper value through; `bangbang` snaps at `gripper_threshold`. |
| `gripper_threshold` | `0.5` | float `[0,1]` | Snap point for `bangbang` gripper mode. |
| `stereo` | `false` | bool | Open every camera as a stereo ZED X (expands each into `<name>_left` / `<name>_right` views). |
| `resolution` | _native SVGA (960×600)_ | ZED resolution name (`SVGA`, `HD1080`, `HD1200`, …) | ZED capture resolution; must be one of the native `ZED_RESOLUTION_DIMS`. |
| `camera_fps` | _native (60)_ | int | ZED capture frame rate. |
| `restart_zed_daemon` | `true` | bool | Restart the Jetson `zed_x_daemon` before opening cameras so a GMSL camera plugged in after boot is enumerable (needs passwordless sudo). |

Bool values accept `1/true/yes/on` (case-insensitive); anything else is false.

## `--camera` declarations

`--camera <name>=<serial>`

- `<name>` — observation key; **must match the policy's training camera keys** (e.g.
  `overhead` / `left_arm` / `right_arm`).
- `<serial>` — the **ZED serial number** (integer) of the GMSL-attached camera, opened
  onboard the Jetson by `almond_axol`'s native ZED camera.

With `stereo=true`, each camera expands into `<name>_left` and `<name>_right` views.

## Joint names & units

16 keys: `left_<joint>.pos` (7 arm joints + gripper) then the `right_*` block, in the
`almond_axol` `Joint` enum order. Arm joints are **radians**; the gripper is the last
index of each arm. Axol has **no `RobotProfile` yet**, so manual `action()` is unavailable
(it fails closed) — the engine/policy path (`send_action` per tick) does not need one.

## Examples

```bash
# Bimanual policy session, three onboard ZEDs by serial, stiffness matched to training
interlatent-node run --robot axol \
  --robot-arg left_stiffness=200 --robot-arg right_stiffness=200 \
  --robot-arg resolution=HD1080 --robot-arg camera_fps=30 \
  --camera overhead=41234567 --camera left_arm=42345678 --camera right_arm=43456789
```

**Host requirements:** `pip install 'interlatent[axol]'` (Python ≥ 3.13; pulls `lerobot`
via `almond-axol`), the ZED SDK + `pyzed` host-installed, and a Jetson with the arms on
CAN.
