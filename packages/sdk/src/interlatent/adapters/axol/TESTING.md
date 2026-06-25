# Axol adapter — hardware test checklist

Items that can only be verified on the real Axol robot + connected ZED cameras.
The automated bring-up check (`hardware_check.py`, run onboard the Jetson via
`python -m interlatent.adapters.axol.hardware_check --camera <name>=<serial> ...`)
covers the action-write + observation-read subset; this checklist covers the
rest. The full loop runs via `interlatent-node run --robot axol ...` with
`interlatent[axol]` installed.

## Joint / action contract

- [ ] `list(Joint)` order (left arm → right arm, gripper last) matches the order
      `Axol.left/right.positions` returns.
- [ ] That same order matches the order the served policy was trained on
      (state and action not transposed).
- [ ] `Axol.left.positions` / `right.positions` are each length 8, gripper at index 7.
- [ ] Served policy action dim == 16; values land on the correct joints.
- [ ] `observation.state` vector order matches `action_features` (no image array
      leaking into the state vector).

## Motion / safety

- [ ] Gripper command convention is normalized `[0, 1]` (not radians/meters); the
      0/1 bang-bang snap actually opens/closes the gripper.
- [ ] `motion_control` drops over-`max_step_rad` commands (does not clamp or
      execute them); the adapter's held-target bookkeeping stays in sync.
- [ ] `max_step_rad` field exists on the native `AxolConfig` and is in radians
      (gate not silently falling back to 0.5).
- [ ] Grippers are exempt from the `max_step_rad` gate on the real robot.
- [ ] A blocked/stalled `motion_control` (contact) raising after the 1s timeout
      ends the episode safely (acceptable failure mode).
- [ ] `disconnect` disables motors even when a camera teardown hangs.
- [ ] Configured arm stiffness matches what training used.

## Camera / ZED

- [ ] Each `--camera <name>=<serial>` opens the intended physical camera (the SDK
      verifies the opened serial matches the requested one); `find_cameras()`
      lists the connected serials.
- [ ] `restart_zed_daemon` brings up a camera plugged in after boot; on a node
      without passwordless sudo, `--robot-arg restart_zed_daemon=false` still
      connects when the daemon is already up.
- [ ] In a container, the host `zed_x_daemon` is shared in: mount the WHOLE host
      `/tmp` (`-v /tmp:/tmp`), not just `/tmp/argus_socket` — the daemon's control
      socket is a separate file in `/tmp`. Mounting only the Argus socket fails
      with "Failed to connect to daemon socket: No such file or directory". Also
      mount `/var/nvidia/nvcam/settings/`, `/usr/local/zed/{settings,resources}/`,
      run with `--runtime nvidia --ipc host --pid host -e NVIDIA_DRIVER_CAPABILITIES=all`,
      and pass `restart_zed_daemon=false` (the container can't restart the host service).
- [ ] Onboard capture is single-clock: `read_at_or_after` aligns frames by the
      local capture timestamp (no PTP / `zed.sync-clocks` needed).
- [ ] Live camera fps/width/height match the config (default SVGA 960x600 @ 60),
      overridable via `--robot-arg resolution=... camera_fps=...`; connect succeeds.
- [ ] Degraded/dropped link: `read_latest` stale-frame path does not hard-crash
      the episode uncaught.
- [ ] Link recovery (`async_grab_camera_recovery`) reconnects with the installed
      ZED SDK version.
- [ ] No SDK-handle / FD leak over repeated reconnects during a long soak.
- [ ] BGRA→RGB conversion produces correct color (ZED returns 4-channel).
- [ ] CAN channel names (`CAN_LEFT` / `CAN_RIGHT`) exist on the Jetson
      (`axol can.setup` was run there).

## Install / runtime

- [ ] `interlatent[axol]` installs cleanly via `uv` on Python >= 3.13
      (`pip` does not honor almond-axol's source pins).
- [ ] Private `almond-axol` git dep authenticates and resolves.
- [ ] Full loop sustains target FPS (telemetry 120 Hz + observe + send_action
      within the loop period).
- [ ] Feature-element-names get reported to the backend (episode records with
      named state/action elements).
