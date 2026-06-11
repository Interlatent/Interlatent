# Teleoperate an SO-101 from your laptop

`interlatent-teleop` is a standalone laptop ↔ Pi teleoperation stack: gRPC streaming over
LAN or Tailscale, a 50 Hz control loop with a safety gate (workspace/velocity clamps,
deadman switch, staleness detection) on the robot side, and either keyboard control or
MediaPipe hand tracking on the laptop side.

## On the Pi (or whatever drives the arm)

```bash
pip install 'interlatent-teleop[so101]'
interlatent-teleop-pi --port /dev/ttyACM0
# gRPC server on :50061, homes the arm, waits for a laptop client
```

No SO-101 handy? Use the mock driver to try the loop end to end:

```bash
interlatent-teleop-pi --driver mock
```

## On the laptop

Keyboard control:

```bash
pip install 'interlatent-teleop[laptop]'
interlatent-teleop-keyboard --pi <pi-host>:50061
```

Hand tracking (MediaPipe, webcam):

```bash
interlatent-teleop-laptop --pi <pi-host>:50061
```

See [`packages/teleop/README.md`](../packages/teleop/README.md) for the keymap, retargeting
tunables (rates, leads, kinematics lengths), and safety-gate configuration.

## Recording demonstrations

To record while teleoperating, run the policy path with DAgger takeover instead: serve a
policy ([02_serve_policy.md](02_serve_policy.md)), drive the robot through the server, and
use the teleop relay (`:50052`) to take over when the policy drifts. The server's
`RecordTick` RPC captures every control tick — including which ticks were human overrides
(`control_source: "teleop"`), which is exactly the data you want for DAgger-style
fine-tuning.
