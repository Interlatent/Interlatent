# interlatent-teleop

Direct laptop ↔ Pi teleoperation for Interlatent-supported arms (SO-101 for v0). gRPC over Tailscale (or LAN), no GPU in the loop.

This is an isolated package — it does not import from `interlatent` (the SDK / engine) and the SDK does not import from it. The top-level Python module is `interlatent_teleop` so the namespaces stay disjoint.

## Architecture

```
Laptop (MediaPipe hand tracker)
   │  gRPC bidi stream (Tailscale / LAN)
   ▼
Pi (TeleopServer)
   ├── SafetyGate: workspace clamp, velocity clamp, deadman, staleness
   ├── ControlLoop: 50 Hz, owns the SO-101 motor writes
   └── SO101Driver: lerobot wrapper (or MockDriver for dev)
```

The Pi advertises joint names, limits, and max velocity on `OpenTeleop`. The laptop calibrates a neutral hand pose against the home joints and sends linear-offset target poses at ~30 Hz. The Pi interpolates to 50 Hz under a velocity clamp.

## Install

On the **Pi** (the one wired to the SO-101):

```bash
pip install 'interlatent-teleop[so101]'        # or from a checkout: pip install -e './packages/teleop[so101]'
```

On the **laptop** (the producer with the webcam):

```bash
pip install 'interlatent-teleop[laptop]'       # or from a checkout: pip install -e './packages/teleop[laptop]'
```

Both ends share the base package (gRPC + numpy + protocol stubs); the extras only differ in hardware-side deps.

## Run

Pi:

```bash
interlatent-teleop-pi --driver so101 --port /dev/ttyACM0 --control-hz 50
# or, on a dev machine without an arm:
interlatent-teleop-pi --driver mock
```

Laptop:

```bash
interlatent-teleop-laptop --pi 100.x.y.z:50061
```

Replace `100.x.y.z` with the Pi's Tailscale IP (or LAN IP if you're testing on the same network). Default port is `50061`.

### Controls

| Key       | Action                                          |
|-----------|-------------------------------------------------|
| SPACE     | Toggle deadman (motion is only commanded while ARMED) |
| `c`       | Re-calibrate the neutral hand pose              |
| `q` / ESC | Quit                                            |

Auto-calibration runs the first time a hand is detected, so the typical flow is: launch the laptop CLI → put your hand in a neutral pose → press SPACE to arm.

## Auth

Pass `--session-token <secret>` to the Pi and the same value to the laptop (or set `INTERLATENT_TELEOP_TOKEN` in the environment on both). When unset, the Pi accepts any client — fine for closed-network use over Tailscale, not for exposing the Pi on the public internet.

## Safety envelope

Defined in `interlatent_teleop/common/config.py`. SO-101 defaults:

- Joint limits: per joint, `±100°`–`±180°` for the arm joints, `0..100` for the gripper
- Max velocity: 50–240 °/s per joint (gravity-loaded joints are capped lowest), 400 °/s for gripper
- Producer-side: low-pass smoothing (`α=0.4`) on retargeted joints, target clamp before send
- Pi-side: workspace clamp → velocity clamp → 200 ms staleness freeze → deadman → optional confidence threshold
- E-stop: a failed motor write latches an e-stop that freezes the arm. Release and
  re-press the deadman to clear it; a persistent hardware fault re-latches on the
  next write, so a broken arm never resumes for more than one tick.

Adjust limits in `common/config.py` after you have validated the loop end-to-end on real hardware. Defaults are intentionally conservative.

## Protocol

`src/interlatent_teleop/protocol/teleop.proto` is the source of truth. Regenerate stubs after editing:

```bash
bash packages/teleop/scripts/gen_proto.sh
```

## MVP scope

Implemented (v0.1):

- gRPC bidi protocol (`OpenTeleop` / `Stream` / `CloseTeleop`)
- MediaPipe hand tracker with handedness preference
- 3DOF + gripper retargeter (locked wrist); auto-calibration on first detection
- Pi-side safety gate, 50 Hz control loop, lerobot SO-101 driver + mock driver
- CLIs on both ends

Not yet:

- Wrist flex / wrist roll (need palm-orientation extraction from landmarks)
- Recording teleop sessions as demonstration data (would reuse the SDK's collection layer)
- Discovery via the site backend (today: laptop hardcodes the Pi address)
- Browser/phone producer (WebRTC data channels speaking the same proto)
- TLS / mutual auth on the gRPC channel (today: insecure_channel + token)

These are intentionally out of v0 — the loop has to feel right with the simplest possible producer before adding surface area.
