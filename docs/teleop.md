# Teleoperation

**Teleop** is a human driving the robot in real time, in VR, over the network.
Its purpose is **remote human demonstration**: a
[teleop recording](#teleop-recordings-no-policy) is a session with **no policy
loaded**, where the human drives end-to-end and every episode is captured as
training data — the usual path from a pretrained base policy to full
automation.

> **Live intervention:** engaging teleop *during* an inference session
> takes over from the running policy mid-rollout — the human-driven steps
> record as `control_source="intervention"` (the DAgger correction label), and
> releasing the deadman hands control back to the policy within about one
> control tick. Everything below applies the same way; the only difference is
> that a policy is running underneath.

The split is **producer in the browser, thin receiver on the robot** (see
[ADR 0012](adr/0012-teleop-receiver-stub-open-core-boundary.md)): everything
that *computes* a joint target — VR pose retargeting, IK — runs off-robot, and
the node keeps only a receiver plus the last-hop safety clamp:

```
browser producer                                        robot (node)
────────────────                                        ────────────
WebXR overlay ─ in-browser IK ─ targets ─(QUIC)─▶ QuicTeleopChannel ─▶ SafetyGate ─▶ send_action
```

Teleop runs over **QUIC/WebTransport only** — there is no WebSocket path
(`teleop.factory` builds the one channel, so you configure nothing). The
browser owns IK and solves against a compact kinematic spec the node itself
serves from its installed robot data. The node never runs IK, and **all motion
converges on one node-side action-driving interface**: absolute joint target →
`SafetyGate` → `send_action` — the same interface every action, human or
policy, passes through (see
[the action interface](action-interface.md#safety)). Wiring details:
[QUIC transport & process model](#quic-transport--process-model).

The producer is the browser-native teleop web app in
[`teleop/teleop-web/`](../teleop/teleop-web) — build it once and serve `dist/`;
there is nothing to install on the operator's machine beyond a headset's
browser.

## Driving a recording

1. **Run the node.** Teleop needs no flags: `interlatent-node` opens the teleop
   channel automatically whenever its coordinator assigns it a session. The node
   must have `aioquic` installed (the `interlatent[teleop-quic]` extra), and
   the robot kind needs a
   [`RobotProfile`](../packages/sdk/src/interlatent/node/teleop/robot_profile.py)
   plus its robot data shipped with the node's `interlatent` version — see
   [Adding a teleop-capable robot](#adding-a-teleop-capable-robot). Without a
   profile, teleop is disabled for the session (the node reports
   `teleop_configured=false`).
2. **Start a teleop recording** from the teleop web app: pick a node +
   environment and hit *Start recording*. It needs no GPU or policy choice, and
   enters VR on its own once the recording goes live; the node picks it up
   through its normal assignment poll.
3. **Enter VR:** open the session page in the headset's browser (Meta Quest)
   and *Enter VR*. The **grip/squeeze button is a clutch**: the robot follows
   your hand only while it is held, anchored where the robot's end-effector
   actually is at the moment you engage — so you can ratchet: move, release,
   reposition your hand, re-engage. The **trigger** drives the gripper (fully
   pulled = closed). Your hand pose becomes an absolute end-effector target and
   is solved to joint targets in the browser.

Releasing the clutch (the deadman) is a *soft* hand-back: the node holds the
current pose rather than snapping anywhere. Engaged ticks record
`control_source="teleop"`; disengaged ticks record `control_source="hold"`.

## Safety

Teleop safety runs **on the robot, next to the motors** — never on the relay or
in the browser:

- The **`SafetyGate`**
  ([`safety.py`](../packages/sdk/src/interlatent/node/teleop/safety.py)) is the
  single safety authority for human-driven motion. Every teleop target passes a
  **staleness freeze** (a target older than 200 ms holds the arm, and the
  channel stops surfacing frames older than 250 ms, so a dropped connection or
  hung producer can't leave the arm chasing a stale target), the **deadman**
  (release holds pose), a **workspace clamp** (profile joint limits), and a
  **velocity clamp** anchored to the last-*commanded* pose.
- The robot's **delta clamp** (`--robot-arg max_step=…`; `max_step_rad` on some
  adapters) caps the per-tick joint jump for *every* action, human-driven or
  not.

Both need the kind's `RobotProfile` (joint limits, velocity caps, rest pose) —
start conservative and widen only after watching real hardware.

### E-stop

The overlay's e-stop is the operator's **hard stop**, distinct from releasing
the deadman. It is latched sticky at frame-decode time, so a panic press can
never be lost to a dropped frame, staleness, or a disconnect — on receipt the
control loop latches the `SafetyGate` and the robot stops taking motion (see
[ADR 0016](adr/0016-teleop-estop-ingress-human-only-reset.md)).

Clearing is **never automatic**: end the session and start a new one. Robots
whose own daemon/firmware additionally hard-latches robot-side need an explicit
robot-side reset first — see your adapter's `CONFIG.md` for the recovery
procedure.

## Adding a teleop-capable robot

Teleop needs two things from a robot kind: a **`RobotProfile`** (what the
`SafetyGate` enforces) and the kind's **robot data** (what IK solves against).
The full bring-up path — profile, adapter, runtime knobs — is
[ROBOT.md](../ROBOT.md); this section covers the robot data.

Each kind ships a small data bundle under
[`interlatent_robots/`](../packages/sdk/src/interlatent_robots/) (see
[ADR 0017](adr/0017-robot-data-ships-in-the-sdk.md) and the
[`interlatent_robots` README](../packages/sdk/src/interlatent_robots/README.md)):

```
packages/sdk/src/interlatent_robots/<kind>/
    __init__.py           # data-only marker; copy an existing kind's
    <robot>.urdf          # KINEMATICS-ONLY: links + joints + inertials
    ik_config.json        # hand-authored IK/tuning (repo-only; not in the wheel)
    kinematic_spec.json   # GENERATED from the URDF + ik_config
```

The directory name **is** the `robot_kind` — it must equal the string the live
node reports (`--robot <kind>`).

**1. The URDF.** Kinematics only: links, joints, inertials — no
`<visual>`/`<collision>` geometry, no mesh references. IK is a function of the
joint tree alone (origins, axes, limits, tool frame).

**2. `ik_config.json`** — the hand-authored tuning surface: solver damping,
reach limits, scales, mounting frame (`webxr_to_base_R`), gripper range.

**3. Compile `kinematic_spec.json`.** The spec is the compact serial-chain
descriptor the in-browser solver walks — **generated, never hand-edited**.
Produce it with `packaging/kinematic_spec.py`, the MuJoCo-backed exporter (a
maintainer tool that ships in this repo but not in the wheel — nothing at
runtime needs MuJoCo, because the node serves the generated JSON verbatim):

```bash
pip install mujoco numpy
python packaging/kinematic_spec.py packages/sdk/src/interlatent_robots/<kind>
```

Pass `--stdout` to print the spec instead of writing it into the bundle —
useful for diffing a regenerated spec against the committed one before
replacing it.

Regenerate it after **any** URDF or `ik_config.json` change, or the browser
solver and the robot's real joint tree silently disagree. A bundle missing the
spec fails in-browser IK loudly rather than solving against kinematics it isn't
driving.

**4. Verify.** `packaging/verify_urdf.py` compiles the URDF exactly as the
exporter does, confirms `ik_config.json` resolves (`ee_body` + every joint), and
runs FK parity between the compiled model and `kinematic_spec.json` — a spec
that drifted from the URDF fails loudly. It reads the two files independently of
the exporter, so it catches a spec the exporter never produced:

```bash
python packaging/verify_urdf.py packages/sdk/src/interlatent_robots/<kind>
```

**5. Ship it.** Add the kind to `[tool.setuptools.package-data]` in
`packages/sdk/pyproject.toml` (or its data files won't ship);
`tests/test_robots.py` fails on a kind that is incomplete or mis-named.

Finally, register the kind's `RobotProfile` in
[`robot_profile.py`](../packages/sdk/src/interlatent/node/teleop/robot_profile.py)
— joint names and order, software limits, per-joint velocity caps, rest pose.
[ROBOT.md](../ROBOT.md) walks through authoring one.

## Teleop recordings (no policy)

A **teleop recording** is a full VR-teleop session with no policy loaded — the
human drives end-to-end and the episode is captured for training. Start one
from the teleop web app; the node picks it up through the same assignment poll
as an inference session and runs its normal loop with the policy disabled.
Engaged ticks record `control_source="teleop"`; disengaged ticks hold pose and
record `control_source="hold"` (keeping the episode continuous). Stopping the
recording publishes it through the standard dataset path — it lands in the same
per-environment LeRobot dataset as policy rollouts.

## Recording-uplink pacing

During teleop the node ships full-resolution record ticks live over your
uplink — three 480p cameras at 30 Hz offer ~30 Mbit/s, the same order as a
typical residential connection. `INTERLATENT_REC_MAX_KBPS` (KiB/s, default
`8000`; `0` disables pacing) caps that stream, and its job is **bufferbloat
headroom, not throttling**. Both failure directions are real:

- **Unpaced** at full uplink utilization: recording keeps up, but poses, the
  state heartbeat, and headset video queue behind saturated buffers — control
  latency runs hundreds of ms with multi-second spikes.
- **Paced below the offered recording bitrate**: the backlog grows in the
  node's disk spool (ticks journal locally and are deleted only on server
  ack), so the dataset can no longer *thin* — instead the spool fills toward
  its cap (`INTERLATENT_SPOOL_MAX_MB`, default 6 GiB) and capture
  **hard-stops** with a loud error, auto-resuming once the backlog drains.

**Rule: set the cap to ~80% of your measured uplink, and never below the
offered recording bitrate.** On a ~30 Mbit/s uplink with three cameras, `3000`
is a confirmed operating point. Adding a camera or raising resolution moves the
offered bitrate — redo the arithmetic (≈ cameras × pixels × fps × 1.2 bits/px
at q85; three 720p cameras ≈ 50–90 Mbit/s, wired-ethernet territory).

**Low-bandwidth recipe** (uplink below the offered bitrate): set a low cap
(e.g. `INTERLATENT_REC_MAX_KBPS=300`) and `INTERLATENT_PREVIEW_HZ=10`. The
session trickles; the backlog banks in the disk spool; the close-time drain
runs **unpaced** at line rate and blocks session close until the spool is
acked. The drain's hard ceiling scales with the banked bytes (assuming a
≥250 KiB/s link, floor 600 s; `INTERLATENT_REC_DRAIN_CEILING_S` forces a fixed
value), so a long session's tail is never guillotined mid-drain. If a drain
does give up (dead link), the un-acked tail is retained on disk and the close
log names the spool path — but a *completed* session's tail resumes only if
that session is re-assigned, and spool GC deletes it after 7 days, so treat
that warning as a call to action. Live preview competes with the recording
stream for the same uplink; on the QUIC transport the preview rate backs off
automatically under congestion (see [node-encoding.md](node-encoding.md)).

## QUIC transport & process model

The aioquic WebTransport connection runs in a **dedicated child process**
(`python -m interlatent.node.teleop._quic_proc`), not in the node's main
process. QUIC's handshake and loss timers live in userspace Python, and a busy
robot-driver GIL (e.g. i2rt's ~270 Hz threads) would starve them in process —
the child has its own GIL, so timer starvation is structurally impossible
([ADR 0021](adr/0021-quic-teleop-child-process.md)).

```
browser ⇄ (QUIC/WebTransport) ⇄ relay ⇄ (QUIC) ⇄ child proc ⇄ (loopback UDP) ⇄ parent (node)
  IK + preview decode          Fly           _quic_proc      _quic_ipc      QuicTeleopChannel
```

- **Parent** (`QuicTeleopChannel`) owns all protocol logic: dedupe, staleness,
  state echo, telemetry, and the preview **rate control** (`PreviewBackoff`). It
  supervises the child (stdin-EOF is the lifetime tether — no orphans; a crash
  respawns with 1→15 s backoff).
- **Child** (`_quic_proc`) is a dumb pipe: connect/handshake/reconnect, plus the
  video tee's stream **mechanism** (in-flight cap + TTL, `_VideoGovernor`). It
  never inspects payloads.
- **Transports on the wire:** control (targets browser→node, joint state
  node→browser) rides **unreliable datagrams** — small, latest-wins, must never
  queue. Preview video rides one short-lived **unidirectional stream per JPEG
  frame** (reliable within a frame, independent across frames; aioquic never
  GCs these on its own, so the connection discards each once acked —
  [ADR 0020](adr/0020-aioquic-uni-stream-discard.md)).
- **`request_spec` handshake:** on connect the browser sends a `request_spec`
  datagram; the node answers on a uni stream with its installed kinematic_spec
  (the browser builds its IK solver from *this node's* robot data, with no
  round-trip to anything else). If the node has no local data for its
  `robot_kind`, QUIC teleop does not start and the node logs it loudly.

Two knobs here are **dev-only**, not in the tables below:

- `INTERLATENT_TELEOP_INSECURE=1` disables TLS certificate verification on the
  node's QUIC connection — for a self-signed relay cert in local development
  **only**. Never set it in production (it defeats relay authentication).
- `INTERLATENT_LOG_LEVEL` (default `INFO`) sets the child process's log level;
  set `DEBUG` to trace reconnects. Non-default teleop knobs are echoed once at
  child startup (`teleop knob overrides: …`).

## Bandwidth knob reference

Everything the node sends over the uplink during a teleop session, and the
knob that controls each stream. Three streams share the link, in order of
latency-sensitivity: **control** (pose/state datagrams, ~tens of kbit/s,
not tunable and must never queue), **live preview** (what the operator
steers by), and **recording** (full-res ticks; laggy by design).

### Live preview (QUIC video tee)

Offered preview bitrate ≈ `HZ × cameras × frame_bytes`, where
`frame_bytes` is set by `MAX_DIM` and `JPEG_QUALITY` (~5–8 KB at 320 px
q70; ~4–6 KB at q50). Example: 24 Hz × 3 cams × 6 KB ≈ 430 KiB/s
≈ 3.5 Mbit/s.

| Env var | Default | Effect on bandwidth / latency |
|---|---|---|
| `INTERLATENT_PREVIEW_HZ` | `30` (clamp 1–30) | Per-camera frame-rate **ceiling**. Linear in bandwidth. Mean perceived video staleness ≈ half the period, so this is also the dominant smoothness knob. Read once at node start. |
| `INTERLATENT_PREVIEW_MAX_DIM` | `320` (64–1280) | Long-side downscale before JPEG encode. Bytes scale ~quadratically with dimension: 640 px is ~4× the bits of 320 px. Live-dialable (read per frame). |
| `INTERLATENT_PREVIEW_JPEG_QUALITY` | `70` (10–95) | JPEG q. q50 ≈ 25–30 % smaller than q70; above ~q85 bytes balloon fast. Live-dialable. |
| `INTERLATENT_QUIC_VIDEO` | `1` | Kill switch: `0` sends **zero** preview bytes (control unaffected). |
| `INTERLATENT_QUIC_VIDEO_INFLIGHT` | `2` per cam (1–16) | Not a bitrate knob — the **queue-depth** knob. Frames in flight per camera (global cap 3×). Higher hides RTT (more fps on a fast, long-RTT link) but builds a standing queue on a thin link: every extra in-flight frame is ~one frame of added glass-to-eye latency under congestion. `1` = fully closed-loop, lowest latency, fps capped at 1/completion-time. |
| `INTERLATENT_QUIC_VIDEO_TTL_MS` | `350` (50–5000) | Freshness ceiling: a frame still undelivered after this is RESET (bytes abandoned, not retransmitted). Bounds worst-case preview age; TTL resets are also the **only** congestion signal that backs the preview rate off. |
| `INTERLATENT_PREVIEW_ADAPTIVE` | `1` | `0` disables the staleness-driven rate backoff (rate stays pinned at `PREVIEW_HZ` no matter what). Backoff floor is 1 Hz. |

The preview governor drops frames at admission when the link can't keep
up (`drop_cap` in the `quic-proc pumped` log line) — that is free,
intended pacing, not loss. The `qs=` gauge on the same line is aioquic
stream bookkeeping and should stay in the single digits
([ADR 0020](adr/0020-aioquic-uni-stream-discard.md)).

### Recording uplink + disk spool

Offered recording bitrate ≈ `cameras × pixels × fps × ~1.2 bits/px` at
q85 (three 480p cams @ 30 fps ≈ 30 Mbit/s; three 720p ≈ 50–90 Mbit/s).
See “Recording-uplink pacing” above for the failure modes.

| Env var | Default | Effect |
|---|---|---|
| `INTERLATENT_REC_MAX_KBPS` | `8000` KiB/s (`0` = unpaced) | Caps the live recording stream. Rule: ~80 % of measured uplink, and not below the offered bitrate unless you are deliberately banking to the spool (low-bandwidth recipe above). This cap is what keeps recording from bufferbloating control and preview. |
| `INTERLATENT_SPOOL_MAX_MB` | `6144` (6 GiB) | Disk cap for the banked backlog. Full spool ⇒ capture **hard-stops** (with hysteresis) until the backlog drains. |
| `INTERLATENT_SPOOL_MIN_FREE_MB` | `2048` | Free-disk floor — spool stops growing before the device does. |
| `INTERLATENT_SPOOL_DIR` | `~/.interlatent/spool` | Where the backlog lives. Orphan spools from interrupted sessions are GC'd after 7 days. |
| `INTERLATENT_REC_DRAIN_CEILING_S` | auto (scales with backlog, floor 600 s) | Forces a fixed ceiling on the unpaced close-time drain. |

### Inference uplink

During policy sessions the node also ships observation frames to the
DRTC box each inference call. `--image-resize <px>` /
`INTERLATENT_IMAGE_RESIZE` resizes frames to
that square edge before encode; leave it unset to send native camera
resolution. Resizing to the policy's input size cuts this stream the
same quadratic way `PREVIEW_MAX_DIM` cuts the preview.

### Interactions worth knowing

- All three streams share one uplink. The recording cap exists to
  protect the other two; the preview knobs exist to fit what's left.
- Preview fps on a thin link is delivery-limited, not offer-limited:
  raising `PREVIEW_HZ` past what the link completes just raises
  `drop_cap`, it does not add bandwidth. Lower `MAX_DIM`/`QUALITY` to
  make frames cheaper before raising the rate.
- If preview latency grows *within* a session while fps falls and
  `reset_ttl` stays 0, that is not a bandwidth problem — check the
  `qs=` gauge first ([ADR 0020](adr/0020-aioquic-uni-stream-discard.md)).

## What lands in the dataset

Every recorded step carries its provenance in
`annotation.interlatent.control_source` — `"teleop"` for human-driven ticks in
a policy-less recording, `"intervention"` for human-driven ticks that overrode
a running policy mid-rollout, `"hold"` for disengaged/no-motion ticks,
`"policy"` for policy-driven steps in inference sessions. Downstream training
uses this to distinguish human demonstration and correction from policy
behavior — intervention frames are the DAgger corrections and are upweighted.
Filtering an episode on this field is how you find the human-driven steps in a
mixed rollout.
