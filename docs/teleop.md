# Teleoperation

**Teleop** is a human driving the robot in real time, in VR, over the network.
Today it exists for one purpose: **remote human demonstration** — a
[teleop recording](#teleop-recordings-no-policy) is a session with **no policy
loaded**, where the human drives end-to-end and every episode is captured as
training data. This is typically how you get from a pretrained base policy to
full automation: collect demonstrations in your environment, then train on
them.

> **Coming in a future release:** live *intervention* — taking over mid-rollout
> during a hosted inference session to correct the policy. It is not yet fully
> implemented and tested, and is not covered further here.

The split is **engine on the platform, thin receiver on the robot** (see
[ADR 0012](adr/0012-teleop-receiver-stub-open-core-boundary.md)): everything
that *computes* a joint target — VR pose retargeting, IK — runs off-robot, and
the node keeps only a receiver plus the last-hop safety clamp:

```
browser producer (dashboard)                                   robot (node)
────────────────────────────                                   ─────────────
WebXR VR overlay ── pose ──▶ hosted IK ── joint targets ──▶ ┐   TeleopChannel
                                                            ├──▶  │ SafetyGate
WebXR VR overlay ── in-browser IK ── joint targets ───────▶ ┘     │ send_action
```

Two paths exist — pose frames solved to joint targets on hosted compute, or a
low-latency path that runs IK **in the browser** and sends joint-target frames
directly (fed by a compact kinematic spec the node itself serves from its
installed robot data). The node selects automatically; you don't configure
this. Whichever path is in play, the node never runs IK, and **all motion
converges on one node-side action-driving interface**: absolute joint target →
`SafetyGate` → `send_action` — the same interface every action, human or
policy, ultimately passes through (see
[the action interface](action-interface.md#safety)).

The producer is browser-native and lives in the dashboard — there is nothing
to install on the operator's machine beyond a headset's browser.

## Driving a recording

1. **Run the node.** Teleop needs no flags: `interlatent-node` opens the teleop
   channel automatically whenever the dashboard assigns it a session. The only
   robot-side requirement is that the robot kind has a
   [`RobotProfile`](../packages/sdk/src/interlatent/node/teleop/robot_profile.py)
   and its robot data — see
   [Adding a teleop-capable robot](#adding-a-teleop-capable-robot). Without a
   profile, teleop is disabled for the session (the node reports
   `teleop_configured=false`).
2. **Start a teleop recording** from the dashboard. The node picks it up
   through its normal assignment poll.
3. **Enter VR:** open the session page in the headset's browser (Meta Quest)
   and *Enter VR*. The **grip button is a clutch**: the robot follows your hand
   only while it is held, anchored where the robot's end-effector actually is
   at the moment you engage — so you can ratchet: move, release, reposition
   your hand, re-engage. **Pinch** drives the gripper. Your hand pose becomes
   an absolute end-effector target and is solved to joint targets off-robot.

Releasing the clutch (the deadman) is a *soft* hand-back: the node holds the
current pose rather than snapping anywhere. Engaged ticks record
`control_source="teleop"`; disengaged ticks record `control_source="hold"`.

## Safety

Teleop safety runs **on the robot, next to the motors** — never on the relay or
in the browser:

- The **`SafetyGate`**
  ([`safety.py`](../packages/sdk/src/interlatent/node/teleop/safety.py)) is the
  single safety authority for human-driven motion. Every teleop target passes,
  in order: a **workspace clamp** (profile joint limits), a **velocity clamp**
  anchored to the last-commanded pose, the **deadman** (release holds pose),
  and a **staleness freeze** — the channel stops surfacing frames older than
  ~250 ms, so a dropped connection or hung producer holds the arm rather than
  chasing a stale target.
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
[ROBOT.md](../ROBOT.md); this section covers the teleop-specific half: the
robot data.

Each kind ships a small data bundle under
[`interlatent_robots/`](../packages/sdk/src/interlatent_robots/) (see
[ADR 0017](adr/0017-robot-data-ships-in-the-sdk.md) and the
[`interlatent_robots` README](../packages/sdk/src/interlatent_robots/README.md)):

```
packages/sdk/src/interlatent_robots/<kind>/
    __init__.py           # data-only marker; copy an existing kind's
    <robot>.urdf          # KINEMATICS-ONLY: links + joints + inertials
    ik_config.json        # hand-authored IK/tuning
    kinematic_spec.json   # GENERATED from the URDF + ik_config
```

The directory name **is** the `robot_kind` — it must equal the string the live
node reports (`--robot <kind>`).

**1. The URDF.** Kinematics only: links, joints, inertials — no
`<visual>`/`<collision>` geometry, no mesh references. IK is a function of the
joint tree alone (origins, axes, limits, tool frame), so meshes are
deliberately off the critical path.

**2. `ik_config.json`** — the hand-authored half: solver damping, reach
limits, scales, mounting frame (`webxr_to_base_R`), gripper range. This is the
tuning surface, read by the retarget stage that solves your hand pose to joint
targets.

**3. Compile `kinematic_spec.json`.** The spec is the compact serial-chain
descriptor the in-browser solver walks — **generated, never hand-edited**.
Produce it with the MuJoCo-backed exporter (an engine-side maintainer tool; the
shipped SDK has no MuJoCo dependency):

```bash
python -m interlatent.inference.server.retarget.kinematic_spec <bundle-dir>
```

Regenerate it after **any** URDF or `ik_config.json` change, or the browser
solver and the hosted solver silently disagree. A bundle missing the spec fails
in-browser IK loudly rather than solving against kinematics it isn't driving.

**4. Verify.** `packaging/verify_urdf.py` compiles the URDF exactly as the
engine does, confirms `ik_config.json` resolves (`ee_body` + every joint), and
runs FK parity between the compiled model and `kinematic_spec.json` — a spec
that drifted from the URDF fails loudly:

```bash
pip install mujoco numpy
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
from the dashboard; the node picks it up through the same assignment poll as an
inference session and runs its normal loop with the policy disabled. Engaged
ticks record `control_source="teleop"`; disengaged ticks hold pose and record
`control_source="hold"` (keeping the episode continuous). Stopping the
recording uploads it through the standard dataset path — it lands in the same
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
- **Paced below the offered recording bitrate**: the drop-oldest queue pins
  full, the recorded dataset *thins* (measured fps collapses), and the
  surviving frames are seconds old.

**Rule: set the cap to ~80% of your measured uplink, and never below the
offered recording bitrate.** On a ~30 Mbit/s uplink with three cameras, `3000`
is a confirmed operating point. Adding a camera or raising resolution moves the
offered bitrate — redo the arithmetic.

## What lands in the dataset

Every recorded step carries its provenance in
`annotation.interlatent.control_source` — `"teleop"` for human-driven ticks,
`"hold"` for disengaged ticks, `"policy"` for policy-driven steps in inference
sessions. Downstream training uses this to distinguish human demonstration
from policy behavior. Episodes with any teleop steps are flagged in the
dashboard.
