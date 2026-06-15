# Concepts

## The problem DRTC solves

A VLA policy takes 100–2000 ms per inference. A robot needs an action every 33 ms (30 Hz).
Request/response inference therefore can't drive a robot — the arm would freeze between
requests.

**DRTC (Distributed Real-Time Chunking)** decouples the two clocks:

1. Policies emit **action chunks** — a window of future actions per inference (e.g. 32–50
   steps), not a single action.
2. The robot streams observations continuously and keeps a **schedule** of upcoming
   actions; each control tick consumes the next scheduled action — never waiting on the
   network.
3. New chunks **overlap** old ones. The client merges them last-writer-wins on monotonic
   control timestamps, so a fresher inference always overrides stale plans.
4. The server conditions each inference on the actions the robot has already committed to
   ("RTC in-painting"), so chunk boundaries stitch into a continuous trajectory.
5. A latency estimator (Jacobson–Karels, the TCP RTT algorithm) splits round-trip into
   network vs. compute, so the client knows how far ahead it must stay scheduled
   (`min_execution_horizon`) and how often to request inference (`cooldown_steps`).

The result: smooth 30 Hz control over a model that thinks in seconds — across a LAN, a
tailnet, or the public internet.

## Sessions

A robot opens a **session** against a server (`OpenSession`) binding it to a policy URI and
metadata (language `task`, `fps`, optional recording). The server loads the policy once per
process per `(backend, policy_uri)` and reuses it across sessions — that's why a warm
server starts sessions instantly.

## Observations and actions on the wire

Observations are opaque payloads (default codec: numpy `.npz`) whose keys mirror LeRobot
features: `observation.images.<camera>` (uint8 HWC), `observation.state` (float32),
`task` (str). Actions come back as float32 vectors of the policy's `action_dim`,
timestamped per control step.

## Environments and episodes

An **environment** is a label for one robot/policy collection (e.g. `"so101-kitchen"`); an
**episode** is one rollout. Offline these are just names stamped into your local files. On
Interlatent Cloud they're first-class objects: the environment owns a canonical hosted
LeRobot dataset accumulated across sessions, and episodes get a dashboard viewer and
analysis.

## Datasets

Everything records to **LeRobot v3.0 datasets** — parquet frames + MP4 video + JSON
metadata, the lingua franca of open robot learning. Two recording paths:

- **Client-side** (`watch()`/`tick()` in the SDK): steps stage to local SQLite + JPEGs;
  `LeRobotRebuilder` emits the dataset on your disk.
- **Server-side** (`RecordTick` RPC): the GPU box persists each control tick — including
  whether it was policy or human teleop — and builds the dataset at session close. This is
  how DAgger-style takeover data gets captured without the robot staging anything. The
  finished dataset is published to a **destination**: the hosted inbox (Cloud), a local
  directory, or an S3-compatible bucket. The local/S3 destinations *merge-on-stop* — each
  session is appended into one flat, training-ready LeRobot dataset, no account required.

## The node

`interlatent-node` is a long-running daemon for robots that should be remotely operable: it
pairs the machine to a **coordinator** (your own, or Interlatent Cloud), heartbeats, and
converges to whatever inference session is assigned to it (policy, cameras, DAgger keyboard
takeover). The node is the managed counterpart of hand-writing the `connect_drtc()` loop —
it needs a coordinator for session assignment, while the loop in
[examples/03](../examples/03_run_on_so101.py) is fully self-contained.

## The coordinator

The **coordinator** is the control plane that assigns sessions to nodes. It is the
self-hosted, offline replacement for Interlatent Cloud's session assignment: `interlatent up`
runs a small local HTTP service that speaks the same API the node polls, and
`interlatent gpu add` / `interlatent session start` register GPU boxes and drive sessions —
no account, no dashboard. Stopping a session **unassigns** it; the node then closes the DRTC
session, which is what triggers the server to build and publish the recorded dataset. The
coordinator is *only* a control plane — the inference link is direct node↔GPU and keeps
running even if the coordinator goes down. See [self-hosting.md](self-hosting.md) and
[ADR-0001](adr/0001-offline-coordinator-control-plane.md).

## Teleoperation

Two distinct surfaces:

- **`interlatent-teleop`** — standalone laptop ↔ Pi teleop (keyboard / MediaPipe hand
  tracking → 50 Hz safety-gated control loop). No server, no policy involved.
- **Teleop relay** (server `:50052`) — lets an operator take over *during a policy
  rollout* (DAgger). Override ticks are flagged `control_source: "teleop"` in recordings.
