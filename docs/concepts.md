# Concepts

The mental model. Precise term definitions live in [CONTEXT.md](../CONTEXT.md); the code
map lives in [ARCHITECTURE.md](../ARCHITECTURE.md).

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
4. The GPU pod conditions each inference on the actions the robot has already committed to
   ("RTC in-painting"), so chunk boundaries stitch into a continuous trajectory.
5. A latency estimator (Jacobson–Karels, the TCP RTT algorithm) splits round-trip into
   network vs. compute, so the client knows how far ahead it must stay scheduled
   (`min_execution_horizon`) and how often to request inference (`cooldown_steps`).

The result: smooth 30 Hz control over a model that thinks in seconds — across a LAN, a
VPN, or the public internet.

Policies too slow to win that race run in sequential mode instead — see *Chunk scheduling*
in [CONTEXT.md](../CONTEXT.md).

## Sessions

A robot opens a **session** against a GPU pod (`OpenSession`), binding it to a policy URI
and metadata (language `task`, `fps`, optional recording). The dashboard provisions the
pod, keeps policies warm-pooled, and returns the endpoint per-session.

## Observations and actions on the wire

Observations are opaque payloads (default codec: numpy `.npz`) whose keys mirror LeRobot
features: `observation.images.<camera>` (uint8 HWC), `observation.state` (float32),
`task` (str). Actions come back as float32 vectors of the policy's `action_dim`,
timestamped per control step.

## Environments and episodes

An **environment** is a label for one robot/policy collection (e.g. `"so101-kitchen"`); an
**episode** is one rollout. On Interlatent Cloud they're first-class objects: the
environment owns a canonical hosted LeRobot dataset accumulated across sessions, and
episodes get a dashboard viewer and analysis.

## Datasets

Everything records to **LeRobot v3.0 datasets** — parquet frames + MP4 video + JSON
metadata. Recording is **streaming-first**: your node JPEG-encodes each camera frame per
control tick and streams `RecordTick`s to the hosted recorder (the session's GPU pod, or a
teleop recorder pod), which persists every tick and builds the dataset at session close.
The finished dataset goes to a **destination** configured on the dashboard: the hosted
inbox, a local directory, or an S3-compatible bucket. The local/S3 destinations
*merge-on-stop* — each session is appended into one flat LeRobot dataset.

The uplink is lossless by design: ticks journal to a disk spool on the node and are
deleted only after the server acknowledges them, so a link drop or node crash never
silently thins an episode. The old client-side path (`watch()`/`tick()`/`upload()`) was
removed in SDK 2.0.0 —
[ADR 0018](adr/0018-collection-verbs-removed-streaming-only.md); datasets already on disk
enter the platform through the dashboard's HF import.

## The node

`interlatent-node` pairs a robot machine to your account
(`interlatent-node pair --name <name> --api-key ilat_…`), then polls the
[dashboard](https://interlatent.com) and converges to whatever inference session is
assigned to it (policy, cameras, DRTC endpoint). It is the managed counterpart of
hand-writing the `connect_drtc()` loop, which drives its own session —
[examples/03](../examples/03_run_on_so101.py).

## The dashboard CLI

`interlatent` is a thin client over the dashboard API — it is **not** a daemon. Authenticate
with `--api-key` or `INTERLATENT_API_KEY` (`ilat_…`); the base URL defaults to
https://interlatent.com (override with `--api-base` / `INTERLATENT_API_BASE`). Commands:

- `interlatent gpus ls` — GPU pods available to your account
- `interlatent nodes ls` — robot nodes paired to your account
- `interlatent session ls | start | stop` — e.g.
  `interlatent session start --node my-arm --gpu a100-0 --policy lerobot/smolvla_base`
- `interlatent env create --slug <slug>` — create an environment
- `interlatent behavior ls | validate | run` — offline named behaviors, no account needed
  (see [behaviors.md](behaviors.md))

Stopping a session closes the DRTC link, which is what triggers the pod to build and publish
any recorded dataset.
