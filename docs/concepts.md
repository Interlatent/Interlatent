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
4. The GPU pod conditions each inference on the actions the robot has already committed to
   ("RTC in-painting"), so chunk boundaries stitch into a continuous trajectory.
5. A latency estimator (Jacobson–Karels, the TCP RTT algorithm) splits round-trip into
   network vs. compute, so the client knows how far ahead it must stay scheduled
   (`min_execution_horizon`) and how often to request inference (`cooldown_steps`).

The result: smooth 30 Hz control over a model that thinks in seconds — across a LAN, a
VPN, or the public internet.

## Sessions

A robot opens a **session** against a managed GPU pod (`OpenSession`) binding it to a policy
URI and metadata (language `task`, `fps`, optional recording). The dashboard provisions the
pod and keeps policies warm-pooled — that's why a session starts inference quickly. The pod
endpoint is provisioned per-session by the dashboard.

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
  `LeRobotRebuilder` emits the dataset on your disk. Fully offline, no account.
- **Pod-side** (`RecordTick` RPC): the GPU pod persists each control tick and builds the
  dataset at session close. The finished dataset is published to a **destination**: the
  hosted inbox (Cloud), a local directory, or an S3-compatible bucket. The local/S3
  destinations *merge-on-stop* — each session is appended into one flat, training-ready
  LeRobot dataset.

## The node

`interlatent-node` is a long-running daemon for robots that should be remotely operable: it
pairs the machine to your account (`interlatent-node pair --name <name> --api-key ilat_…`), polls the
[dashboard](https://interlatent.com), and converges to whatever inference session is
assigned to it (policy, cameras). The DRTC GPU endpoint is provided per-session by the
dashboard. The node is the managed counterpart of hand-writing the `connect_drtc()` loop —
it relies on the dashboard for session assignment, while the loop in
[examples/03](../examples/03_run_on_so101.py) drives a session itself.

## The dashboard CLI

`interlatent` is a thin client over the dashboard API — it is **not** a daemon. Authenticate
with `--api-key` or `INTERLATENT_API_KEY` (`ilat_…`); the base URL defaults to
https://interlatent.com (override with `--api-base` / `INTERLATENT_API_BASE`). Commands:

- `interlatent gpus ls` — GPU pods available to your account
- `interlatent nodes ls` — robot nodes paired to your account
- `interlatent session ls | start | stop` — e.g.
  `interlatent session start --node my-arm --gpu a100-0 --policy lerobot/smolvla_base`

Stopping a session closes the DRTC link, which is what triggers the pod to build and publish
any recorded dataset.
