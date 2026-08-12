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

A robot opens a **session** against a GPU box (`OpenSession`) binding it to a policy URI and
metadata (language `task`, `fps`, optional recording). The coordinator names the box and
returns its endpoint per-session. A hosted coordinator additionally provisions the box and
keeps policies warm-pooled, which is why a hosted session starts inference quickly; a
self-hosted one brokers boxes you registered yourself.

## Observations and actions on the wire

Observations are opaque payloads (default codec: numpy `.npz`) whose keys mirror LeRobot
features: `observation.images.<camera>` (uint8 HWC), `observation.state` (float32),
`task` (str). Actions come back as float32 vectors of the policy's `action_dim`,
timestamped per control step.

## Environments and episodes

An **environment** is a label for one robot/policy collection (e.g. `"so101-kitchen"`); an
**episode** is one rollout. Every coordinator owns environments — they key the dataset a
session records into. On Interlatent Cloud they are additionally first-class objects with a
canonical hosted dataset accumulated across sessions, an episode viewer, and analysis.

## Datasets

Everything records to **LeRobot v3.0 datasets** — parquet frames + MP4 video + JSON
metadata. Recording is **streaming-first**:
your node JPEG-encodes each camera frame per control tick and streams `RecordTick`s to
the hosted recorder (the session's GPU pod, or a teleop recorder pod), which persists
every tick and builds the dataset at session close. The finished dataset is published to a **destination**: the hosted inbox, a local
directory, or an S3-compatible bucket. The local and S3 destinations *merge-on-stop* —
each session is appended into one flat, training-ready LeRobot dataset — and need no
account. Configure it once on your coordinator (`interlatent config --output-dir …`),
which stamps it onto every session it issues; the node forwards that block verbatim to
the box. `interlatent-serve --output-dir` / `--s3-uri` sets a per-box fallback.

The uplink is lossless by design: ticks journal to a disk spool on the node and are
deleted only after the server acknowledges them, so a link drop or node crash never
silently thins an episode. The old client-side path (`watch()`/`tick()`/`upload()`) was
removed in SDK 2.0.0 —
[ADR 0018](adr/0018-collection-verbs-removed-streaming-only.md); datasets already on disk
enter the platform through the dashboard's HF import.

## The coordinator

A **coordinator** is whatever service assigns work: it pairs nodes, tracks GPU boxes, brokers
inference and teleop sessions, and answers the long-poll each node converges against. The
[Interlatent dashboard](https://interlatent.com) is one; `interlatent up` runs another on your
own machine. They are two deployments of **one contract** — see
[the coordinator protocol](coordinator-protocol.md) — and the SDK never asks which one it is
talking to.

A coordinator address is required everywhere (`--coordinator`, `INTERLATENT_COORDINATOR`).
There is no default: silently defaulting to a hosted control plane is how a self-hosted fleet
ends up quietly phoning home.

A coordinator is never in the data path. DRTC is direct node↔box and teleop is
browser↔relay↔node, so a running session survives the coordinator's absence — the node keeps
driving the robot and its poll just backoff-retries.

## The node

`interlatent-node` is a long-running daemon for robots that should be remotely operable: it
pairs the machine to a coordinator (`interlatent-node pair --name <name> --coordinator <url>
--api-key …`), long-polls it, and converges to whatever session is assigned to it (policy,
cameras). The DRTC GPU endpoint is provided per-session by the coordinator. The node is the
managed counterpart of hand-writing the `connect_drtc()` loop — it relies on a coordinator for
session assignment, while the loop in [examples/03](../examples/03_run_on_so101.py) drives a
session itself.

## The CLI

`interlatent` is a session manager. It can **run** a coordinator, and it is a client of one:

```bash
interlatent up                       # start a coordinator here; prints an operator key
interlatent gpu add --name rig --url 10.0.0.7:50051
interlatent gpus ls                  # GPU boxes
interlatent nodes ls                 # robot nodes
interlatent session start --node my-arm --gpu rig --policy lerobot/smolvla_base
interlatent session stop <session-id>
interlatent config --output-dir /data/lerobot   # where recordings land
interlatent down                     # refuses while a session is live
```

The same commands drive the hosted dashboard with `--coordinator https://interlatent.com` —
it is one protocol, so there is no second set of verbs.

Stopping a session **unassigns** it. That is not a detail: the node's own teardown is what
sends `CloseSession`, which is the only trigger for the dataset build, and a box discards any
recording whose session never closed. Stopping by killing something loses the episode.
