# Architecture

A contributor-facing map of how the pieces fit. The mental model (DRTC, sessions,
datasets) is in [docs/concepts.md](docs/concepts.md); the vocabulary is in
[CONTEXT.md](CONTEXT.md).

## The shape of the system

Big policies can't run on robot compute, and naive request/response inference makes arms
stutter. Interlatent's answer is **DRTC — Distributed Real-Time Chunking**
([how it works](docs/concepts.md#the-problem-drtc-solves)). Both ends of the loop live in
this repo — `packages/sdk` is the client, `packages/server` is the GPU box it talks to.

```
robot (client)                              GPU box (yours)
──────────────                              ───────────────
 sender thread  ── Observation stream ──▶   gRPC inference endpoint
                                              │ decode payload (npz/jpeg)
                                              │ policy forward()
 receiver thread ◀── ActionChunk stream ──   │ chunk buffer + schedule
      │
 LWW merge into action schedule
      │
 step() → next action at control rate
```

Where each property of that loop lives: `inference/client/merge.py` (last-writer-wins
schedule keyed on monotonic control timestamps), `inference/client/latency.py`
(Jacobson-Karels network-vs-compute split), `server/schedule.py` (RTC in-painting).

## Packages

### `packages/sdk` — pip `interlatent`, import `interlatent` (robot side)

| Area | Modules | Role |
|---|---|---|
| DRTC client | `inference/client/` (controller, sender, receiver, merge, latency, cooldown) | The real-time loop above |
| Wire protocol | `inference/protocol/` | Generated stubs from `proto/messages.proto` |
| Integration | `inference/integration/` (connect, preflight) | `connect_drtc()` — one-call session against a GPU box; `interlatent-preflight` |
| Node daemon | `node/` (cli, daemon, control, movement, looprunner) | `interlatent-node` — pairs to your coordinator, long-polls it, runs assigned sessions on real hardware. `movement.CommandBus` owns the motion path and `looprunner.run_control_loop` is the one tick skeleton every loop shares ([ADR 0022](docs/adr/0022-command-bus-owns-the-motion-path.md)) |
| Robot adapters | `adapters/` (base, lerobot, axol, yam, nori, dimos) | Per-vendor implementations of the adapter contract, imported lazily behind `interlatent[<kind>]` — see [ROBOT.md](ROBOT.md) |
| Teleop receiver | `node/teleop/` (factory, quic_channel, frame, safety, robot_profile) | Thin receiver for VR teleop — see below |
| Robot data | `interlatent_robots/<kind>/`, read via `robots.py` | Per-kind URDF + `kinematic_spec.json` shipped in the wheel ([ADR 0017](docs/adr/0017-robot-data-ships-in-the-sdk.md)) |
| Offline behaviors | `robot.py`, `behaviors/` | `il.Robot("so101").act("home")` — named min-jerk moves, no coordinator, no policy ([docs/behaviors.md](docs/behaviors.md)) |
| Coordinator | `coordinator/` (server, state, auth, protocol, supervisor, relay, certs) | The control plane itself: serves the HTTP contract in [docs/coordinator-protocol.md](docs/coordinator-protocol.md), issues the `ilop_`/`ilnode_`/`ilbox_` keys, and supervises the teleop QUIC relay. `protocol.py` is the route table the doc is pinned to |
| Coordinator CLI | `cli/main.py` | `interlatent` — runs a coordinator (`up`/`down`/`status`/`logs`/`config`, via `coordinator/supervisor.py`) and is a client of one ([commands](docs/concepts.md#the-cli)) |
| Tick spool | `inference/client/spool.py` | Write-through disk journal for the RecordTick uplink: delete-after-ack, drain-done at close, hard-stop when full |
| JPEG encode | `node/jpeg.py`, `node/nvjpeg.py`, `node/gpujpeg.py` | Capability-adaptive frame encoder, resolved once at runtime: nvJPEG → GPUJPEG → PyTurboJPEG → OpenCV → PIL |
| HTTP client | `_client.py`, `_resources.py` | `Interlatent` — environments/episodes API surface used by the daemon and CLI |

Collection is **streaming-first** ([ADR 0018](docs/adr/0018-collection-verbs-removed-streaming-only.md)):
devices never build datasets, they stream `RecordTick`s to the recorder on the GPU box,
which writes the finished dataset to a directory or an S3 bucket you own. Details in
[docs/concepts.md](docs/concepts.md#datasets).

### Teleop (VR remote demonstration)

A human drives the robot remotely in VR and every human-driven step is recorded — see
`control_source` in [CONTEXT.md](CONTEXT.md) for how demonstration and mid-policy
intervention are labelled apart. The split is **producer in the browser, thin receiver on
the robot** ([ADR 0012](docs/adr/0012-teleop-receiver-stub-open-core-boundary.md)):

- The WebXR producer (`teleop/teleop-web`, below) solves IK in the browser and streams
  **absolute joint targets** as unreliable datagrams over a WebTransport/QUIC relay.
- `node/teleop/` keeps only the receiver: `make_teleop_channel` builds a
  `QuicTeleopChannel` that decodes `TeleopFrame`s, tees live state and preview JPEGs back,
  and serves the robot's `kinematic_spec.json` to the browser. The aioquic connection runs
  in a child process ([ADR 0021](docs/adr/0021-quic-teleop-child-process.md)); this one is
  optional (`interlatent[teleop-quic]`). The `CommandBus` applies engaged `mode="targets"`
  frames through the `SafetyGate` before `send_action`.

Client-side safety is **layered** and runs next to the motors, never across the network:
the per-adapter delta clamp bounds every action, the `SafetyGate` adds
workspace/velocity/deadman limits on the teleop path. Both are defined in
[CONTEXT.md](CONTEXT.md).

### `packages/server` — pip `interlatent-server`, import `interlatent_server` (GPU side)

The other end of the DRTC loop
([ADR 0023](docs/adr/0023-self-hosted-policy-server-returns.md)). Run it on any CUDA
machine you control — a workstation under the desk or a box you rent from RunPod, Lambda
or Vast — and it registers with your coordinator as an available compute box, addressed at
whatever `--advertise-address` you give it. See
[docs/self-hosting.md](docs/self-hosting.md).

| Area | Modules | Role |
|---|---|---|
| Entry point | `cli.py` | `interlatent-serve` — mint/persist a box UUID, detect the GPU, register with the coordinator, then serve |
| Launcher | `serve_gpu.py`, `credentials.py`, `box_status.py` | Warmup, CPU isolation, gRPC bind, identity (the `ilop_` operator key of the coordinator it registers with), status self-reporting |
| Servicer | `server/transport.py`, `server/chunk_buffer.py`, `server/schedule.py`, `server/chunk_seam.py` | The RPCs, the chunk buffer, RTC in-painting reconstruction, and seam smoothing for backends that can't in-paint |
| Policy backends | `server/policy_runtime.py`, `server/lerobot_backend.py`, `server/molmoact2_backend.py`, `server/dreamzero_backend.py` (+ `server/dreamzero_sidecar.py`, `interlatent_dreamzero/`) | Load and run the policy; `torch`/`lerobot` are imported lazily so a recording-only box needs neither |
| Auth | `server/auth.py` | Owner-checked `x-api-key` on every RPC, on by default |
| Recording | `server/recorder.py`, `storage/lerobot_rebuild.py`, `storage/lerobot_live.py` | Ingest `RecordTick`s, build a LeRobot v3.0 dataset (live-encoded, with a rebuild fallback), land it in a local directory or an S3 bucket you own — or, with neither stamped on the session, push it through the protocol's optional presigned upload-url tier |

The two Python dists share **no** code — they run on different machines and are versioned
independently. They meet only at `proto/messages.proto`.

### `teleop/teleop-web`

A standalone WebXR PWA: the open-source VR producer for teleoperation. It solves IK in the
browser and streams absolute joint targets over WebTransport/QUIC to the node. Its engine
files carry the app's only real algorithmic content and its test suite — see its
[README](teleop/teleop-web/README.md).

### `proto/`

`messages.proto` is the single wire contract, and the single source of truth: both
`packages/sdk` and `packages/server` hold *mirrored* copies plus generated stubs, written
in one pass by `./proto/gen_proto.sh`. Never edit a mirror — `tests/test_proto_sync.py`
fails the build when one drifts. Compatibility rule: additive changes only. Details in
[proto/README.md](proto/README.md).

## Networking

Inference is gRPC (HTTP/2): `OpenSession` / `Stream` / `Infer` / `RecordTick` /
`RecordTicks` / `CloseSession`. The DRTC GPU endpoint is named per-session by the
coordinator and returned to the client (or node) when a session is assigned. Plain LAN, a
VPN, or the public internet all work — the client merges chunks the same way regardless of
the path's latency.

## The two contracts

There are **two contracts in this system**, and keeping them apart explains most of the
layout. `proto/messages.proto` is the *data* plane: the gRPC the client and node speak to
a GPU box. [`docs/coordinator-protocol.md`](docs/coordinator-protocol.md) is the *control*
plane: the HTTP a node, a box, the CLI and the teleop app speak to whatever assigns them
work. Both are additive-only.

The service on the other end of that second contract is a **coordinator**: a deployment has
exactly one — the one `interlatent up` runs for you — and the world has as many as there
are deployments. Nothing in the SDK branches on which coordinator it is talking to; it
speaks the one contract and goes to the address it is given. That address
is required everywhere (`--coordinator`, `INTERLATENT_COORDINATOR`) and has **no default**,
so a fleet never quietly phones somewhere you didn't point it. See
[ADR 0038](docs/adr/0038-coordinator-protocol-one-control-plane.md).

A coordinator is never in the data path. DRTC is direct node↔box, so a running session
survives the coordinator's absence — the node keeps driving the robot and its poll and
heartbeat simply backoff-retry.

Every runtime piece is yours to run and all of it is Apache-2.0: `packages/sdk` on the
robot, `packages/server` on the GPU machine, the coordinator wherever you like it, and
`teleop/teleop-web` for VR. Episodes are recorded through sessions (ADR 0022) and land in
a directory or an S3 bucket you own. What remains private is the offline half — the
dataset/canonical store and merge pipeline, offline policy improvement, and the annotation
stack.
