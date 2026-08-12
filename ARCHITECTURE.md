# Architecture

A contributor-facing map of how the pieces fit. The mental model (DRTC, sessions,
datasets) is in [docs/concepts.md](docs/concepts.md); the vocabulary is in
[CONTEXT.md](CONTEXT.md).

## The shape of the system

Big policies can't run on robot compute, and naive request/response inference makes arms
stutter. Interlatent's answer is **DRTC — Distributed Real-Time Chunking**
([how it works](docs/concepts.md#the-problem-drtc-solves)). Both ends of the loop live in
this repo — `packages/sdk` is the client, `packages/server` is the GPU box it talks to.
Interlatent's managed boxes run the same server code.

```
robot (client)                              GPU box (managed or your own)
──────────────                              ─────────────────────────────
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
| Integration | `inference/integration/` (connect, preflight) | `connect_drtc()` — one-call session against a dashboard-provisioned GPU pod; `interlatent-preflight` |
| Node daemon | `node/` (cli, daemon, control, movement, looprunner) | `interlatent-node` — pairs to your account, polls the dashboard, runs assigned sessions on real hardware. `movement.CommandBus` owns the motion path and `looprunner.run_control_loop` is the one tick skeleton every loop shares ([ADR 0022](docs/adr/0022-command-bus-owns-the-motion-path.md)) |
| Robot adapters | `adapters/` (base, lerobot, axol, yam, nori, dimos) | Per-vendor implementations of the adapter contract, imported lazily behind `interlatent[<kind>]` — see [ROBOT.md](ROBOT.md) |
| Teleop receiver | `node/teleop/` (factory, quic_channel, frame, safety, robot_profile) | Thin receiver for VR teleop — see below |
| Robot data | `interlatent_robots/<kind>/`, read via `robots.py` | Per-kind URDF + `kinematic_spec.json` shipped in the wheel ([ADR 0017](docs/adr/0017-robot-data-ships-in-the-sdk.md)) |
| Offline behaviors | `robot.py`, `behaviors/` | `il.Robot("so101").act("home")` — named min-jerk moves, no account, no policy ([docs/behaviors.md](docs/behaviors.md)) |
| Coordinator CLI | `cli/main.py` | `interlatent` — thin client over the dashboard API ([commands](docs/concepts.md#the-dashboard-cli)) |
| Tick spool | `inference/client/spool.py` | Write-through disk journal for the RecordTick uplink: delete-after-ack, drain-done at close, hard-stop when full |
| JPEG encode | `node/jpeg.py`, `node/nvjpeg.py`, `node/gpujpeg.py` | Capability-adaptive frame encoder, resolved once at runtime: nvJPEG → GPUJPEG → PyTurboJPEG → OpenCV → PIL |
| HTTP client | `_client.py`, `_resources.py` | `Interlatent` — environments/episodes API surface used by the daemon and CLI |

Collection is **streaming-first** ([ADR 0018](docs/adr/0018-collection-verbs-removed-streaming-only.md)):
devices never build datasets, they stream `RecordTick`s to a hosted recorder. Details in
[docs/concepts.md](docs/concepts.md#datasets).

### Teleop (VR remote demonstration)

A human drives the robot remotely in VR and every human-driven step is recorded — see
`control_source` in [CONTEXT.md](CONTEXT.md) for how demonstration and mid-policy
intervention are labelled apart. The split is **producer in the browser, thin receiver on
the robot** ([ADR 0012](docs/adr/0012-teleop-receiver-stub-open-core-boundary.md)):

- The WebXR producer — the dashboard's, or the `teleop/teleop-web` fork below — solves IK
  in the browser and streams **absolute joint targets** as unreliable datagrams over a
  hosted WebTransport/QUIC relay.
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

The other end of the DRTC loop, and the same code Interlatent's hosted boxes run
([ADR 0023](docs/adr/0023-self-hosted-policy-server-returns.md)). Run it on your own
CUDA machine and it registers with the dashboard as a self-hosted compute box — see
[docs/self-hosting.md](docs/self-hosting.md).

| Area | Modules | Role |
|---|---|---|
| Entry point | `cli.py` | `interlatent-serve` — mint/persist a box UUID, detect the GPU, register with the dashboard, then serve |
| Launcher | `serve_gpu.py`, `credentials.py`, `box_status.py` | Warmup, CPU isolation, gRPC bind, identity (hosted admin key vs owner `ilat_` key), status self-reporting |
| Servicer | `server/transport.py`, `server/chunk_buffer.py`, `server/schedule.py`, `server/chunk_seam.py` | The RPCs, the chunk buffer, RTC in-painting reconstruction, and seam smoothing for backends that can't in-paint |
| Policy backends | `server/policy_runtime.py`, `server/lerobot_backend.py`, `server/molmoact2_backend.py`, `server/dreamzero_backend.py` (+ `server/dreamzero_sidecar.py`, `interlatent_dreamzero/`) | Load and run the policy; `torch`/`lerobot` are imported lazily so a recording-only box needs neither |
| Auth | `server/auth.py` | Owner-checked `x-api-key` on every RPC, on by default for self-hosted boxes |
| Recording | `server/recorder.py`, `storage/lerobot_rebuild.py`, `storage/lerobot_live.py` | Ingest `RecordTick`s, build a LeRobot v3.0 dataset (live-encoded, with a rebuild fallback), upload via backend-issued presigned URLs |

The two Python dists share **no** code — they run on different machines and are versioned
independently. They meet only at `proto/messages.proto`.

### `teleop/teleop-web`

A standalone WebXR PWA: the open-source VR producer for teleoperation. It solves IK in the
browser and streams absolute joint targets over WebTransport/QUIC to the node. It is a
deliberate fork of the dashboard's teleop engine rather than a shared package — see its
[README](teleop/teleop-web/README.md) for the provenance rule (fixes land in both copies).

### `proto/`

`messages.proto` is the single wire contract, and the single source of truth: both
`packages/sdk` and `packages/server` hold *mirrored* copies plus generated stubs, written
in one pass by `./proto/gen_proto.sh`. Never edit a mirror — `tests/test_proto_sync.py`
fails the build when one drifts. Compatibility rule: additive changes only. Details in
[proto/README.md](proto/README.md).

## Networking

Inference is gRPC (HTTP/2): `OpenSession` / `Stream` / `Infer` / `RecordTick` /
`RecordTicks` / `CloseSession`. The DRTC GPU endpoint is provisioned per-session by the
dashboard and returned to the client (or node) when a session is assigned. Plain LAN, a
VPN, or the public internet all work — the client merges chunks the same way regardless of
the path's latency.

## Relationship to Interlatent Cloud

There are **two contracts in this system**, and keeping them apart explains most of the
layout. `proto/messages.proto` is the *data* plane: the gRPC the client and node speak to
a GPU box. [`docs/coordinator-protocol.md`](docs/coordinator-protocol.md) is the *control*
plane: the HTTP a node, a box, the CLI and the teleop app speak to whatever assigns them
work. Both are additive-only.

The service on the other end of that second contract is a **coordinator**. The
[Interlatent dashboard](https://interlatent.com) is one implementation of it — currently
the only one, with a self-hosted implementation landing in the CLI. They are two
deployments of one contract, not two modes: nothing in the SDK branches on which it is
talking to. See [ADR 0038](docs/adr/0038-coordinator-protocol-one-control-plane.md), which
supersedes ADR 0023's "the dashboard remains the only control plane".

A coordinator is never in the data path. DRTC is direct node↔box, so a running session
survives the coordinator's absence — the node keeps driving the robot and its poll and
heartbeat simply backoff-retry.

The **serving stack itself is open** (`packages/server`) — a box you provision on the
dashboard and a box you run yourself execute the same code, differing only in identity
(a managed system key vs. your own `ilat_` key) and in who pays for the hardware.

Episode recording happens through hosted sessions (ADR 0022), so collection requires an
account; the client, node, server, and protocol are all Apache-2.0. Existing stock LeRobot
datasets can be imported through the dashboard's HF import. What remains private is the
platform around the boxes: provisioning and warm pools, the dataset/canonical store and
merge pipeline, offline policy improvement, and the annotation stack.
