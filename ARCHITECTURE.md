# Architecture

A contributor-facing map of how the pieces fit. User-facing docs live in [docs/](docs/).

## The shape of the system

Big policies can't run on robot compute, and naive request/response inference makes arms
stutter. Interlatent's answer is **DRTC — Distributed Real-Time Chunking**:

Both ends of that loop live in this repo — `packages/sdk` is the client, `packages/server`
is the GPU box it talks to. Interlatent's managed boxes run the same server code.

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

Key properties:

- The client sends observations continuously and never blocks on inference.
- The pod returns overlapping **action chunks**; `merge.py` joins them with
  last-writer-wins semantics keyed on monotonic control timestamps.
- `latency.py` (Jacobson-Karels estimator) splits round-trip into network vs. compute so
  the client knows how far ahead to schedule.
- RTC "in-painting": on each inference the pod reconstructs the actions already
  scheduled on the robot and conditions the policy on them, so chunk
  boundaries stay continuous.

## Packages

### `packages/sdk` — pip `interlatent`, import `interlatent` (robot side)

| Area | Modules | Role |
|---|---|---|
| DRTC client | `inference/client/` (controller, sender, receiver, merge, latency, cooldown) | The real-time loop described above |
| Wire protocol | `inference/protocol/` | Generated stubs from `proto/messages.proto` |
| Integration | `inference/integration/connect.py` | `connect_drtc()` — one-call session against a cloud-provisioned GPU pod (`api_key=`) |
| Node daemon | `node/` (cli, daemon, control) | `interlatent-node` — long-running daemon that pairs to your account, polls the dashboard, and runs assigned inference sessions on real hardware (LeRobot robot classes) |
| Teleop stub | `node/teleop/` (channel, frame, safety, robot_profile) | Thin receiver for hosted VR teleop (remote human demonstration) — see below |
| CLI | `cli/` | `interlatent` — session manager: `up`/`down`/`status` to run a coordinator, `gpu`, `gpus ls`, `nodes ls`, `session ls\|start\|stop`, `config` |
| Coordinator | `coordinator/` | The control plane itself: `protocol` (the frozen contract), `state`, `server` (`/api/v1/*` only), `auth`, `relay` (vendored WebTransport relay), `certs` |
| Tick spool | `inference/client/spool.py` | Write-through disk journal for the RecordTick uplink: delete-after-ack, drain-done at close, hard-stop when full (ADR 0023) |
| JPEG encode | `node/jpeg.py` | Capability-adaptive frame encoder: PyTurboJPEG → OpenCV → PIL, resolved at runtime (`interlatent[turbo]`) |
| HTTP client | `_client.py` | `Interlatent` — environments/episodes API surface used by the daemon and CLI |

Collection is **streaming-first** (ADR 0022): the node JPEG-encodes each camera
frame per control tick and streams `RecordTicks` to the hosted recorder, which
builds and uploads the LeRobot dataset server-side. Devices never build
datasets; the old client-side `watch()`/`tick()`/`upload()` staging path was
removed in 2.0.0.

### Teleop (VR remote demonstration)

A human drives the robot remotely in VR and every human-driven step is recorded
— `control_source="teleop"` for policy-less demonstration recordings,
`control_source="intervention"` for a mid-policy takeover during a hosted
inference session (engaging teleop preempts the policy; the node keeps
shadow-stepping the inference client so handing control back costs ≈1 control
tick). The split is **engine on
the platform, thin stub on the client** (see
[docs/adr/0012](docs/adr/0012-teleop-receiver-stub-open-core-boundary.md)):

- The hosted platform runs the teleop *engine* — WebXR pose IK, retargeting —
  and streams **absolute joint targets** to the robot.
- `node/teleop/` keeps only the receiver: `TeleopChannel` (a channel to the
  hosted relay) decodes `TeleopFrame`s; the control loop applies engaged
  `mode="targets"` frames through the **`SafetyGate`** (the last-hop
  workspace/velocity/deadman clamp) before driving the arm, and records the
  commanded action as `control_source="teleop"`.

**Layered client-side safety** (both run next to the motors, never across the
network): the per-adapter **delta clamp** (`--robot-arg max_step=…`) caps the per-tick
joint jump for *all* actions — policy and teleop alike — and the `SafetyGate`
adds workspace/velocity/deadman limits on the teleop path.

### `packages/server` — pip `interlatent-server`, import `interlatent_server` (GPU side)

The other end of the DRTC loop, and the same code Interlatent's hosted boxes run
([ADR 0023](docs/adr/0023-self-hosted-policy-server-returns.md)). Run it on your own
CUDA machine and it registers with the dashboard as a self-hosted compute box — see
[docs/self-hosting.md](docs/self-hosting.md).

| Area | Modules | Role |
|---|---|---|
| Entry point | `cli.py` | `interlatent-serve` — mint/persist a box UUID, detect the GPU, register with the dashboard, then serve |
| Launcher | `serve_gpu.py`, `credentials.py`, `box_status.py` | Warmup, CPU isolation, gRPC bind, identity (hosted admin key vs owner `ilat_` key), status self-reporting |
| Servicer | `server/transport.py`, `server/chunk_buffer.py`, `server/schedule.py` | The RPCs, the chunk buffer, and RTC in-painting reconstruction |
| Policy backends | `server/policy_runtime.py`, `server/lerobot_backend.py`, `server/molmoact2_backend.py` | Load and run the policy; `torch`/`lerobot` are imported lazily so a recording-only box needs neither |
| Auth | `server/auth.py` | Owner-checked `x-api-key` on every RPC, on by default for self-hosted boxes |
| Recording | `server/recorder.py`, `storage/lerobot_rebuild.py`, `storage/lerobot_live.py` | Ingest `RecordTicks`, build a LeRobot v3.0 dataset (live-encoded, with a rebuild fallback), upload via backend-issued presigned URLs |

The two Python dists share **no** code — they run on different machines and are versioned
independently. They meet only at `proto/messages.proto`.

### `teleop/teleop-web`

A standalone WebXR PWA: the VR producer for teleoperation. It solves IK in the browser
and streams absolute joint targets over WebTransport/QUIC to the node. It is a deliberate
fork of the dashboard's teleop engine rather than a shared package — see its
[README](teleop/teleop-web/README.md) for the provenance rule (fixes land in both copies).

### `proto/`

`messages.proto` is the single wire contract, and the single source of truth: both
`packages/sdk` and `packages/server` hold *mirrored* copies plus generated stubs, written
in one pass by `./proto/gen_proto.sh`. Never edit a mirror — `tests/test_proto_sync.py`
fails the build when one drifts. Compatibility rule: additive changes only. Details in
[proto/README.md](proto/README.md).

## Networking

Inference is gRPC (HTTP/2): OpenSession / Stream / Infer / RecordTick / CloseSession.
The DRTC GPU endpoint is provisioned per-session by the dashboard and returned to the
client (or node) when a session is assigned. Plain LAN, a VPN, or the public internet
all work — the client merges chunks the same way regardless of the path's latency.

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
