# Architecture

A contributor-facing map of how the pieces fit. User-facing docs live in [docs/](docs/).

## The shape of the system

Big policies can't run on robot compute, and naive request/response inference makes arms
stutter. Interlatent's answer is **DRTC — Distributed Real-Time Chunking**:

```
robot (client)                              managed GPU pod (cloud)
──────────────                              ──────────────────────
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
| Dashboard CLI | `cli/` | `interlatent` — thin client over the dashboard API: `gpus ls`, `nodes ls`, `session ls\|start\|stop` |
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
(`control_source="teleop"`) — today for policy-less demonstration recordings;
mid-policy takeover (live intervention) is coming in a future release. The split is **engine on
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

### `proto/`

`messages.proto` is the single wire contract between the SDK and the cloud-managed GPU
pods. Generated stubs are committed in the SDK; regenerate with
`./proto/gen_proto.sh`. Compatibility rule: additive changes only.

## Networking

Inference is gRPC (HTTP/2): OpenSession / Stream / Infer / RecordTick / CloseSession.
The DRTC GPU endpoint is provisioned per-session by the dashboard and returned to the
client (or node) when a session is assigned. Plain LAN, a VPN, or the public internet
all work — the client merges chunks the same way regardless of the path's latency.

## Relationship to Interlatent Cloud

Inference runs on managed GPU pods through the [Interlatent dashboard](https://interlatent.com).
The client and node speak the gRPC contract in `proto/messages.proto` to the pod, and
authenticate to the dashboard with an API key (`ilat_…`) to discover pods, pair nodes, and
drive sessions. Episode recording happens through those hosted sessions (ADR 0022), so
collection requires an account; the client, node, and protocol themselves stay Apache-2.0.
Existing stock LeRobot datasets can be imported through the dashboard's HF import. Cloud-only
capabilities (managed warm GPUs, hosted datasets and dashboard, Robometer reward labeling)
live in a separate private codebase.
