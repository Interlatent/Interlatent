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
| Teleop stub | `node/teleop/` (channel, frame, safety, robot_profile) | Thin receiver for hosted DAgger takeover — see below |
| Dashboard CLI | `cli/` | `interlatent` — thin client over the dashboard API: `gpus ls`, `nodes ls`, `session ls\|start\|stop` |
| Collection | `_client.py`, `_watcher.py`, `_db.py`, `_step_source.py` | `watch()/tick()/collect()` — stage per-step state/action/reward into local SQLite |
| Dataset build | `storage/lerobot_rebuild.py`, `_dataset.py` | Turn the staging cache into a LeRobot v3.0 dataset on disk |

Collection is **local-first**: `watch()`/`tick()` write only to local SQLite + JPEG staging.
Uploading to the hosted platform is a separate, optional step.

### Teleop (DAgger takeover)

A human can take over a robot mid-policy and have the intervention recorded
(`control_source="teleop"`). The split is **engine on the platform, thin stub on
the client** (see [docs/adr/0012](docs/adr/0012-teleop-receiver-stub-open-core-boundary.md)):

- The hosted platform runs the teleop *engine* — keyboard integration, WebXR
  pose IK, retargeting — and streams **absolute joint targets** to the robot.
- `node/teleop/` keeps only the receiver: `TeleopChannel` (a WebSocket to the
  hosted relay) decodes `TeleopFrame`s; the control loop applies engaged
  `mode="targets"` frames through the **`SafetyGate`** (the last-hop
  workspace/velocity/deadman clamp) before driving the arm, and records the
  commanded action as `control_source="teleop"`.

**Layered client-side safety** (both run next to the motors, never across the
network): the per-adapter **delta clamp** (`--robot.max_step`) caps the per-tick
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
drive sessions. Local LeRobot dataset collection still runs with zero account. Cloud-only
capabilities (managed warm GPUs, hosted datasets and dashboard, Robometer reward labeling)
live in a separate private codebase.
