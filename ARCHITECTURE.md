# Architecture

A contributor-facing map of how the pieces fit. User-facing docs live in [docs/](docs/).

## The shape of the system

Big policies can't run on robot compute, and naive request/response inference makes arms
stutter. Interlatent's answer is **DRTC — Distributed Real-Time Chunking**:

```
robot (client)                              GPU box (server)
──────────────                              ────────────────
 sender thread  ── Observation stream ──▶   gRPC servicer (transport.py)
                                              │ decode payload (npz/jpeg)
                                              │ PolicyRuntime.forward()
 receiver thread ◀── ActionChunk stream ──   │ chunk_buffer + schedule
      │                                       └ optional SessionRecorder
 LWW merge into action schedule
      │
 step() → next action at control rate
```

Key properties:

- The client sends observations continuously and never blocks on inference.
- The server returns overlapping **action chunks**; `merge.py` joins them with
  last-writer-wins semantics keyed on monotonic control timestamps.
- `latency.py` (Jacobson-Karels estimator) splits round-trip into network vs. compute so
  the client knows how far ahead to schedule.
- RTC "in-painting": on each inference the server reconstructs the actions already
  scheduled on the robot (`schedule.py`) and conditions the policy on them, so chunk
  boundaries stay continuous.

## Packages

### `packages/sdk` — pip `interlatent`, import `interlatent` (robot side)

| Area | Modules | Role |
|---|---|---|
| DRTC client | `inference/client/` (controller, sender, receiver, merge, latency, cooldown) | The real-time loop described above |
| Wire protocol | `inference/protocol/` | Generated stubs from `proto/messages.proto` |
| Integration | `inference/integration/connect.py` | `connect_drtc()` — one-call session against any server (self-hosted or cloud) |
| Node daemon | `node/` (cli, daemon, control, keyboard_action, teleop_channel) | `interlatent-node` — long-running daemon that runs assigned inference sessions on real hardware (LeRobot robot classes), with DAgger keyboard takeover |
| Collection | `_client.py`, `_watcher.py`, `_db.py`, `_step_source.py` | `watch()/tick()/collect()` — stage per-step state/action/reward into local SQLite |
| Dataset build | `storage/lerobot_rebuild.py`, `_dataset.py` | Turn the staging cache into a LeRobot v3.0 dataset on disk |

Collection is **local-first**: `watch()`/`tick()` write only to local SQLite + JPEG staging.
Uploading to the hosted platform is a separate, optional step.

### `packages/server` — pip `interlatent-server`, import `interlatent_server` (GPU side)

| Module | Role |
|---|---|
| `server/app.py` | `interlatent-serve` CLI — optional `--policy` pre-warm, then serve |
| `server/transport.py` | gRPC servicer: sessions, streaming, recording hooks |
| `server/policy_runtime.py` | Backend registry + router registry (`register_router` / `resolve_backend` — URI-based dispatch to transformers-native families) + process-wide `(backend, policy_uri)` cache — a policy is loaded/compiled once per process |
| `server/lerobot_backend.py` | LeRobot policies (SmolVLA, ACT, Pi0, Diffusion, VQ-BeT, TDMPC) with RTC + torch.compile |
| `server/molmoact2_backend.py` | MolmoAct2 (transformers-native), routed transparently |
| `server/spatialvla_backend.py` | SpatialVLA (Shanghai AI Lab, transformers-native); image+instruction → action chunk |
| `server/rdt_backend.py` | RDT-1B (Tsinghua, diffusion, proprio-aware) via the upstream `create_model` API |
| `server/chunk_buffer.py`, `server/schedule.py` | Per-session chunk storage + RTC in-painting reconstruction |
| `server/recorder.py` | Optional server-side episode recording → LeRobot dataset (`storage/lerobot_rebuild.py`) |
| `server/teleop_relay.py` | WebSocket relay (`:50052`) pairing browser/laptop operators with robot sessions for DAgger |
| `server/auth.py` | Optional API-key gate (off for plain self-hosting) |

Heavy deps (`torch`, `lerobot`) are lazy — the server imports and runs with only the base
install, using the `echo`/`tiny_torch` test backends.

### `packages/teleop` — pip `interlatent-teleop`, import `interlatent_teleop`

Standalone laptop ↔ Pi teleoperation over gRPC: MediaPipe hand tracking or keyboard on the
laptop, a 50 Hz control loop with a safety gate (workspace/velocity clamps, deadman,
staleness) on the Pi. Shares no Python imports with the other packages — only hardware.

### `proto/`

`messages.proto` is the single wire contract between the SDK, the self-hosted server, and
Interlatent Cloud. Generated stubs are committed in both packages; regenerate with
`./proto/gen_proto.sh`. Compatibility rule: additive changes only.

## Ports & networking

| Port | Protocol | What |
|---|---|---|
| 50051 | gRPC (HTTP/2) | Inference: OpenSession / Stream / Infer / RecordTick / CloseSession |
| 50052 | WebSocket | Teleop relay (DAgger takeover) |

The reference deployment is a persistent GPU box on Tailscale; plain LAN works identically.
`docker/` ships a CUDA image with optional Tailscale join. `teleop-proxy/` is an optional
public WS relay for browsers on networks that block direct connections.

## Relationship to Interlatent Cloud

The hosted platform consumes these packages from PyPI and speaks the same gRPC/HTTP
contracts — the dependency only points that way. Nothing in this repo imports cloud code,
and everything here runs with zero account. Cloud-only capabilities (managed warm GPUs,
hosted datasets and dashboard, Robometer reward labeling) live in a separate private
codebase.
