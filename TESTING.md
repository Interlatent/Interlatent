# Test plan

What must be exercised before declaring a release stable. The repo ships three things
and they are released independently, so read this per-deliverable:

| Deliverable | Test roots | CI job |
|---|---|---|
| `packages/sdk` (`interlatent`) — DRTC client, node, dashboard CLI | `tests/`, `packages/sdk/tests/` | `test` (x86 + ARM, py3.11/3.12) |
| `packages/server` (`interlatent-server`) — the DRTC policy server | `packages/server/tests/` | `server` (x86, py3.11/3.12) |
| `teleop/teleop-web` — the WebXR VR producer | *(none yet — typecheck + build only)* | `teleop-web` |

Both test roots for the SDK matter: `packages/sdk/tests/` holds the loop-runner,
movement-arbitration, and Nori-guard suites, and is **not** a subset of `tests/`.

## What CI covers vs. what it doesn't

Everything runs with **no GPU and no robot**.

- **SDK.** The DRTC client control loop against the `echo` / `tiny_torch` test backends
  (chunk merging, latency estimation, scheduling), motion-path arbitration, the robot
  adapters' pure logic, and the dashboard CLI argument/URL plumbing.
- **Server.** Identity resolution, the owner-checked gRPC auth wrapper, box status
  reporting, `interlatent-serve` registration, and an import walk of every module on a
  bare install (no torch, no lerobot). Not the policy path: loading a real checkpoint and
  running `forward()` needs a GPU.
- **Protocol.** `tests/test_proto_sync.py` asserts the two packages' mirrored `.proto`
  and generated descriptors agree — a client from PyPI has to talk to a server from PyPI.
- **teleop-web.** `tsc --noEmit` and a production build. There are no runtime tests; WebXR
  needs a headset.

It does **not** exercise a real policy on a real GPU box end to end — that is Tier 4,
below, and is run by hand.

> **Stale below.** Tier 3 still describes the client-side collection API
> (`watch()` / `tick()` / `collect()`, local SQLite staging) that ADR 0022 removed.
> Collection is streaming-first and server-side now; those checks want rewriting against
> the recorder in `packages/server`.

---

## Tier 1 — DRTC client + routing (CI, no hardware)

- [ ] **Chunk merge** — overlapping chunks merge last-writer-wins on monotonic
  control timestamps; a fresher inference overrides stale plans.
- [ ] **Latency estimator** — Jacobson-Karels split tracks network vs. compute and
  drives `min_execution_horizon` / `cooldown_steps` sanely.
- [ ] **`connect_drtc(api_key=…)`** — resolves the account and dials the
  per-session GPU endpoint the dashboard returns; `INTERLATENT_API_KEY` env var is
  honored; `--api-base` / `INTERLATENT_API_BASE` override works.
- [ ] **`step()` returns `None` while the first chunk is in flight**, then streams
  actions at the control rate.

## Tier 2 — Node + dashboard CLI (CI where possible)

- [ ] **`interlatent-node pair`** — registers the robot against the dashboard with
  an API key (mock the API for CI).
- [ ] **`interlatent-node run`** — polls the dashboard and converges to the
  assigned session; keeps driving the robot while a session is assigned.
- [ ] **CLI** — `interlatent gpus ls`, `interlatent nodes ls`,
  `interlatent session ls|start|stop` build correct requests and parse responses.
- [ ] **Session lifecycle** — `session start` → node converges → `session stop`
  closes the DRTC link and triggers any recorded dataset to build/publish.
- [x] **Nori adapter conformance + session client** — outbound frames validate
  against the vendored Nori-Protocol schemas, golden fixtures replay through
  the inbound parser, and the liveness-tied keep-alive pump / fail-closed
  handshake / reconnect paths run against a fake in-process NDJSON daemon.
  CI-safe: loopback sockets only, no hardware, no network
  (`test_nori_protocol_conformance.py`, `test_nori_client.py`,
  `test_nori_adapter.py`, `test_nori_cameras.py`, `test_teleop_estop_frame.py`).

## Tier 3 — Local dataset collection + storage

- [ ] **`watch()` / `tick()` / `collect()`** stage per-step state/action/reward to
  local SQLite + JPEG staging with no account.
- [ ] **`LeRobotRebuilder`** emits a valid LeRobot v3.0 dataset on disk (parquet
  frames + MP4 video + JSON metadata).
- [ ] **Recording destinations** flush on stop: local `output_dir` and `s3_uri`
  merge-on-stop into one flat dataset; the hosted inbox path requires an API key.

## Tier 4 — Real policy on a cloud pod (manual, against the dashboard)

- [ ] **End-to-end rollout** — pair a node, start a session against a real pod with
  a real policy (e.g. SmolVLA), confirm smooth control at the loop rate.
- [ ] **Latency at the control rate** — measure round-trip and confirm the client
  stays scheduled ahead; the arm does not stutter.
