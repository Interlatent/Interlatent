# Test plan

What must be exercised before declaring a release stable. The repo ships three things
and they are released independently, so read this per-deliverable:

| Deliverable | Test roots | CI job |
|---|---|---|
| `packages/sdk` (`interlatent`) — DRTC client, node, CLI | `tests/`, `packages/sdk/tests/` | `test` (x86 + ARM, py3.11/3.12) |
| `packages/server` (`interlatent-server`) — the DRTC policy server | `packages/server/tests/` | `server` (x86, py3.11/3.12) |
| …its dataset writers against the **real** lerobot, + one full DRTC rollout | `test_lerobot_real_build.py`, `test_drtc_end_to_end.py` | `server-lerobot` |
| `teleop/teleop-web` — the WebXR VR producer | `teleop/teleop-web/src/**/__tests__/` (vitest) | `teleop-web` |

Both test roots for the SDK matter: `packages/sdk/tests/` holds the loop-runner,
movement-arbitration, and Nori-guard suites, and is **not** a subset of `tests/`.
CI runs `pytest tests/ packages/sdk/tests/`; run both locally too.

Two more jobs gate every PR: `extras resolve` (dry-run resolution of every extra in
`packages/sdk/pyproject.toml`; `[yam]`, `[axol]`, `[dimos]` are advisory) and `dco`
(every commit needs `Signed-off-by:`). `ruff check .` runs in the `test` job.

## What CI covers vs. what it doesn't

Everything runs with **no GPU and no robot**.

- **SDK.** The DRTC client control loop against the `echo` / `tiny_torch` test backends
  (chunk merging, latency estimation, scheduling), motion-path arbitration, the robot
  adapters' pure logic, the CLI argument/URL plumbing, and the **node daemon**
  (heartbeat payload, long-poll assignment dispatch, the convergence table, the
  disk-pressure spool gate, and OpenSession metadata assembly) — `tests/test_node_daemon.py`.
- **Coordinator.** The control plane runs in-process under CI: `test_coordinator.py`
  (state machine, long-poll wakeup, one-session-per-box, persistence across reload),
  `test_coordinator_auth.py` (the `ilop_`/`ilnode_`/`ilbox_` principals and the
  cross-scope rejections), `test_coordinator_resolution.py` (address resolution and the
  no-default `CoordinatorNotConfigured`), `test_coordinator_teleop.py` (tokens,
  certificates, relay pairing), `test_coordinator_protocol.py` (pins the route table in
  [`docs/coordinator-protocol.md`](docs/coordinator-protocol.md) to
  `interlatent.coordinator.protocol` — edit one and the other must move with it), and
  `test_coordinator_e2e.py` (a real `NodeDaemon` against a real coordinator over real
  HTTP, asserting that stopping a session unassigns rather than kills).
- **Server.** Identity resolution, the owner-checked gRPC auth wrapper, box status
  reporting, `interlatent-serve` registration, an import walk of every module on a
  bare install (no torch, no lerobot), and the two halves of streaming-first
  collection: the **session recorder** (admission control, tick dedupe, measured
  fps, the live-encode lane and its fallbacks, and destination-sink selection across
  all three sinks — local directory, S3, and the protocol's presigned upload-url
  inbox) and the **LeRobot v3 writer** (feature discovery,
  row→frame conversion, and the parquet
  post-edits that carry `episode_uuid` / `control_source` / `failure_type`).
  `LeRobotDataset` itself is stubbed — lerobot drags torch — so what is covered is
  everything on this side of that seam. Not the policy path: loading a real checkpoint
  and running `forward()` needs a GPU.
- **Protocol.** `tests/test_proto_sync.py` asserts the two packages' mirrored `.proto`
  and generated descriptors agree — a client from PyPI has to talk to a server from PyPI.
- **teleop-web.** `tsc --noEmit`, `npm test` (vitest), and a production build, covering the
  pure math the arm's motion comes from: quaternion helpers, FK and the geometric Jacobian
  (against a finite-difference derivative), the DLS IK's clamps and units seam, clutch pose
  mapping including the swing–twist split and the slipping clutch, and the wrist-pivot
  solve. The XR session itself is untested; that needs a headset.

It does **not** exercise a real policy on a real GPU box end to end — that is Tier 4,
run by hand.

---

## Tier 1 — DRTC client + routing (CI, no hardware)

- [ ] **Chunk merge** — overlapping chunks merge last-writer-wins on monotonic
  control timestamps; a fresher inference overrides stale plans.
- [ ] **Latency estimator** — Jacobson-Karels split tracks network vs. compute and
  drives `min_execution_horizon` / `cooldown_steps` sanely.
- [ ] **`connect_drtc(server_address=…, environment=…)`** — dials the GPU endpoint and
  opens a session; `INTERLATENT_DRTC_URL` is honored (`--server` on
  `interlatent-preflight`), and calling with no address at all raises instead of
  guessing one.
- [x] **`step()` returns `None` while the first chunk is in flight**, then streams
  actions at the control rate. `test_drtc_end_to_end.py` runs a real rollout —
  `connect_drtc()` over a real localhost gRPC socket into the `echo` backend, ticks
  journalled through the real spool, then the recorder's real rebuild and upload
  through the protocol's `upload-urls` / `upload-complete` routes, served by a local
  HTTP stub. The assertion is that `upload-complete` was called and that the uploaded
  tree is a complete LeRobot dataset — the only externally visible difference between
  "the recording survived" and "the rebuild raised and the staging was deleted".

## Tier 2 — Coordinator + node + CLI (CI where possible)

- [x] **Coordinator** — `interlatent up`'s service under test in-process:
  pair/heartbeat/poll, session assign and unassign, box registration and the `authz`
  probe, key scoping, and address resolution (`test_coordinator.py`,
  `test_coordinator_auth.py`, `test_coordinator_resolution.py`,
  `test_coordinator_teleop.py`, `test_coordinator_protocol.py`,
  `test_coordinator_e2e.py`).
- [ ] **`interlatent-node pair --name … --coordinator … --api-key …`** — registers the
  robot against a coordinator (mock the API for CI).
- [x] **`interlatent-node run`** — long-polls the coordinator and converges to the
  assigned session; keeps driving the robot while a session is assigned.
  `tests/test_node_daemon.py` drives the daemon against a scripted HTTP client
  and a stubbed `connect_drtc`: the full convergence table (start / noop /
  endpoint-moved restart / swap / clear), the ADR 0023 refusal under disk
  pressure, teleop-recording vs inference-session dispatch, and the forced
  client close when a wedged robot teardown would otherwise lose the recording.
  What it does NOT cover is the control loop actually driving hardware.
- [ ] **CLI** — `interlatent gpus ls`, `interlatent nodes ls`, `interlatent env create`,
  `interlatent session ls|start|stop` build correct requests and parse responses.
- [x] **Offline CLI** — `interlatent behavior ls|validate|run` lists, validates and runs
  behaviors, and `interlatent-act` reads a pose, settles a move, and rejects unknown /
  missing / out-of-range joints — all with no key and no coordinator
  (`test_behavior_cli.py`, `test_behaviors*.py`, `test_act_cli.py`).
- [ ] **Session lifecycle** — `session start` → node converges → `session stop`
  closes the DRTC link and triggers any recorded dataset to build/publish.
- [x] **Nori adapter conformance + session client** — outbound frames validate
  against the vendored Nori-Protocol schemas, golden fixtures replay through
  the inbound parser, and the liveness-tied keep-alive pump / fail-closed
  handshake / reconnect paths run against a fake in-process NDJSON daemon.
  CI-safe: loopback sockets only, no hardware, no network
  (`test_nori_protocol_conformance.py`, `test_nori_client.py`,
  `test_nori_adapter.py`, `test_nori_cameras.py`, `test_teleop_estop_frame.py`).

## Tier 3 — Server-side recording + dataset storage

Collection is streaming-first: the node streams `RecordTick`s to the recorder on the
GPU box, which builds and publishes the dataset. The client-side verbs
(`watch()` / `tick()` / `collect()`) were removed in ADR 0018 and now raise.

- [x] **`LeRobotRebuilder`** emits a valid LeRobot v3.0 dataset on disk (parquet
  frames + MP4 video + JSON metadata). Two layers, because one is not enough:
  `test_lerobot_rebuild.py` covers schema discovery, frame conversion and the parquet
  post-edits against a stubbed writer, and `test_lerobot_real_build.py` drives the
  **real** `LeRobotDataset` + encoder and reads the result back with stock lerobot —
  including that the video really is H.264 and not lerobot's AV1 default.
  Stubs alone are how the `vcodec=` breakage shipped: upstream renamed the parameter, the
  stub took `**kwargs` and accepted the dead call, and a real close-session silently
  discarded the episode. See `storage/lerobot_codec.py`.
- [ ] **Recording destinations** — the `output_dir` / `s3_uri` keys the node puts in
  the OpenSession metadata pick the server's sink and flush on stop into one flat
  dataset; a local directory needs no coordinator at all (`test_sinks.py`).

## Tier 4 — Real policy on a real GPU box (manual)

- [ ] **End-to-end rollout** — the control-plane half of this is covered in Tier 2; what
  is manual is the hardware. Bring up a coordinator, pair a node, register a GPU box,
  start a session with a real policy (e.g. SmolVLA), confirm smooth control at the loop
  rate.
- [ ] **Latency at the control rate** — measure round-trip and confirm the client
  stays scheduled ahead; the arm does not stutter.
