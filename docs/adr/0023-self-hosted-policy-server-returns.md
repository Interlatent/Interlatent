# 0023 — The self-hosted policy server returns: `interlatent-server`, dashboard-registered

Status: accepted (2026-07-30)

## Context

PR #2 ("no-server", 2026-06-23) deleted this repo's entire self-hosting
stack — the offline coordinator control plane (ADR 0001), the BYO-box
registration client (ADR 0003), `packages/server/`, `docker/`, and
`docs/self-hosting.md` — leaving the repo "just client and CLI" and the
serving code closed-source in the platform's engine. The platform-side
half of ADR 0003 (`POST /api/v1/compute/boxes/register`, owner-key status
reports) shipped anyway and sat unused.

Users want to run inference on their own GPUs. What nobody wants back is
the *offline coordinator*: a second control plane forked every flow into
two modes and is why the 2026-06 stack collapsed.

## Decision

Resurrect **half** of the deleted stack — the box, not the control plane.

- `packages/server/` returns as the **`interlatent-server`** dist
  (top-level module `interlatent_server`, so it can coexist with the
  `interlatent` SDK dist in one environment): the DRTC servicer, policy
  runtime, LeRobot backends, session recorder, `serve_gpu` launcher, and
  the wire protocol (`interlatent.inference.v1` — byte-identical to the
  stubs the SDK client ships). This is the same code that powers hosted
  boxes, moved from the engine — not a fork (platform ADR 0035 governs
  the single-source arrangement).
- **`interlatent-serve`** implements deleted ADR 0003's registration
  exactly: dial out with the owner's `ilat_` key, UUID persisted at
  `~/.interlatent/box-id`, `--advertise-address` handed to nodes
  verbatim, status self-reports (+ `stopped` on graceful exit).
- **The dashboard remains the only control plane** (deleted ADR 0001
  stays dead). No offline mode, no local session brokering: a self-hosted
  box with no API key is just a bare local `serve_gpu` for smoke tests.
- Unlike the 2026-06 stack ("the network is the trust boundary"), the
  gRPC port is **guarded by default** with an owner-scoped key check
  against the backend; `--insecure` opts out.
- torch remains CPU-pinned in the `interlatent` SDK dist; only
  `interlatent-server` wants CUDA wheels. This split is why the server is
  a second dist rather than an SDK extra.

## Consequences

- `docs/self-hosting.md` and `docker/` return (the docker image installs
  `packages/server`; it contains no repo-clone hot-reload and no baked
  credentials — unlike the hosted image it replaces for this use case).
- The `teleop/teleop-web/` standalone teleop producer (same release) makes
  the OSS story symmetric: BYO compute for inference, BYO headset app for
  teleop, hosted control plane for both.
- CONTEXT.md's "GPU pod" entry now names both flavors; "no self-hosted
  control plane" under **Node** still stands, deliberately.
