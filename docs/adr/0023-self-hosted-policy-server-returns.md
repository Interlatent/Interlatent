# 0023 — The self-hosted policy server returns: `interlatent-server`

Status: accepted (2026-07-30)

> **Amended 2026-08-06:** superseded in part by
> [ADR 0038](0038-coordinator-protocol-one-control-plane.md), which reverses
> the "no control plane comes back with the box" bullet below: `interlatent up`
> now runs the coordinator, and it is the only control plane there is. The rest
> of this record stands as written.

## Context

PR #2 ("no-server", 2026-06-23) deleted this repo's entire self-hosting
stack — the offline coordinator control plane (ADR 0001), the BYO-box
registration client (ADR 0003), `packages/server/`, `docker/`, and
`docs/self-hosting.md` — leaving the repo "just client and CLI" and the
serving code closed-source in the engine. The registration half of ADR
0003 (`POST /api/v1/compute/boxes/register`, owner-key status reports)
shipped anyway and sat unused.

Users want to run inference on their own GPUs. What nobody wants back is
the *offline coordinator*: a second control plane forked every flow into
two modes and is why the 2026-06 stack collapsed. So as of this decision
the control plane a box registers with stays an **upstream** service, one
the operator does not run — the box is theirs, the brokering is not.

## Decision

Resurrect **half** of the deleted stack — the box, not the control plane.

- `packages/server/` returns as the **`interlatent-server`** dist
  (top-level module `interlatent_server`, so it can coexist with the
  `interlatent` SDK dist in one environment): the DRTC servicer, policy
  runtime, LeRobot backends, session recorder, `serve_gpu` launcher, and
  the wire protocol (`interlatent.inference.v1` — byte-identical to the
  stubs the SDK client ships). This is the serving code moved out of the
  engine — not a fork (platform ADR 0035 governs the single-source
  arrangement).
- **`interlatent-serve`** implements deleted ADR 0003's registration
  exactly: dial out to that upstream control plane with the owner's key,
  UUID persisted at `~/.interlatent/box-id`,
  `--advertise-address` handed to nodes verbatim, status self-reports
  (+ `stopped` on graceful exit).
- **No control plane comes back with the box** (deleted ADR 0001 stays
  dead). No offline mode, no local session brokering: a box with no key
  is just a bare local `serve_gpu` for smoke tests. *(Reversed by ADR
  0038 — see the amendment above.)*
- Unlike the 2026-06 stack ("the network is the trust boundary"), the
  gRPC port is **guarded by default** with an owner-scoped key check
  against that control plane; `--insecure` opts out.
- torch remains CPU-pinned in the `interlatent` SDK dist; only
  `interlatent-server` wants CUDA wheels. This split is why the server is
  a second dist rather than an SDK extra.

## Consequences

- `docs/self-hosting.md` and `docker/` return (the docker image installs
  `packages/server`; it contains no repo-clone hot-reload and no baked
  credentials).
- The `teleop/teleop-web/` standalone teleop producer (same release) makes
  the OSS story symmetric: BYO compute for inference, BYO headset app for
  teleop.
- CONTEXT.md's "GPU pod" entry now covers a box you run yourself; "no
  self-hosted control plane" under **Node** still stands, deliberately.
