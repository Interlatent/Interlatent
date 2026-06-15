---
status: accepted
---

# Offline operation uses a local coordinator the node polls, not a direct daemon mode

To run the node fully offline (no Interlatent Cloud) we add a small local **coordinator** — a
background HTTP control plane that speaks the exact `/api/v1/nodes/*` API the node daemon
already long-polls (pair, heartbeat, poll for session assignment). The node is reused
**unchanged**; starting a session means the coordinator assigns one and the node picks it up on
its normal poll, and **stopping a session means the coordinator unassigns it**, which routes
through the node's existing `_converge(None) → client.close() → CloseSession` teardown.

We chose this over a "direct/static" daemon mode (a CLI subcommand that constructs a session
locally and runs the control loop, with `session stop` killing the process) for one decisive
reason: `CloseSession` is the **only** trigger for the GPU server's dataset build+merge+upload,
and the server's idle-GC **discards** (does not upload) any recording whose session was never
closed (`packages/server/.../server/transport.py`, idle-GC loop). A process-kill stop would
skip `CloseSession` and silently lose the recorded episode. Routing stop through unassign makes
graceful teardown the default and needs ~zero node changes.

## Considered options

- **Direct/static daemon mode** — rejected: process-kill stop trips the idle-GC discard
  (silent data loss); also duplicates the converge/assignment logic the node already has.
- **Foreground coordinator** (blocks a terminal) vs **background daemon** — chose the
  background daemon (`up`/`down`/`status`/`logs`, pidfile + log under `~/.interlatent/`).
- **CLI supervises the GPU server + node too** — rejected for now: coordinator-only. The user
  launches `interlatent-serve` and `interlatent-node` themselves, so the user-run GPU server
  outlives the session and finishes the merge; the CLI never owns those lifecycles.

## Consequences

- **Control plane vs. data plane.** The DRTC link is **direct node↔GPU**; the coordinator is
  not in the data path. A running session therefore **survives the coordinator's absence** — if
  the coordinator crashes, the node keeps driving the robot and its poll/heartbeat just
  backoff-retry. Intentional shutdown is always graceful: `interlatent down` refuses while a
  session is active unless `--force`, and `--force` unassigns + waits for teardown before
  exiting (never orphans a moving robot). Only an unexpected crash leaves a session running with
  no remote stop.
- **State file is load-bearing.** Active assignments are persisted so `up` after a crash/`down`
  re-serves the same session to a still-running node (`changed:false`, keeps running). Losing
  the state would make the coordinator answer `session:null` and spuriously tear the node down —
  so it is written atomically.
- **Trust boundary is the network.** The coordinator binds `0.0.0.0` with no auth on `/admin/*`,
  consistent with the existing self-hosted stance ("the network is the trust boundary", see
  `packages/server/.../server/auth.py`).
