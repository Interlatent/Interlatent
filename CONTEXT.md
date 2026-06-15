# Interlatent — Domain Glossary

The open-source robot runtime: run VLA policies on real robots (DRTC inference), record
demonstrations, and collect LeRobot datasets — self-hosted or against Interlatent Cloud. This
file pins the canonical vocabulary. Prose explanations live in [docs/concepts.md](docs/concepts.md);
this file is the tight glossary domain experts should agree on.

## Language

**Coordinator**:
The self-hosted control plane that assigns inference **Sessions** to **Nodes** — the offline
replacement for Interlatent Cloud's session assignment. It speaks the `/api/v1/nodes/*` API the
Node already polls. It is *not* in the inference data path.
_Avoid_: controller, control plane (as a proper name), orchestrator, broker.

**Node**:
A long-running daemon on a robot machine (`interlatent-node`) that pairs to a Coordinator (or
Cloud), heartbeats, and converges to whatever **Session** is assigned to it. Owns the robot
hardware config (robot type, port, cameras) where it runs.
_Avoid_: agent, worker, device.

**Session**:
One robot↔server inference engagement — a policy URI + task + fps + recording flag, opened
against a GPU server (`OpenSession`). Created by assignment, ended by unassignment.
_Avoid_: job, run (a **run** is a rollout/episode, not a Session).

**Environment**:
A label for one robot/policy collection that owns a **canonical dataset** accumulated across
Sessions. Offline it is just a name stamped into local files; on Cloud it is a first-class
object. _Avoid_: project, workspace.

**Canonical dataset**:
The single LeRobot v3 dataset that a recording destination accumulates across Sessions via
merge-on-stop. Offline this is one flat dataset per destination; Environment-keyed canonical
datasets are a Cloud concern.

**GPU server**:
The DRTC inference server (`interlatent-serve`) that loads the policy and serves action chunks.
Just a URL to the Coordinator and Node — it is never registered with any platform.
_Avoid_: compute box (Cloud term), backend (overloaded with "policy backend").

**Route / routing method**:
How a **Node** reaches a **GPU server** for a session, captured as a descriptor
`{method, address}`. `direct` (the only method today) means "dial this address as-is". The
method is a seam for future connectivity (relay, tunnel, MagicDNS) — it is *not* the network
transport (gRPC vs gRPC-web), which is inferred from the address.
_Avoid_: transport, endpoint (the address is one field of a route).

## Relationships

- A **Coordinator** assigns **at most one** **Session** to a **Node** at a time (busy-node guard).
- **Stopping a Session = the Coordinator unassigning it**; the Node then tears down its control
  loop and the **GPU server** builds + publishes the episode to the **canonical dataset**.
- The **Coordinator** is the *control plane*; the **Node ↔ GPU server** DRTC link is the *data
  plane* and survives the Coordinator's absence (see [ADR-0001](docs/adr/0001-offline-coordinator-control-plane.md)).
- A **Node** records one **episode** per **Session**; merge-on-stop appends it to the
  destination's **canonical dataset**.

## Example dialogue

> **Dev:** "When I `session stop`, does the Coordinator kill the Node process?"
> **Domain expert:** "No — it *unassigns* the Session. The Node sees `session:null` on its next
> poll, closes the DRTC client (`CloseSession`), and the GPU server merges the episode into the
> canonical dataset. The Node process keeps running, idle, ready for the next assignment."

## Flagged ambiguities

- **"controller" vs "coordinator"** — resolved: **Coordinator** is the offline control plane;
  **control/controller** is reserved for the real-time robot/inference path (`node/control.py`'s
  control loop, `inference/client/controller.py`'s DRTC client, "control tick/timestamp").
- **"the Node requires Interlatent Cloud for session assignment"** (stated in
  [docs/concepts.md](docs/concepts.md)) — now **stale**: a self-hosted **Coordinator** also
  provides assignment. concepts.md should be updated when the coordinator ships.
