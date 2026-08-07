# The Interlatent Coordinator Protocol

Version: `interlatent.coordinator/1`

A **coordinator** is the service that assigns work. It pairs nodes, tracks GPU
boxes, brokers inference and teleop sessions, and tells each node what to
converge to. The robot-side stack in this repo talks to *a* coordinator and
never asks which one it is:

- The hosted [Interlatent dashboard](https://interlatent.com) implements it.
- `interlatent up` runs a self-hosted one on your own machine.

Those are two deployments of one contract, **not two modes**. The SDK contains
no branch on "is this the dashboard?" — see
[ADR 0038](adr/0038-coordinator-protocol-one-control-plane.md) for why that rule
is load-bearing rather than stylistic.

This document is normative. Its route table is pinned to
`interlatent.coordinator.protocol` by `tests/test_coordinator_protocol.py`, so
the two cannot drift.

## Addressing

A coordinator address is a **bare origin**: scheme, host, optional port. No
trailing slash, no `/api/v1` suffix.

```
https://interlatent.com
http://192.168.1.20:8900
```

Callers append `/api/v1/...` themselves. Two conflicting conventions used to
coexist in this repo — some call sites stored the `/api/v1` suffix in the base
URL and others did not — and were reconciled at runtime in three separate
places. There is one convention now, resolved in one place.

Resolution order, everywhere: explicit flag (`--coordinator`) →
`INTERLATENT_COORDINATOR` → the stored config value → **error**. There is no
default. A missing coordinator address is a configuration error with an
actionable message, never a silent fallback to a hosted service.

### Trailing slashes

`POST /api/v1/inference/sessions` and `POST /api/v1/inference/sessions/` MUST
both be accepted. The CLI has historically sent the trailing-slash form and the
teleop web app the bare one; a coordinator that honours only one of them breaks
a caller that is already in the field.

## Authentication

Every request carries `x-api-key`. Three principal kinds exist:

| Prefix | Principal | Held by |
|---|---|---|
| `ilop_` | Operator | The CLI, and whoever administers the coordinator |
| `ilnode_` | Node | One paired node, scoped to its own routes |
| `ilat_` | User | The hosted dashboard's own user keys |

A coordinator is the authority for the keys it issues. `POST /api/v1/nodes`
mints an `ilnode_` token; the coordinator MUST reject that token on another
node's routes.

`GET /api/v1/environments` doubles as the **auth probe**: `interlatent-server`
validates a presented key by calling it and treating any 2xx as "this key is
real". That has been an implicit contract discoverable only by reading
`packages/server/src/interlatent_server/server/auth.py`; it is part of the
protocol and a coordinator must serve it for a GPU box to accept any RPC.

## Tiers

**mandatory** — a coordinator MUST serve these. Absent any one of them,
something concrete breaks: a node cannot pair, a box refuses to boot, or every
gRPC call is rejected.

**optional** — a coordinator MAY serve these. Every SDK caller degrades on 404
without failing the operation; teleop, for instance, is simply disabled for the
session (`node/teleop/factory.py` treats 401/403/404 as definitive).

**coordinator-only** — the hosted dashboard does not serve these and 404s them.
The CLI reports that by name rather than surfacing a bare 404.

`GET /api/v1/capabilities` lets a caller find out in advance instead of
discovering it at session teardown. It is itself optional; a 404 means "assume
everything is served".

```json
{
  "protocol": "interlatent.coordinator/1",
  "optional_supported": ["/api/v1/episodes", "..."]
}
```

## Routes

| Method | Path | Tier | Purpose |
|---|---|---|---|
| `POST` | `/api/v1/nodes` | mandatory | Pair a node; mints its node id and node token. |
| `POST` | `/api/v1/nodes/{node_id}/heartbeat` | mandatory | Liveness plus the node's recording-spool and safety telemetry. |
| `GET` | `/api/v1/nodes/{node_id}/poll` | mandatory | Long-poll for the node's current assignment. |
| `POST` | `/api/v1/nodes/{node_id}/hardware` | mandatory | Report robot type, port, cameras and robot args. |
| `POST` | `/api/v1/nodes/{node_id}/robot-features` | mandatory | Report feature element names and the teleop profile. |
| `POST` | `/api/v1/compute/boxes/register` | mandatory | Register a GPU box; idempotent on box_id. |
| `POST` | `/api/v1/compute/boxes/{box_id}/status` | mandatory | Box self-reports ready/running/uploading/stopped. |
| `GET` | `/api/v1/compute/boxes/{box_id}/warmup-target` | mandatory | Policy and camera keys to pre-warm; 404 means 'no target'. |
| `GET` | `/api/v1/compute/boxes/{box_id}/authz` | mandatory | Per-RPC authorization probe for the box's gRPC port. |
| `GET` | `/api/v1/nodes` | mandatory | List nodes. |
| `GET` | `/api/v1/gpus` | mandatory | List GPU boxes available to the caller. |
| `GET` | `/api/v1/environments` | mandatory | List environments. Doubles as the box auth probe. |
| `POST` | `/api/v1/environments` | mandatory | Create an environment. |
| `GET` | `/api/v1/environments/{env_id}/config` | mandatory | Observation schema: action_dim, camera_names, num_cameras. |
| `GET` | `/api/v1/inference/sessions` | mandatory | List inference sessions. |
| `POST` | `/api/v1/inference/sessions` | mandatory | Create an inference session and assign it to a node. |
| `DELETE` | `/api/v1/inference/sessions/{session_id}` | mandatory | Stop a session by unassigning it. MUST NOT kill the node. |
| `POST` | `/api/v1/episodes` | optional | Register an episode row. 409 means 'already exists', tolerated. |
| `POST` | `/api/v1/episodes/{episode_id}/upload-urls` | optional | Exchange file keys for presigned PUT urls. |
| `POST` | `/api/v1/episodes/{episode_id}/upload-complete` | optional | Signal the inbox that every file landed. |
| `GET` | `/api/v1/capabilities` | optional | Protocol version and which optional tiers are served. |
| `GET` | `/api/v1/teleop-recordings` | optional | List teleop recordings. |
| `POST` | `/api/v1/teleop-recordings` | optional | Create a teleop recording and assign it to a node. |
| `POST` | `/api/v1/teleop-recordings/{recording_id}/stop` | optional | Stop a teleop recording. |
| `POST` | `/api/v1/inference/sessions/{session_id}/teleop-token` | optional | Mint a teleop token for role=node|browser. |
| `POST` | `/api/v1/teleop-recordings/{recording_id}/teleop-token` | optional | Mint a teleop token for role=node|browser. |
| `GET` | `/api/v1/environments/{env_id}/episodes` | optional | List an environment's episodes. |
| `POST` | `/api/v1/environments/{env_id}/process` | optional | Kick the hosted merge pipeline. |
| `GET` | `/api/v1/environments/{env_id}/processing-status` | optional | Poll the hosted merge pipeline. |
| `POST` | `/api/v1/environments/{env_id}/cancel-processing` | optional | Cancel the hosted merge pipeline. |
| `POST` | `/api/v1/environments/{env_id}/analyze` | optional | Request hosted policy analysis. |
| `GET` | `/api/v1/episodes/{episode_id}` | optional | Fetch one episode row. |
| `GET` | `/api/v1/episodes/{episode_id}/status` | optional | Poll an episode's processing status. |
| `GET` | `/api/v1/episodes/{episode_id}/results` | optional | Fetch analysis results. |
| `GET` | `/api/v1/episodes/{episode_id}/meta` | optional | Fetch episode metadata. |
| `GET` | `/api/v1/episodes/{episode_id}/chunks/{chunk}` | optional | Fetch one dataset chunk. |
| `POST` | `/api/v1/episodes/{episode_id}/inbox-gc` | optional | Drop a partially uploaded inbox session. |
| `PUT` | `/api/v1/coordinator/recording` | coordinator-only | Set the recording destination stamped onto every session. |

## Invariants a coordinator must not break

### Stopping a session means unassigning it

`DELETE /api/v1/inference/sessions/{id}` MUST take the assignment away and let
the node tear itself down through its normal convergence path
(`_converge(None)` → `client.close()` → gRPC `CloseSession`). It MUST NOT kill
the node process or the box.

This is not stylistic. `CloseSession` is the **only** trigger for the GPU
server's dataset build, merge and upload, and the server's idle-GC *discards*
any recording whose session was never closed. A coordinator that stops a
session by killing something silently destroys the episode. This is the reason
the original design chose a poll/assign control plane over a "run the loop
directly" CLI mode, and it still holds.

### The coordinator is not in the data path

The DRTC link is direct node↔GPU-box, and teleop is browser↔relay↔node. A
running session therefore **survives the coordinator's absence** — if it
crashes, the node keeps driving the robot and its poll and heartbeat simply
backoff-retry.

The consequence is that intentional shutdown must be graceful: `interlatent
down` refuses while a session is active unless forced, and `--force` unassigns
and waits for teardown before exiting. Only an unexpected crash leaves a
session running with nothing able to stop it remotely.

### Assignment state is durable

A coordinator that forgets its assignments answers `session: null` on the next
poll, which tears down a node that was happily driving a robot. Active
assignments MUST survive a coordinator restart.

### Additive changes only

Same rule as [`proto/messages.proto`](../proto/README.md). New optional fields
and new routes are fine at any time. Removing a route, renaming a field, or
tightening a type is a protocol version bump.

## Node assignment payload

`GET /api/v1/nodes/{node_id}/poll` takes `known_session_id`, `known_endpoint`
and `wait` (seconds) and blocks until the assignment changes or `wait` elapses.
It returns the typed envelope, and for an inference session SHOULD also mirror
the payload at the top level for older nodes:

```json
{
  "changed": true,
  "assignment": {"type": "inference_session", "session": { }},
  "session": { }
}
```

`type` is `inference_session` or `teleop_recording`; a teleop assignment carries
its payload under `recording` instead.

Fields the node reads off an inference-session payload:

| Field | Meaning |
|---|---|
| `id` | Session id; also the episode id and the teleop-token path segment |
| `route` | `{address, method}` — the DRTC endpoint descriptor |
| `drtc_endpoint` | Legacy flat form of the same thing |
| `policy_uri` | Policy to load |
| `policy_backend` | Defaults to `lerobot` |
| `task` | Language task string |
| `task_id` | Optional link to a durable task |
| `chunk_size` | Defaults to 50 |
| `action_dim` | Defaults to 6 |
| `fps` | Defaults to 30 |
| `environment_id` | Environment id |
| `collection_context.env_slug` | Environment slug; defaults to `default` |
| `synchronous` | Sequential rather than overlapping chunking |
| `recording` | Opaque; forwarded verbatim into gRPC `OpenSession` metadata |

`recording` is the seam that lets a coordinator keep episodes off the hosted
inbox entirely: it is passed through the node untouched and interpreted by the
GPU box's recorder, so stamping `{"output_dir": ...}` or `{"s3_uri": ...}` onto
every session is enough to make the whole inbox tier unnecessary. A coordinator
that does **not** stamp a destination MUST serve the inbox tier instead.

`synchronous` is worth calling out: a policy whose successive plans disagree
will fight itself under the default overlapping chunking. Sequential chunking is
a per-policy fact and a session pins one policy, so the session payload is its
natural home.
