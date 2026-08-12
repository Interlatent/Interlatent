# The Interlatent Coordinator Protocol

Version: `interlatent.coordinator/2`

A **coordinator** is the service that assigns work. It pairs nodes, tracks GPU
boxes, brokers inference and teleop sessions, and tells each node what to
converge to. `interlatent up` runs one on your own machine; this document is
what it serves, and what anything else claiming to be a coordinator must serve.

The robot-side stack in this repo reaches a coordinator by address and knows
nothing else about it. There is one control plane and **no modes**: the SDK
contains no branch on which coordinator it is talking to — see
[ADR 0038](adr/0038-coordinator-protocol-one-control-plane.md) for why that rule
is load-bearing rather than stylistic.

This document is normative. Its route table is pinned to
`interlatent.coordinator.protocol` by `tests/test_coordinator_protocol.py`, so
the two cannot drift.

## Addressing

A coordinator address is a **bare origin**: scheme, host, optional port. No
trailing slash, no `/api/v1` suffix.

```
http://192.168.1.20:8900
https://coordinator.internal.example
```

Callers append `/api/v1/...` themselves. Two conflicting conventions used to
coexist in this repo — some call sites stored the `/api/v1` suffix in the base
URL and others did not — and were reconciled at runtime in three separate
places. There is one convention now, resolved in one place.

Resolution order, everywhere: explicit flag (`--coordinator`) →
`INTERLATENT_COORDINATOR` → the stored config value → **error**. There is no
default. A missing coordinator address is a configuration error with an
actionable message, never a silent fallback to some remote service. Nothing in
this stack phones home.

### Trailing slashes

`POST /api/v1/inference/sessions` and `POST /api/v1/inference/sessions/` MUST
both be accepted. The CLI has historically sent the trailing-slash form and the
teleop web app the bare one; a coordinator that honours only one of them breaks
a caller that is already in the field.

## Authentication

Every request carries `x-api-key`. Three principal kinds exist:

| Prefix | Principal | Held by |
|---|---|---|
| `ilop_` | Operator | The CLI, `interlatent-serve`, the teleop web app, and whoever administers the coordinator |
| `ilnode_` | Node | One paired node, scoped to its own routes |
| `ilbox_` | Box | One registered GPU box, scoped to its own routes |

`interlatent up` prints the operator key on first start; it is the root
credential, and the one the CLI, a registering GPU box, and the teleop web app
present.

A coordinator is the authority for the keys it issues. `POST /api/v1/nodes`
mints an `ilnode_` token and `POST /api/v1/compute/boxes/register` mints an
`ilbox_` key; the coordinator MUST reject either on another node's or another
box's routes. A box's per-RPC `authz` probe is authorized by the `ilbox_` key
it was given at registration.

`GET /api/v1/environments` doubles as the **auth probe**: `interlatent-server`
validates a presented key by calling it and treating any 2xx as "this key is
real". That has been an implicit contract discoverable only by reading
`packages/server/src/interlatent_server/server/auth.py`; it is part of the
protocol and a coordinator must serve it for a GPU box to accept any RPC.

## One tier

Every route in the table below is **mandatory**: a coordinator serves all of
them or it is not a coordinator. Absent any one of them something concrete
breaks — a node cannot pair, a box refuses to boot, every gRPC call is
rejected, or an operator verb has no spelling at all.

Earlier versions of this document split the table into *mandatory*, *optional*
and *coordinator-only*. Those tiers only ever described what a **second**
implementation did not serve, and there is no second implementation: the
thirteen routes that nothing shipping ever served were deleted in
[ADR 0039](adr/0039-one-coordinator-one-protocol-tier.md) rather than left
advertised, and the remainder became one tier.

One runtime condition sits on top of the table rather than inside it: **teleop
needs a relay**. `interlatent up` ships an embedded one, but a coordinator
running without a relay answers the token mints with a definitive 404 and the
node turns teleop off for the session (`node/teleop/factory.py` treats
401/403/404 as final). `GET /api/v1/capabilities` is how a caller asks in
advance instead of discovering it mid-session:

```json
{
  "protocol": "interlatent.coordinator/2",
  "optional_supported": ["/api/v1/teleop-recordings", "..."]
}
```

`optional_supported` lists the conditional surfaces that are **live right
now**, full paths, and it is empty when none are. The field keeps the name it
had in `interlatent.coordinator/1` because shipped callers parse it. A list
that lies is worse than no list, so a coordinator MUST NOT advertise a path it
would 404.

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
| `GET` | `/api/v1/capabilities` | mandatory | Protocol version and which conditional surfaces are live. |
| `GET` | `/api/v1/teleop-recordings` | mandatory | List teleop recordings. |
| `POST` | `/api/v1/teleop-recordings` | mandatory | Create a teleop recording and assign it to a node. |
| `POST` | `/api/v1/teleop-recordings/{recording_id}/stop` | mandatory | Stop a teleop recording. |
| `POST` | `/api/v1/inference/sessions/{session_id}/teleop-token` | mandatory | Mint a teleop token for role=node|browser. |
| `POST` | `/api/v1/teleop-recordings/{recording_id}/teleop-token` | mandatory | Mint a teleop token for role=node|browser. |
| `PUT` | `/api/v1/coordinator/recording` | mandatory | Set the recording destination stamped onto every session. |

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

`interlatent.coordinator/2` is exactly such a bump, taken deliberately:
[ADR 0039](adr/0039-one-coordinator-one-protocol-tier.md) removed thirteen
routes and collapsed the tiers. A `/1` client that only called what `/2` still
carries is unaffected; one that called the inbox or analysis routes has nothing
left to call.

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

`recording` is the seam that decides where a finished episode lands: it is
passed through the node untouched and interpreted by the GPU box's recorder, so
stamping `{"output_dir": ...}` or `{"s3_uri": ...}` onto every session is how a
coordinator chooses a destination. It is also the *only* way — there is no
upload plane to fall back on. A coordinator that stamps nothing leaves the box
writing to its own `--output-dir`, which defaults to `~/.interlatent/episodes`;
`PUT /api/v1/coordinator/recording` is how an operator sets the stamp.

`synchronous` is worth calling out: a policy whose successive plans disagree
will fight itself under the default overlapping chunking. Sequential chunking is
a per-policy fact and a session pins one policy, so the session payload is its
natural home.
