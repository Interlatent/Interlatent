# `interlatent` CLI — coordinator API reference

The `interlatent` CLI (`cli/main.py`) is a thin coordinator client; this is the contract it
expects from the coordinator it is pointed at. The CLI's word for a GPU box is **gpu**
(`interlatent gpus ls`, `--gpu`); on the wire it is a *compute box*, and the session field is
named **pod**. `interlatent behavior` is offline and hits no endpoint
([docs/behaviors.md](../../../../../docs/behaviors.md)). The normative contract for every
route below is
[docs/coordinator-protocol.md](../../../../../docs/coordinator-protocol.md).

## Transport & auth (all endpoints)

Requests go through `interlatent._http.HTTPClient`.

- **Base URL:** the coordinator's bare origin, e.g. `http://10.0.0.5:8900`, from
  `--coordinator` (alias `--api-base`) or `INTERLATENT_COORDINATOR`. There is **no default**:
  an unset address is a configuration error, never a silent fallback.
- **Auth:** `x-api-key: ilop_…` on every request — the operator key the coordinator minted.
  The coordinator resolves the key to a principal + access rights and **scopes every response
  to it**.
- **Headers sent:** `Accept: application/json`, and `x-api-key` when a key is set.
- **Error semantics the client depends on:**
  - `401` / `403` → CLI prints "authentication failed — check your INTERLATENT_API_KEY".
  - `404` → CLI prints "not found".
  - `5xx` → the client retries, up to 3 attempts total 5s apart; return `5xx` only for
    genuinely transient failures.
  - JSON error bodies: the client reads `detail` or `message` for the displayed text.
- **List-shape flexibility:** any list endpoint may return *either* a bare JSON array *or* an
  object wrapping it under a named key (e.g. `{"gpus": [...]}`). Either parses.
- **Field tolerance:** the documented fields are what the CLI table renders. Unknown fields
  are ignored and missing ones render blank — but `session start` needs `id` back. Every
  `ls` also takes `--json`, which dumps the rows verbatim.

---

## 1. List GPU boxes — `GET /api/v1/gpus`

GPU boxes the caller can run sessions on.

```json
[
  {"id": "pod_a1b2", "name": "a100-0", "status": "ready",
   "gpu": "A100-40GB", "region": "us-east"}
]
```

- `status` is free text the CLI prints verbatim (e.g. `ready` / `busy` / `starting`).
- Bare array or `{"gpus": [...]}`.

## 2. List nodes — `GET /api/v1/nodes`

The paired robot nodes (read-only; same resource the node daemon pairs against).

```json
[
  {"id": "node_9f3", "name": "my-arm", "status": "online", "robot_type": "so101"}
]
```

- Bare array or `{"nodes": [...]}`.

## 3. List sessions — `GET /api/v1/inference/sessions/`

Active inference sessions.

```json
[
  {"id": "sess_77", "node": "my-arm", "pod": "a100-0",
   "policy_uri": "lerobot/smolvla_base", "status": "running"}
]
```

- Bare array or `{"sessions": [...]}`.

## 4. Start a session — `POST /api/v1/inference/sessions/`

The one write action on the session resource. Request body the CLI sends:

```json
{
  "node": "my-arm",                   // required — name or id (--node)
  "pod": "a100-0",                    // required — name or id (--gpu)
  "policy": "lerobot/smolvla_base",   // required (--policy)
  "backend": "lerobot",               // always sent; CLI default "lerobot"
  "task": "pick up the cube",         // optional — omitted when empty
  "env_slug": "my-arm",               // optional
  "fps": 30,                          // optional (float)
  "chunk_size": 50,                   // optional (int)
  "action_dim": 6                     // optional (int)
}
```

Coordinator responsibilities:

- Authorize that the caller owns `node` and `pod`.
- Enforce one-session-per-node and one-session-per-pod.
- Bind the box's DRTC endpoint to the session (the box is attached to the resolved
  environment; the session's endpoint resolves from it).
- **Persist the session so the node's existing poll picks it up** (the node converges to it).
- `env_slug` (defaulting to the node name when omitted) **must reference an existing
  environment** — a missing one is a `400`. Create it first with `interlatent env create`.

Response — either form is accepted; the CLI only reads `.id`:

```json
{"session": {"id": "sess_77"}}   // or just {"id": "sess_77"}
```

## 5. Stop a session — `DELETE /api/v1/inference/sessions/{id}`

Cancel / unassign a session. Any 2xx is success; the node converges to idle on its next poll.

## 6. Create an environment — `POST /api/v1/environments`

Sessions collect into an **environment** (a data collection), which must exist before
`session start`. Request body the CLI sends:

```json
{
  "slug": "my-arm",                 // required — environment name (--slug)
  "display_name": "my-arm",         // --display-name; defaults to slug
  "robot_type": "so101",            // optional (--robot-type)
  "task_description": "pick cube"   // optional (--task)
}
```

Returns the created environment; the CLI reads `slug` and `environment_id` (falling back
to `id`) for display.

## 7. Register / remove a GPU box — `POST /api/v1/compute/boxes/register`, `DELETE /api/v1/gpus/{name}`

`interlatent gpu add` posts the same registration body `interlatent-serve` sends, so a box
can be entered by address instead of self-registering:

```json
{
  "box_id": "box0",                       // --box-id, defaults to --name
  "name": "box0",                         // --name
  "endpoint": "10.0.0.7:50051",           // --url — host:port your nodes dial
  "provider": "manual",
  "warmup_policy": "lerobot/smolvla_base" // optional (--warm-policy)
}
```

The CLI prints `name` back. `interlatent gpu rm <name>` deletes the row; any 2xx is success.

## 8. Recording destination — `GET`/`PUT /api/v1/coordinator/recording`

Where every session's finished dataset is published. `interlatent config` with no flags GETs
it; with `--output-dir` or `--s3-uri` (plus the optional `--s3-endpoint-url`,
`--s3-access-key`, `--s3-secret-key`, `--s3-region`) it PUTs
`{"recording": {...}}`. Both return `{"recording": {...}}`, empty when nothing is set — the
CLI then prints that sessions will run but not be saved. A coordinator that does not manage
recording destinations returns `404` and the CLI reports that by name rather than surfacing a
bare not-found.

---

## Notes

- These endpoints are the **only** demands the CLI places on a coordinator. The robot
  node daemon (`interlatent-node`) talks to the coordinator independently and is already
  covered by the existing nodes API (pair / heartbeat / poll / hardware / robot-features).
- Two auth identities exist: the operator key (`ilop_…`, used by this CLI and for DRTC
  inference) and the node token (`ilnode_…`, minted at pair time for the node daemon). This
  CLI only ever uses the operator key.
