# `interlatent` CLI — backend API reference

The `interlatent` CLI (`cli/main.py`) is a thin client for the Interlatent dashboard.
This document is the contract it expects from the backend, so the dashboard API can be
implemented to match. Endpoints marked **TODO(api)** are net-new and the client currently
codes against the placeholder paths below — pin them down here (or change the client to
match the real contract).

## Transport & auth (all endpoints)

Requests go through `interlatent._http.HTTPClient`.

- **Base URL:** `https://interlatent.com` (override `--api-base` / `INTERLATENT_API_BASE`).
- **Auth:** `x-api-key: ilat_…` on every request. The backend resolves the key to a user +
  access rights and **scopes every response to that user**.
- **Headers sent:** `Accept: application/json` (plus `x-vercel-protection-bypass` when configured).
- **Error semantics the client depends on:**
  - `401` / `403` → CLI prints "authentication failed — check your INTERLATENT_API_KEY".
  - `404` → CLI prints "not found".
  - `5xx` → client auto-retries up to 3× (5s apart); return `5xx` only for genuinely
    transient failures.
  - JSON error bodies: the client reads `detail` or `message` for the displayed text.
- **List-shape flexibility:** any list endpoint may return *either* a bare JSON array *or* an
  object wrapping it under a named key (e.g. `{"pods": [...]}`). Either parses.
- **Field tolerance:** the documented fields are what the CLI table renders. Unknown fields
  are ignored and missing ones render blank, so the shape is forgiving — but `session start`
  needs `id` back.

---

## 1. List pods — `GET /api/v1/pods`  **TODO(api)**

GPU pods the user can run sessions on.

```json
[
  {"id": "pod_a1b2", "name": "a100-0", "status": "ready",
   "gpu": "A100-40GB", "region": "us-east"}
]
```

- `status` is free text the CLI prints verbatim (e.g. `ready` / `busy` / `starting`).
- Bare array or `{"pods": [...]}`.

## 2. List nodes — `GET /api/v1/nodes`

The user's paired robot nodes (read-only view; same resource the node daemon pairs against).

```json
[
  {"id": "node_9f3", "name": "my-arm", "status": "online", "robot_type": "so101"}
]
```

- Bare array or `{"nodes": [...]}`.

## 3. List sessions — `GET /api/v1/inference/sessions/`  **TODO(api)**

Active inference sessions for the user.

```json
[
  {"id": "sess_77", "node": "my-arm", "pod": "a100-0",
   "policy_uri": "lerobot/smolvla_base", "status": "running"}
]
```

- Bare array or `{"sessions": [...]}`.

## 4. Start a session — `POST /api/v1/inference/sessions/`  **TODO(api)**

The one write action. Request body the CLI sends:

```json
{
  "node": "my-arm",                   // required — name or id
  "pod": "a100-0",                    // required — name or id
  "policy": "lerobot/smolvla_base",   // required
  "backend": "lerobot",               // defaults to "lerobot"
  "task": "pick up the cube",         // optional — omitted when empty
  "env_slug": "my-arm",               // optional
  "fps": 30,                          // optional (float)
  "chunk_size": 50,                   // optional (int)
  "action_dim": 6                     // optional (int)
}
```

Backend responsibilities:

- Authorize that the user owns `node` and `pod`.
- Enforce one-session-per-node and one-session-per-pod.
- Bind the pod's DRTC endpoint to the session.
- **Persist the session so the node's existing poll picks it up** (the node converges to it).

Response — either form is accepted; the CLI only reads `.id`:

```json
{"session": {"id": "sess_77"}}   // or just {"id": "sess_77"}
```

## 5. Stop a session — `DELETE /api/v1/inference/sessions/{id}`  **TODO(api)**

Cancel / unassign a session. Any 2xx is success; the node converges to idle on its next poll.

---

## Notes

- These five endpoints are the **only** demands the CLI places on the backend. The robot
  node daemon (`interlatent-node`) talks to the dashboard independently and is already
  covered by the existing nodes API (pair / heartbeat / poll / hardware / robot-features).
- Two auth identities exist: the user key (`ilat_…`, used by this CLI and for DRTC inference)
  and the node token (`ilnode_…`, minted at pair time for the node daemon). This CLI only
  ever uses the user key.
