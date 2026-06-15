---
status: accepted
---

# A self-hosted GPU box registers with the hosted dashboard by dialing out with the owner's API key

A box you run yourself (`interlatent-serve` on your own hardware) is invisible to the hosted
dashboard, which only knows about boxes it provisioned. When the operator supplies their own
Interlatent API key (`--api-key` / `INTERLATENT_API_KEY`), the box **dials out** and announces
itself, so it appears as a launchable box — mirroring how a provisioned box reports its activity
over a path the backend can't reach inbound (e.g. Tailscale/NAT).

Two phases against the backend's `compute/boxes` API:

- **register** — a one-time handshake (`POST /api/v1/compute/boxes/register`) carrying the box's
  identity, reachable DRTC endpoint, GPU label, and warmup policy, authenticated with the
  operator's `x-api-key`. It **upserts** a row keyed by a UUID the box mints once and persists
  to `~/.interlatent/box-id` — so a restart re-attaches to the same dashboard box instead of
  orphaning a new one. `INTERLATENT_BOX_ID` overrides the persisted id.
- **status** — lightweight activity transitions (`POST /boxes/{box_id}/status`) reusing the same
  key: `ready` when idle, `running` while serving, `uploading` while flushing, and a best-effort
  `stopped` on clean shutdown.

## Considered options

- **Put everything on `/status`** (no register handshake) — rejected: a personal box isn't known
  to the dashboard, so the first contact must carry endpoint/GPU/policy to create the row. A
  separate register keeps `/status` lightweight and lower-traffic.
- **Mint a per-box capability token at register and present it on `/status`** — more secure (the
  status credential is box-scoped, not the user's full key) but adds a token to mint, return,
  persist, and validate. Deferred; the relaxed-auth approach below is enough for the single-owner
  model.
- **Ephemeral (per-process) id** — rejected: every restart would create a new dashboard box.

## Consequences

- **`/boxes/{box_id}/status` auth is relaxed.** It previously accepted only the system admin key;
  it now also accepts the **owning user's** API key (resolve the user, assert `box.user_id`).
  Low-risk: that key can only set activity states on a box it already owns. The owner may also
  report `stopped`; the admin path keeps its existing `{ready,running,uploading}` set.
- **No liveness/TTL.** `status` is the only signal and a BYO box has no provider to detect a
  crash, so a box that dies *ungracefully* lingers as `ready` until the operator removes it in the
  dashboard. Graceful shutdown reports `stopped`. A `last_seen`/TTL is a deliberate future step
  (it would touch the DB schema + frontend).
- **Reporting is off by default.** With no API key, `interlatent-serve` makes no outbound calls
  and behaves exactly as before. The whole feature is gated on the key.
- **Backend scope.** Only `site/app/routers/compute.py` changes (new `register` route + the
  `/status` auth relax); the `ComputeBox` model already supports BYO (`provider="byo"`, nullable
  cost, `public_ip`+`endpoint_port`), so there is **no DB migration**.
