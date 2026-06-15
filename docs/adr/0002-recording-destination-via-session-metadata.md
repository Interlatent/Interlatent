---
status: accepted
---

# Recording destination is coordinator-configured and passed to the GPU server via session metadata

The recording destination for offline runs (a local directory or an S3-compatible
`uri`+endpoint+credentials) is configured **once on the Coordinator** and injected into each
session's `recording` block, which the Node forwards verbatim into the DRTC `OpenSession`
metadata; the GPU server builds its dataset sink from that metadata per session. We chose this
over baking the destination into each `interlatent-serve` process because, in the open-source
self-hosted model, **whoever runs the Coordinator also owns the GPU box and the Node** — there
is no privilege boundary between them, so a single config point on the Coordinator is the
convenient and natural home, and routing it through the Node (the only party with a channel to
the server) is the only way to get it there.

## Considered options

- **Destination on `interlatent-serve`** (server-level flags) — kept as a *standalone fallback*
  for running the server by hand, but not the primary path: it would force the destination to be
  reconfigured per GPU process and split config across machines.
- **Destination per `session start`** — folded in: `session start` could override the
  coordinator default, but the default lives on the coordinator.

## Consequences

- **S3 credentials transit coordinator → node → server gRPC metadata.** Acceptable only because
  of the single-owner trust model and a co-owned LAN/tailnet (same boundary as ADR-0001). This
  would be unacceptable in a multi-tenant deployment and must be revisited before any
  shared/hosted coordinator. The hosted path does **not** use this — it keeps the presigned-URL
  inbox (`BackendInboxSink`) and never ships credentials to the server.
- **A local `output_dir` is on the GPU server's filesystem**, since recording is server-side.
  Co-located dev boxes share one disk; a remote GPU box writes locally to itself — use `s3_uri`
  to land data elsewhere.
- **Sink resolution precedence** is fixed at: session metadata → `interlatent-serve` flags →
  `BackendInboxSink`. The `OpenSession` metadata gains a stable `recording` contract
  (`output_dir` | `s3_uri`,`s3_endpoint_url`,`s3_access_key`,`s3_secret_key`,`s3_region`).
