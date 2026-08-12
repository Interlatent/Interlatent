---
status: accepted
---

# One coordinator implementation, so one protocol tier

The upstream control plane is gone; `interlatent up` is the only coordinator
that exists. The three-tier route model in `coordinator/protocol.py` —
mandatory, optional, coordinator-only — described a two-implementation world
(`TIER_COORDINATOR_ONLY` was defined as the routes the *other* implementation
404s), so it collapses to a single tier, and the thirteen routes nothing will
ever serve are deleted rather than left advertised.

## What was deleted

- **The inbox plane** (3 routes): `POST /episodes`,
  `POST /episodes/{id}/upload-urls`, `POST /episodes/{id}/upload-complete`,
  along with `BackendInboxSink` and the three `SessionRecorder` methods that
  drove it. `interlatent up` never served these — it answers only
  `/capabilities` from the optional tier (`server.py:_OPTIONAL_SERVED`).
- **The analysis and dataset surfaces** (10 routes): `analyze`, `process`,
  `processing-status`, `cancel-processing`, `episodes/{id}` and its `status` /
  `results` / `meta` / `chunks`, and `inbox-gc`. Product surface, never protocol.
- **The box's credential *kinds*** (`interlatent_server/credentials.py`):
  `INTERLATENT_ADMIN_KEY` was the provisioning identity and the `ilat_` prefix
  was an upstream account key. `is_system` and its branches go with them — every
  box now self-reports `stopped` and guards its gRPC port unless `--insecure`,
  which was already the self-hosted behavior in both cases.

  The `kind` field went too, rather than collapsing to a single `operator`
  value, because that value would have been wrong. `interlatent_server/cli.py`
  sets `INTERLATENT_API_KEY` to `box_key or args.api_key`, where `box_key` is
  the box-scoped `ilbox_` key the coordinator mints at registration
  (`state.py:285`) — so a normally registered box presents `ilbox_`, not
  `ilop_`, and the old `startswith("ilop_")` test had been labelling every real
  box as a `user`. A box presents whichever key its coordinator handed back and
  never inspects it; nothing downstream branched on the field.

`PROTOCOL_VERSION` moves to `interlatent.coordinator/2`. This is a breaking
change under the additive-only rule the constant documents.

## What was deliberately kept

`GET /compute/boxes/{box_id}/warmup-target`, `POST /compute/boxes/{id}/status`
and the `/environments` auth probe **stay**. They read as upstream code because
of their naming (`_fetch_warmup_target_from_backend`, "the backend", "the env
attached to this box upstream"), but each dials whichever coordinator the box
registered with — which is now always one you run. They are renamed and
redocumented as coordinator calls, not deleted.

Deleting the warmup fetch specifically would have broken a working self-hosted
feature: `interlatent session start --policy X` writes `warm_policy` onto the
box (`state.py:568`), and `state.warmup_target()` hands it back so the box
pre-compiles the policy it is about to be given. Without it the box only ever
warms whatever `--warmup-policy` was typed at launch, and the first `Infer` of
any reassigned session eats a cold `torch.compile`. It also supplies
`image_keys` from the environment's `camera_names`, which `serve_gpu._warmup`
argues keeps warm and first-session configs in agreement by construction —
hand-typed keys can poison `PolicyRuntime`'s `(backend, policy_uri)` cache with
a wrong-camera runtime that the first real session then inherits.

Note that `sdk/coordinator/auth.py`'s `KIND_OPERATOR` / `KIND_NODE` /
`KIND_BOX` (`ilop_` / `ilnode_` / `ilbox_`) are a **different enum** — the keys
a coordinator mints and validates. All three are untouched by this ADR.

## Considered options

- **Keep the tiers as a third-party extension point** — someone else could
  implement the protocol and serve the inbox. Rejected: a route table that
  advertises surfaces no shipping coordinator serves is a table that lies, and
  `server.py:_OPTIONAL_SERVED` already carries a comment explaining that a
  capability list which lies is worse than none.
- **Delete the analysis block but keep the inbox plane** — plausibly protocol
  rather than product. Rejected for the same reason: nothing serves it, and
  `LocalDirSink` / `S3Sink` already cover getting a dataset off the box.

## Consequences

- **Recording no longer has a remote fallback**, so a session with no
  destination would silently record nothing. `serve_gpu`'s `--output-dir` gains
  a real default of `~/.interlatent/episodes` — the same home `box-id`,
  `s3-cache` and `failed-publish` already use. The default lives on the flag
  rather than deeper in the stack so that ADR-0002's precedence stays literally
  two-step (session metadata → `interlatent-serve` flags) and the destination
  is announced in `--help` and in the existing boot log.
- **`requires_api_key()` disappears from the sink protocol.** `LocalDirSink`
  and `S3Sink` both returned False; only the inbox needed a key. Recording is
  now unconditionally account-free, which removes the last coupling between
  having a key and being able to collect data.
- **A coordinator that did serve the inbox tier can no longer be talked to.**
  Acceptable: none exists.
