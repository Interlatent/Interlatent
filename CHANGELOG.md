# Changelog

## 3.0.0 — the SDK runs on a coordinator you own

**Breaking.** A coordinator address is now required everywhere. Anything that
invoked `interlatent`, `interlatent-preflight`, `interlatent-serve` or
`Interlatent()` without one was relying on a hardcoded default that no longer
exists. The fix is one line:

```bash
export INTERLATENT_COORDINATOR=http://10.0.0.2:8900   # the one you run
```

Paired nodes are unaffected: `node.toml` has always stored the address, and the
old `api_base` key is still read.

### The CLI is a session manager, not a thin client

`interlatent up` runs a **coordinator** on your machine — the control plane that
pairs nodes, tracks GPU boxes, and brokers inference and teleop sessions:

```bash
interlatent up                                   # prints an operator key, once
interlatent gpu add --name rig --url 10.0.0.7:50051
interlatent session start --node arm --gpu rig --policy lerobot/smolvla_base
interlatent config --output-dir /data/lerobot    # where recordings land
interlatent down                                 # refuses while a session is live
```

`--coordinator` (or `INTERLATENT_COORDINATOR`) points the same verbs at any
coordinator you run — your laptop, a box on the LAN, a rented GPU host. That is
the whole point: **one protocol, one client, no second set of verbs.** ADR 0023
blamed a coordinator for the 2026-06 collapse; the actual cause it names is that
the old one served a bespoke `/admin/*` *alongside* `/api/v1/*`, forking every
operator flow. This one serves `/api/v1/*` only. See
[ADR 0038](docs/adr/0038-coordinator-protocol-one-control-plane.md).

`interlatent down` waits for nodes to converge to idle before exiting, because
the node's teardown is what sends `CloseSession` — the only trigger for the
dataset build. Stopping a session unassigns it; it never kills anything.

### Recording lands where you point it

`sinks.py` returns: a finished dataset publishes to **a local directory or an
S3-compatible bucket you own**, both merging on stop into one flat
training-ready LeRobot dataset. `docs/concepts.md` has documented this since
2026-07 and it has not been true since 2026-06; it is true again.

Configure it once on the coordinator and it is stamped onto every session it
issues. A box publishing locally needs no credentials at all — the recorder's
auth gate now asks the *destination* whether a key is required instead of
demanding one unconditionally.

### Teleop runs on a relay you run

`interlatent up` can serve the WebTransport relay itself (`pip install
'interlatent[teleop-relay]'`), so a VR session needs nothing off the LAN.
Because no public CA issues certificates for `10.0.0.5`, the coordinator mints a
short-lived ECDSA P-256 certificate and hands its SHA-256 to the browser as
`serverCertificateHashes`, rotating well inside the 14-day cap Chromium imposes.
Nodes pin it with `INTERLATENT_TELEOP_CA_FILE` — a real trust anchor, rather
than `INTERLATENT_TELEOP_INSECURE`, which disables chain *and* hostname checking
together and would have undone the auth model at the teleop layer.

**The teleop browser now reconnects.** It never has: `onClose` fired once and
nothing re-dialled, so any relay blip ended the VR session permanently and the
operator had to take the headset off. It now runs the same 1→15 s re-mint-and-
redial ladder the node has always had, and clears `specReceived` so a
reconnected browser re-requests its kinematic spec.

### The coordinator is the token authority

Not "the network is the trust boundary" — that was deleted ADR 0001's stance and
it is rejected, not restored. `interlatent up` mints an `ilop_` operator key
(`O_CREAT|O_EXCL`, 0600 — no write-then-chmod window), the coordinator issues
scoped `ilnode_` and `ilbox_` credentials, and **only hashes are persisted**. A
node's token is refused on another node's routes.

One trap worth naming: `/compute/boxes/{id}/authz` accepts the operator key
*and* any node token the coordinator issued, because the node presents
`drtc_api_key or token` on the box's gRPC metadata. Accepting only the operator
key returns `UNAUTHENTICATED` for every `Infer`.

### One coordinator address, resolved in one place

Eight hardcoded copies of a default coordinator address in two incompatible
spellings (bare origin vs `/api/v1`-suffixed), reconciled at runtime by three
separate fixups, collapse into `interlatent._coordinator.resolve()` and a twin
in the server dist. One convention: a coordinator address is a bare origin, and
callers append `/api/v1/…`. `INTERLATENT_API_BASE` is still read for one minor,
with a warning.

Also gone: `DEFAULT_DRTC_URL`, which pointed at one specific Modal deployment.
`INTERLATENT_BYPASS_KEY`, `--bypass-key` and the `bypass_key` node.toml key,
which existed only to send `x-vercel-protection-bypass` past the hosted
deployment's preview auth. With that deployment gone the header has nothing to
get past, so the whole chain is deleted rather than renamed — the SDK stops
naming a hosting vendor in its own configuration surface. A `bypass_key` line
left in an existing `~/.interlatent/node.toml` is simply ignored.

### The control-plane contract has a name: the coordinator protocol

The HTTP surface the node, the GPU box, the CLI and the teleop web app have
always spoken is now written down as the **Interlatent Coordinator Protocol** —
`docs/coordinator-protocol.md`, with a machine-readable
twin at `interlatent.coordinator.protocol` and a test
(`tests/test_coordinator_protocol.py`) that fails if the two disagree by so much
as a reworded summary.

Nothing changes at runtime; this is the surface `interlatent up` already serves,
written down so a third party can implement it too. Routes are tiered: **mandatory** (a node cannot
pair, a box cannot boot, or every RPC is rejected without them), **optional**
(every SDK caller already degrades on 404 — teleop, for instance, simply turns
itself off for the session), and **coordinator-only** (operator routes such as
setting the recording destination, where a 404 must be reported by name rather
than as a generic failure).

Two implicit contracts are promoted to documented ones because they were
discoverable only by reading the source: `GET /api/v1/environments` doubles as
the auth probe a GPU box uses to validate any presented key, and
`DELETE /api/v1/inference/sessions/{id}` **must** unassign rather than kill
anything — `CloseSession` is the only trigger for the dataset build and the
server's idle-GC discards recordings whose session was never closed.

See [ADR 0038](docs/adr/0038-coordinator-protocol-one-control-plane.md), which
supersedes ADR 0023's refusal to ship a control plane at all. The short
version: ADR 0023 blamed a coordinator for a collapse that was actually caused
by *two* control-plane surfaces (`/api/v1/*` and a bespoke `/admin/*`) forking
every operator flow. One protocol with one client is not that, and the
`/api/v1`-only rule is what keeps the distinction real.

### A failed publish no longer deletes the episode it just built

`SessionRecorder.upload()` ended in `finally: self._cleanup_working_dir()`, and the
dataset lives *inside* that working dir — both lanes build it there (`dataset/v3`
for the rebuild lane, `live/v3` for the ADR-0016 live-encode one). So any
exception between "dataset is complete on disk" and "backend acked the upload"
logged a traceback and then `rmtree`'d the finished LeRobot dataset. The episode
was unrecoverable: the ticks are deleted with the staging, and the node already
dropped its spool entries as they were acked over gRPC.

Every step before that point is careful about this — a rebuild failure returns
early, zero-step sessions skip, the 409 case is tolerated — but the one path where
the data is *most* valuable, because it survived the whole build, was the one that
threw it away. It only ever fired when a publish failed outright, which is why
it went unnoticed; with local and S3 destinations landing next, a wrong bucket
or an expired credential makes it routine.

On publish failure the built dataset is now moved out of the working directory to
`~/.interlatent/failed-publish/<episode_id>/` (override with
`INTERLATENT_FAILED_PUBLISH_DIR`) and the path is logged at ERROR. Repeated
failures for one episode id get a `.1`, `.2` suffix rather than clobbering. The
working directory is still cleaned up — this is not a change of tidiness policy,
only a refusal to count a finished dataset as garbage. Quarantine is itself
best-effort and cannot mask the original upload exception.

## Unreleased

### `--warmup-image-keys` fills a partial backend warmup target

The fallback added above only ran when the backend returned *no* target, but the
backend answers with the policy the box registered (`_register` posts
`warmup_policy`) even with no env attached — and then `image_keys` is empty. That
lands in the target branch, which never consulted the operator's flag, so a
MolmoAct2 pre-warm was unreachable: the guard skips for want of cameras, and there
is no env to go configure. The override now fills an empty `image_keys` and only
that; keys the backend did supply still win, and now say so in the log instead of
leaving the flag silently inert.

### MolmoAct2: camera ORDER is now reconciled against the checkpoint

lerobot's MolmoAct2 processor collects frames by iterating `cfg.image_keys` **in
order**, and released checkpoints state that order — `MolmoAct2-BimanualYAM`'s
`norm_stats.json` declares `[top, left, right]`, and its model card spells it out
("`images` should preserve camera order"). The node's order is just the order the
operator passed `--camera` flags, so it agreed only by luck. A permutation was
accepted by every layer and fed the overhead view where the model expects a wrist
view: wrong actions, no error, nothing in the logs.

The backend took the node's list unconditionally, on the assumption that "the
released checkpoint's `camera_keys` are empty". That is true of SO100/101 and false
of BimanualYAM. `_reconcile_camera_keys` now compares the two:

- same set, different order → reordered to the checkpoint's, with a warning naming
  both. The names match exactly, so the intent is unambiguous.
- different count → `RuntimeError` at session open, naming both sides, instead of a
  silent misfeed.
- same count, different names → node order kept (a deployment may rename cameras;
  position is then the only signal) and warned about, because nothing verifies it.
- checkpoint declares no `camera_keys` → unchanged.

### `interlatent-serve` logs the endpoint it registered

`--advertise-address` without a port gets `--port` appended — the *container's*
port, which is wrong whenever a provider proxies an external one (RunPod's TCP
proxy, a NAT forward). The box serves happily and the node gets `UNAVAILABLE:
Connection refused` against a port nobody listens on. Registration now logs the
resolved endpoint and how to override it.

### A self-hosted box no longer ignores its own `--warmup-policy`

Pre-warm config is backend-first: whatever `GET /compute/boxes/{id}/warmup-target`
returns wins, because it derives the policy *and* the camera keys from the attached
environment, so the warm can't disagree with what the node later asks for. The
fallback when that fetch returns nothing was gated on `_has_box_identity()` — but
that is true for an owner-key box too, not just the admin-provisioned boxes it was
written for. So on a self-hosted box, `interlatent-serve --warmup-policy ...` was
silently dropped the moment registration succeeded, which is always. It now falls
back for owner-key and unidentified boxes; a box whose warmup target the coordinator
provisions still skips, which is the case the rule was guarding.

- Added `--warmup-image-keys` / `DRTC_WARMUP_IMAGE_KEYS`. MolmoAct2 can't build its
  feature dict without camera keys, so before this there was no way to pre-warm one
  without an environment attached on the coordinator — it was skipped with a message
  telling you to go configure one. Bare names are normalized
  (`cam_high` → `observation.images.cam_high`). Match the node's `--camera` names:
  `PolicyRuntime` caches on `(backend, policy_uri)` and ignores session metadata on
  reuse, so a mismatched warm is *inherited* by the first real session, not discarded.
- The skip log said "Box has system identity" for owner-key boxes, which sent you
  looking for an admin key that was never there.

### `docker/install-bare-metal.sh` — the image's layers, without the image

Provisions a GPU box to run `interlatent-serve` directly: Python ≥ 3.12 check, torch
matched to the driver's CUDA from `nvidia-smi`, ffmpeg, lerobot at the commit
`docker/Dockerfile` pins (parsed from it, so the pin has one home), proto stubs
regenerated against the installed protobuf, and an optional systemd unit using
`KillSignal=SIGINT` so shutdown reports `stopped` instead of leaving a ghost box.

### `interlatent-server` 0.1.0 — the policy server is open source (ADR 0023)

The DRTC serving stack moved out of the closed engine into `packages/server/`,
published as a second dist. `pip install 'interlatent-server[lerobot]'` on a CUDA
machine, run `interlatent-serve --advertise-address <ip>`, and the box registers with
your coordinator as a compute box. Same server code, same wire protocol, same
episode recording every session has always used. See [docs/self-hosting.md](docs/self-hosting.md).

- Fixed: the dist could not be imported at all. Two relative imports carried over from
  the engine's package layout (`...cloud.box_status`, `...storage.lerobot_*`) resolved
  past the top of `interlatent_server`, and the recorder's dataset writers
  (`storage/lerobot_rebuild.py`, `storage/lerobot_live.py`) were never moved with it.
  The wheel built and passed `twine check` regardless — `packages/server/tests/test_import_surface.py`
  now walks and imports every module, on a bare install with no torch and no lerobot.
- `[lerobot]` now declares `pyarrow` and `av` explicitly. They arrive transitively, but
  without them the recorder's parquet post-edits (episode-uuid injection, `control_source`
  int→string) warn and skip, producing a session the merge cannot fold.
- Added a release workflow: tag `interlatent-server-v<version>` or `interlatent-v<version>`
  to publish via PyPI Trusted Publishing. The tag version must match `pyproject.toml`.

### `proto/messages.proto` is the single source of truth

Both packages carry mirrored copies plus generated stubs; `./proto/gen_proto.sh` writes
all of them in one pass, and `tests/test_proto_sync.py` fails the build if a mirror
drifts or the two packages' descriptors disagree. Pin `grpcio-tools==1.74.0` to
regenerate — a different version rewrites the stubs' version stamps and the diff reads
as a protocol change.

- `RecordTickRequest.control_source` now documents all four values
  (`policy` / `teleop` / `intervention` / `hold`). Both copies described two, and the
  server's called an intervention a teleop — the distinction ADR 0034 (platform repo) introduced, and
  the one training upweights. Comment-only; the wire is unchanged.

### CI

- `packages/sdk/tests/` now runs. 36 tests — loop runner, movement arbitration, Nori
  guard, and the ADR 0034 intervention coverage — were never executed, because the test
  step named only `tests/`.
- New `server` job: import surface, auth/CLI tests, lint, dependency resolution, wheel
  completeness, `twine check`, and a clean-venv install of the built wheel.
- New `teleop-web` job: `npm ci`, `tsc --noEmit`, and a production build for the WebXR
  producer, which landed with no build gate.

## 2.0.0 — 2026-07-18

### BREAKING: client-side collection removed (ADR 0022)

Collection is streaming-first and server-side: a robot node streams JPEG
RecordTicks to a server-side recorder (the DRTC GPU box or the teleop
recorder), which builds the LeRobot dataset and publishes it. The
long-deprecated local path — stage steps + JPEGs on-device,
build a LeRobot dataset locally, `upload()` it — is gone.

- Removed `Interlatent.watch() / collect() / tick() / add_frame() /
  checkpoint() / upload() / register_cameras() / sb3_callback()`. For one
  release these raise a `RuntimeError` pointing at streaming collection;
  the stubs disappear in the next release.
- Removed the staging internals: `_media` (MediaBuffer), `_db`,
  `_storage`, `_schema`, `_watcher`, `_step_source`, `_dataset`,
  `_metrics`, and `interlatent.storage.lerobot_rebuild` (the server-side
  recorder keeps its own copy in `interlatent-engine`).
- Removed the `interlatent-sync-rollout` CLI and the
  `adapters.lerobot.sync_inference` package.
- `Interlatent(db_path=...)` is accepted but ignored (DeprecationWarning).
- Collection now requires a running session; the client, node,
  and protocol remain Apache-2.0.

### Added (ADR 0023: lossless node uplink)

- **Write-through tick spool** (`inference/client/spool.py`): every
  captured tick is journaled to disk (`~/.interlatent/spool`, override
  `INTERLATENT_SPOOL_DIR`) and deleted only after the server's honest
  accepted-prefix ack. Link failures and node crashes no longer lose
  frames; session close blocks on drain-done, and an undrained tail is
  retained on disk and surfaced at the next daemon start.
  Sizing knobs: `INTERLATENT_SPOOL_MAX_MB` (default 6144),
  `INTERLATENT_SPOOL_MIN_FREE_MB` (default 2048). When the spool fills,
  capture HARD-STOPS (loud error, auto-resume on drain) — never silent
  frame thinning.
- **Capability-adaptive JPEG encoder** (`node/jpeg.py`): resolves
  PyTurboJPEG → OpenCV → PIL at runtime; same interface on RPi/Jetson/x86.
  New optional extra `interlatent[turbo]` (requires system libturbojpeg).
- Node heartbeat now reports `recording` state (spool backlog,
  `drain_done`, hard-stop `blocked`) so the backend can gate the next
  session launch on drain completion.
