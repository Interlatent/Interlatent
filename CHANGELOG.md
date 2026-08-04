# Changelog

## Unreleased

### `interlatent-server` 0.1.0 — the policy server is open source (ADR 0023)

The DRTC serving stack moved out of the closed engine into `packages/server/`,
published as a second dist. `pip install 'interlatent-server[lerobot]'` on a CUDA
machine, run `interlatent-serve --advertise-address <ip>`, and the box registers with
the dashboard as a self-hosted compute box. Same server code, same wire protocol, same
episode recording as Interlatent's managed boxes. See [docs/self-hosting.md](docs/self-hosting.md).

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
  server's called an intervention a teleop — the distinction ADR 0034 introduced, and
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
RecordTicks to a hosted recorder (DRTC GPU box or teleop recorder pod),
which builds the LeRobot dataset and uploads it through the inbox→merge
path. The long-deprecated local path — stage steps + JPEGs on-device,
build a LeRobot dataset locally, `upload()` it — is gone.

- Removed `Interlatent.watch() / collect() / tick() / add_frame() /
  checkpoint() / upload() / register_cameras() / sb3_callback()`. For one
  release these raise a `RuntimeError` pointing at hosted collection;
  the stubs disappear in the next release.
- Removed the staging internals: `_media` (MediaBuffer), `_db`,
  `_storage`, `_schema`, `_watcher`, `_step_source`, `_dataset`,
  `_metrics`, and `interlatent.storage.lerobot_rebuild` (the server-side
  recorder keeps its own copy in `interlatent-engine`).
- Removed the `interlatent-sync-rollout` CLI and the
  `adapters.lerobot.sync_inference` package.
- `Interlatent(db_path=...)` is accepted but ignored (DeprecationWarning).
- Collection now requires an account/hosted session; the client, node,
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
