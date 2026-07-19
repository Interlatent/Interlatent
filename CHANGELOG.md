# Changelog

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
