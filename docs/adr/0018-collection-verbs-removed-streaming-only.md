# 0018 — Collection verbs removed: recording is streaming-only

Status: accepted (2026-07-18)

## Context

The SDK carried two collection paths. The streaming one — the node streams
per-tick JPEG `RecordTick`s to a server-side recorder that builds and
publishes the LeRobot dataset — is how every real session (inference or
teleop) records. The client-side one — `watch()`/`tick()` staging to
local SQLite + JPEGs, an on-device `LeRobotRebuilder` build at
`upload()` — belonged to local-policy sessions, which were deprecated
product-side, yet the code and the public docs still advertised it as
"local-first".

The mismatch became untenable when the target node hardware became
Jetson Orin Nano / RPi-class devices with multi-camera high-resolution
rigs: the Orin Nano has no video-encode hardware at all, so the
on-device dataset build (software AV1 encode) could never meet the
pixel rate — while JPEG capture + streaming comfortably does. The
platform-side decisions are recorded in the monorepo's ADR 0022
(streaming-first collection) and ADR 0023 (node spool / lossless
uplink); this SDK ADR records the public-surface consequence.

## Decision

Remove the client-side collection surface in SDK **2.0.0**:

- `Interlatent.watch / collect / tick / add_frame / checkpoint /
  upload / register_cameras / sb3_callback` become `RuntimeError` stubs
  for one release (clear pointer to streaming collection), then disappear.
- The staging internals are deleted (`_media`, `_db`, `_storage`,
  `_schema`, `_watcher`, `_step_source`, `_dataset`, `_metrics`,
  `storage/lerobot_rebuild`), along with the `interlatent-sync-rollout`
  CLI and `adapters.lerobot.sync_inference`.
- `Interlatent` remains as the HTTP API surface (environments,
  episodes, routing) used by the node daemon and CLI.
- The node capture path gains the write-through tick spool
  (`inference/client/spool.py`, delete-after-ack, drain-done) and the
  capability-adaptive JPEG encoder (`node/jpeg.py`, turbojpeg→cv2→PIL,
  `interlatent[turbo]`).

## Consequences

- **Recording requires a running session.** Recording is a property of a
  session now, not of a local staging directory. The open-core boundary
  moves with it: what stays Apache-2.0 is the client, node, CLI, and wire
  protocol — not a standalone offline collector. Docs (README,
  ARCHITECTURE, concepts, self-hosting) were rewritten accordingly.
- Old integrations fail loudly with a pointer, not silently: the verbs
  raise, and `Interlatent(db_path=...)` warns and ignores.
- The SDK's wheel no longer ships `interlatent.storage`; the engine
  keeps its own `LeRobotRebuilder` (it is the server-side recorder's
  builder).
