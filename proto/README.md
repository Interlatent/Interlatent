# Wire protocol

`messages.proto` in **this directory** is the single source of truth for the DRTC
inference contract. Everything else is generated or mirrored from it.

Spoken by:

| Side | Package | Mirror |
|---|---|---|
| Robot-side client | `packages/sdk` (`interlatent`) | `src/interlatent/inference/protocol/` |
| Policy server | `packages/server` (`interlatent-server`) | `src/interlatent_server/protocol/` |

Interlatent's hosted GPU boxes run the same protocol; until the hosted image
builds from `packages/server` (ADR 0035), the closed `interlatent-engine` carries
its own copy that must not drift from this one.

Each mirror holds a copy of `messages.proto` next to its stubs so editors, type
checkers, and the in-image regeneration (`docker/gen_proto.sh`) can find it. Those
copies are **outputs** — never edit them. Edit this file and regenerate:

```bash
pip install 'grpcio-tools==1.74.0'
./proto/gen_proto.sh
```

Pin the toolchain version. The stubs embed the generator's version stamps, so a
different `grpcio-tools` rewrites those lines and the diff reads as a protocol
change when nothing on the wire moved.

`tests/test_proto_sync.py` fails the build if a mirror drifts from this file or if
the two packages' generated descriptors disagree.

The wire package name is `interlatent.inference.v1` and does **not** track the
Python package names — it is the compatibility surface for every deployed robot
and GPU box.

## Compatibility rules

- **Additive changes only** — new fields with new numbers. Never renumber, remove,
  or repurpose an existing field: old robots and the hosted cloud speak this
  contract.
- Unknown metadata keys in `OpenSession` must be ignored by servers.
- Comments are free to change (they do not affect the serialized descriptor), but
  keep them true — `control_source`'s four-value contract is a live data-integrity
  rule, not documentation.
