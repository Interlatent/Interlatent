# Wire protocol

`messages.proto` in **this directory** is the single source of truth for the DRTC
inference contract. Everything else is generated or mirrored from it.

Spoken by:

| Side | Package | Mirror |
|---|---|---|
| Robot-side client | `packages/sdk` (`interlatent`) | `src/interlatent/inference/protocol/` |
| Policy server | `packages/server` (`interlatent-server`) | `src/interlatent_server/protocol/` |

Interlatent's hosted GPU boxes run the same protocol; until the hosted image builds
from `packages/server` ([ADR 0023](../docs/adr/0023-self-hosted-policy-server-returns.md)),
the closed `interlatent-engine` carries its own copy that must not drift from this one.

The wire package name is `interlatent.inference.v1` and does **not** track the
Python package names — it is the compatibility surface for every deployed robot
and GPU box.

## What's in it

`service InferenceService`:

| RPC | Purpose |
|---|---|
| `OpenSession(OpenSessionRequest) → OpenSessionResponse` | Bootstrap: policy, chunk size, codec, `min_execution_horizon`. Returns the `session_id`. |
| `Stream(stream Observation) → stream ActionChunk` | The hot path — pipelined observations, chunks pushed back as they're ready. |
| `Infer(Observation) → ActionChunk` | Unary fallback for gRPC-Web / simpler clients. |
| `RecordTick(RecordTickRequest) → RecordTickResponse` | One captured control tick into the server's recorder, decoupled from `Infer` so recording runs at the control rate. |
| `RecordTicks(RecordTicksRequest) → RecordTicksResponse` | Batched `RecordTick`; the drain coalesces queued ticks so the RTT is amortized. Falls back to unary against older servers. |
| `CloseSession(CloseSessionRequest) → CloseSessionResponse` | Tear down; the server builds and uploads the episode. |

Messages: `Action` (one control-rate action + its `action_step` and monotonic
`control_timestamp`, which drives the client's last-writer-wins merge), `Span` (a
contiguous run of already-scheduled steps, sent up for RTC in-painting),
`Observation` (opaque `payload` + `payload_codec`, `next_action_step`,
`scheduled_spans`, `inference_delay`), `ActionChunk` (actions plus
`server_compute_ns` for the compute-vs-network latency split), the four session
messages, and `RecordTickRequest` (`observation_state`, `action`, per-camera
`jpegs`, `control_source`) with its two responses.

## Changing it

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

## Compatibility rules

- **Additive changes only** — new fields with new numbers. Never renumber, remove,
  or repurpose an existing field: old robots and the hosted cloud speak this
  contract.
- Unknown metadata keys in `OpenSession` must be ignored by servers.
- Comments are free to change (they do not affect the serialized descriptor), but
  keep them true — `control_source`'s four-value contract (`policy` / `teleop` /
  `intervention` / `hold`) is a live data-integrity rule, not documentation.
