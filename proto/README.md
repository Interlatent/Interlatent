# Wire protocol

`messages.proto` is the single source of truth for the DRTC inference contract spoken by:

- the robot-side client (`packages/sdk` → `interlatent/inference/protocol/`)
- the cloud-managed GPU pods (Interlatent's hosted endpoint)

Generated `*_pb2.py` stubs are committed in the SDK. After editing the proto, regenerate:

```bash
pip install grpcio-tools
./proto/gen_proto.sh
```

Compatibility rules:

- **Additive changes only** — new fields with new numbers. Never renumber, remove, or
  repurpose an existing field: old robots and the hosted cloud speak this contract.
- Unknown metadata keys in `OpenSession` must be ignored by servers.
