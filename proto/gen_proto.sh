#!/usr/bin/env bash
# Regenerate the Python gRPC stubs from the protos in this directory into
# every package that speaks the wire.
#
# `proto/messages.proto` is the SINGLE source of truth. Each package holds
# a mirrored .proto (so editors, type checkers, and the in-image
# regeneration in docker/gen_proto.sh can find it next to the stubs) — but
# those mirrors are outputs, not inputs. Never edit them; edit this one
# and re-run. `tests/test_proto_sync.py` fails the build if they drift.
#
# Consumers:
#   - packages/sdk    (interlatent)        — the robot-side client
#   - packages/server (interlatent-server) — the DRTC policy server
#
# Requires: pip install 'grpcio-tools==1.74.0'
#
# Pin the version. The stubs carry the generator's version stamps
# (GRPC_GENERATED_VERSION, the protobuf gencode runtime assertion), so a
# different grpcio-tools rewrites those lines and the diff looks like a
# protocol change when nothing about the wire moved. 1.74.0 reproduces the
# committed stubs byte-for-byte.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="$ROOT/proto"

TARGET_DIRS=(
  "$ROOT/packages/sdk/src/interlatent/inference/protocol"
  "$ROOT/packages/server/src/interlatent_server/protocol"
)

for OUT_DIR in "${TARGET_DIRS[@]}"; do
  if [ ! -d "$OUT_DIR" ]; then
    echo "error: target dir does not exist: $OUT_DIR" >&2
    exit 1
  fi

  # Mirror the source of truth, then generate beside it.
  cp "$SRC_DIR/messages.proto" "$OUT_DIR/messages.proto"
  python3 -m grpc_tools.protoc \
    -I"$SRC_DIR" \
    --python_out="$OUT_DIR" \
    --grpc_python_out="$OUT_DIR" \
    "$SRC_DIR/messages.proto"

  # Post-process generated stubs:
  #   1. Rewrite `import X_pb2` to a relative import so the generated
  #      module imports cleanly inside our packages.
  #   2. Strip `_registered_method=True` kwargs — recent grpcio-tools
  #      emits these for a client-side optimization, but sonora's
  #      WebChannel doesn't accept the kwarg, which breaks gRPC-Web
  #      clients. Removing it is safe (the optimization just doesn't
  #      kick in) and keeps stubs portable across gRPC client variants.
  python3 - "$OUT_DIR/messages_pb2_grpc.py" <<'PY'
import pathlib, re, sys
for arg in sys.argv[1:]:
    p = pathlib.Path(arg)
    src = p.read_text()
    src = re.sub(r'^import (\w+_pb2) as', r'from . import \1 as', src, flags=re.M)
    src = re.sub(r',\s*_registered_method=True', '', src)
    p.write_text(src)
PY
done

echo "Generated stubs in:"
for OUT_DIR in "${TARGET_DIRS[@]}"; do
  echo "  $OUT_DIR"
done
