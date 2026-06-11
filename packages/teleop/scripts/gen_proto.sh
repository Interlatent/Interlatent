#!/usr/bin/env bash
# Regenerate Python gRPC stubs for the teleop protocol.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROTO_DIR="$ROOT/src/interlatent_teleop/protocol"
PROTO_FILE="$PROTO_DIR/teleop.proto"

python3 -m grpc_tools.protoc \
  -I"$PROTO_DIR" \
  --python_out="$PROTO_DIR" \
  --grpc_python_out="$PROTO_DIR" \
  "$PROTO_FILE"

# Post-process: rewrite the generated `import teleop_pb2` to a relative
# import so the module loads cleanly inside the package, and strip
# `_registered_method=True` (sonora's WebChannel doesn't accept it; it's
# a no-op optimization otherwise). Mirrors the DRTC stub treatment.
python3 -c "
import pathlib, re
p = pathlib.Path('$PROTO_DIR/teleop_pb2_grpc.py')
src = p.read_text()
src = re.sub(r'^import teleop_pb2 as', 'from . import teleop_pb2 as', src, flags=re.M)
src = re.sub(r',\s*_registered_method=True', '', src)
p.write_text(src)
"

echo "Generated stubs in: $PROTO_DIR"
