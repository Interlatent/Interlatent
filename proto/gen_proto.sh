#!/usr/bin/env bash
# Regenerate Python gRPC stubs from the protos in this directory into the
# SDK, server, and teleop packages. proto/ is the source of truth; the
# per-package .proto copies are mirrored from here.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="$ROOT/proto"

SDK_PROTO_DIR="$ROOT/packages/sdk/src/interlatent/inference/protocol"
SERVER_PROTO_DIR="$ROOT/packages/server/src/interlatent_server/protocol"
TELEOP_PROTO_DIR="$ROOT/packages/teleop/src/interlatent_teleop/protocol"

# --- DRTC inference protocol (messages.proto) -> sdk + server ----------
for OUT_DIR in "$SDK_PROTO_DIR" "$SERVER_PROTO_DIR"; do
  cp "$SRC_DIR/messages.proto" "$OUT_DIR/messages.proto"
  python3 -m grpc_tools.protoc \
    -I"$SRC_DIR" \
    --python_out="$OUT_DIR" \
    --grpc_python_out="$OUT_DIR" \
    "$SRC_DIR/messages.proto"
done

# --- Teleop protocol (teleop.proto) -> teleop ---------------------------
cp "$SRC_DIR/teleop.proto" "$TELEOP_PROTO_DIR/teleop.proto"
python3 -m grpc_tools.protoc \
  -I"$SRC_DIR" \
  --python_out="$TELEOP_PROTO_DIR" \
  --grpc_python_out="$TELEOP_PROTO_DIR" \
  "$SRC_DIR/teleop.proto"

# Post-process generated stubs:
#   1. Rewrite `import X_pb2` to a relative import so the generated
#      module imports cleanly inside our packages.
#   2. Strip `_registered_method=True` kwargs — recent grpcio-tools
#      emits these for a client-side optimization, but sonora's
#      WebChannel doesn't accept the kwarg, which breaks gRPC-Web
#      clients. Removing it is safe (the optimization just doesn't
#      kick in) and keeps stubs portable across gRPC client variants.
python3 - "$SDK_PROTO_DIR/messages_pb2_grpc.py" \
          "$SERVER_PROTO_DIR/messages_pb2_grpc.py" \
          "$TELEOP_PROTO_DIR/teleop_pb2_grpc.py" <<'PY'
import pathlib, re, sys
for arg in sys.argv[1:]:
    p = pathlib.Path(arg)
    src = p.read_text()
    src = re.sub(r'^import (\w+_pb2) as', r'from . import \1 as', src, flags=re.M)
    src = re.sub(r',\s*_registered_method=True', '', src)
    p.write_text(src)
PY

echo "Generated stubs in:"
echo "  $SDK_PROTO_DIR"
echo "  $SERVER_PROTO_DIR"
echo "  $TELEOP_PROTO_DIR"
