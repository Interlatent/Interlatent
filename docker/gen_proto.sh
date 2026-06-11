#!/usr/bin/env bash
# Regenerate the DRTC protobuf / gRPC Python stubs in place, against the
# protobuf + grpcio-tools installed in the current environment.
#
# Run during the Docker build. The checked-in *_pb2.py files carry a
# hard runtime-version assertion, so the generated code must match the
# protobuf runtime that ships in the image. Generating the stubs here
# makes gencode == runtime and removes that whole class of drift.
#
# This is the image-side counterpart of the repo's proto/gen_proto.sh
# — the GPU image does not contain the SDK package.
#
# Usage: gen_proto.sh <protocol-dir>   (the dir holding messages.proto)
set -euo pipefail

PROTO_DIR="${1:?usage: gen_proto.sh <protocol-dir>}"
PROTO_FILE="$PROTO_DIR/messages.proto"

python -m grpc_tools.protoc \
    -I"$PROTO_DIR" \
    --python_out="$PROTO_DIR" \
    --grpc_python_out="$PROTO_DIR" \
    "$PROTO_FILE"

# Post-process the gRPC stub, exactly as scripts/gen_proto.sh does:
#   1. rewrite `import messages_pb2` to a relative import so it resolves
#      inside the package;
#   2. strip `_registered_method=True` — a kwarg recent grpcio-tools
#      emit that sonora's gRPC-Web channel does not accept.
python - "$PROTO_DIR/messages_pb2_grpc.py" <<'PY'
import pathlib, re, sys

p = pathlib.Path(sys.argv[1])
s = p.read_text()
s = re.sub(r'^import messages_pb2 as', 'from . import messages_pb2 as', s, flags=re.M)
s = re.sub(r',\s*_registered_method=True', '', s)
p.write_text(s)
PY

echo "[gen_proto] regenerated DRTC stubs in $PROTO_DIR"
