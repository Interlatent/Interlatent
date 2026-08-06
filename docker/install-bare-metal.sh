#!/usr/bin/env bash
# install-bare-metal.sh — provision a GPU box to run `interlatent-serve`
# WITHOUT Docker.
#
# The pip path in docs/self-hosting.md is two commands, but it leaves the
# operator holding every environment assumption docker/Dockerfile encodes:
# a Python new enough for lerobot, a torch matched to the driver's CUDA,
# ffmpeg for the dataset writers, and protobuf stubs that agree with the
# installed runtime. This script is those layers, minus the container.
#
# Deliberately NOT a curl|sh bootstrap: it wants the repo checked out
# anyway (it reads the lerobot pin out of the Dockerfile so the two can
# never drift), and a self-hosted GPU box is exactly where piping a remote
# script into a root shell is a bad habit to teach.
#
# Usage, from the repo root:
#
#   sudo ./docker/install-bare-metal.sh                  # system deps + venv
#   ./docker/install-bare-metal.sh --no-system           # skip apt (no sudo)
#   ./docker/install-bare-metal.sh --venv ~/ilat --cuda cu126
#   ./docker/install-bare-metal.sh --systemd \
#       --api-key ilat_xxx --advertise-address 203.0.113.7
#
# Idempotent: re-running upgrades in place and rewrites the unit file.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKERFILE="$REPO_ROOT/docker/Dockerfile"

VENV="${VENV:-/opt/interlatent}"
CUDA_TAG=""            # cu128 / cu126 / cpu — auto-detected when empty
PYTHON_BIN=""          # auto-detected when empty
DO_SYSTEM=1
DO_SYSTEMD=0
API_KEY="${INTERLATENT_API_KEY:-}"
ADVERTISE="${INTERLATENT_ADVERTISE_ADDRESS:-}"
PORT="${INTERLATENT_PORT:-50051}"
WARMUP_POLICY="${DRTC_WARMUP_POLICY:-}"
WARMUP_IMAGE_KEYS="${DRTC_WARMUP_IMAGE_KEYS:-}"
LEROBOT_EXTRAS="dataset,smolvla,pi0,molmoact2"

# torch/torchvision pair. Kept in step with docker/Dockerfile — the CUDA
# tag varies per host driver, so it is resolved separately below.
TORCH_VERSION="2.7.1"
TORCHVISION_VERSION="0.22.1"

# lerobot must be >= this for `import lerobot.datasets` to work at all.
MIN_PYTHON_MINOR=12

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m==> WARNING:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m==> ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

usage() {
    # The header comment IS the help text — print it up to the first line
    # of actual code, so the two can't drift.
    awk 'NR==1 {next} /^#/ {sub(/^# ?/, ""); print; next} {exit}' \
        "${BASH_SOURCE[0]}"
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        --venv)               VENV="$2"; shift 2 ;;
        --cuda)               CUDA_TAG="$2"; shift 2 ;;
        --python)             PYTHON_BIN="$2"; shift 2 ;;
        --lerobot-extras)     LEROBOT_EXTRAS="$2"; shift 2 ;;
        --api-key)            API_KEY="$2"; shift 2 ;;
        --advertise-address)  ADVERTISE="$2"; shift 2 ;;
        --port)               PORT="$2"; shift 2 ;;
        --warmup-policy)      WARMUP_POLICY="$2"; shift 2 ;;
        --warmup-image-keys)  WARMUP_IMAGE_KEYS="$2"; shift 2 ;;
        --no-system)          DO_SYSTEM=0; shift ;;
        --systemd)            DO_SYSTEMD=1; shift ;;
        -h|--help)            usage ;;
        *) die "unknown argument: $1 (try --help)" ;;
    esac
done

[ -f "$DOCKERFILE" ] || die "run this from a checkout — $DOCKERFILE not found"

# --- 1. the lerobot pin, read from the Dockerfile -------------------------
# Single source of truth. Duplicating the SHA here is exactly the drift the
# in-image proto regen was built to eliminate, so we parse it instead.
LEROBOT_REF="$(
    sed -n 's/^ARG LEROBOT_REF="\{0,1\}\([0-9a-f]\{7,\}\)"\{0,1\}.*/\1/p' \
        "$DOCKERFILE" | head -1
)"
[ -n "$LEROBOT_REF" ] || die \
    "could not read ARG LEROBOT_REF from $DOCKERFILE — the pin moved; fix \
the sed in this script rather than hardcoding a SHA"
log "lerobot pin (from docker/Dockerfile): $LEROBOT_REF"

# --- 2. system packages ---------------------------------------------------
# ffmpeg is the one that matters at runtime: the dataset writers shell out
# to it, and without it recording fails only once an episode is uploaded.
# libgl1/libglib2.0-0 back lerobot's opencv import.
if [ "$DO_SYSTEM" = 1 ]; then
    [ "$(id -u)" = 0 ] || die "system deps need root — use sudo, or pass --no-system"
    log "installing system packages (ffmpeg, libgl1, python venv toolchain)"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends \
        build-essential git ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 \
        ca-certificates curl python3 python3-dev python3-venv python3-pip
else
    log "skipping system packages (--no-system)"
    command -v ffmpeg >/dev/null 2>&1 || warn \
        "ffmpeg not on PATH — the dataset writers need it; recording will \
fail at upload time with a rebuild error"
fi

# --- 3. a Python new enough for lerobot -----------------------------------
if [ -z "$PYTHON_BIN" ]; then
    for cand in python3.13 python3.12 python3; do
        command -v "$cand" >/dev/null 2>&1 || continue
        minor="$("$cand" -c 'import sys; print(sys.version_info[1])')"
        if [ "$minor" -ge "$MIN_PYTHON_MINOR" ]; then PYTHON_BIN="$cand"; break; fi
    done
fi
[ -n "$PYTHON_BIN" ] || die \
    "no python3.$MIN_PYTHON_MINOR+ found. lerobot 0.5.x requires it — this is \
the reason docker/Dockerfile builds on ubuntu24.04 instead of a \
pytorch/pytorch tag. Install one (deadsnakes PPA, uv, conda) and re-run \
with --python."
log "python: $PYTHON_BIN ($("$PYTHON_BIN" --version))"

# --- 4. CUDA wheel index --------------------------------------------------
# Match torch to the DRIVER's CUDA, not the toolkit's. A cu128 wheel on a
# 12.4 driver imports fine and then fails at the first kernel launch.
if [ -z "$CUDA_TAG" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
        cuda_ver="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9]*\)\.\([0-9]*\).*/\1\2/p' | head -1)"
        case "$cuda_ver" in
            12[89]|1[3-9]*) CUDA_TAG="cu128" ;;
            12[67])         CUDA_TAG="cu126" ;;
            "")             CUDA_TAG="cu128"; warn "could not parse CUDA version from nvidia-smi; assuming cu128" ;;
            *)              CUDA_TAG="cu126"; warn "CUDA $cuda_ver is older than torch $TORCH_VERSION's cu126 floor; trying cu126 anyway" ;;
        esac
        log "GPU driver $drv, CUDA $cuda_ver -> torch index $CUDA_TAG"
    else
        CUDA_TAG="cpu"
        warn "nvidia-smi not found — installing CPU torch. The server will \
start and serve, but every policy load lands on CPU and inference will be \
unusably slow. Pass --cuda cu128 if this is wrong."
    fi
fi

# --- 5. venv + python deps ------------------------------------------------
log "creating venv at $VENV"
"$PYTHON_BIN" -m venv "$VENV"
PIP="$VENV/bin/pip"
"$PIP" install --upgrade pip setuptools wheel

log "installing torch $TORCH_VERSION ($CUDA_TAG)"
if [ "$CUDA_TAG" = "cpu" ]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
else
    TORCH_INDEX="https://download.pytorch.org/whl/$CUDA_TAG"
fi
"$PIP" install --retries 10 --timeout 600 --index-url "$TORCH_INDEX" \
    "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION"

# Pin the resolved torch/CUDA stack before lerobot's transitive deps can
# drag in a different torch. Same trick as the image, same reason.
CONSTRAINTS="$(mktemp)"
trap 'rm -f "$CONSTRAINTS"' EXIT
"$PIP" freeze | grep -iE "^(torch|torchvision|torchaudio|nvidia-|triton)==" \
    > "$CONSTRAINTS" || true
log "torch constraints pinned ($(wc -l < "$CONSTRAINTS") entries)"

log "installing lerobot @ $LEROBOT_REF [$LEROBOT_EXTRAS]"
"$PIP" install --retries 10 --timeout 600 -c "$CONSTRAINTS" \
    "lerobot[$LEROBOT_EXTRAS] @ git+https://github.com/huggingface/lerobot@$LEROBOT_REF" \
    || {
        warn "extras [$LEROBOT_EXTRAS] did not resolve; retrying plain lerobot"
        "$PIP" install --retries 10 --timeout 600 -c "$CONSTRAINTS" \
            "lerobot @ git+https://github.com/huggingface/lerobot@$LEROBOT_REF"
    }

# Explicit floors so the no-extras fallback above still yields a working
# install (pip leaves already-satisfied versions untouched).
"$PIP" install -c "$CONSTRAINTS" \
    "transformers>=4.40" "accelerate>=0.30" "diffusers>=0.27" "einops>=0.7" \
    "peft>=0.10" "scipy>=1.11" sentencepiece num2words safetensors \
    hf_transfer hf_xet "datasets>=2.19" "pyarrow>=15" "av>=12"

log "installing interlatent-server from $REPO_ROOT/packages/server"
"$PIP" install "$REPO_ROOT/packages/server"

# --- 6. proto stubs against THIS runtime ----------------------------------
# packages/server/pyproject.toml declares protobuf>=4.25, but the checked-in
# stubs assert a >=6.31.1 runtime at import. A fresh resolve gets 6.x and is
# fine; anything holding protobuf lower breaks on import. The image dodges
# this by regenerating in-layer — do the same here.
log "regenerating protobuf/gRPC stubs against the installed runtime"
"$PIP" install "grpcio-tools>=1.80" "protobuf>=6.31.1"
PROTO_DIR="$("$VENV/bin/python" -c \
    'import interlatent_server.protocol as p, pathlib; print(pathlib.Path(p.__file__).parent)')"
PATH="$VENV/bin:$PATH" "$REPO_ROOT/docker/gen_proto.sh" "$PROTO_DIR"

# --- 7. verify ------------------------------------------------------------
log "verifying the install"
"$VENV/bin/python" - <<'PY'
import sys
import interlatent_server
from interlatent_server.protocol import messages_pb2  # runtime-version assert
import torch

print(f"  interlatent_server  {interlatent_server.__file__}")
print(f"  torch               {torch.__version__}")
print(f"  cuda available      {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device              {torch.cuda.get_device_name(0)} x{torch.cuda.device_count()}")
try:
    import lerobot
    print(f"  lerobot             {getattr(lerobot, '__version__', 'unknown')}")
    import lerobot.datasets  # noqa: F401  — the recorder's whole write path
    print("  lerobot.datasets    importable")
except Exception as e:
    print(f"  lerobot             UNAVAILABLE: {type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(1)
PY
"$VENV/bin/interlatent-serve" --help > /dev/null
log "interlatent-serve entry point OK"

# --- 8. optional systemd unit --------------------------------------------
if [ "$DO_SYSTEMD" = 1 ]; then
    [ "$(id -u)" = 0 ] || die "--systemd needs root to write /etc/systemd/system"
    [ -n "$API_KEY" ]   || die "--systemd needs --api-key (or INTERLATENT_API_KEY)"
    [ -n "$ADVERTISE" ] || die "--systemd needs --advertise-address (or INTERLATENT_ADVERTISE_ADDRESS)"

    UNIT=/etc/systemd/system/interlatent-server.service
    log "writing $UNIT"

    EXTRA_ENV=""
    [ -n "$WARMUP_POLICY" ] && EXTRA_ENV="${EXTRA_ENV}Environment=DRTC_WARMUP_POLICY=$WARMUP_POLICY"$'\n'
    [ -n "$WARMUP_IMAGE_KEYS" ] && EXTRA_ENV="${EXTRA_ENV}Environment=DRTC_WARMUP_IMAGE_KEYS=$WARMUP_IMAGE_KEYS"$'\n'

    cat > "$UNIT" <<UNITEOF
[Unit]
Description=Interlatent DRTC policy server (self-hosted)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
# HOME must be set: the box id persists at ~/.interlatent/box-id, and
# without it every restart mints a new UUID and orphans a dashboard row.
Environment=HOME=/root
Environment=INTERLATENT_API_KEY=$API_KEY
Environment=INTERLATENT_ADVERTISE_ADDRESS=$ADVERTISE
${EXTRA_ENV}ExecStart=$VENV/bin/interlatent-serve --port $PORT
Restart=always
RestartSec=5
# SIGINT, not the default SIGTERM: cli.main catches KeyboardInterrupt and
# reports status=stopped on the way out (wait=True). A SIGTERM skips that
# and the dashboard keeps showing a ghost "ready" box until re-register.
KillSignal=SIGINT
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
UNITEOF

    chmod 600 "$UNIT"   # the unit file carries the API key
    systemctl daemon-reload
    systemctl enable interlatent-server
    log "systemd unit installed. Start it with:"
    echo "    systemctl start interlatent-server && journalctl -fu interlatent-server"
else
    cat <<EOF

$(log "done — start the server with:")

    export INTERLATENT_API_KEY=ilat_xxx
    $VENV/bin/interlatent-serve \\
        --advertise-address <IP-your-robots-can-reach> --port $PORT

Re-run with --systemd (plus --api-key/--advertise-address) to install a
unit file instead. Open ONLY port $PORT to your nodes.
EOF
fi
