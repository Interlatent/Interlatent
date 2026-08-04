# syntax=docker/dockerfile:1.7
# interlatent-server — self-hosted DRTC policy server GPU image.
#
# Runs `interlatent-serve`: registers this machine with the Interlatent
# dashboard as a self-hosted compute box (your API key, your hardware),
# then serves policies to robot nodes over native gRPC. Portable across
# GPU providers (RunPod, Lambda Labs, Vast.ai, Prime Intellect, bare
# metal).
#
# Build (from the repo root):
#   docker build -f docker/Dockerfile -t interlatent-server:latest .
#
#   # Slim image: only SmolVLA, no Pi0/diffusion stack
#   docker build -f docker/Dockerfile \
#     --build-arg LEROBOT_EXTRAS=dataset,smolvla \
#     -t interlatent-server:smolvla .
#
# Run (single GPU, expose 50051):
#   docker run --rm --gpus all -p 50051:50051 \
#     -v interlatent-cache:/root/.cache \
#     -e INTERLATENT_API_KEY=ilat_xxx \
#     -e INTERLATENT_ADVERTISE_ADDRESS=<IP-your-robots-can-reach> \
#     -e HF_TOKEN=hf_xxx \
#     interlatent-server:latest
#
# The `interlatent-cache` volume persists torch.compile + HF model
# downloads across container restarts — without it, every restart
# re-pays the multi-minute torch.compile for VLA-class policies.

# --- base ---------------------------------------------------------------
# NVIDIA CUDA runtime on Ubuntu 24.04: lerobot 0.5.x needs Python >= 3.12
# and no pytorch/pytorch tag ships it, so we install torch cu128 wheels
# ourselves on the system py3.12. See the hosted image's Dockerfile
# history for the full rationale.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    HOME=/root \
    TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor \
    TRITON_CACHE_DIR=/root/.cache/triton \
    TORCHINDUCTOR_FX_GRAPH_CACHE=1 \
    HF_HOME=/root/.cache/huggingface \
    DRTC_COMPILE_MODE=auto

# ffmpeg + libGL/libglib for lerobot's camera/opencv path; git for
# pip's git-based installs (the pinned lerobot ref); tini for PID-1
# signal handling; build-essential for sdist-only deps (evdev).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        ca-certificates \
        curl \
        tini \
        python3 \
        python3-dev \
        python3-venv \
        python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip setuptools wheel

# --- python deps --------------------------------------------------------
# protobuf/grpcio floors match the checked-in *_pb2.py stubs' hard
# runtime-version assertion (regenerated below anyway).
RUN pip install \
        "grpcio>=1.80" \
        "grpcio-tools>=1.80" \
        "protobuf>=6.31.1" \
        "httpx>=0.24" \
        "numpy>=1.24" \
        "pillow"

# torch cu128 pair matched to the base's CUDA 12.8. Retry loop + pip
# cache mount so a mid-download drop on a multi-GB wheel doesn't restart
# the whole set from byte 0.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    PIP_NO_CACHE_DIR=0 sh -eu -c 'for i in 1 2 3 4 5 6 7 8; do \
        if pip install --cache-dir /root/.cache/pip \
              --retries 10 --timeout 600 \
              --index-url https://download.pytorch.org/whl/cu128 \
              "torch==2.7.1" "torchvision==0.22.1"; then \
            exit 0; \
        fi; \
        echo "==> torch install attempt $i failed; sleeping $((i*10))s" >&2; \
        sleep $((i*10)); \
    done; \
    echo "torch install failed after 8 attempts" >&2; exit 1'

# `dataset` is required even on a pure inference box: the session
# recorder rebuilds a LeRobotDataset at upload time. MolmoAct2 is only
# on lerobot main (post-0.5.1) — pinned ref until a release ships it.
ARG LEROBOT_EXTRAS="dataset,smolvla,pi0,molmoact2"
ARG LEROBOT_REF="24017e960c39a24fe1b6ea6248522460fa5aa4b3"
ARG EXTRA_PIP_PACKAGES=""

# Constrain the installed torch/CUDA stack before lerobot's transitive
# deps can drag in a different torch pin and double the image size.
RUN pip freeze | grep -iE "^(torch|torchvision|torchaudio|nvidia-|triton)==" \
    > /tmp/torch-constraints.txt \
    && echo "--- torch-constraints.txt ---" && cat /tmp/torch-constraints.txt

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    PIP_NO_CACHE_DIR=0 sh -eu -c 'for i in 1 2 3 4 5 6 7 8; do \
        if pip install --cache-dir /root/.cache/pip \
              --retries 10 --timeout 600 \
              -c /tmp/torch-constraints.txt \
              "lerobot[${LEROBOT_EXTRAS}] @ git+https://github.com/huggingface/lerobot@${LEROBOT_REF}"; then \
            exit 0; \
        fi; \
        if pip install --cache-dir /root/.cache/pip \
              --retries 10 --timeout 600 \
              -c /tmp/torch-constraints.txt \
              "lerobot @ git+https://github.com/huggingface/lerobot@${LEROBOT_REF}"; then \
            exit 0; \
        fi; \
        echo "==> lerobot install attempt $i failed; sleeping $((i*10))s" >&2; \
        sleep $((i*10)); \
    done; \
    echo "lerobot install failed after 8 attempts" >&2; exit 1'

# Explicit policy-family floors so the no-extras fallback branch above
# still yields a working image (pip leaves satisfied versions untouched).
RUN pip install -c /tmp/torch-constraints.txt \
        "transformers>=4.40" \
        "accelerate>=0.30" \
        "diffusers>=0.27" \
        "einops>=0.7" \
        "peft>=0.10" \
        "scipy>=1.11" \
        "sentencepiece" \
        "num2words" \
        "safetensors" \
        "hf_transfer" \
        "hf_xet" \
        "datasets>=2.19"

RUN if [ -n "${EXTRA_PIP_PACKAGES}" ]; then \
        pip install -c /tmp/torch-constraints.txt ${EXTRA_PIP_PACKAGES}; \
    fi

# Re-pin protobuf in case the lerobot/transformers layer downgraded it.
RUN pip install "protobuf>=6.31.1"

# --- the server package -------------------------------------------------
WORKDIR /app
COPY docker/gen_proto.sh /usr/local/bin/gen_proto.sh
COPY packages/server /app/packages/server

# Regenerate the protobuf/gRPC stubs against THIS image's protobuf, so
# gencode == runtime and the stubs' hard version assertion always holds.
RUN chmod +x /usr/local/bin/gen_proto.sh \
    && /usr/local/bin/gen_proto.sh \
        /app/packages/server/src/interlatent_server/protocol

RUN pip install /app/packages/server

# --- entrypoint ---------------------------------------------------------
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 50051

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import socket,sys; s=socket.socket(); s.settimeout(3); s.connect(('127.0.0.1',50051)); s.close()" || exit 1

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/entrypoint.sh"]
CMD []
