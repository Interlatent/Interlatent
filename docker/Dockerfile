# syntax=docker/dockerfile:1.7
# Interlatent DRTC inference server — GPU container image.
#
# This image runs `interlatent-serve`, which is the
# persistent-GPU policy server the Pi-side node daemon connects to over
# native gRPC. Designed to be portable across GPU providers (RunPod,
# Lambda Labs, Vast.ai, Prime Intellect, bare metal).
#
# Policy support:
#   The DRTC server uses LeRobot's `PreTrainedConfig.from_pretrained` +
#   `make_policy` path, so any policy LeRobot ships with works in
#   principle — ACT, Diffusion Policy, VQ-BeT, TDMPC, Pi0/Pi0.5,
#   SmolVLA, and third-party policies registered through
#   `register_third_party_plugins`. The default install pulls runtime
#   deps for the popular families; less common deps can be added at
#   build time via `--build-arg EXTRA_PIP_PACKAGES=...`.
#
# Build:
#   # Default broad image (recommended)
#   docker build -f docker/Dockerfile -t interlatent-drtc:latest .
#
#   # Slim image: only SmolVLA, no Pi0/diffusion stack
#   docker build -f docker/Dockerfile \
#     --build-arg LEROBOT_EXTRAS=smolvla \
#     --build-arg EXTRA_PIP_PACKAGES="" \
#     -t interlatent-drtc:smolvla .
#
#   # Add an out-of-tree policy's deps
#   docker build -f docker/Dockerfile \
#     --build-arg EXTRA_PIP_PACKAGES="openvla flash-attn" \
#     -t interlatent-drtc:openvla .
#
# Run (single GPU, expose 50051):
#   docker run --rm --gpus all -p 50051:50051 \
#     -v interlatent-cache:/root/.cache \
#     -e DRTC_WARMUP_POLICY=lerobot/smolvla_base \
#     -e HF_TOKEN=hf_xxx \
#     interlatent-drtc:latest
#
# The `interlatent-cache` volume persists torch.compile + HF model
# downloads across container restarts — without it, every restart
# re-pays the multi-minute torch.compile for VLA-class policies.

# --- base ---------------------------------------------------------------
#
# We base on NVIDIA's CUDA runtime image rather than a `pytorch/pytorch`
# tag, because lerobot 0.5.x requires Python >= 3.12 and every
# `pytorch/pytorch:2.7.x` tag still ships Python 3.11 (PyTorch's Docker
# build matrix never bumped its conda default past 3.11). Ubuntu 24.04
# ships Python 3.12 as the system `python3`, which is the floor lerobot
# 0.5 demands, so we install torch ourselves from the cu128 wheel index
# on top of it.
#
# Trade-off vs the old pytorch base: torch is no longer pre-baked, so
# the build pulls the ~3 GB torch + nvidia-* cu128 wheels itself. The
# image is correspondingly larger, but it's the only path that gets us
# py3.12 + torch 2.7 + lerobot 0.5 (MolmoAct2 and other 0.5-era
# policies) in one interpreter.
#
# Version history of this base and why:
# - runpod/pytorch:2.4.0 (py3.11): original.
# - pytorch/pytorch:2.6.0-cuda12.4 (py3.11): required by lerobot 0.4.x,
#   pinned <0.5 to dodge 0.5's py3.12 floor.
# - nvidia/cuda:12.8.1 + py3.12 + torch 2.7.1 (current): lerobot 0.5.x
#   needs py3.12, which no pytorch/pytorch tag provides. Installing
#   torch 2.7.1+cu128 by hand is the supported way to get there. torch
#   2.7.1 ↔ torchvision 0.22.1 is the matched pair for this index.
#
# lerobot RTC has historically had Dynamo-tracer bugs on older torch
# (see lerobot_backend._patch_lerobot_rtc_device_bug for the device
# fix, and DRTC_RTC=0 in compute.py for the kill switch). RTC stays
# disabled by default and the box runs plain chunking — fine for
# DAgger/teleop, smooth enough for most autonomous work. Re-validate
# RTC on torch 2.7 / lerobot 0.5 before enabling it by default.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # All Python work happens in a venv at /opt/venv (the CUDA base has
    # no Python, and Ubuntu 24.04's system pip is PEP-668 externally
    # managed). Putting it first on PATH makes `python`/`pip` resolve to
    # the 3.12 venv for every build layer, the entrypoint, and the
    # healthcheck — none of which hardcode an interpreter path.
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    # Persist torch.compile artifacts here. interlatent-serve sets these too,
    # but exporting up-front means HF downloads + any sub-process also
    # land under /root/.cache, which is what the volume mounts.
    HOME=/root \
    TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor \
    TRITON_CACHE_DIR=/root/.cache/triton \
    TORCHINDUCTOR_FX_GRAPH_CACHE=1 \
    HF_HOME=/root/.cache/huggingface \
    # Default torch.compile mode. NOTE: must NOT enable CUDA graphs.
    # `reduce-overhead` and `max-autotune` both capture CUDA graphs, which
    # crash flow-matching policies (SmolVLA / Pi0 / Pi0.5) at every forward
    # with "Offset increment outside graph capture encountered unexpectedly"
    # — their noise sampling advances the Philox RNG offset inside the
    # captured graph. See docs/adr/0004-no-cudagraphs-for-flow-matching.md.
    # `default` gives inductor's kernel fusion without cudagraphs. On a box
    # with a persistent /root/.cache volume, override at runtime with
    # -e DRTC_COMPILE_MODE=max-autotune-no-cudagraphs for the best
    # steady-state latency (slow first compile, then cached).
    DRTC_COMPILE_MODE=default

# --- system packages ----------------------------------------------------
#
# ffmpeg + libGL/libglib are required by lerobot's camera + opencv path.
# git is required by `pip install` of HF/lerobot when they pull
# git-based deps. tini gives us PID-1 signal handling so docker stop
# actually shuts the server down cleanly instead of waiting 10s for
# SIGKILL. iptables / iproute2 are pulled in for Tailscale even in
# userspace mode (some subcommands shell out to `ip`).
# build-essential supplies gcc/make: a few lerobot deps (notably
# `evdev`) ship sdists only and compile a C extension at install time.
# python3 / python3-venv / python3-pip: Ubuntu 24.04 ships Python 3.12
# as `python3`; the CUDA base image strips it, so reinstall it plus the
# venv module we build /opt/venv from. python-is-python3 isn't used —
# the venv's bin/ already provides `python`.
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
        iproute2 \
        iptables \
        tini \
        python3 \
        python3-dev \
        python3-venv \
        python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip setuptools wheel

# --- tailscale ----------------------------------------------------------
#
# Bake the Tailscale static binaries into the image so the container
# joins its tailnet on startup (entrypoint.sh handles `tailscale up`
# when TS_AUTHKEY is set). We use the static .tgz, not the apt repo —
# fewer moving parts on rebuilds, no key/repo churn, and the binaries
# are statically linked.
#
# Userspace networking (--tun=userspace-networking in the entrypoint)
# means the container does NOT need NET_ADMIN or /dev/net/tun. The
# tradeoff is that inbound TCP from tailnet peers requires
# `tailscale serve --tcp <port>` to proxy from the tailnet listener to
# the in-container loopback — the entrypoint sets that up for the
# DRTC port automatically.
ARG TAILSCALE_VERSION=1.74.1
RUN curl -fsSL "https://pkgs.tailscale.com/stable/tailscale_${TAILSCALE_VERSION}_amd64.tgz" \
        -o /tmp/tailscale.tgz \
    && tar -xzf /tmp/tailscale.tgz -C /tmp \
    && mv "/tmp/tailscale_${TAILSCALE_VERSION}_amd64/tailscale"  /usr/local/bin/ \
    && mv "/tmp/tailscale_${TAILSCALE_VERSION}_amd64/tailscaled" /usr/local/bin/ \
    && rm -rf /tmp/tailscale.tgz "/tmp/tailscale_${TAILSCALE_VERSION}_amd64" \
    && mkdir -p /var/lib/tailscale /tmp/tailscale

# --- python deps --------------------------------------------------------
#
# Install in three phases so the heavy lerobot/transformers layer can
# stay cached across small changes to the engine source.

# 1. Engine runtime deps that aren't in pyproject.toml.
#    - grpcio: interlatent-serve uses native gRPC, but engine pyproject doesn't
#      declare it (it's optional in other engine code paths).
#    - httpx: server-side episode recorder talks to S3 via httpx.
#    - boto3 is intentionally NOT installed — uploads use presigned URLs
#      from the Interlatent backend, no AWS creds on the GPU box.
#    - protobuf / grpcio are pinned to >= the versions the checked-in
#      *_pb2.py stubs were generated with (protobuf 6.31.1, grpc 1.80).
#      Those stubs run a hard version check at import and abort
#      interlatent-serve if the runtime is older.
RUN pip install \
        "grpcio>=1.80" \
        "grpcio-tools>=1.80" \
        "protobuf>=6.31.1" \
        "httpx>=0.24" \
        "numpy>=1.24" \
        "pillow"

# 2. torch + LeRobot + policy runtime deps.
#
#    The CUDA base ships no torch, so install the matched
#    torch 2.7.1 / torchvision 0.22.1 cu128 pair from PyTorch's wheel
#    index first. cu128 matches the base image's CUDA 12.8; the wheels
#    bundle their own nvidia-* CUDA libs.
#
#    Reliability: this is ~3 GB across several wheels (torch 2.8 GB,
#    nvidia-cusolver-cu12 ~260 MB, nvidia-cudnn-cu12 ~700 MB, ...). On
#    flaky links the per-file download breaks mid-stream and pip's
#    internal retry restarts the *file* from byte 0, so a connection
#    that drops near the end of a big wheel can loop forever within one
#    pip invocation. Two layers of resilience:
#      1. ``--mount=type=cache`` gives pip a wheel cache that persists
#         across RUN invocations (even with --no-cache build) and across
#         retries of THIS RUN — completed wheels from a failed attempt
#         are reused by the next attempt instead of being redownloaded.
#         We override PIP_NO_CACHE_DIR=1 by passing --cache-dir
#         explicitly so the env-level disable doesn't defeat the mount.
#      2. Bash retry loop wraps the pip invocation: up to 8 attempts
#         with exponential-ish backoff. Each attempt benefits from the
#         cache populated by previous attempts.
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

#    The DRTC backend loads any policy that `PreTrainedConfig` can
#    decode, but the policy classes themselves each have heavyweight
#    runtime deps that aren't in lerobot's base install. We bake in the
#    union for the popular families so the same image runs:
#      - ACT          (base lerobot only)
#      - Diffusion    (diffusers)
#      - VQ-BeT       (einops + base)
#      - TDMPC        (base)
#      - SmolVLA      (transformers, accelerate, num2words)
#      - Pi0 / Pi0.5  (transformers, sentencepiece)
#      - MolmoAct2    (transformers, peft, scipy)
#
#    LEROBOT_EXTRAS controls which lerobot pip extras get installed.
#    LEROBOT_REF is the git ref we install from (see below).
#    EXTRA_PIP_PACKAGES is for out-of-tree policies (e.g. OpenVLA via
#    a third-party plugin). All build args so users can slim down or
#    extend the image without forking this Dockerfile.
#
#    This is the slow layer — keep it before the COPY so it doesn't
#    invalidate when engine source changes.
# `dataset` is required even on a pure inference box: the server-side
# session recorder rebuilds a LeRobotDataset at upload time, which imports
# lerobot.datasets and hard-fails without the `datasets` package. In
# lerobot 0.5.x/main it moved out of the base install into this extra.
ARG LEROBOT_EXTRAS="dataset,smolvla,pi0,molmoact2"
# MolmoAct2 is only on lerobot `main` (post-0.5.1); the latest PyPI
# release, 0.5.1, predates it. Pin the exact commit that added MolmoAct2
# so the build is reproducible. Switch to a `>=0.5.2` PyPI pin once a
# release actually ships MolmoAct2.
ARG LEROBOT_REF="24017e960c39a24fe1b6ea6248522460fa5aa4b3"
ARG EXTRA_PIP_PACKAGES=""

# Pin the torch / CUDA stack we just installed to a pip constraints
# file *before* installing lerobot. Without this, lerobot's transitive
# deps drag in a different torch pin and pip re-downloads multi-GB of
# torch + cuDNN + cuBLAS + nvjitlink on top of what's already there,
# doubling the image size. The constraint forces pip to treat the
# installed versions as satisfied.
RUN pip freeze | grep -iE "^(torch|torchvision|torchaudio|nvidia-|triton)==" \
    > /tmp/torch-constraints.txt \
    && echo "--- torch-constraints.txt ---" && cat /tmp/torch-constraints.txt

# Install lerobot from the pinned git ref (the post-0.5.1 commit that
# ships MolmoAct2 + requires Python 3.12, which this image now provides).
# Requires git, installed in the system-packages layer above. Try the
# requested extras first; if a ref doesn't expose one (extras names
# drift) fall back to base — every model family still loads, just
# without the convenience extra. RTC bugs are gated by the DRTC_RTC env
# var injected from the backend (see compute.py).
#
# Same retry+cache pattern as the torch layer above — lerobot's
# transitive closure pulls transformers, diffusers, peft, scipy and
# friends (several GB), and any one of them can drop mid-download on a
# flaky link. The cache mount makes earlier successes survive across
# retries so we don't redownload everything from scratch.
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

# Explicit policy-family runtime deps. lerobot's extras pin most of
# these, but install them directly too so the no-extras fallback branch
# above still yields a working image. peft + scipy are MolmoAct2's extra
# deps (lerobot[molmoact2] == transformers + peft + scipy). These are
# floors: run AFTER the lerobot install, pip leaves any already-satisfied
# version untouched, so lerobot's own pins stand and we only fill in
# whatever the no-extras fallback left missing.
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

# Pin protobuf to the version the image will ship: the lerobot /
# transformers layer above can quietly downgrade it via a transitive
# pin. Pinning here — before the stubs are regenerated against it —
# keeps gencode == runtime. No-op when nothing downgraded it.
RUN pip install "protobuf>=6.31.1"

# 3. The server itself. Only copy what we need — keeps the build
#    context tight and the layer small.
WORKDIR /app
COPY docker/gen_proto.sh /usr/local/bin/gen_proto.sh
COPY packages/server/pyproject.toml /app/packages/server/pyproject.toml
COPY packages/server/README.md /app/packages/server/README.md
COPY packages/server/src /app/packages/server/src

# Regenerate the protobuf / gRPC stubs from messages.proto against the
# protobuf + grpcio-tools installed in this image, so the generated
# code's hard runtime-version assertion always matches the runtime that
# ships here — no checked-in *_pb2.py drift. Image-only counterpart of
# proto/gen_proto.sh. Run before the install so it picks up the fresh
# stubs.
RUN chmod +x /usr/local/bin/gen_proto.sh \
    && /usr/local/bin/gen_proto.sh \
        /app/packages/server/src/interlatent_server/protocol

RUN pip install /app/packages/server

# --- entrypoint ---------------------------------------------------------
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# 50051 = native gRPC (DRTC inference). 50052 = browser teleop WS relay
# (DAgger). The relay only listens when INTERLATENT_TELEOP_SECRET is set
# in the container env; otherwise this port is documentation only.
EXPOSE 50051 50052

# Healthcheck: bind a TCP probe at 50051. We deliberately don't run a
# full gRPC health probe because that pulls in grpc_health_probe and
# adds a layer. A port check is enough to know the server is up — gRPC
# itself reports per-method errors if something deeper is wrong.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import socket,sys; s=socket.socket(); s.settimeout(3); s.connect(('127.0.0.1',50051)); s.close()" || exit 1

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/entrypoint.sh"]
CMD []
