# Interlatent Axol adapter image — onboard Jetson Orin (JetPack 6.2 / L4T r36.4).
#
# IMPORTANT: the ZED_IMAGE L4T revision MUST match the host's L4T, or `import
# pyzed.sl` dies at load with an `undefined symbol` from the NVIDIA camera /
# multimedia libs the `nvidia` runtime injects from the host. Check the host
# with `head -1 /etc/nv_tegra_release` (e.g. "R36 (release), REVISION: 4.4" =>
# l4t-r36.4) and pin the matching `-l4t-r36.4` tag. Note the Stereolabs tags use
# the MAJOR.MINOR form (`-l4t-r36.4`), not a patch suffix (`r36.4.0` does NOT
# exist). See `docker manifest inspect` / the Docker Hub tag list to confirm.
#
# Based on the Stereolabs ZED image, which already bundles the ZED SDK + CUDA +
# L4T userspace (and `pyzed` for the base's system Python). We layer our SDK on
# top. Two things make this more than a base swap:
#
#   1. Python version. The JP6 / L4T r36 base is Ubuntu 22.04 → Python 3.10, but
#      the Interlatent SDK needs >=3.11 and almond-axol installs cleanly only on
#      >=3.13 (see TESTING.md). So we provision Python 3.13 with `uv`.
#   2. pyzed for the venv. The base's pyzed is built for its system Python 3.10
#      and is NOT importable from the 3.13 venv. We rebuild it by running the ZED
#      SDK's own `get_python_api.py` inside the venv (it fetches the wheel
#      matching the running interpreter + the installed SDK/CUDA).
#
# The ZED L4T images are arm64-only; build natively on the Jetson or on Apple
# Silicon. Pin ZED_IMAGE to the tag matching your JetPack/L4T + ZED SDK
# (Orin / JP6.2 → ZED SDK 5.3, L4T r36.4):
#
#   docker buildx build --platform linux/arm64 \
#     --build-arg ZED_IMAGE=stereolabs/zed:5.3-py-runtime-l4t-r36.4 \
#     --build-arg VENV_PYTHON=3.13 \
#     -f src/interlatent/adapters/axol/Dockerfile \
#     -t interlatent-axol --load \
#     packages/interlatent-sdk
#
# Run on the robot with the GPU + GMSL cameras + CAN exposed. The ZED X (GMSL)
# cameras need the host's `zed_x_daemon` shared in — and its control socket lives
# in host `/tmp` alongside the Argus socket, so mount the WHOLE `/tmp` (mounting
# only `/tmp/argus_socket` leaves the daemon socket invisible → the SDK fails with
# "Failed to connect to daemon socket: No such file or directory"). This matches
# Stereolabs' own zed-docker GMSL run set:
#   docker run --rm -it --runtime nvidia --privileged \
#     --network host --ipc host --pid host \
#     -e NVIDIA_DRIVER_CAPABILITIES=all \
#     -v /dev:/dev \
#     -v /tmp:/tmp \
#     -v /var/nvidia/nvcam/settings/:/var/nvidia/nvcam/settings/ \
#     -v /usr/local/zed/settings/:/usr/local/zed/settings/ \
#     -v /usr/local/zed/resources/:/usr/local/zed/resources/ \
#     -v /etc/systemd/system/zed_x_daemon.service:/etc/systemd/system/zed_x_daemon.service \
#     interlatent-axol
# The host's zed_x_daemon must already be running; the container can't restart it,
# so pass `--robot-arg restart_zed_daemon=false` to the node/hardware_check.

ARG ZED_IMAGE=stereolabs/zed:5.3-py-runtime-l4t-r36.4
FROM ${ZED_IMAGE} AS base

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# git is required to install almond-axol (the [axol] extra pulls it from a
# GitHub URL); the ZED runtime base does not ship it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# uv (provisions Python 3.13 and resolves almond-axol's source pins, which bare
# pip mishandles). Installed with the base's system pip.
RUN pip3 install --no-cache-dir uv

# A Python venv (default 3.13). `--seed` installs pip/setuptools/wheel into it
# so the ZED SDK's get_python_api.py (which shells out to pip) works inside the
# venv. Drop VENV_PYTHON to 3.12 if get_python_api has no cp313 pyzed wheel for
# your ZED SDK version (3.12 still satisfies the SDK's requires-python >=3.11).
ARG VENV_PYTHON=3.13
ENV VIRTUAL_ENV=/opt/venv
RUN uv venv --seed --python "${VENV_PYTHON}" "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /opt/interlatent-sdk

# Copy the SDK source (build context = packages/interlatent-sdk).
COPY pyproject.toml README.md ./
COPY src ./src

# Build pyzed for THIS venv's Python 3.13 from the ZED SDK baked into the base
# image. get_python_api.py picks the wheel matching the interpreter + SDK/CUDA.
RUN uv pip install numpy cython requests \
    && python /usr/local/zed/get_python_api.py

# Interlatent SDK + the Axol extra (almond-axol + lerobot).
RUN uv pip install '.[axol]'

# Entrypoint script: runs any passed command, then drops into an interactive
# bash shell. Lives alongside this Dockerfile in the axol adapter.
COPY src/interlatent/adapters/axol/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /work

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
