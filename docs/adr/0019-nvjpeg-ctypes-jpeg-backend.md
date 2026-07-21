# 0019 — CUDA JPEG encode via ctypes bindings (nvJPEG + GPUJPEG)

Status: accepted (2026-07-21); amended same day — see Amendment

## Amendment (2026-07-21): nvJPEG does not exist on Jetson

Field-verified on the Orin Nano fleet the same day this was accepted:
**JetPack ships no CUDA nvJPEG** (no `nvjpeg.h`, no CUDA `libnvjpeg` —
the Tegra `/usr/lib/aarch64-linux-gnu/tegra/libnvjpeg.so` is an
unrelated libjpeg-API library with the same soname), and **the Orin
Nano has no NVJPG hardware block** (Orin NX/AGX do). So the nvJPEG
backend below is effectively x86-CUDA-only, and the GPU path on the
actual target board is the originally-rejected **CESNET GPUJPEG**
(CUDA-SM encode, no fixed-function hardware needed), built from source
by the operator and bound in `node/gpujpeg.py`:

- Resolver order: `nvjpeg → gpujpeg → turbojpeg → cv2 → pil`; env
  choices gain `gpujpeg`; the GPU routing threshold is shared
  (`INTERLATENT_GPU_JPEG_MIN_PIXELS`, old name accepted as alias).
- **ABI pin**: GPUJPEG passes parameter structs by pointer and their
  layout moves between releases; the binding pins the v0.25+ layout
  (validated against v0.27.13), refuses older versions via
  `gpujpeg_version()`, fills all defaults through the library's own
  initializers, and the probe round-trips a color-asymmetric frame
  through PIL so a layout/channel-order mismatch fails at resolve
  time.
- Build recipe (Jetson): install the CUDA toolkit (`sudo apt install
  cuda-toolkit` / `nvidia-jetpack`), then
  `git clone --branch v0.27.13 --depth 1 https://github.com/CESNET/GPUJPEG.git
  && cd GPUJPEG && cmake -B build -DCMAKE_BUILD_TYPE=Release
  && cmake --build build -j$(nproc) && sudo cmake --install build && sudo ldconfig`.

The "rejected: CESNET GPUJPEG (out-of-tree build)" reasoning below
stands for platforms where nvJPEG exists; on Tegra the out-of-tree
build is the only GPU option, which outweighs it.

## Context

The node capture path JPEG-encodes every camera frame per control tick
(recording q85 full-res, preview tee q70 @ 320px, inference uplink at
the policy's input size). The encoder is capability-resolved in
`node/jpeg.py` (turbojpeg → cv2 → PIL); the monorepo's ADR 0023
deferred "a CUDA JPEG path (nvJPEG/GPUJPEG)" as a later optimization.

The trigger is the Jetson Orin Nano fleet at the ADR 0022 design
ceiling (3 × 720p × 30 fps): the Orin Nano has **no NVENC and no NVJPG
fixed-function block**, so CPU turbojpeg encode (~6–8 ms per 720p
frame) eats most of a 33 ms tick budget across three cameras. The
Orin's GPU is otherwise idle on a node (inference is hosted), and
nvJPEG runs JPEG encode on CUDA SMs.

## Decision

Add an **nvJPEG backend as a thin ctypes wrapper over `libnvjpeg.so`**
(`node/nvjpeg.py`), resolved first in the `node/jpeg.py` chain when a
CUDA device is visible (`cudaGetDeviceCount` via `libcudart`) and a
16×16 probe encode succeeds. Alternatives rejected:

- **CESNET GPUJPEG** — faster, but an out-of-tree source build users
  must produce per platform; no maintained Python binding.
- **torchvision CUDA `encode_jpeg`** — trivial code, but drags CUDA
  torch + torchvision onto the node; the SDK's posture is CPU-only
  deps.
- **PyPI nvJPEG bindings** — unmaintained and CUDA-ABI-fragile.

`libnvjpeg` + `libcudart` ship with every JetPack/CUDA toolkit, so the
binding adds **zero pip dependencies**; a box without them silently
resolves the CPU chain.

Shape of the integration:

- **Routing, not replacement**: nvJPEG takes RGB frames at or above
  `_NVJPEG_MIN_PIXELS` (default 150 000 px, post-resize;
  `INTERLATENT_NVJPEG_MIN_PIXELS` overrides). The per-call GPU cost
  (H2D copy + launch + sync + bitstream retrieve) is ~fixed while CPU
  cost scales with area — preview/uplink frames (≤ 77k px) stay on the
  CPU chain, recording frames (≥ 307k px) go GPU. Mono frames always
  stay CPU (no grayscale path in v1).
- **Kill-switch**: `INTERLATENT_JPEG_BACKEND`
  (`auto|nvjpeg|turbojpeg|cv2|pil`) starts the chain at the named
  backend; a forced backend that fails to probe warns and falls
  through — the node never ends up encoder-less.
- **Failure semantics unchanged**: a per-call nvJPEG error logs once
  (WARNING, then debug) and falls back to the best CPU encoder; the
  control loop never crashes on encode.
- **4:2:0 subsampling, `NVJPEG_INPUT_RGBI`**, quality 1:1 — matching
  turbojpeg's defaults so backends stay interchangeable; the
  cross-backend parity test pins the color order (auto-skipped on
  GPU-less CI).

## Consequences

- Recording encode cost on CUDA nodes moves off the control thread's
  CPU budget; turbojpeg remains the recommended install for the
  small-frame classes and non-CUDA boxes.
- Every foreign call declares ctypes `argtypes`/`restype` (implicit
  int truncation of 64-bit pointers on aarch64 is the classic failure);
  the probe encode surfaces ABI breakage once at resolve time, never
  at 30 Hz.
- Pinned host staging, CUDA streams, and a grayscale path are
  documented later micro-optimizations, not v1.

## Verification (Jetson)

Run the node and confirm `node JPEG encoder backend: nvjpeg (cpu
fallback: …)`. Microbench: 300 × `encode_jpeg` on a random 640×480×3
frame under `INTERLATENT_JPEG_BACKEND=nvjpeg` vs `=turbojpeg`; decode
one output via PIL and sanity-check per-channel means; watch
`tegrastats` CPU during a 3-camera session before/after.
