# 0019 — CUDA JPEG encode via a ctypes nvJPEG binding

Status: accepted (2026-07-21)

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
