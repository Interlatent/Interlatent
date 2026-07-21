# Node video encoding & GPU acceleration

The node JPEG-encodes every camera frame on every control tick — for the
recording uplink (full-res, q85), the live teleop preview (downscaled,
q70), and the inference uplink (policy input size). The encoder backend
is **capability-resolved once per process** and logged at session start:

```
node JPEG encoder backend: gpujpeg (cpu fallback: turbojpeg)
```

That log line is the source of truth for what your node is actually
using. The chain, fastest available wins:

| Backend | What it is | When it resolves |
|---|---|---|
| `nvjpeg` | CUDA toolkit's nvJPEG via ctypes (no pip dep) | x86 box with a CUDA GPU. **Not available on Jetson** — JetPack ships no CUDA nvJPEG. |
| `gpujpeg` | [CESNET GPUJPEG](https://github.com/CESNET/GPUJPEG), JPEG on CUDA SMs | Any CUDA GPU with an operator-built `libgpujpeg` — **the GPU path on Jetson** (see below). |
| `turbojpeg` | libjpeg-turbo (SIMD/NEON) | `pip install 'interlatent[turbo]'` + system `libturbojpeg`. Install this on every node. |
| `cv2` | OpenCV `imencode` | OpenCV present. |
| `pil` | Pillow | Always-works fallback. |

GPU backends take RGB frames at or above ~150k pixels (recording
frames); preview/inference-sized and mono frames stay on the CPU chain,
where fixed per-call GPU overhead would dominate. Any GPU failure falls
back to CPU per-frame — encoding can never crash the control loop.

## Jetson (Orin) setup

Platform facts that decide the setup (see SDK ADR 0019):

- **JetPack ships no CUDA nvJPEG.** The `libnvjpeg.so` under
  `/usr/lib/aarch64-linux-gnu/tegra/` is an unrelated Tegra library
  with the same name — the `nvjpeg` backend cannot work on Jetson.
- **Orin Nano has no JPEG hardware** (no NVJPG block, no NVENC). Orin
  NX / AGX have NVJPG, but the SDK does not use it yet either way.
- The only GPU-accelerated JPEG encode on an Orin Nano is therefore
  **GPUJPEG**, which runs on CUDA cores and must be built once from
  source.

One-time install on the Jetson:

```bash
# 1. CUDA toolkit (skip if /usr/local/cuda already exists)
sudo apt update && sudo apt install -y cuda-toolkit cmake build-essential

# 2. Build + install GPUJPEG — use this tag; the SDK pins its ABI (v0.25+,
#    validated against v0.27.13) and refuses older builds.
#    The toolkit does NOT put nvcc on PATH, and CMake's CUDA-arch
#    auto-detect fails without it ("Failed to detect a default CUDA
#    architecture") — point at nvcc and name the arch explicitly
#    (Orin family = 87, Xavier = 72).
git clone --branch v0.27.13 --depth 1 https://github.com/CESNET/GPUJPEG.git
cd GPUJPEG
export PATH=/usr/local/cuda/bin:$PATH
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=87
cmake --build build -j$(nproc)
sudo cmake --install build && sudo ldconfig
# If /usr/local/cuda/bin/nvcc is missing, the metapackage skipped the
# compiler — install the versioned toolkit matching your JetPack
# (e.g. `sudo apt install cuda-toolkit-12-6` on JetPack 6) and retry.

# 3. CPU fallback for small/preview frames (NEON, ~3x faster than cv2)
pip install 'interlatent[turbo]' && sudo apt install -y libturbojpeg
```

Then run the node and confirm the log says
`node JPEG encoder backend: gpujpeg (cpu fallback: turbojpeg)`. If it
says `cv2` or `pil`, run the probe with debug logging to see why:

```bash
python -c "import logging; logging.basicConfig(level=logging.DEBUG); \
  from interlatent.node import gpujpeg; print(gpujpeg.probe())"
```

## Tuning knobs

| Env var | What it does |
|---|---|
| `INTERLATENT_JPEG_BACKEND` | `auto` (default) \| `nvjpeg` \| `gpujpeg` \| `turbojpeg` \| `cv2` \| `pil`. Starts the chain at the named backend — the kill-switch when an encoder misbehaves in the field. A forced backend that fails to probe warns and falls through. |
| `INTERLATENT_GPU_JPEG_MIN_PIXELS` | Pixel area (post-resize) below which frames stay on the CPU chain even when a GPU backend resolved. Default `150000`. |
| `INTERLATENT_PREVIEW_HZ` | Live teleop preview rate **ceiling**, clamped [1, 30], default 10. Read once at node start — set it in the environment the node process actually inherits. On the QUIC transport the effective rate backs off (down to 1 Hz) when the uplink drops video streams and recovers automatically when it clears. |
| `INTERLATENT_PREVIEW_ADAPTIVE` | `0` disables the QUIC congestion backoff — the preview runs at the fixed configured rate. Default on. |
| `INTERLATENT_REC_DRAIN_CEILING_S` | Force a fixed close-drain hard ceiling (seconds). Default: scales with the banked spool bytes at an assumed ≥250 KiB/s link, floor 600 s. |

## Know what encode does — and does not — buy you

Encode cost is a CPU-budget problem; **bandwidth is a separate one**.
3 cameras × 640×480 @ 30 Hz at q85 offer ~3.5 MB/s to the recording
uplink regardless of which backend produced the bytes. If your uplink
can't carry that, the disk spool absorbs the difference losslessly
(ADR 0023) — but it grows until the session closes (close blocks until
the spool drains; killing the node instead leaves an orphan spool and a
truncated episode), and a saturated uplink also degrades the live
preview and teleop responsiveness. The levers for bandwidth are camera
resolution (`--camera name=dev,width=…,height=…`), the preview rate
(`INTERLATENT_PREVIEW_HZ` — every preview frame competes with the
recording and teleop traffic), and `INTERLATENT_REC_MAX_KBPS` (keep it
just under your measured uplink; it is bufferbloat headroom, not a way
to shrink the data). USB-side bandwidth (camera pixel formats, shared
USB2 domains) is covered in [ROBOT.md](../ROBOT.md#usb-bandwidth-on-multi-camera-rigs).
