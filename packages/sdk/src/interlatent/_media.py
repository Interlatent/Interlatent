"""Local media buffer for frames and future sensor modalities.

Stores rendered frames as JPEGs to a temp directory during collection.
Frame encoding (PIL → JPEG) and disk writes are offloaded to a
background thread so they never block the inference loop.

The upload-time LeRobot rebuild (see ``_dataset.py``) walks the buffer
on a per-episode basis, decodes each JPEG to a numpy array, and feeds
it to ``LeRobotDataset.add_frame(...)``. The lerobot library handles
mp4 encoding internally at ``save_episode`` time, so we no longer
shell out to ffmpeg from the SDK.
"""
from __future__ import annotations

import atexit
import logging
import re
import shutil
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator


_FRAME_RE = re.compile(r"^frame_(?:([a-zA-Z]\w*)_)?(\d+)\.(jpg|jpeg|png)$")

_LOG = logging.getLogger(__name__)


class MediaBuffer:
    """Buffer that saves rendered frames to a local directory.

    Accepts numpy arrays (HWC uint8), PIL Images, or file paths.
    PIL / numpy are imported lazily so the SDK has no hard imaging dependency.

    Heavy work (image encoding + disk I/O) runs in a background thread
    so ``add_frame()`` returns almost immediately.
    """

    def __init__(self, base_dir: str | Path, *, frames_dir: str | Path | None = None) -> None:
        self._base = Path(base_dir)
        # If frames_dir is supplied the caller owns that directory — we use it
        # directly and never delete it on cleanup.
        if frames_dir is not None:
            self._frames_dir = Path(frames_dir)
            self._external_frames = True
        else:
            self._frames_dir = self._base / "frames"
            self._external_frames = False
        self._frames_dir.mkdir(parents=True, exist_ok=True)
        self._count = 0

        # Single-thread executor for background frame encoding + writes.
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="media-buf")
        self._pending: list[Future] = []
        atexit.register(self._drain)

    @property
    def frame_count(self) -> int:
        return self._count

    def add_frame(
        self,
        step: int,
        image: Any,
        *,
        episode_id: str | None = None,
        camera_name: str | None = None,
        format: str = "jpg",
        quality: int = 85,
    ) -> None:
        """Queue a frame for background encoding and disk write.

        For numpy arrays the data is copied before handing off to the
        background thread so the caller can safely reuse the buffer.

        Args:
            step: Environment step number (used in filename).
            image: One of:
                - numpy ndarray (H, W, C) uint8
                - PIL Image
                - str or Path to an existing image file
            episode_id: Episode UUID. When provided, frames are stored
                under a per-episode subdirectory.
            camera_name: Optional camera identifier for multicamera setups.
                When provided, filename becomes ``frame_{camera_name}_{step}.{format}``.
            format: Image format extension (default ``"jpg"``).
            quality: JPEG quality 1-100 (default 85).
        """
        # Snapshot mutable data before submitting to background thread.
        if _is_ndarray(image):
            image = image.copy()

        fut = self._pool.submit(
            self._write_frame, step, image, episode_id, camera_name, format, quality,
        )
        self._pending.append(fut)
        self._count += 1

        # Periodically prune completed futures to avoid unbounded list growth.
        if len(self._pending) > 64:
            self._pending = [f for f in self._pending if not f.done()]

    def _write_frame(
        self,
        step: int,
        image: Any,
        episode_id: str | None,
        camera_name: str | None,
        format: str,
        quality: int,
    ) -> Path:
        """Encode and write a single frame (runs in background thread)."""
        if camera_name:
            fname = f"frame_{camera_name}_{step:07d}.{format}"
        else:
            fname = f"frame_{step:07d}.{format}"

        if episode_id is not None:
            ep_dir = self._frames_dir / f"episode_{episode_id}"
            ep_dir.mkdir(parents=True, exist_ok=True)
            dest = ep_dir / fname
        else:
            dest = self._frames_dir / fname

        if isinstance(image, (str, Path)):
            src = Path(image)
            if src != dest:
                shutil.copy2(src, dest)
        elif _is_pil_image(image):
            _save_pil(image, dest, format, quality)
        elif _is_ndarray(image):
            pil_img = _ndarray_to_pil(image)
            _save_pil(pil_img, dest, format, quality)
        else:
            raise TypeError(
                f"Unsupported image type: {type(image).__name__}. "
                "Expected numpy array, PIL Image, or file path."
            )
        return dest

    def _drain(self) -> None:
        """Wait for all pending background writes to complete."""
        for fut in self._pending:
            fut.result()  # propagate exceptions
        self._pending.clear()

    def flush(self) -> None:
        """Block until all queued frame writes are finished.

        Call this before upload or checkpoint to ensure all frames
        are on disk.
        """
        self._drain()

    def iter_episode_frames(
        self, episode_uuid: str
    ) -> Iterator[tuple[int, str | None, Path]]:
        """Yield ``(step, camera_name, path)`` triples for one episode.

        Drains pending writes first so the listing is complete. Used by
        the upload-time LeRobot rebuild to feed JPEGs into
        ``LeRobotDataset.add_frame``.
        """
        self._drain()
        ep_dir = self._frames_dir / f"episode_{episode_uuid}"
        if not ep_dir.is_dir():
            return
        for f in sorted(ep_dir.iterdir()):
            if not f.is_file():
                continue
            m = _FRAME_RE.match(f.name)
            if not m:
                continue
            cam = m.group(1)
            step = int(m.group(2))
            yield step, cam, f

    def episode_uuids(self) -> list[str]:
        """Return all episode UUIDs that have frames staged."""
        self._drain()
        result: list[str] = []
        for d in sorted(self._frames_dir.iterdir()):
            if d.is_dir() and d.name.startswith("episode_"):
                result.append(d.name[len("episode_"):])
        return result

    def cameras_for_episode(self, episode_uuid: str) -> list[str | None]:
        """Return the sorted list of camera names present for an episode.

        ``None`` indicates a single-camera setup (no camera prefix in
        filenames).
        """
        cams: set[str | None] = set()
        for _step, cam, _path in self.iter_episode_frames(episode_uuid):
            cams.add(cam)
        if None in cams and len(cams) > 1:
            # Mixed single-cam and multi-cam — should not happen but
            # be defensive.
            cams.discard(None)
        return sorted(cams, key=lambda x: (x is None, x or ""))

    def cleanup(self) -> None:
        """Remove the temp directory and all stored media.

        External frame directories (supplied via ``frames_dir``) are never
        deleted — only the internally-created temp base dir is removed.
        """
        self._drain()
        self._pool.shutdown(wait=False)
        if self._external_frames:
            return
        if self._base.exists():
            shutil.rmtree(self._base, ignore_errors=True)


# ---------------------------------------------------------------------------
# Lazy helpers — avoid hard PIL / numpy dependency
# ---------------------------------------------------------------------------


def _save_pil(img: Any, dest: Any, format: str, quality: int) -> None:
    """Save a PIL image, falling back to PNG if JPEG encoding fails."""
    save_kwargs: dict[str, Any] = {}
    if format in ("jpg", "jpeg"):
        save_kwargs["quality"] = quality
    try:
        img.save(str(dest), **save_kwargs)
    except (TypeError, OSError):
        # JPEG encoder broken (Pillow 11.x regression) — fall back to PNG
        png_dest = dest.with_suffix(".png") if hasattr(dest, "with_suffix") else str(dest).rsplit(".", 1)[0] + ".png"
        img.save(str(png_dest))


def _is_pil_image(obj: Any) -> bool:
    try:
        from PIL import Image
        return isinstance(obj, Image.Image)
    except ImportError:
        return False


def _is_ndarray(obj: Any) -> bool:
    try:
        import numpy as np
        return isinstance(obj, np.ndarray)
    except ImportError:
        return False


def _ndarray_to_pil(arr: Any) -> Any:
    import numpy as np
    from PIL import Image
    # Ensure uint8 HWC format for PIL
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    # Ensure contiguous array (some renderers return non-contiguous views)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return Image.fromarray(arr, mode="RGB" if arr.ndim == 3 and arr.shape[2] == 3 else None)
