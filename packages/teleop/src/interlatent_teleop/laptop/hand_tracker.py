"""MediaPipe hand tracker wrapper.

Uses the modern `mediapipe.tasks.vision.HandLandmarker` API — the
legacy `mp.solutions.hands` module was removed in mediapipe 0.10.14+.
The HandLandmarker needs a `.task` model file; we auto-download it
into the user's cache directory on first run (~7.6 MB).
"""
from __future__ import annotations

import logging
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

_LOG = logging.getLogger("interlatent_teleop.laptop.hand_tracker")

# Google-hosted official HandLandmarker model. Small enough to fetch
# transparently on first run; mirror it locally if you don't want a
# network dependency at startup.
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def _default_model_path() -> Path:
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache / "interlatent_teleop" / "hand_landmarker.task"


def _ensure_model(path: Path) -> Path:
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    _LOG.info("downloading HandLandmarker model -> %s", path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    urllib.request.urlretrieve(_MODEL_URL, tmp)
    tmp.replace(path)
    return path


@dataclass
class HandObservation:
    landmarks: np.ndarray   # shape (21, 3); x,y in [0,1] image space, z relative depth
    handedness: str         # "Left" / "Right"
    confidence: float       # tracking confidence in [0, 1]
    frame_bgr: np.ndarray   # raw camera frame for optional preview
    timestamp_ns: int


class HandTracker:
    """Wraps cv2.VideoCapture + mediapipe HandLandmarker (tasks API)."""

    def __init__(
        self,
        camera_index: int = 0,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
        preferred_handedness: Optional[str] = "Right",
        model_path: Optional[Path] = None,
    ) -> None:
        self.camera_index = camera_index
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.preferred_handedness = preferred_handedness
        self.model_path = model_path or _default_model_path()
        self._cap = None
        self._landmarker = None
        self._mp = None  # mediapipe top-level handle (for Image / ImageFormat)
        self._t0_ns: Optional[int] = None

    def open(self) -> None:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        _ensure_model(self.model_path)

        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"failed to open camera {self.camera_index}")

        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=self.max_num_hands,
            min_hand_detection_confidence=self.min_detection_confidence,
            min_hand_presence_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._mp = mp

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:  # noqa: BLE001
                pass
            self._landmarker = None

    def frames(self) -> Iterator[tuple[np.ndarray, Optional[HandObservation]]]:
        """Yield `(frame_bgr, obs)` per camera frame; `obs` is None when no hand is detected.

        Always yielding the raw frame lets the caller render every tick
        and keep the OS event loop pumping — important on macOS, where
        an OpenCV window that isn't refreshed gets marked unresponsive
        and freezes.
        """
        import cv2
        import time

        if self._cap is None or self._landmarker is None or self._mp is None:
            raise RuntimeError("HandTracker not opened")

        mp = self._mp
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            ts_ns = time.monotonic_ns()
            if self._t0_ns is None:
                self._t0_ns = ts_ns
            # HandLandmarker.detect_for_video requires monotonically
            # increasing millisecond timestamps.
            ts_ms = (ts_ns - self._t0_ns) // 1_000_000

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._landmarker.detect_for_video(mp_image, ts_ms)

            if not result.hand_landmarks:
                yield frame, None
                continue

            # Pick preferred handedness if multiple detected. Note: the
            # tasks API reports handedness from the image's perspective,
            # so a user's *right* hand shows up as "Left" in a non-mirrored
            # camera feed. We don't mirror here; if you find the
            # preferred-handedness picking off, swap the label or run
            # the camera through a horizontal flip upstream.
            chosen_idx = 0
            if result.handedness and self.preferred_handedness:
                for i, h in enumerate(result.handedness):
                    if h and h[0].category_name == self.preferred_handedness:
                        chosen_idx = i
                        break

            lm = result.hand_landmarks[chosen_idx]
            arr = np.array([(p.x, p.y, p.z) for p in lm], dtype=np.float32)
            handedness = "Unknown"
            score = 1.0
            if result.handedness and result.handedness[chosen_idx]:
                cat = result.handedness[chosen_idx][0]
                handedness = cat.category_name
                score = float(cat.score)
            yield frame, HandObservation(
                landmarks=arr,
                handedness=handedness,
                confidence=score,
                frame_bgr=frame,
                timestamp_ns=ts_ns,
            )
