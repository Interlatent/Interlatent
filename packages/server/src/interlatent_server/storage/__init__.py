"""LeRobot v3.0 dataset writers used by the DRTC recorder.

Two writers over one feature/frame contract:

  - :mod:`.lerobot_rebuild` — the offline rebuild from a
    :class:`~.lerobot_rebuild.StepSource` (JSONL + JPEG staging dir).
  - :mod:`.lerobot_live` — the live, during-session encoder (ADR 0016).
    Any failure here falls back to the rebuild path, so the two must
    stay artifact-equivalent.

Both are import-light: numpy, pyarrow, and PyAV are imported inside the
functions that need them, so importing this package costs nothing on a
box that only serves inference.
"""
