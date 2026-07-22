"""LeRobot-native rollout/record helpers (``interlatent.adapters.lerobot``).

Adapts LeRobot's own ``Robot``/``Teleoperator`` stack to the Interlatent
platform. (The old ``sync_inference`` submodule / ``interlatent-sync-rollout``
CLI — local record-and-upload — was removed with client-side collection;
see ADR 0022: recording is streaming-first through a hosted session.)

Dependency-heavy (pulls in ``lerobot``); imported lazily, never at package load.
Install with ``pip install 'interlatent[lerobot]'``.
"""
