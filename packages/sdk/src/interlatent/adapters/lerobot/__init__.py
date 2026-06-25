"""LeRobot-native rollout/record helpers (``interlatent.adapters.lerobot``).

Adapts LeRobot's own ``Robot``/``Teleoperator`` stack to the Interlatent
platform. The ``sync_inference`` submodule provides ``interlatent-sync-rollout``
— a drop-in replacement for ``lerobot-record`` that captures policy activations
during recording and uploads them for interpretability analysis.

Dependency-heavy (pulls in ``lerobot``); imported lazily, never at package load.
Install with ``pip install 'interlatent[lerobot]'``.
"""
