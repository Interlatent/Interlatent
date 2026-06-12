try:
    from interlatent.isaaclab.collection_env import IsaacSimCollectionEnv
except ImportError as _e:
    raise ImportError(
        "interlatent.isaaclab requires Isaac Lab, which is installed separately "
        "from source (it is not on PyPI). See "
        "https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html"
    ) from _e

__all__ = ["IsaacSimCollectionEnv"]
