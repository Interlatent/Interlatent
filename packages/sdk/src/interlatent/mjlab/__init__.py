"""interlatent.mjlab — MuJoCo/Isaac Lab (mjlab) integration for Interlatent.

Install with:
    pip install 'interlatent[mjlab]'

Usage:
    from interlatent.mjlab import CollectionEnv
"""

try:
    from interlatent.mjlab.collection_env import CollectionEnv
except ImportError as _e:
    raise ImportError(
        "interlatent.mjlab requires mjlab, which is installed separately from "
        "source (it is not on PyPI). See https://github.com/mujocolab/mjlab"
    ) from _e

__all__ = ["CollectionEnv"]
