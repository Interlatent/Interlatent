"""DRTC server. See parent package docstring."""

# Importing registers backends with policy_runtime. Each backend imports
# its heavy deps (torch / lerobot / transformers) lazily inside __init__,
# so these imports are safe even without those engines installed.
from . import policy_runtime  # noqa: F401  (registers echo, tiny_torch)
from . import lerobot_backend  # noqa: F401  (registers "lerobot")
from . import molmoact2_backend  # noqa: F401  (registers "molmoact2")
from . import spatialvla_backend  # noqa: F401  (registers "spatialvla")
from . import rdt_backend  # noqa: F401  (registers "rdt")

