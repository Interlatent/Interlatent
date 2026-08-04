"""DRTC server. See parent package docstring."""

# Importing registers backends with policy_runtime. lerobot_backend
# imports lazily so this is safe even without lerobot installed.
from . import policy_runtime  # noqa: F401  (registers echo, tiny_torch)
from . import lerobot_backend  # noqa: F401  (registers "lerobot")
from . import molmoact2_backend  # noqa: F401  (registers "molmoact2")

