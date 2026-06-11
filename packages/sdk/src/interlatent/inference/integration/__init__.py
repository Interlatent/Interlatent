"""Glue between the DRTC client and existing SDK surfaces.

- `connect`: one-call helper, `connect_drtc(api_key=..., ...)` -> opened DRTCClient.
- `sdk_adapter`: wires DRTCClient into Interlatent.watch() / tick().
- `rollout`: backend for the `interlatent-rollout` console script,
  which after this refactor is a thin wrapper that launches a DRTC
  client against the hosted server.
"""

from .connect import connect_drtc, DEFAULT_DRTC_URL  # noqa: F401

