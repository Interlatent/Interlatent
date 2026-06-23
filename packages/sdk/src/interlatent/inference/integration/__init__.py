"""Glue between the DRTC client and existing SDK surfaces.

- `connect`: one-call helper, `connect_drtc(api_key=..., ...)` -> opened DRTCClient.
- `sdk_adapter`: wires DRTCClient into Interlatent.watch() / tick().
- `preflight`: backend for the `interlatent-preflight` console script —
  a connectivity check that opens a real DRTC session with synthetic
  observations and reports a latency verdict.
"""

from .connect import connect_drtc, DEFAULT_DRTC_URL  # noqa: F401

