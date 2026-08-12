"""Self-hosted control plane for the Interlatent SDK.

A **coordinator** assigns work: it pairs nodes, tracks GPU boxes, brokers
inference and teleop sessions, and tells each node what to converge to. The
contract it implements is :mod:`interlatent.coordinator.protocol`, and
``interlatent up`` runs one on your own machine.

The SDK talks to *a* coordinator and never asks which one — see
``docs/adr/0038-coordinator-protocol-one-control-plane.md``.

Nothing heavy is imported here: ``protocol`` is stdlib-only, and the server
implementation is imported lazily by the CLI so that merely importing
``interlatent`` never pulls in an HTTP server.
"""

from __future__ import annotations

from .state import Coordinator, PolicyChangeError
from .protocol import (
    API_PREFIX,
    MANDATORY,
    PROTOCOL_VERSION,
    ROUTES,
    Route,
)

__all__ = [
    "API_PREFIX",
    "Coordinator",
    "PolicyChangeError",
    "MANDATORY",
    "PROTOCOL_VERSION",
    "ROUTES",
    "Route",
]
