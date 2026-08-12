"""Box identity resolution — one code path for hosted and self-hosted boxes.

Every backend call a box makes (warmup-target fetch, status reports, the
gRPC auth probe) authenticates as one of two identities:

- **Hosted / provisioned box**: the dashboard injects ``INTERLATENT_BOX_ID``,
  ``INTERLATENT_COORDINATOR`` and the shared system secret
  ``INTERLATENT_ADMIN_KEY`` at provision time.
- **Self-hosted (BYO) box**: the operator supplies their own key
  (``INTERLATENT_API_KEY``) — an ``ilat_`` user key against the hosted
  dashboard, or the ``ilop_`` operator key a self-hosted coordinator
  minted. ``interlatent-serve`` persists the box id and registers via
  ``POST /api/v1/compute/boxes/register``.

The admin key wins when both are present (a dashboard-provisioned box that
also happens to have a user key in the environment must keep its system
identity). Everything downstream asks :func:`resolve` instead of reading
env vars, so the two deployment shapes never fork the code path.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from .coordinator import normalize


#: Which kind of principal ``BoxCredentials.api_key`` is. Determines what a
#: coordinator will let the box do, and (via ``is_system``) whether the box may
#: report a graceful ``stopped`` — a hosted box's lifecycle is the dashboard's
#: to narrate, an operator's box narrates its own.
KIND_SYSTEM = "system"     # the dashboard's shared admin secret
KIND_OPERATOR = "operator"  # ilop_ — minted by a self-hosted coordinator
KIND_USER = "user"          # ilat_ — a dashboard user key


@dataclass(frozen=True)
class BoxCredentials:
    box_id: str
    api_base: str
    api_key: str
    kind: str = KIND_USER

    @property
    def is_system(self) -> bool:
        """Kept as a property rather than a field: call sites like
        ``box_status.report_status`` branch on it and predate ``kind``."""
        return self.kind == KIND_SYSTEM

    @property
    def api_root(self) -> str:
        """``https://host`` base with no trailing slash, no /api/v1."""
        return normalize(self.api_base)


def resolve() -> BoxCredentials | None:
    """The box's backend identity, or None (local dev / smoke tests —
    every backend interaction becomes a no-op, exactly as before)."""
    box_id = os.environ.get("INTERLATENT_BOX_ID", "").strip()
    api_base = (
        os.environ.get("INTERLATENT_COORDINATOR", "").strip()
        or os.environ.get("INTERLATENT_API_BASE", "").strip()
    )
    if not (box_id and api_base):
        return None
    admin_key = os.environ.get("INTERLATENT_ADMIN_KEY", "").strip()
    if admin_key:
        return BoxCredentials(box_id, api_base, admin_key, kind=KIND_SYSTEM)
    user_key = os.environ.get("INTERLATENT_API_KEY", "").strip()
    if user_key:
        kind = KIND_OPERATOR if user_key.startswith("ilop_") else KIND_USER
        return BoxCredentials(box_id, api_base, user_key, kind=kind)
    return None
