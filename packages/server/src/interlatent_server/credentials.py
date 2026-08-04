"""Box identity resolution — one code path for hosted and self-hosted boxes.

Every backend call a box makes (warmup-target fetch, status reports, the
gRPC auth probe) authenticates as one of two identities:

- **Hosted / provisioned box**: the dashboard injects ``INTERLATENT_BOX_ID``,
  ``INTERLATENT_API_BASE`` and the shared system secret
  ``INTERLATENT_ADMIN_KEY`` at provision time.
- **Self-hosted (BYO) box**: the operator supplies their own user API key
  (``INTERLATENT_API_KEY``, an ``ilat_...`` key); ``interlatent-serve``
  mints/persists the box id and registers it via
  ``POST /api/v1/compute/boxes/register``.

The admin key wins when both are present (a dashboard-provisioned box that
also happens to have a user key in the environment must keep its system
identity). Everything downstream asks :func:`resolve` instead of reading
env vars, so the two deployment shapes never fork the code path.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class BoxCredentials:
    box_id: str
    api_base: str
    api_key: str
    # True when api_key is the shared system/admin secret (hosted box);
    # False when it is the owner's ilat_ user key (self-hosted box).
    is_system: bool

    @property
    def api_root(self) -> str:
        """``https://host`` base with no trailing slash, no /api/v1."""
        return self.api_base.rstrip("/")


def resolve() -> BoxCredentials | None:
    """The box's backend identity, or None (local dev / smoke tests —
    every backend interaction becomes a no-op, exactly as before)."""
    box_id = os.environ.get("INTERLATENT_BOX_ID", "").strip()
    api_base = os.environ.get("INTERLATENT_API_BASE", "").strip()
    if not (box_id and api_base):
        return None
    admin_key = os.environ.get("INTERLATENT_ADMIN_KEY", "").strip()
    if admin_key:
        return BoxCredentials(box_id, api_base, admin_key, is_system=True)
    user_key = os.environ.get("INTERLATENT_API_KEY", "").strip()
    if user_key:
        return BoxCredentials(box_id, api_base, user_key, is_system=False)
    return None
