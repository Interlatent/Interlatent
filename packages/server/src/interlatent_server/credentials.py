"""The box's identity when it calls its coordinator.

Every coordinator call a box makes — the warmup-target fetch, status
reports, the gRPC auth probe — presents one key: the one the operator
supplied as ``INTERLATENT_API_KEY``, alongside the ``INTERLATENT_BOX_ID``
and ``INTERLATENT_COORDINATOR`` that say who is calling and where. There
is one shape of box now (you run the coordinator, you run the box), so
there is one identity.

``interlatent-serve`` is what normally populates the three: it persists a
box id, registers via ``POST /api/v1/compute/boxes/register``, and puts
the key the coordinator handed back into the environment (a box-scoped
``ilbox_`` key when the coordinator minted one, otherwise the ``ilop_``
operator key it registered with). Either way the box just presents what
it holds — it never inspects the key, and neither does this module.

Everything downstream asks :func:`resolve` instead of reading env vars,
so no caller has to know which of the two it ended up with.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from .coordinator import normalize


@dataclass(frozen=True)
class BoxCredentials:
    box_id: str
    api_base: str
    api_key: str

    @property
    def api_root(self) -> str:
        """``https://host`` base with no trailing slash, no /api/v1."""
        return normalize(self.api_base)


def resolve() -> BoxCredentials | None:
    """The box's coordinator identity, or None (local dev / smoke tests —
    every coordinator interaction becomes a no-op, exactly as before)."""
    box_id = os.environ.get("INTERLATENT_BOX_ID", "").strip()
    api_base = (
        os.environ.get("INTERLATENT_COORDINATOR", "").strip()
        or os.environ.get("INTERLATENT_API_BASE", "").strip()
    )
    if not (box_id and api_base):
        return None
    api_key = os.environ.get("INTERLATENT_API_KEY", "").strip()
    if not api_key:
        return None
    return BoxCredentials(box_id, api_base, api_key)
