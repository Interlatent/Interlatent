"""Thin compatibility shim over :mod:`interlatent.storage.lerobot_rebuild`.

The actual rebuilder is parametrised by the
:class:`~interlatent.storage.lerobot_rebuild.StepSource` protocol and
lives in ``interlatent.storage.lerobot_rebuild`` (canonical home in
the engine package, duplicated into the SDK wheel). This module wraps
it with the legacy ``build_from_staging(db_path=, media=, ...)``
entrypoint that :class:`interlatent.Interlatent.upload` still uses,
so the call site in ``_client.py`` does not need to know about the
generalised :class:`StepSource` interface.

Activations are no longer written as dataset columns. If a caller
still passes ``record_activations=True``, the flag is accepted for
backward compatibility but ignored (with a one-time deprecation
warning at INFO level) — the SAE/latent interpretability path has
been retired and reroutes through a separate pipeline if it returns.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

from ._step_source import CollectionDBStepSource
from .storage.lerobot_rebuild import LeRobotRebuilder as _EngineRebuilder

_LOG = logging.getLogger(__name__)

# Re-exported so ``from interlatent._dataset import StepRow`` keeps
# resolving for any external code that wandered in here.
from .storage.lerobot_rebuild import StepRow, StepSource  # noqa: F401


class LeRobotRebuilder(_EngineRebuilder):
    """SDK-flavored rebuilder with the legacy SQLite+MediaBuffer entrypoint.

    :meth:`build_from_staging` adapts the SDK's staging cache (the
    transient SQLite + :class:`MediaBuffer`) into a
    :class:`CollectionDBStepSource` and hands it to the underlying
    engine rebuilder.

    The constructor signature is unchanged from the previous standalone
    implementation, so the call site in
    :meth:`interlatent.Interlatent.upload` is a no-op rename.
    """

    def build_from_staging(
        self,
        *,
        db_path: str,
        media,
    ) -> Tuple[Path, list[str]]:
        """Read the SDK staging cache and produce an on-disk LeRobot dataset.

        Returns ``(root, episode_uuids)``.
        """
        if not Path(db_path).exists():
            _LOG.warning("build_from_staging: db_path %s does not exist", db_path)
            return self.root, []

        source = CollectionDBStepSource(db_path, media)
        try:
            return self.build_from_source(source)
        finally:
            source.close()


__all__ = ["LeRobotRebuilder", "StepRow", "StepSource"]
