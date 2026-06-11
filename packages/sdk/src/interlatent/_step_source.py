"""Adapter that exposes the SDK's staging cache as a :class:`StepSource`.

The SDK collects to a transient SQLite at ``./interlatent_*.db`` (per-step
``_ctx`` rows written by :class:`interlatent._watcher.Watcher`) and a
:class:`MediaBuffer` directory of JPEG frames. At ``upload()`` time
:class:`interlatent.storage.lerobot_rebuild.LeRobotRebuilder` reads
everything through this adapter and emits a LeRobot v3.0 dataset.

Activations, if they were ever staged, are silently ignored — the
rebuilder no longer materializes them as dataset columns.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple

from .storage.lerobot_rebuild import StepRow

_LOG = logging.getLogger(__name__)

# Sentinel layer name written by :class:`interlatent._watcher.Watcher`
# for the per-step context row. Mirrors the constant in ``_watcher.py``
# — kept duplicated here to avoid a tight import cycle.
_CTX_LAYER = "_ctx"


class CollectionDBStepSource:
    """Reads per-step rows + frames from the SDK staging cache.

    The SDK constructor opens (and owns) the SQLite connection for the
    lifetime of the rebuild call; :meth:`close` releases it. The
    :class:`MediaBuffer` is borrowed — the caller still owns its
    cleanup.

    Episode order is the first-appearance order of ``episode_id`` in
    the staged ``_ctx`` rows, which equals the order the watcher wrote
    them. This matches the previous rebuilder's behavior.
    """

    def __init__(self, db_path: str | Path, media) -> None:
        self._db_path = str(db_path)
        self._media = media
        # Drain pending media writes once up-front so the directory
        # listing is complete before the rebuilder probes shape.
        if media is not None:
            media.flush()

        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row

        # Cache: episode order + per-episode row lists. The staging
        # cache is small (one user session), so reading the whole thing
        # into memory once is simpler and faster than streaming.
        self._episode_ids: list[str] = []
        self._rows_by_episode: dict[str, list[StepRow]] = {}
        self._load()

    # ------------------------------------------------------------------
    # StepSource protocol
    # ------------------------------------------------------------------

    def episode_ids(self) -> list[str]:
        return list(self._episode_ids)

    def iter_steps(self, episode_id: str) -> Iterable[StepRow]:
        return iter(self._rows_by_episode.get(episode_id, ()))

    def cameras_for_episode(self, episode_id: str) -> list[Optional[str]]:
        if self._media is None:
            return []
        return list(self._media.cameras_for_episode(episode_id))

    def iter_frames(
        self, episode_id: str
    ) -> Iterator[Tuple[int, Optional[str], Path]]:
        if self._media is None:
            return iter(())
        return iter(self._media.iter_episode_frames(episode_id))

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Materialize all ``_ctx`` rows into :class:`StepRow`s.

        Only the ``_ctx`` layer carries the per-step context the rebuilder
        needs; real activation rows (if any) are ignored here.
        """
        rows = self._conn.execute(
            """
            SELECT episode_id, step, context
            FROM activations
            WHERE layer = ?
            ORDER BY episode_id, step
            """,
            (_CTX_LAYER,),
        ).fetchall()

        seen_episodes: set[str] = set()
        for row in rows:
            ep_id = row["episode_id"]
            step = int(row["step"])
            try:
                ctx = json.loads(row["context"] or "{}")
            except (json.JSONDecodeError, TypeError):
                ctx = {}

            step_row = StepRow(
                episode_id=ep_id,
                step=step,
                observation=list(ctx.get("observation") or []),
                action=list(ctx.get("action") or []),
                reward=float(ctx.get("reward") or 0.0),
                done=bool(ctx.get("done") or False),
                truncated=bool(ctx.get("truncated") or False),
                metrics=dict(ctx.get("metrics") or {}),
                failure_type=ctx.get("failure_type") or None,
                control_source=ctx.get("control_source") or None,
            )
            self._rows_by_episode.setdefault(ep_id, []).append(step_row)
            if ep_id not in seen_episodes:
                seen_episodes.add(ep_id)
                self._episode_ids.append(ep_id)


__all__ = ["CollectionDBStepSource"]
