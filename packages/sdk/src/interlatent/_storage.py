"""Simplified SQLite backend — write path only for SDK collection."""
from __future__ import annotations

import json
import pathlib
import sqlite3
from typing import Sequence

from ._schema import ActivationEvent


def _dict_factory(cursor, row):
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}


class CollectionSQLiteBackend:
    """SQLite write-only driver for activation collection."""

    def __init__(self, path: str) -> None:
        self._path = pathlib.Path(path).expanduser().resolve()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = _dict_factory
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS activations (
              episode_id  TEXT,
              step        INTEGER,
              layer       TEXT,
              tensor      TEXT,
              context     TEXT,
              PRIMARY KEY (episode_id, step)
            ) WITHOUT ROWID;
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_activations_layer_step
            ON activations(layer, step);
            """
        )
        self._conn.commit()

    def write_events(self, events: Sequence[ActivationEvent]) -> None:
        """Batch INSERT OR REPLACE activation events (full-tensor fast path)."""
        if not events:
            return

        def _json_fallback(obj):
            if hasattr(obj, "item"):
                try:
                    return obj.item()
                except Exception:
                    pass
            return str(obj)

        # Group by (episode_id, step, layer) to merge batch samples
        batches: dict[tuple[str, int, str], dict] = {}
        for ev in events:
            key = (ev.episode_id, ev.step, ev.layer)
            batch = batches.setdefault(
                key,
                {"context": ev.context, "full_tensor": None},
            )
            if ev.tensor and len(ev.tensor) > 1:
                batch["full_tensor"] = ev.tensor
            if not batch["context"]:
                batch["context"] = ev.context

        rows = []
        for (episode_id, step, layer), batch in batches.items():
            tensor = batch.get("full_tensor") or []
            tensor = [float(v) for v in tensor]
            context = batch["context"]
            if isinstance(context, str):
                try:
                    parsed = json.loads(context)
                    context = parsed if isinstance(parsed, dict) else {}
                except Exception:
                    context = {}

            tensor_json = json.dumps(tensor, default=_json_fallback)
            context_json = json.dumps(context or {}, default=_json_fallback)
            rows.append((str(episode_id), int(step), str(layer), tensor_json, context_json))

        cur = self._conn.cursor()
        cur.executemany(
            """
            INSERT OR REPLACE INTO activations
            (episode_id, step, layer, tensor, context)
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )
        self._conn.commit()

    def write_event(self, ev: ActivationEvent) -> None:
        self.write_events([ev])

    def update_step_contexts(
        self,
        *,
        contexts: dict[int, dict],
    ) -> None:
        if not contexts:
            return
        cur = self._conn.cursor()
        rows = [
            (json.dumps(ctx or {}), (ctx or {}).get("episode_id", ""), int(step))
            for step, ctx in contexts.items()
        ]
        cur.executemany(
            """
            UPDATE activations
            SET context = ?
            WHERE episode_id = ? AND step = ?
            """,
            rows,
        )
        self._conn.commit()

    def flush(self) -> None:
        self._conn.commit()
        self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")

    def close(self) -> None:
        self._conn.commit()
        self._conn.close()
