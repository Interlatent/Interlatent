"""Write-through disk spool for the RecordTick uplink (ADR 0023).

Every captured tick is journaled to local disk *before* it is offered to
the network; the sender drains the spool at whatever rate the link
allows and a tick is deleted only after the server's honest ack
(``RecordTicksResponse.accepted`` prefix — see the engine's transport).
Recording therefore survives link failure and node-process crash: an
un-acked tick is still on disk. What it does NOT survive is a
recorder-host crash (staged data on the pod is lost with the pod) or
node power loss for the last few unsynced writes — we deliberately skip
fsync-per-tick to spare SD cards; ``.part``-then-rename keeps the
journal free of torn files.

Layout, one directory per session under the spool root::

    <root>/<session_id>/
      meta.json            # session_id, server_address, created_at
      00000042.tick        # one serialized RecordTickRequest per file

Spool-full policy is HARD-STOP with hysteresis (never drop-oldest):
``blocked`` flips True when pending bytes cross the cap (or the disk's
free-space floor is hit) and back to False only once the backlog drains
below ``_RESUME_FRACTION`` of the cap — the capture path refuses ticks
while blocked, loudly, and auto-resumes.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_TICK_SUFFIX = ".tick"
_SEQ_WIDTH = 8

# Hysteresis: hard-stop above the cap, auto-resume below this fraction.
_RESUME_FRACTION = 0.8


def spool_root() -> Path:
    """Spool root: ``INTERLATENT_SPOOL_DIR`` or ``~/.interlatent/spool``."""
    env = os.environ.get("INTERLATENT_SPOOL_DIR", "").strip()
    if env:
        return Path(env).expanduser()
    return Path.home() / ".interlatent" / "spool"


def _max_bytes() -> int:
    """Spool cap from INTERLATENT_SPOOL_MAX_MB (default 6 GiB ≈ 10 min of
    a 3×720p30 rig — the ADR 0023 design ceiling)."""
    try:
        mb = float(os.environ.get("INTERLATENT_SPOOL_MAX_MB", "") or 6144.0)
    except (TypeError, ValueError):
        mb = 6144.0
    # Floor of 1 KiB guards against zero/negative misconfiguration while
    # still allowing tiny caps (tests, constrained devices).
    return int(max(1024.0, mb * 1024 * 1024))


def _min_free_bytes() -> int:
    """Free-disk floor from INTERLATENT_SPOOL_MIN_FREE_MB (default 2 GiB)."""
    try:
        mb = float(os.environ.get("INTERLATENT_SPOOL_MIN_FREE_MB", "") or 2048.0)
    except (TypeError, ValueError):
        mb = 2048.0
    return int(max(0.0, mb) * 1024 * 1024)


class TickSpool:
    """Per-session append-only tick journal with delete-after-ack.

    Thread contract: ``append`` is called from the control thread,
    ``peek_batch``/``ack`` from the sender thread; all index state is
    behind one lock. Disk I/O is done outside no lock-sensitive hot
    path — a spool append is one small file write.
    """

    def __init__(
        self,
        session_id: str,
        *,
        server_address: str = "",
        root: Optional[Path] = None,
    ) -> None:
        self.session_id = str(session_id)
        self.dir = (root or spool_root()) / self.session_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._max_bytes = _max_bytes()
        self._min_free = _min_free_bytes()
        self._blocked = False
        self._blocked_logged = False

        # Rebuild the index from disk (same-session resume after a crash:
        # whatever was journaled but never acked is still pending).
        pending: dict[int, int] = {}
        for p in self.dir.glob(f"*{_TICK_SUFFIX}"):
            try:
                pending[int(p.stem)] = p.stat().st_size
            except (ValueError, OSError):
                continue
        for p in self.dir.glob("*.part"):  # torn writes from a crash
            try:
                p.unlink()
            except OSError:
                pass
        self._pending = pending                      # seq -> size
        self._order = sorted(pending)                # ascending seqs
        self._bytes = sum(pending.values())
        self._next_seq = (self._order[-1] + 1) if self._order else 0

        meta = self.dir / "meta.json"
        if not meta.exists():
            try:
                meta.write_text(json.dumps({
                    "session_id": self.session_id,
                    "server_address": server_address,
                    "created_at": time.time(),
                }))
            except OSError:
                log.warning("spool meta.json write failed", exc_info=True)

    # -- capture side ---------------------------------------------------

    @property
    def blocked(self) -> bool:
        """Hard-stop state, with hysteresis (ADR 0023)."""
        with self._lock:
            over_cap = self._bytes >= self._max_bytes
            if not over_cap and self._blocked:
                # Only resume once drained below the hysteresis line.
                over_cap = self._bytes >= _RESUME_FRACTION * self._max_bytes
            low_disk = False
            if not over_cap:
                try:
                    low_disk = shutil.disk_usage(self.dir).free < self._min_free
                except OSError:
                    low_disk = False
            now_blocked = over_cap or low_disk
            if now_blocked and not self._blocked_logged:
                log.error(
                    "tick spool %s FULL (%.0f MB pending, cap %.0f MB, "
                    "free-floor %.0f MB) — capture hard-stopped; will "
                    "auto-resume once the uplink drains the backlog",
                    self.session_id, self._bytes / 1e6,
                    self._max_bytes / 1e6, self._min_free / 1e6,
                )
                self._blocked_logged = True
            if not now_blocked and self._blocked:
                log.warning(
                    "tick spool %s drained below resume threshold — "
                    "capture resumed", self.session_id,
                )
                self._blocked_logged = False
            self._blocked = now_blocked
            return now_blocked

    def append(self, data: bytes) -> Optional[int]:
        """Journal one serialized tick. Returns its seq, or None if the
        write failed (disk error) — the caller must treat None as a
        refused (not recorded) tick."""
        with self._lock:
            seq = self._next_seq
            self._next_seq += 1
        name = f"{seq:0{_SEQ_WIDTH}d}{_TICK_SUFFIX}"
        part = self.dir / f"{name}.part"
        final = self.dir / name
        try:
            part.write_bytes(data)
            os.replace(part, final)
        except OSError:
            log.error("tick spool append failed (disk?)", exc_info=True)
            try:
                part.unlink(missing_ok=True)
            except OSError:
                pass
            return None
        with self._lock:
            self._pending[seq] = len(data)
            self._order.append(seq)
            self._bytes += len(data)
        return seq

    # -- sender side ----------------------------------------------------

    def peek_batch(
        self, max_ticks: int, max_bytes: int,
    ) -> list[tuple[int, bytes]]:
        """Oldest unacked ticks, capped by count and cumulative bytes.
        Always returns at least one tick when any is pending (a lone
        oversized tick still goes out solo)."""
        with self._lock:
            seqs = list(self._order[: max(1, int(max_ticks))])
        out: list[tuple[int, bytes]] = []
        total = 0
        for seq in seqs:
            path = self.dir / f"{seq:0{_SEQ_WIDTH}d}{_TICK_SUFFIX}"
            try:
                data = path.read_bytes()
            except OSError:
                # File vanished underneath us — drop it from the index
                # rather than wedging the sender on it forever.
                log.warning("spooled tick %d unreadable; skipping", seq)
                self._forget(seq)
                continue
            if out and total + len(data) > max_bytes:
                break
            out.append((seq, data))
            total += len(data)
        return out

    def ack(self, through_seq: int) -> None:
        """Delete every pending tick with seq <= through_seq (the server
        durably accepted them — RecordTicks prefix semantics)."""
        with self._lock:
            acked = [s for s in self._order if s <= through_seq]
        for seq in acked:
            path = self.dir / f"{seq:0{_SEQ_WIDTH}d}{_TICK_SUFFIX}"
            try:
                path.unlink(missing_ok=True)
            except OSError:
                log.debug("spool ack unlink failed for %d", seq, exc_info=True)
            self._forget(seq)

    def _forget(self, seq: int) -> None:
        with self._lock:
            size = self._pending.pop(seq, None)
            if size is not None:
                self._bytes -= size
                try:
                    self._order.remove(seq)
                except ValueError:
                    pass

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._order)

    @property
    def pending_bytes(self) -> int:
        with self._lock:
            return self._bytes

    def dispose(self) -> None:
        """Remove the whole session dir (fully drained at close)."""
        try:
            shutil.rmtree(self.dir, ignore_errors=True)
        except OSError:
            pass


# ---------------------------------------------------------------------
# Orphan handling (daemon startup)
# ---------------------------------------------------------------------


def orphan_sessions(root: Optional[Path] = None) -> list[dict]:
    """Spool dirs left behind by a crashed node process, oldest first.
    Each entry: {session_id, dir, meta, pending_count, pending_bytes}."""
    base = root or spool_root()
    if not base.is_dir():
        return []
    out = []
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        ticks = list(d.glob(f"*{_TICK_SUFFIX}"))
        meta: dict = {}
        try:
            meta = json.loads((d / "meta.json").read_text())
        except (OSError, ValueError):
            pass
        out.append({
            "session_id": d.name,
            "dir": d,
            "meta": meta,
            "pending_count": len(ticks),
            "pending_bytes": sum(p.stat().st_size for p in ticks),
        })
    return out


def disk_pressure(root: Optional[Path] = None) -> Optional[str]:
    """A human-readable reason when starting a NEW recording session would
    immediately hard-stop (free disk under the floor, or the accumulated
    spool backlog already at the cap) — or None when healthy. The daemon
    checks this before accepting an assignment (ADR 0023)."""
    base = root or spool_root()
    try:
        base.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(base).free
    except OSError:
        return None
    floor = _min_free_bytes()
    if free < floor:
        return (
            f"free disk {free / 1e9:.1f} GB is below the spool floor "
            f"{floor / 1e9:.1f} GB"
        )
    total = sum(o["pending_bytes"] for o in orphan_sessions(base))
    cap = _max_bytes()
    if total >= cap:
        return (
            f"spool backlog {total / 1e6:.0f} MB is at the cap "
            f"{cap / 1e6:.0f} MB (unsent recordings pending)"
        )
    return None


def gc_orphans(
    root: Optional[Path] = None, max_age_s: float = 7 * 24 * 3600.0,
) -> int:
    """Delete orphan spools older than ``max_age_s`` (bounded retention —
    never silent: each removal logs what was lost). Returns count removed."""
    removed = 0
    for orphan in orphan_sessions(root):
        created = float(orphan["meta"].get("created_at") or 0.0)
        if created and (time.time() - created) < max_age_s:
            continue
        if orphan["pending_count"]:
            log.warning(
                "GC-ing orphan tick spool %s: %d unsent ticks (%.0f MB) "
                "older than %.0fh — this data is now lost",
                orphan["session_id"], orphan["pending_count"],
                orphan["pending_bytes"] / 1e6, max_age_s / 3600.0,
            )
        shutil.rmtree(orphan["dir"], ignore_errors=True)
        removed += 1
    return removed
