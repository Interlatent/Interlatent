"""Best-effort bus arbitration so a behavior run never silently fights the node.

When the node daemon is running an inference session on a robot it holds that robot's
bus. If a :class:`~interlatent.robot.Robot` opened the same bus at the same time, the
two would issue conflicting commands. Full daemon integration is out of scope (the
daemon does not yet publish a session lock we can read) — so this module fails *loud*
instead of corrupting a live session, using the two signals available on the client:

1. **Interlatent lockfile** — every ``Robot`` writes ``~/.interlatent/locks/<bus>.lock``
   with its PID while connected. A second ``Robot``/``behavior run`` on the same bus
   sees a live PID and refuses.
2. **OS serial lock** — many serial stacks drop a ``LCK..<device>`` lock in
   ``/run/lock`` or ``/var/lock``. If one exists for the target port with a live PID,
   we treat the bus as busy.

Either check can be overridden with ``force=True`` (documented as dangerous — it can
corrupt a live session). Neither is a hard guarantee; they catch the common case
(another Interlatent client on the same machine) cleanly rather than surfacing an
opaque "device busy" from deep inside the driver.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Optional

from .schema import BehaviorError

_LOG = logging.getLogger("interlatent.behaviors.arbitration")

_LOCK_DIR = Path.home() / ".interlatent" / "locks"
_OS_LOCK_DIRS = (Path("/run/lock"), Path("/var/lock"))


class RobotBusyError(BehaviorError):
    """The robot's bus appears to be held by another process (e.g. the node daemon)."""


def _sanitize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", text).strip("_") or "robot"


def _bus_key(robot_kind: str, port: Optional[str], extra: Optional[dict]) -> str:
    """A stable filesystem-safe key identifying the physical bus."""
    if port:
        return _sanitize(port)
    extra = extra or {}
    channels = [str(extra[k]) for k in ("left_channel", "right_channel") if extra.get(k)]
    if channels:
        return _sanitize("+".join(channels))
    return _sanitize(robot_kind)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but owned by someone else
    except OSError:
        return False
    return True


def _os_serial_conflict(port: Optional[str]) -> Optional[str]:
    """Return a reason string if an OS serial lock for ``port`` is held by a live PID."""
    if not port:
        return None
    device = os.path.basename(port)
    for d in _OS_LOCK_DIRS:
        lock = d / f"LCK..{device}"
        try:
            if not lock.is_file():
                continue
            raw = lock.read_text(errors="ignore").strip().split()
            pid = int(raw[0]) if raw else 0
        except (OSError, ValueError):
            continue
        if _pid_alive(pid):
            return f"OS serial lock {lock} held by pid {pid}"
    return None


class BusLock:
    """The acquired lock; call :meth:`release` (idempotent) to drop it."""

    def __init__(self, path: Optional[Path]) -> None:
        self._path = path
        self._released = False

    def release(self) -> None:
        if self._released or self._path is None:
            return
        self._released = True
        try:
            # Only remove it if it is still ours.
            if self._path.is_file():
                raw = self._path.read_text(errors="ignore").strip()
                if raw.isdigit() and int(raw) == os.getpid():
                    self._path.unlink()
        except OSError:
            _LOG.debug("could not remove bus lock %s", self._path, exc_info=True)


def acquire_bus_lock(
    robot_kind: str,
    port: Optional[str] = None,
    extra: Optional[dict] = None,
    *,
    force: bool = False,
) -> BusLock:
    """Claim the bus for this process, or raise :class:`RobotBusyError`.

    ``force=True`` logs and overrides both checks (dangerous — may corrupt a live
    node session). Returns a :class:`BusLock` to release on ``close()``.
    """
    conflict = _os_serial_conflict(port)
    key = _bus_key(robot_kind, port, extra)
    lock_path = _LOCK_DIR / f"{key}.lock"

    if conflict is None and lock_path.is_file():
        try:
            raw = lock_path.read_text(errors="ignore").strip()
            pid = int(raw) if raw.isdigit() else 0
        except OSError:
            pid = 0
        if pid and pid != os.getpid() and _pid_alive(pid):
            conflict = f"Interlatent lock {lock_path} held by pid {pid}"

    if conflict is not None:
        msg = (
            f"robot bus {key!r} is already in use ({conflict}). Another Interlatent "
            "process — likely a running node daemon or a second Robot — is driving it. "
            "Pass force=True to override (this can corrupt a live inference session)."
        )
        if not force:
            raise RobotBusyError(msg)
        _LOG.warning("force=True: overriding bus arbitration — %s", conflict)

    try:
        _LOCK_DIR.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(str(os.getpid()), encoding="utf-8")
    except OSError:
        # If we cannot write the lock we still allow the move (a read-only home dir
        # should not block a manual behavior); just skip cooperative locking.
        _LOG.debug("could not write bus lock %s; continuing unlocked", lock_path, exc_info=True)
        return BusLock(None)
    return BusLock(lock_path)


__all__ = ["acquire_bus_lock", "BusLock", "RobotBusyError"]
