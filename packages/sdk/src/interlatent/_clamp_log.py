"""Dedicated, rate-limited logger for per-tick delta-clamp warnings.

The execution-safety delta clamp can fire on *every* control tick (30-50 Hz)
when a policy or teleop stream repeatedly overshoots ``max_step``. Emitting a
WARNING each time drowns the periodic latency/throughput reports, so every
clamp warning in the codebase is funnelled through here instead of a
per-module logger. That buys two independent knobs:

  * **Isolation** — all clamp warnings share the ``interlatent.clamp`` logger,
    so ``interlatent-node run --quiet-clamp`` can raise *just* that logger to
    ERROR without hiding any other warning (or the INFO latency reports, which
    live on a different logger). A blanket ``--log-level`` can't do this: the
    clamp lines are WARNING and the latency lines are INFO, so no single level
    keeps one while dropping the other.

  * **Rate-limiting** — a continuously-clamping joint would still spam even
    when enabled, so :func:`warn_clamp` logs the first few per source verbatim
    and then only one in every ``_EVERY`` after that. Nothing is silenced
    permanently; the throttle just thins a sustained flood.

The clamp itself always runs — this module only governs how often it is
*logged*, never whether the safety guard fires.
"""
from __future__ import annotations

import logging
import threading

#: Loggers named this (or below it) carry only delta-clamp warnings.
LOGGER_NAME = "interlatent.clamp"
_LOG = logging.getLogger(LOGGER_NAME)

# Log the first _HEAD occurrences per source verbatim, then one in every
# _EVERY. Tuned so a joint clamping every tick at 50 Hz costs ~1 line/sec
# instead of 50 while still surfacing the very first hits immediately.
_HEAD = 5
_EVERY = 50

_counts: dict[str, int] = {}
_lock = threading.Lock()


def warn_clamp(source: str, msg: str, *args: object) -> None:
    """Emit a rate-limited clamp WARNING on the ``interlatent.clamp`` logger.

    ``source`` is a short, stable key that groups related clamps for throttling
    (e.g. ``"control:policy"``, ``"yam:left"``); each key is counted and thinned
    independently. ``msg``/``args`` are printf-style exactly as passed to
    :meth:`logging.Logger.warning`. Skipped occurrences are folded into a
    ``(source total: N)`` suffix so a throttled line still conveys scale.
    """
    if not _LOG.isEnabledFor(logging.WARNING):
        return  # short-circuit when silenced, e.g. --quiet-clamp
    with _lock:
        n = _counts.get(source, 0) + 1
        _counts[source] = n
    if n <= _HEAD or n % _EVERY == 0:
        _LOG.warning(msg + " (%s total: %d)", *args, source, n)


def reset_counts() -> None:
    """Clear the per-source throttle counters (test/session-boundary helper)."""
    with _lock:
        _counts.clear()
