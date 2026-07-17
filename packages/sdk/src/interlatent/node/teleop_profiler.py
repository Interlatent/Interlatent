"""Per-session control-loop profiler for the node (Raspberry Pi).

Aggregates one row per second of :func:`control.lerobot_control_loop`:
total per-tick loop time, time-to-command (loop start -> ``robot.
send_action`` returning), capture/recording overhead (JPEG encode +
queueing in ``_capture_tick``), and — on teleop ticks — how old the
executed frame was (the same window the loop already logs every 5s as
"teleop exec latency"). Answers "why did the arm get slow/erratic
partway through the session" from the node's own point of view.

Two output channels, in priority order:

  1. **Log line** (``_LOG.info``, tag ``TELEOP-PROFILE``) — the primary,
     load-bearing channel. Uses the exact same ``logging`` call the
     control loop's pre-existing "teleop exec latency" line already
     uses, so it reaches wherever that line reaches (journalctl,
     console, whatever log aggregation is already configured) with no
     new moving parts. This is emitted unconditionally once a second,
     independent of whether the CSV file below is working.
  2. **CSV file** (``~/.interlatent/teleop_profiles/``) — a convenience
     copy for opening in a spreadsheet later. Best-effort only: if the
     directory can't be created, the file can't be opened, or a write
     fails partway through a session (disk full, permissions changing
     mid-session, SD card hiccup), the CSV is silently dropped and
     logging keeps going uninterrupted. A missing/empty CSV file is
     therefore NOT evidence that profiling isn't running — check the
     log line first.

Reliability is the whole point of this module, not an afterthought:
inference/teleop must NEVER be interrupted by profiling, and the
primary channel must not depend on a local file write succeeding.

  * Every public method catches its own exceptions.
  * Off switch: ``INTERLATENT_NODE_PROFILE=0`` disables entirely (zero
    overhead — the constructor returns immediately, no log line either).
  * CSV output directory: ``~/.interlatent/teleop_profiles/`` by default
    (same base as ``~/.interlatent/node.toml`` — see node/cli.py),
    overridable via ``INTERLATENT_PROFILE_DIR``.

Not in scope: this profiles what the NODE can see (its own loop timing).
It does not know about the relay or the browser — see the browser-side
profiler (``site/src/lib/teleop/teleopProfiler.ts`` in interlatent-main)
for the operator-visible half of the same investigation.
"""
from __future__ import annotations

import csv
import logging
import os
import time
from pathlib import Path
from typing import Optional, TextIO

_LOG = logging.getLogger("interlatent.node.teleop_profile")

_ENV_ENABLE = "INTERLATENT_NODE_PROFILE"
_ENV_DIR = "INTERLATENT_PROFILE_DIR"


def _profile_enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


def _default_dir() -> Path:
    override = os.environ.get(_ENV_DIR, "").strip()
    if override:
        return Path(override).expanduser()
    return Path("~/.interlatent/teleop_profiles").expanduser()


def _safe_slug(s: str, max_len: int = 40) -> str:
    out = "".join(c if (c.isalnum() or c in "-_") else "_" for c in (s or "unknown"))
    return out[:max_len] or "unknown"


def _hostname() -> str:
    try:
        return os.uname().nodename  # type: ignore[attr-defined]
    except Exception:
        return os.environ.get("COMPUTERNAME", "unknown-host")


_CSV_COLUMNS = [
    "t_uptime_s",
    "ticks",
    "engaged_ticks",
    "teleop_ticks",
    "policy_ticks",
    "estop_ticks",
    "loop_dt_avg_ms",
    "loop_dt_max_ms",
    "cmd_dt_avg_ms",
    "cmd_dt_max_ms",
    "capture_dt_avg_ms",
    "capture_dt_max_ms",
    "frame_age_avg_ms",
    "frame_age_max_ms",
    "over_period_ticks",
]


class NodeTeleopProfiler:
    """One instance per :func:`control.lerobot_control_loop` call.

    Usage (see control.py):
        prof = NodeTeleopProfiler(session_id=..., robot_kind=..., fps=...)
        ... in the loop, after send_action / _capture_tick ...
        prof.record_tick(loop_dt_s=..., cmd_dt_s=..., capture_dt_s=...,
                          frame_age_ms=..., engaged=..., teleop_ok=...,
                          estop=..., over_period=...)
        ... in the finally block ...
        prof.close()
    """

    def __init__(
        self,
        *,
        session_id: str,
        robot_kind: str,
        fps: int,
        teleop_configured: bool,
        out_dir: Optional[Path] = None,
    ) -> None:
        self.enabled = _profile_enabled()
        self.path: Optional[Path] = None
        self._f: Optional[TextIO] = None
        self._writer = None
        self._closed = False
        self._t0 = time.perf_counter()
        self._reset_window()

        if not self.enabled:
            return

        # Log-only header line FIRST, unconditionally — this is the one
        # guaranteed signal that profiling is active for this session,
        # independent of whether the CSV file below can be opened.
        _LOG.info(
            "TELEOP-PROFILE start session_id=%s robot_kind=%s fps=%s "
            "teleop_configured=%s host=%s",
            session_id, robot_kind, fps, teleop_configured, _hostname(),
        )

        # CSV file: best-effort convenience copy. Any failure here is
        # logged once and otherwise ignored — record_tick/close below
        # never check self.enabled to decide whether to LOG, only
        # whether to WRITE THE FILE, so a failure here never silences
        # the log line.
        try:
            out = out_dir or _default_dir()
            out.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d-%H%M%S")
            fname = f"{ts}_{_safe_slug(session_id)}_node.csv"
            self.path = out / fname
            self._f = open(self.path, "w", newline="", encoding="utf-8")
            for line in (
                "# interlatent node teleop profile",
                f"# session_id: {session_id}",
                f"# robot_kind: {robot_kind}",
                f"# fps_configured: {fps}",
                f"# teleop_configured: {teleop_configured}",
                f"# host: {_hostname()}",
                f"# started_at: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
                "# 1 row = ~1s of the control loop; blank cells mean that "
                "metric had no samples in that second",
            ):
                self._f.write(line + "\n")
            self._writer = csv.writer(self._f)
            self._writer.writerow(_CSV_COLUMNS)
            self._f.flush()
            _LOG.info("teleop profiler: also writing CSV to %s", self.path)
        except Exception:
            _LOG.warning(
                "teleop profiler: could not open CSV file (dir=%s) — "
                "continuing with the TELEOP-PROFILE log line only, no "
                "file will be produced for this session",
                out_dir or _default_dir(), exc_info=True,
            )
            self._safe_close_file()

    # ---------- hot path ----------

    def record_tick(
        self,
        *,
        loop_dt_s: float,
        cmd_dt_s: Optional[float],
        capture_dt_s: Optional[float],
        frame_age_ms: Optional[float] = None,
        engaged: bool,
        teleop_ok: bool,
        estop: bool = False,
        over_period: bool,
    ) -> None:
        """Record one control-loop tick. Never raises."""
        if not self.enabled:
            return
        try:
            self._ticks += 1
            if engaged:
                self._engaged_ticks += 1
            if estop:
                self._estop_ticks += 1
            elif teleop_ok:
                self._teleop_ticks += 1
            else:
                self._policy_ticks += 1
            self._loop_dt_sum += loop_dt_s
            if loop_dt_s > self._loop_dt_max:
                self._loop_dt_max = loop_dt_s
            if cmd_dt_s is not None:
                self._cmd_n += 1
                self._cmd_dt_sum += cmd_dt_s
                if cmd_dt_s > self._cmd_dt_max:
                    self._cmd_dt_max = cmd_dt_s
            if capture_dt_s is not None:
                self._capture_n += 1
                self._capture_dt_sum += capture_dt_s
                if capture_dt_s > self._capture_dt_max:
                    self._capture_dt_max = capture_dt_s
            if frame_age_ms is not None:
                self._frame_age_n += 1
                self._frame_age_sum += frame_age_ms
                if frame_age_ms > self._frame_age_max:
                    self._frame_age_max = frame_age_ms
            if over_period:
                self._over_period += 1

            now = time.perf_counter()
            if now - self._window_start >= 1.0:
                self._flush_window(now)
        except Exception:
            # The counting/bookkeeping above is pure arithmetic and
            # should never raise; if it somehow does, disable rather
            # than risk repeating the failure on every future tick.
            _LOG.warning(
                "teleop profiler: record_tick failed; disabling for the "
                "rest of this session (control loop unaffected)",
                exc_info=True,
            )
            self.enabled = False
            self._safe_close_file()

    def close(self) -> None:
        """Flush the final partial window, log a close line, close the
        file. Never raises."""
        if self._closed:
            return
        self._closed = True
        if not self.enabled:
            return
        try:
            self._flush_window(time.perf_counter())
        except Exception:
            pass
        _LOG.info(
            "TELEOP-PROFILE end path=%s", self.path if self._f or self.path else "(log-only, no file)",
        )
        self._safe_close_file()

    # ---------- internals ----------

    def _reset_window(self) -> None:
        self._window_start = time.perf_counter()
        self._ticks = 0
        self._engaged_ticks = 0
        self._teleop_ticks = 0
        self._policy_ticks = 0
        self._estop_ticks = 0
        self._loop_dt_sum = 0.0
        self._loop_dt_max = 0.0
        self._cmd_n = 0
        self._cmd_dt_sum = 0.0
        self._cmd_dt_max = 0.0
        self._capture_n = 0
        self._capture_dt_sum = 0.0
        self._capture_dt_max = 0.0
        self._frame_age_n = 0
        self._frame_age_sum = 0.0
        self._frame_age_max = 0.0
        self._over_period = 0

    def _row_values(self, now: float) -> list:
        cmd_avg = round((self._cmd_dt_sum / self._cmd_n) * 1000, 2) if self._cmd_n else None
        cmd_max = round(self._cmd_dt_max * 1000, 2) if self._cmd_n else None
        cap_avg = round((self._capture_dt_sum / self._capture_n) * 1000, 2) if self._capture_n else None
        cap_max = round(self._capture_dt_max * 1000, 2) if self._capture_n else None
        age_avg = round(self._frame_age_sum / self._frame_age_n, 1) if self._frame_age_n else None
        age_max = round(self._frame_age_max, 1) if self._frame_age_n else None
        return [
            round(now - self._t0, 1),
            self._ticks, self._engaged_ticks, self._teleop_ticks,
            self._policy_ticks, self._estop_ticks,
            round((self._loop_dt_sum / self._ticks) * 1000, 2),
            round(self._loop_dt_max * 1000, 2),
            cmd_avg, cmd_max, cap_avg, cap_max, age_avg, age_max,
            self._over_period,
        ]

    def _flush_window(self, now: float) -> None:
        if self._ticks == 0:
            # Nothing happened this window (e.g. between sessions) — skip
            # rather than log/write an all-blank row.
            self._window_start = now
            return
        values = self._row_values(now)

        # 1. LOG — primary channel, always attempted, independent of the
        #    CSV file's state.
        try:
            (
                t_uptime, ticks, eng, tel, pol, estop, loop_avg, loop_max,
                cmd_avg, cmd_max, cap_avg, cap_max, age_avg, age_max, over,
            ) = values
            _LOG.info(
                "TELEOP-PROFILE t=%ss ticks=%d(eng=%d tel=%d pol=%d estop=%d) "
                "loop_ms=%.2f/%.2f cmd_ms=%s/%s capture_ms=%s/%s "
                "frame_age_ms=%s/%s over_period=%d",
                t_uptime, ticks, eng, tel, pol, estop, loop_avg, loop_max,
                cmd_avg, cmd_max, cap_avg, cap_max, age_avg, age_max, over,
            )
        except Exception:
            pass  # logging itself must never break the loop either

        # 2. CSV — best-effort secondary copy. A failure here stops
        #    future file writes but leaves logging (and the profiler as
        #    a whole) running.
        if self._f is not None:
            try:
                self._writer.writerow(values)
                self._f.flush()
            except Exception:
                _LOG.warning(
                    "teleop profiler: CSV write failed; the TELEOP-PROFILE "
                    "log line above (and future ones) is now the only "
                    "record for this session",
                    exc_info=True,
                )
                self._safe_close_file()

        self._reset_window()

    def _safe_close_file(self) -> None:
        try:
            if self._f is not None:
                self._f.close()
        except Exception:
            pass
        finally:
            self._f = None
            self._writer = None


__all__ = ["NodeTeleopProfiler"]
