/**
 * Per-session teleop profiler — browser side.
 *
 * Tags every session with a profile label + metadata header, aggregates
 * per-second rows (XR frame rate, pose-send pacing/backpressure, ee_state
 * receive gap, and the existing video-latency report from
 * `VRTeleopOverlay.recordVideoFrameStats`), and on session end triggers a
 * native browser download of the whole run as a CSV — no server change,
 * no extra command: the file lands in this device's Downloads the moment
 * you leave VR or close the overlay.
 *
 * Column groups (see `snapshot()`):
 *   frame.*  — WebXR rAF pacing. Falling fps here = the Quest browser's
 *              main thread is the bottleneck (JSON parsing, JPEG decode
 *              dispatch, texture uploads all share this thread).
 *   pose.*   — outbound control loop. `sent`/`skippedPace`/`skippedBuffer`
 *              per second, plus the max `ws.bufferedAmount` observed —
 *              growing `skippedBuffer` / `bufferedBytesMax` means the
 *              relay (or the network) isn't draining the socket, so
 *              poses are going stale before they're even sent.
 *   ee.*     — inbound robot-state gap. Growing `gapMaxMs` means ee_state
 *              updates from the pod are arriving late — the arm will feel
 *              laggy/erratic even though the browser's own loop is fine.
 *   video.*  — the pod's own instrumentation (already existed): fps/cam,
 *              lag decomposed into node→pod vs pod→browser legs, decode
 *              time, dropped frames.
 *
 * A run that starts fast and decays should show ONE of these groups
 * degrading first — that's the hop to chase.
 */

export type ProfilerVideoReport = {
  windowS: number;
  frames: number;
  fpsPerCam: number;
  lagAvgMs: number;
  lagMaxMs: number;
  podLagAvgMs: number | null;
  nodeLagAvgMs: number | null;
  bytesAvgKb: number;
  decodeAvgMs: number;
  decodeMaxMs: number;
  dropped: number;
} | null;

type Row = {
  t_uptime_s: number;
  frame_fps: number | null;
  frame_dt_avg_ms: number | null;
  frame_dt_max_ms: number | null;
  pose_sent: number;
  pose_skipped_pace: number;
  pose_skipped_buffer: number;
  pose_buffered_bytes_max: number;
  ee_state_recv: number;
  ee_gap_avg_ms: number | null;
  ee_gap_max_ms: number | null;
  video_fps_per_cam: number | null;
  video_lag_avg_ms: number | null;
  video_lag_max_ms: number | null;
  video_node_lag_avg_ms: number | null;
  video_pod_lag_avg_ms: number | null;
  video_decode_avg_ms: number | null;
  video_decode_max_ms: number | null;
  video_dropped: number | null;
  video_bytes_avg_kb: number | null;
  heap_used_mb: number | null;
};

export type ProfilerMeta = {
  profileLabel: string;
  sessionId: string;
  startedAtIso: string;
  userAgent: string;
  bimanual: boolean;
  robotKind: string | null;
};

function fmtNum(v: number | null): string {
  return v == null ? '' : String(v);
}

/** Short random suffix so two sessions started the same second still get
 *  distinct filenames if the operator immediately re-enters VR. */
function randSuffix(): string {
  return Math.random().toString(36).slice(2, 7);
}

export class TeleopProfiler {
  private meta: ProfilerMeta;
  private t0 = performance.now();
  private rows: Row[] = [];

  // Per-window accumulators, reset on every tick().
  private frameCount = 0;
  private frameDtSum = 0;
  private frameDtMax = 0;
  private lastFrameAt: number | null = null;

  private poseSent = 0;
  private poseSkippedPace = 0;
  private poseSkippedBuffer = 0;
  private poseBufferedMax = 0;

  private eeStateCount = 0;
  private eeGapSum = 0;
  private eeGapMax = 0;
  private lastEeStateAt: number | null = null;

  constructor(meta: Omit<ProfilerMeta, 'profileLabel' | 'startedAtIso'>) {
    this.meta = {
      ...meta,
      profileLabel: `auto-${new Date().toISOString().replace(/[:.]/g, '-')}-${randSuffix()}`,
      startedAtIso: new Date().toISOString(),
    };
  }

  // ---------- recorders (called from the XR frame loop / WS handlers) ----------

  recordFrame(): void {
    const now = performance.now();
    if (this.lastFrameAt != null) {
      const dt = now - this.lastFrameAt;
      this.frameCount++;
      this.frameDtSum += dt;
      if (dt > this.frameDtMax) this.frameDtMax = dt;
    }
    this.lastFrameAt = now;
  }

  recordPoseSent(bufferedAmount: number): void {
    this.poseSent++;
    if (bufferedAmount > this.poseBufferedMax) this.poseBufferedMax = bufferedAmount;
  }

  recordPoseSkipped(reason: 'pace' | 'buffer'): void {
    if (reason === 'pace') this.poseSkippedPace++;
    else this.poseSkippedBuffer++;
  }

  recordEeState(): void {
    const now = performance.now();
    if (this.lastEeStateAt != null) {
      const gap = now - this.lastEeStateAt;
      this.eeStateCount++;
      this.eeGapSum += gap;
      if (gap > this.eeGapMax) this.eeGapMax = gap;
    }
    this.lastEeStateAt = now;
  }

  /** Call once per second with the latest video report (may be the same
   *  object between ticks — the pod-side window is 5s, ours is 1s, so we
   *  just mirror whatever's current rather than re-deriving it). */
  tick(video: ProfilerVideoReport): void {
    const heap = (performance as unknown as { memory?: { usedJSHeapSize: number } }).memory;
    const row: Row = {
      t_uptime_s: Math.round(((performance.now() - this.t0) / 1000) * 10) / 10,
      frame_fps: this.frameCount > 0 ? Math.round((1000 / (this.frameDtSum / this.frameCount)) * 10) / 10 : null,
      frame_dt_avg_ms: this.frameCount > 0 ? Math.round((this.frameDtSum / this.frameCount) * 10) / 10 : null,
      frame_dt_max_ms: this.frameCount > 0 ? Math.round(this.frameDtMax * 10) / 10 : null,
      pose_sent: this.poseSent,
      pose_skipped_pace: this.poseSkippedPace,
      pose_skipped_buffer: this.poseSkippedBuffer,
      pose_buffered_bytes_max: this.poseBufferedMax,
      ee_state_recv: this.eeStateCount,
      ee_gap_avg_ms: this.eeStateCount > 0 ? Math.round((this.eeGapSum / this.eeStateCount) * 10) / 10 : null,
      ee_gap_max_ms: this.eeStateCount > 0 ? Math.round(this.eeGapMax * 10) / 10 : null,
      video_fps_per_cam: video?.fpsPerCam ?? null,
      video_lag_avg_ms: video?.lagAvgMs ?? null,
      video_lag_max_ms: video?.lagMaxMs ?? null,
      video_node_lag_avg_ms: video?.nodeLagAvgMs ?? null,
      video_pod_lag_avg_ms: video?.podLagAvgMs ?? null,
      video_decode_avg_ms: video?.decodeAvgMs ?? null,
      video_decode_max_ms: video?.decodeMaxMs ?? null,
      video_dropped: video?.dropped ?? null,
      video_bytes_avg_kb: video?.bytesAvgKb ?? null,
      heap_used_mb: heap ? Math.round((heap.usedJSHeapSize / 1e6) * 10) / 10 : null,
    };
    this.rows.push(row);

    // Also print every row live (not just the final logToConsole() dump
    // at VR-exit): if the tab/app crashes or is force-closed before the
    // session ever ends cleanly, whatever made it to the console up to
    // that point survives in the Quest Browser's remote-debugging log —
    // console output isn't buffered/lost the way an in-memory `rows`
    // array would be on a hard crash.
    // eslint-disable-next-line no-console
    console.debug(`[teleop profile ${this.meta.profileLabel}]`, row);

    // Reset per-window accumulators. lastFrameAt/lastEeStateAt are NOT
    // reset — they anchor the next gap measurement so a window boundary
    // never fabricates a fake gap.
    this.frameCount = 0;
    this.frameDtSum = 0;
    this.frameDtMax = 0;
    this.poseSent = 0;
    this.poseSkippedPace = 0;
    this.poseSkippedBuffer = 0;
    this.poseBufferedMax = 0;
    this.eeStateCount = 0;
    this.eeGapSum = 0;
    this.eeGapMax = 0;
  }

  get rowCount(): number {
    return this.rows.length;
  }

  // ---------- export ----------

  private toCsv(): string {
    const header = [
      '# interlatent teleop profile',
      `# profile_label: ${this.meta.profileLabel}`,
      `# session_id: ${this.meta.sessionId}`,
      `# started_at: ${this.meta.startedAtIso}`,
      `# bimanual: ${this.meta.bimanual}`,
      `# robot_kind: ${this.meta.robotKind ?? 'unknown'}`,
      `# user_agent: ${this.meta.userAgent}`,
      `# rows: ${this.rows.length} (1 row = ~1s of the session)`,
    ].join('\n');
    const cols: (keyof Row)[] = [
      't_uptime_s', 'frame_fps', 'frame_dt_avg_ms', 'frame_dt_max_ms',
      'pose_sent', 'pose_skipped_pace', 'pose_skipped_buffer', 'pose_buffered_bytes_max',
      'ee_state_recv', 'ee_gap_avg_ms', 'ee_gap_max_ms',
      'video_fps_per_cam', 'video_lag_avg_ms', 'video_lag_max_ms',
      'video_node_lag_avg_ms', 'video_pod_lag_avg_ms',
      'video_decode_avg_ms', 'video_decode_max_ms', 'video_dropped', 'video_bytes_avg_kb',
      'heap_used_mb',
    ];
    const lines = [cols.join(',')];
    for (const row of this.rows) {
      lines.push(cols.map((c) => fmtNum(row[c] as number | null)).join(','));
    }
    return header + '\n' + lines.join('\n') + '\n';
  }

  /** Print the whole session to the browser console — the PRIMARY,
   *  guaranteed-to-work output channel. `a.click()`-triggered downloads
   *  (see downloadCsv() below) require the browser to treat the call as
   *  originating from a direct user gesture; on the Quest Browser (and
   *  most mobile Chromium browsers) a download fired from an async
   *  callback — e.g. the WebXR session's 'end' event, which is what
   *  fires when the operator takes the headset off — silently loses
   *  that trust and the download is dropped with no error, no warning,
   *  nothing. console.log/console.table have no such restriction: they
   *  always work, and the output is retrievable after the fact via the
   *  Quest Browser's remote-debugging console (chrome://inspect over
   *  USB from a PC). Call this BEFORE downloadCsv() so the guaranteed
   *  channel always fires even if the download silently no-ops. */
  logToConsole(): void {
    // eslint-disable-next-line no-console
    console.info(
      `[teleop profile] ${this.meta.profileLabel} — session=${this.meta.sessionId} ` +
      `started=${this.meta.startedAtIso} bimanual=${this.meta.bimanual} ` +
      `robot=${this.meta.robotKind ?? 'unknown'} rows=${this.rows.length}`,
    );
    if (this.rows.length === 0) {
      // eslint-disable-next-line no-console
      console.warn(
        '[teleop profile] zero rows recorded this VR session — either it ' +
        'lasted under 1s, or the profiler never started (token/connect ' +
        'never resolved). No data to show.',
      );
      return;
    }
    // eslint-disable-next-line no-console
    if (typeof console.table === 'function') console.table(this.rows);
    // Raw CSV text too, so it can be copy-pasted into a .csv file
    // directly from the console even without console.table support.
    // eslint-disable-next-line no-console
    console.log(this.toCsv());
  }

  /** Trigger a browser download of the whole session as a CSV. Best-
   *  effort convenience only — see logToConsole() above for why this
   *  can silently do nothing on this platform, and call that FIRST.
   *  No-op if nothing was ever recorded (avoids a zero-row file on an
   *  aborted connect). Safe to call from a `useEffect` cleanup / unmount
   *  path. */
  downloadCsv(): void {
    if (this.rows.length === 0) return;
    try {
      const blob = new Blob([this.toCsv()], { type: 'text/csv' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `teleop_${this.meta.profileLabel}.csv`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      // Revoke on a delay — some browsers cancel the download if the
      // object URL is revoked synchronously.
      setTimeout(() => URL.revokeObjectURL(url), 2000);
      // eslint-disable-next-line no-console
      console.info(
        `teleop profiler: downloaded teleop_${this.meta.profileLabel}.csv (${this.rows.length} rows)`,
      );
    } catch (e) {
      // eslint-disable-next-line no-console
      console.warn('teleop profiler: CSV download failed', e);
    }
  }
}
