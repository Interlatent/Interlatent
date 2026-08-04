// Copied from Interlatent-Main site/src/lib/teleop/quicPoseSocket.ts @ f7e4bfb6 (2026-07-30). Upstream is the dashboard copy; sync fixes both ways.
/**
 * WebSocket-shaped shim for the QUIC teleop path.
 *
 * `VRTeleopOverlay` already: (a) sends `mode:"pose"` frames it computed with
 * the clutch mapper, and (b) consumes `ee_state` messages (the robot's live EE
 * pose) to anchor that mapper. On the QUIC path the browser owns IK, so this
 * shim sits where the WebSocket used to and does the two conversions locally:
 *
 *   send(pose frame)  → run DLS IK → `mode:"targets"` datagram (duplicated)
 *   node qpos datagram → FK → synthesized `ee_state` message → onmessage
 *
 * Because it quacks like a `WebSocket` (`readyState`, `bufferedAmount`,
 * `onopen`/`onmessage`/`onclose`, `send`, `close`), the overlay's existing
 * pose-send and ee_state-handling code runs unchanged — the transport swap is
 * invisible above this line. Video frames arrive on inbound uni streams and
 * pass through `onmessage` untouched as ArrayBuffers, byte-identical to the
 * WS relay's binary video messages, so the overlay's parser is shared (the
 * v1 single-arm-only and no-video restrictions are lifted).
 *
 * Bimanual: a `chains`-shaped spec (bimanual bundle) gets one solver per
 * arm, mirroring the pod's `_process_bimanual` (retarget/stage.py) exactly:
 * left-then-right, each solve overwrites only its own action indices and
 * passes the rest through from its seed, so chaining the second solve's
 * seed from the first solve's output composes both arms into the one flat
 * `joint_targets` vector the node already speaks. `ee_state` back to the
 * overlay is synthesized in the same per-side `chains` shape the pod emits.
 *
 * IK/FK reuse the machine-precision-verified `DlsSolver` + `forwardKinematics`.
 */
import { QuicTeleopLink } from './webtransport';
import { KinematicSpecBundle, isChainsSpec } from './kinematics';
import { DlsSolver } from './dlsSolver';
import { Quat, Vec3 } from './quat';

type MsgHandler = (ev: { data: string | ArrayBuffer }) => void;

const OPEN = 1;
// Cap the pending send-time map (monotonic seq, so we also prune ≤ applied).
const MAX_PENDING = 512;
// Console-log a rolling RTT summary at most this often.
const LATENCY_LOG_MS = 2000;
// How often the browser re-asks the node for the kinematic spec until it
// arrives. The two ends join the relay independently, so the first request can
// land before the node is paired — retry covers that race.
const SPEC_REQUEST_INTERVAL_MS = 250;

const streamDecoder = new TextDecoder();

/** Command round-trip latency tracker (single browser clock).
 *  markSent(seq) at send; onApplied(seq) when the node echoes the executed
 *  seq → RTT = now − sent[seq]. Rolling mean/max/last over a window. */
export class LatencyTracker {
  private sent = new Map<number, number>();
  private n = 0;
  private sum = 0;
  private max = 0;
  private lastMs = 0;
  private lastLogAt = 0;

  markSent(seq: number, now: number): void {
    this.sent.set(seq, now);
    if (this.sent.size > MAX_PENDING) {
      // Evict the oldest (smallest seq) — it was lost or never echoed.
      const oldest = this.sent.keys().next().value;
      if (oldest !== undefined) this.sent.delete(oldest);
    }
  }

  /** Returns the RTT (ms) for this applied seq, or null if unknown. */
  onApplied(appliedSeq: number, now: number): number | null {
    const t = this.sent.get(appliedSeq);
    // Prune everything up to and including the applied seq (stale).
    for (const k of this.sent.keys()) {
      if (k <= appliedSeq) this.sent.delete(k);
      else break;
    }
    if (t === undefined) return null;
    const rtt = now - t;
    this.n += 1;
    this.sum += rtt;
    this.max = Math.max(this.max, rtt);
    this.lastMs = rtt;
    return rtt;
  }

  /** Latest RTT snapshot for the HUD. */
  snapshot(): { last: number; mean: number; max: number; n: number } {
    return {
      last: this.lastMs,
      mean: this.n ? this.sum / this.n : 0,
      max: this.max,
      n: this.n,
    };
  }

  /** Emit a console summary at most every LATENCY_LOG_MS, then reset window. */
  maybeLog(now: number): void {
    if (this.n === 0 || now - this.lastLogAt < LATENCY_LOG_MS) return;
    this.lastLogAt = now;
    // eslint-disable-next-line no-console
    console.info(
      `[teleop:quic] command RTT (n=${this.n}): mean=${(this.sum / this.n).toFixed(0)}ms ` +
        `max=${this.max.toFixed(0)}ms last=${this.lastMs.toFixed(0)}ms ` +
        `(one-way ≈ ${(this.sum / this.n / 2).toFixed(0)}ms)`,
    );
    this.n = 0;
    this.sum = 0;
    this.max = 0;
  }
}

export class QuicPoseSocket {
  readyState = 0; // CONNECTING
  bufferedAmount = 0; // always drained (datagrams are fire-and-forget)
  binaryType = 'arraybuffer';
  onopen: (() => void) | null = null;
  onmessage: MsgHandler | null = null;
  onclose: ((ev: { reason?: string }) => void) | null = null;
  onerror: ((e: unknown) => void) | null = null;

  private link: QuicTeleopLink;
  /** Single-arm solver, or null when this is a bimanual session. */
  private solver: DlsSolver | null = null;
  /** Per-arm solvers for a bimanual bundle, or null for single-arm. */
  private chainSolvers: { left: DlsSolver; right: DlsSolver } | null = null;
  /** True once a spec has been applied (from the node, or an HTTP fallback). */
  private specReceived = false;
  /** Fired when the spec lands, so the overlay can build the clutch mapper(s)
   *  from the SAME object — the spec carries the mapper hints too. */
  private onSpec: ((spec: KinematicSpecBundle) => void) | null;
  private specRequestTimer: ReturnType<typeof setInterval> | null = null;
  /** Latest robot joint state (action order, robot units) — the IK seed. */
  private seed: number[] | null = null;
  private lastStateSeq = -1;
  private latency = new LatencyTracker();
  // Node→browser clock-skew anchor for ABSOLUTE video age (consumed by the
  // overlay's recordVideoFrameStats). State datagrams carry the node's
  // monotonic ts_ms — same clock domain as video-frame headers — and are
  // tiny + drop-don't-queue on the node, so min(arrival − ts_ms) over a
  // short window ≈ clock offset + one datagram transit: a baseline that
  // stays honest while video frames queue (unlike the fastest-video-frame
  // baseline, which absorbs any standing queue delay). Two rotating 5s
  // buckets give a 5–10s window so clock drift and node restarts
  // self-correct.
  private skewBuckets: [number, number] = [Infinity, Infinity];
  private skewRotatedAt = 0;
  private static readonly SKEW_BUCKET_MS = 5000;

  /** Command-RTT snapshot for the HUD (last/mean/max ms, sample count). */
  getLatency(): { last: number; mean: number; max: number; n: number } {
    return this.latency.snapshot();
  }

  /** Node→browser clock skew (ms): min (arrival − node ts_ms) over the last
   *  ~5–10s of state datagrams; null until the first ts_ms-carrying state
   *  arrives (older nodes send none). */
  getStateSkewMs(): number | null {
    const m = Math.min(this.skewBuckets[0], this.skewBuckets[1]);
    return Number.isFinite(m) ? m : null;
  }

  private noteStateSkew(skew: number, now: number): void {
    if (now - this.skewRotatedAt >= QuicPoseSocket.SKEW_BUCKET_MS) {
      this.skewBuckets = [this.skewBuckets[1], Infinity];
      this.skewRotatedAt = now;
    }
    this.skewBuckets[1] = Math.min(this.skewBuckets[1], skew);
  }

  /**
   * @param spec  Optional. When omitted (the node-sourced path), the shim opens
   *   the transport, asks the node for the spec over the relay, and builds the
   *   solver when it arrives. When provided (the HTTP-fallback path), the solver
   *   is built up front. Either way the spec flows out through `opts.onSpec` so
   *   the overlay builds the clutch mapper from the same object.
   */
  constructor(
    url: string,
    token: string,
    spec?: KinematicSpecBundle,
    opts?: { onSpec?: (spec: KinematicSpecBundle) => void },
  ) {
    this.onSpec = opts?.onSpec ?? null;
    if (spec) this.setSpec(spec);
    const full = `${url}${url.includes('?') ? '&' : '?'}token=${encodeURIComponent(token)}`;
    this.link = new QuicTeleopLink(full, {
      onMessage: (m) => this.onNodeMessage(m),
      onStream: (buf) => this.onStreamData(buf),
      onClose: (reason) => {
        this.readyState = 3; // CLOSED
        this.stopSpecRequests();
        this.onclose?.({ reason });
      },
    });
    this.link.ready
      .then(() => {
        this.readyState = OPEN;
        this.onopen?.();
        // No spec yet → pull it from the node (retried; see startSpecRequests).
        if (!this.specReceived) this.startSpecRequests();
      })
      .catch((e) => {
        this.readyState = 3;
        this.stopSpecRequests();
        this.onerror?.(e);
        this.onclose?.({ reason: String(e) });
      });
  }

  /** True once a spec has been applied and the solver(s) exist. */
  get hasSpec(): boolean {
    return this.specReceived;
  }

  /** Apply a kinematic spec: build the DLS solver(s) and surface it to the
   *  overlay for the mapper. First one wins — a late HTTP fallback can't
   *  clobber a spec the node already delivered. */
  setSpec(spec: KinematicSpecBundle): void {
    if (this.specReceived) return;
    try {
      if (isChainsSpec(spec)) {
        this.chainSolvers = {
          left: new DlsSolver(spec.chains.left),
          right: new DlsSolver(spec.chains.right),
        };
      } else {
        this.solver = new DlsSolver(spec);
      }
    } catch (e) {
      this.onerror?.(e);
      return;
    }
    this.specReceived = true;
    this.stopSpecRequests();
    this.onSpec?.(spec);
  }

  private startSpecRequests(): void {
    const ask = () => {
      if (!this.specReceived) this.link.send({ type: 'request_spec' }, 1);
    };
    ask();
    this.specRequestTimer = setInterval(() => {
      if (this.specReceived) {
        this.stopSpecRequests();
        return;
      }
      ask();
    }, SPEC_REQUEST_INTERVAL_MS);
  }

  private stopSpecRequests(): void {
    if (this.specRequestTimer !== null) {
      clearInterval(this.specRequestTimer);
      this.specRequestTimer = null;
    }
  }

  /** Inbound uni stream. A spec stream (envelope header `type:"spec"`) builds
   *  the solver; any other stream (video) passes through the WS-shaped
   *  onmessage untouched as an ArrayBuffer, byte-identical to the WS path. */
  private onStreamData(buf: ArrayBuffer): void {
    try {
      if (buf.byteLength >= 2) {
        const hlen = new DataView(buf).getUint16(0);
        if (buf.byteLength >= 2 + hlen) {
          const header = JSON.parse(
            streamDecoder.decode(new Uint8Array(buf, 2, hlen)),
          );
          if (header && header.type === 'spec') {
            const body = streamDecoder.decode(new Uint8Array(buf, 2 + hlen));
            this.setSpec(JSON.parse(body) as KinematicSpecBundle);
            return;
          }
        }
      }
    } catch {
      // Not a parseable spec envelope — treat as an opaque (video) frame.
    }
    this.onmessage?.({ data: buf });
  }

  /** The overlay sends JSON pose frames here (same call it made on the WS). */
  send(data: string): void {
    if (this.readyState !== OPEN) return;
    let f: Record<string, unknown>;
    try {
      f = JSON.parse(data);
    } catch {
      return;
    }
    const engaged = !!f.engaged && !!f.deadman;
    const seq = typeof f.seq === 'number' ? f.seq : 0;

    // Disengaged / release / non-pose (keys) → tell the node to idle; the
    // SafetyGate holds. No IK needed.
    if (!engaged || f.mode !== 'pose') {
      this.sendHold(seq);
      return;
    }
    // Spec not delivered yet (node-sourced path still in flight) → no solver to
    // IK with; hold. The overlay also gates Enter-VR on the spec, so this is
    // belt-and-suspenders.
    if (!this.solver && !this.chainSolvers) {
      this.sendHold(seq);
      return;
    }
    // Can't IK until we know the robot's current joints (the seed) — hold.
    if (this.seed === null) {
      this.sendHold(seq);
      return;
    }

    if (this.chainSolvers) {
      this.sendBimanual(f, seq);
      return;
    }

    if (!Array.isArray(f.ee_pos)) {
      this.sendHold(seq);
      return;
    }
    const eePos = f.ee_pos as Vec3;
    const eeQuat = (Array.isArray(f.ee_quat) ? f.ee_quat : [0, 0, 0, 1]) as Quat;
    const pinch = typeof f.pinch === 'number' ? f.pinch : 0;
    const targets = this.solver!.solve(eePos, eeQuat, this.seed, pinch);

    this.sendTargets(seq, targets, f);
  }

  /** Idle frame: the node's SafetyGate holds; no IK needed. */
  private sendHold(seq: number): void {
    this.link.send({ engaged: false, deadman: false, seq, mode: 'targets' });
  }

  private sendTargets(
    seq: number,
    targets: number[],
    f: Record<string, unknown>,
  ): void {
    this.latency.markSent(seq, performance.now());
    this.link.send({
      engaged: true,
      deadman: true,
      seq,
      mode: 'targets',
      joint_targets: targets,
      confidence: typeof f.confidence === 'number' ? f.confidence : 1.0,
    }, 3); // duplicate ×3 for loss tolerance
  }

  /** Bimanual pose frame → one combined targets vector. A 1:1 mirror of the
   *  pod's `_process_bimanual` (retarget/stage.py): sides solve left-then-
   *  right with the second seed chained from the first output (disjoint
   *  action indices, everything else passes through); a side that is not
   *  engaged this tick is skipped — its indices pass through unchanged from
   *  the measured seed (hold) — and its warm-start is dropped so the next
   *  engage re-seeds from measurement. */
  private sendBimanual(f: Record<string, unknown>, seq: number): void {
    const chains = f.chains as
      | Record<string, Record<string, unknown> | undefined>
      | undefined;
    if (!chains || typeof chains !== 'object') {
      // Flat pose frame against a bimanual bundle (stale/missing ik_hints
      // put the overlay in single-arm mode) — mirror the pod's
      // bad_frame_no_chains: nothing to solve, hold.
      this.sendHold(seq);
      return;
    }

    let out = this.seed!;
    let anyEngaged = false;
    for (const side of ['left', 'right'] as const) {
      const solver = this.chainSolvers![side];
      const sf = chains[side];
      if (!sf || typeof sf !== 'object' || !sf.engaged) {
        solver.resetWarmstart();
        continue;
      }
      const eePos = sf.ee_pos;
      if (!Array.isArray(eePos) || eePos.length !== 3) continue;
      const eeQuat =
        Array.isArray(sf.ee_quat) && sf.ee_quat.length === 4
          ? (sf.ee_quat as Quat)
          : this.fkQuat(solver, out); // position-only: hold current orientation
      const pinch = typeof sf.pinch === 'number' ? sf.pinch : 0;
      out = solver.solve(eePos as Vec3, eeQuat, out, pinch);
      anyEngaged = true;
    }

    if (!anyEngaged) {
      this.sendHold(seq);
      return;
    }
    this.sendTargets(seq, out, f);
  }

  private fkQuat(solver: DlsSolver, actionState: number[]): Quat {
    try {
      return solver.fk(actionState).quat;
    } catch {
      return [0, 0, 0, 1];
    }
  }

  private onNodeMessage(m: Record<string, unknown>): void {
    if (m.type !== 'state' || !Array.isArray(m.qpos)) return;
    // Seq dedupe: ignore a late duplicate that would clobber newer state.
    const seq = typeof m.seq === 'number' ? m.seq : this.lastStateSeq + 1;
    if (seq <= this.lastStateSeq && seq > this.lastStateSeq - 1000) return;
    this.lastStateSeq = seq;

    const qpos = (m.qpos as number[]).map(Number);
    this.seed = qpos;

    // Command-RTT telemetry: the node echoes the target seq it executed.
    const now = performance.now();
    if (typeof m.ts_ms === 'number') this.noteStateSkew(now - m.ts_ms, now);
    let rttMs: number | null = null;
    if (typeof m.applied_seq === 'number' && m.applied_seq >= 0) {
      rttMs = this.latency.onApplied(m.applied_seq, now);
      this.latency.maybeLog(now);
    }
    // On disengage the solver must re-seed from measurement next engage.
    // (The overlay drives engage/disengage; we key warm-start off it via the
    // send() path — a disengaged frame doesn't call solve(), and the first
    // engaged solve after a gap re-seeds from this measured state.)

    // FK the measured joints → EE pose, synthesize an ee_state message so the
    // clutch mapper anchors to the real robot exactly as on the WS path.
    // Bimanual sessions get the same per-side `chains` shape the pod emits
    // (retarget/stage.py `_build_ee_state`) — the overlay's bimanual branch
    // only ever reads that shape.
    if (this.chainSolvers) {
      const chainsOut: Record<string, unknown> = {};
      for (const side of ['left', 'right'] as const) {
        let fk: { pos: Vec3; quat: Quat };
        try {
          fk = this.chainSolvers[side].fk(qpos);
        } catch {
          continue;
        }
        chainsOut[side] = {
          ready: true,
          ee_pos: fk.pos,
          ee_quat: fk.quat,
          obs_age_ms: 0,
          rtt_ms: rttMs,
          ik: {},
        };
      }
      if (Object.keys(chainsOut).length === 0) return;
      this.onmessage?.({
        data: JSON.stringify({ type: 'ee_state', chains: chainsOut }),
      });
      return;
    }

    let fk: { pos: Vec3; quat: Quat };
    try {
      fk = this.solver!.fk(qpos);
    } catch {
      return;
    }
    const ee_state = {
      type: 'ee_state',
      ready: true,
      ee_pos: fk.pos,
      ee_quat: fk.quat,
      obs_age_ms: 0,
      // Command round-trip latency (ms) for this state's echoed seq, so the
      // overlay HUD can surface live latency. null until the first echo lands.
      rtt_ms: rttMs,
      ik: {},
    };
    this.onmessage?.({ data: JSON.stringify(ee_state) });
  }

  close(): void {
    this.readyState = 3;
    this.stopSpecRequests();
    this.link.close();
  }
}
