// Copied from Interlatent-Main site/src/lib/teleop/webtransport.ts @ f7e4bfb6 (2026-07-30). Upstream is the dashboard copy; sync fixes both ways.
/**
 * Browser WebTransport client for the low-latency teleop control channel.
 *
 * Opens a WebTransport session to the co-located QUIC relay and exchanges
 * **unreliable datagrams**: outbound = joint-target frames the browser IK just
 * solved (duplicated for loss tolerance); inbound = the robot node's live joint
 * state (for FK / clutch-anchor / reconciliation). No reliability, no ordering,
 * no buffering — drop-don't-buffer; the seq dedupe lives at each endpoint.
 *
 * Datagrams are JSON (tiny — a few floats, well under the ~1200 B MTU), so the
 * node reuses `TeleopFrame.from_json` and the browser reuses the same compact
 * frame shape.
 *
 * Video rides the SAME session on inbound **unidirectional streams**: the node
 * ships one FIN-closed uni stream per JPEG frame (uint16-BE header length +
 * JSON header + raw JPEG). Each stream is read fully and delivered as one
 * ArrayBuffer via `onStream`; a reset/aborted stream is silently dropped —
 * that's the node's load shedding doing its job.
 *
 * WebTransport is Chromium-only (Quest Browser is Chromium). Teleop is
 * QUIC-only; a token that isn't `transport === 'quic'` is a misconfiguration,
 * not a fallback (the WS relay path was removed).
 */

export interface QuicLinkOpts {
  /** Called with each inbound datagram's decoded JSON (node→browser state). */
  onMessage: (msg: Record<string, unknown>) => void;
  /** Called with each fully-received inbound uni stream (one video frame). */
  onStream?: (data: ArrayBuffer) => void;
  /** Called once on close/error so the UI can surface disconnect. */
  onClose?: (reason: string) => void;
}

/** Bound on one inbound uni stream — a preview JPEG is ~8-15 KB, so anything
 * near this is a runaway sender; cancel and drop. */
const MAX_STREAM_BYTES = 512 * 1024;

const encoder = new TextEncoder();
const decoder = new TextDecoder();

/** `String(err)` on a WebTransportError yields a bare "WebTransportError:
 *  Opening handshake failed." and drops the two fields that actually localise
 *  the fault: `source` ('session' = the QUIC/HTTP-3 layer, 'stream' = after the
 *  session was up) and `streamErrorCode` (the relay's application close code,
 *  null when the connection never got that far). Keep them. */
export function describeWtError(e: unknown): string {
  if (!e || typeof e !== 'object') return String(e);
  const err = e as {
    name?: string;
    message?: string;
    source?: string;
    streamErrorCode?: number | null;
  };
  const bits = [`${err.name ?? 'Error'}: ${err.message ?? String(e)}`];
  if (err.source) bits.push(`source=${err.source}`);
  if (err.streamErrorCode != null) bits.push(`streamErrorCode=${err.streamErrorCode}`);
  return bits.join(' ');
}

export class QuicTeleopLink {
  private wt: WebTransport | null = null;
  private writer: WritableStreamDefaultWriter<Uint8Array> | null = null;
  private closed = false;
  readonly ready: Promise<void>;

  constructor(url: string, private opts: QuicLinkOpts) {
    this.ready = this.open(url);
  }

  private async open(url: string): Promise<void> {
    // `WebTransport` is not in older TS DOM libs; the cast keeps tsc happy
    // without pulling a lib bump. Feature-detected by the caller.
    const WT = (globalThis as unknown as { WebTransport?: unknown }).WebTransport as
      | (new (u: string) => WebTransport)
      | undefined;
    if (!WT) throw new Error('WebTransport unsupported in this browser');

    const wt = new WT(url);
    this.wt = wt;
    // Subscribe to `closed` BEFORE awaiting `ready`. A failed handshake rejects
    // both, and `closed` usually carries the more specific reason of the two —
    // subscribing after the await lost it *and* left an unhandled rejection.
    wt.closed
      .then(() => this.handleClosed('closed'))
      .catch((e: unknown) => this.handleClosed(describeWtError(e)));
    try {
      await wt.ready;
    } catch (e) {
      // The only place the browser's real reason exists. Every layer above
      // this one collapses it to a generic string, so log it here or it is
      // gone: nothing else in the app writes to console.error.
      const detail = describeWtError(e);
      // eslint-disable-next-line no-console
      console.error(`[teleop:quic] WebTransport handshake failed for ${url}:`, detail, e);
      throw new Error(detail);
    }
    this.writer = wt.datagrams.writable.getWriter();
    this.readLoop(wt.datagrams.readable.getReader());
    this.readUniStreams(wt);
  }

  /** Accept-loop for inbound uni streams (video frames). Each stream is read
   * concurrently so a stalled frame can't block the ones behind it. */
  private async readUniStreams(wt: WebTransport): Promise<void> {
    if (!this.opts.onStream) return;
    // Same older-TS-DOM-lib caveat as the WebTransport constructor above.
    const incoming = (wt as unknown as {
      incomingUnidirectionalStreams?: ReadableStream<ReadableStream<Uint8Array>>;
    }).incomingUnidirectionalStreams;
    if (!incoming) {
      console.warn('[teleop:quic] incomingUnidirectionalStreams unsupported — no video');
      return;
    }
    try {
      const reader = incoming.getReader();
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        if (value) void this.readOneStream(value);
      }
    } catch {
      /* torn down on close */
    }
  }

  private async readOneStream(stream: ReadableStream<Uint8Array>): Promise<void> {
    const chunks: Uint8Array[] = [];
    let total = 0;
    const reader = stream.getReader();
    try {
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        if (!value) continue;
        total += value.byteLength;
        if (total > MAX_STREAM_BYTES) {
          await reader.cancel();
          return;
        }
        chunks.push(value);
      }
      if (this.closed || total === 0) return;
      const buf = new Uint8Array(total);
      let off = 0;
      for (const c of chunks) {
        buf.set(c, off);
        off += c.byteLength;
      }
      this.opts.onStream?.(buf.buffer);
    } catch {
      /* stream reset mid-frame (node TTL / relay kill) — drop the partial */
    }
  }

  private async readLoop(
    reader: ReadableStreamDefaultReader<Uint8Array>,
  ): Promise<void> {
    try {
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        if (!value) continue;
        try {
          this.opts.onMessage(JSON.parse(decoder.decode(value)));
        } catch {
          /* drop malformed datagram */
        }
      }
    } catch {
      /* reader torn down on close */
    }
  }

  /** Send one frame as `dup` duplicate datagrams (loss tolerance). */
  send(frame: Record<string, unknown>, dup = 2): void {
    const w = this.writer;
    if (!w || this.closed) return;
    const bytes = encoder.encode(JSON.stringify(frame));
    for (let i = 0; i < dup; i++) {
      // Fire-and-forget; a full send queue means the link is saturated and
      // the next frame supersedes this one anyway.
      void w.write(bytes).catch(() => {});
    }
  }

  private handleClosed(reason: string): void {
    if (this.closed) return;
    this.closed = true;
    this.writer = null;
    this.opts.onClose?.(reason);
  }

  close(): void {
    if (this.closed) return;
    this.closed = true;
    try {
      this.wt?.close();
    } catch {
      /* already gone */
    }
    this.wt = null;
    this.writer = null;
  }
}
