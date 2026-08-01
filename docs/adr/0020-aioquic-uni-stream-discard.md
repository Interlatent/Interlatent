# 0020 — Per-frame QUIC uni streams must be manually discarded (aioquic leak)

Status: accepted (2026-07-22)

## Context

The QUIC teleop preview ships each video frame as one short-lived
WebTransport **unidirectional stream** (open → write ~6 KB JPEG → FIN;
ADR-adjacent design in `node/teleop/_quic_proc.py`). At 24 Hz × 3
cameras that is ~72 streams/second, thousands per minute, on every
connection that carries preview video — the node child process and the
relay's browser-facing connection alike.

aioquic (pinned `>=1.3,<2`) is supposed to collect closed streams: its
per-packet sweep in `QuicConnection._write_application` discards any
stream whose `QuicStream.is_finished` is true. But `is_finished`
requires **receiver AND sender** halves finished, and
`QuicStreamReceiver.__init__` hardcodes `is_finished = False` — it
ignores its `readable` flag, even though its own docstring promises a
send-only stream's receiver "finishes immediately". So for a
**locally-opened send-only uni stream, `is_finished` is false forever
and the sweep never collects it.** (Receive-only streams are fine: their
*sender* half is born finished, so they collect on FIN receipt.)

Two costs follow, and the second is the one that hurt:

1. **Memory**: each dead stream is a `QuicStream` in `_quic._streams`
   plus an H3-layer entry in `H3Connection._stream` (~200 B). Slow,
   unbounded.
2. **CPU per packet**: `_write_application` iterates
   `self._streams.values()` (stream-limit frames) **and**
   `self._streams_queue` (data scheduling, rebuilt by list comprehension)
   on *every packet build*. With N dead streams, every datagram send,
   every ACK, every frame pays O(N) — and N grows at frame rate.

Field signature (Jetson Orin Nano node → Fly relay → Quest 3, July
2026): with the preview offer pinned at 24 Hz, per-frame delivery time
grew **monotonically ~40 ms → ~100+ ms over one minute**; delivered fps
decayed ~23 → ~9 per camera; headset glass-to-eye `video_lag` grew in
lock-step; `reset_ttl` stayed 0 (no frame ever stale — everything just
uniformly slower); and everything **reset to fast on reconnect**
(per-connection state). Fixing the node alone did not clear it — the
relay exhibits the *same* leak on its browser-facing connection, where
it opens one send-only uni stream per forwarded frame. The relay's
event loop slowdown was visible from the node as growing FIN-ack time.

This is a **class of bug, not a one-off**: any aioquic endpoint that
opens per-frame (or otherwise high-rate) send-only uni streams will
reproduce it, and it is invisible until a session runs long enough for
the O(N) cost to dominate — early-session behavior is always healthy.

## Decision

**Every code path that opens send-only uni streams on aioquic must
actively discard them once the send side is fully acked**, replicating
exactly what aioquic's own sweep does when it fires:

```python
stream = quic._streams.get(sid)
if stream is not None and stream.sender.is_finished:
    quic._streams.pop(sid, None)
    quic._streams_finished.add(sid)      # late peer frames = handled, not error
    quic._streams_queue.remove(stream)   # ValueError → already gone
    http._stream.pop(sid, None)          # H3-layer entry too
```

Timing rule: **discard only when `sender.is_finished`** — true once the
FIN is acked (`QuicStreamSender.on_data_delivery`) *or* once a
RESET_STREAM is acked (`on_reset_delivery`). Discarding earlier would
drop a pending FIN/RESET frame before it is sent and strand the peer's
receive side. A TTL-reset stream therefore parks in a bounded
pending list and retries on subsequent sweeps until its RESET is acked.

Where this lives — in **both endpoints, inside the connection object** (not the
load-shedding policy above it):

- **Node**: the discard is owned by `_WTClientProtocol` in
  `node/teleop/_quic_client.py`, which parks every uni stream it opens in a
  pure `_quic_uni_gc.UniStreamGC` and sweeps it on every `transmit()`
  (piggybacked on packet builds), discarding each stream once
  `_sender_finished` — the same `sender.is_finished` predicate the governor
  uses to free a slot. The `_VideoGovernor` (`_quic_proc.py`) is deliberately
  *not* involved: it is pure load-shedding (in-flight cap + TTL reset) and no
  longer touches aioquic bookkeeping. A TTL-reset stream stays parked in the GC
  until its RESET is acked, then discards.
- **Relay** (monorepo `teleop-quic-relay/server.py`, deployed
  separately on Fly): `RelayProtocol.note_uni_done()` +
  `sweep_uni_discards()`, swept at frame cadence on the destination
  connection and piggybacked on datagram sends. **The fix must exist at
  both endpoints; either one alone re-creates the decay.**

**Observability (the regression tripwire)**: the child's 5 s stats line carries
`qs=` — the live `len(_quic._streams)` gauge (it rides the log line, not the
`vstats` IPC, which the parent has no use for). Healthy: bounded near the video
in-flight cap (single digits).
Growing at ~frame rate: the discard path is not running (old code
deployed, or a private attr moved under an aioquic upgrade). Any future
"fps/latency slowly degrades within a session and resets on reconnect"
report should be checked against `qs` *first*.

## Consequences

- The fix reaches into pinned-version private attrs (`_streams`,
  `_streams_finished`, `_streams_queue`, `H3Connection._stream`) — the
  same caveat already carried by `uni_stream_finished()`. **Bumping the
  aioquic pin requires re-verifying both the receiver-`is_finished` bug
  and these attr names**; every touch point degrades to a safe no-op on
  `AttributeError`, so an upgrade fails soft (leak returns, `qs` gauge
  exposes it) rather than crashing.
- If upstream aioquic ever fixes `QuicStreamReceiver` to honor
  `readable`, the manual discard becomes a harmless no-op (the stream
  is already gone → `discard` returns True).
- Reviewers: treat `create_webtransport_stream(...unidirectional)` +
  `end_stream=True` anywhere in this codebase as incomplete without a
  paired discard path. The pattern "open, write, FIN, forget" is
  exactly the bug.
- Related knobs and the surrounding bandwidth model are documented in
  `docs/teleop.md` (“Bandwidth knob reference”).
