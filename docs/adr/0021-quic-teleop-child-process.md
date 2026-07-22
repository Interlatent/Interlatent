# 0021 — QUIC teleop runs in a dumb-pipe child process

Status: accepted (2026-07-22)

## Context

The node's low-latency teleop transport (`node/teleop/quic_channel.py`) speaks
WebTransport/QUIC to the co-located relay. Unlike TCP, QUIC's handshake,
loss-recovery, and ACK timers live in **userspace Python** (aioquic on
asyncio) — there is no kernel timer doing retransmission on the endpoint's
behalf. The node process also runs the robot drivers, and some of those
(e.g. i2rt's ~270 Hz gravity-comp / CAN threads for YAM) hold the GIL in tight
loops. An in-process aioquic event loop sharing that GIL gets starved: its
handshake timers fire late or not at all, and the WebTransport CONNECT never
completes. The WS transport does not have this problem — TCP retransmission is
kernel-side, so a stalled Python loop only delays application reads, it does not
break the connection.

A second, independent failure compounded the first during bring-up: a local
attribute named `self._connected` on the WebTransport protocol **shadowed
aioquic's own internal handshake boolean of the same name**. Assigning a truthy
Future to it made aioquic's `wait_connected()` return instantly, so the client
tore the handshake down ~1 ms in with a bare `ConnectionTerminated`. This looked
exactly like GIL starvation and was mis-attributed to it at first; the real fix
was renaming the attribute (`_wt_connected`). Both causes had to be removed
before the node handshake was reliable.

## Decision

**Run the aioquic WebTransport connection in a dedicated dumb-pipe child
process** (`node/teleop/_quic_proc.py`, launched `python -m ...`), isolated
from the robot-driver GIL:

- The **child** owns only connect / handshake / reconnect-with-backoff and the
  video tee's stream mechanism (`_VideoGovernor` in-flight cap + TTL). It pumps
  raw datagrams verbatim between the relay and the parent over a loopback UDP
  socket (framing in `_quic_ipc`). DATA payloads are opaque to it.
- The **parent** (`QuicTeleopChannel`) owns all protocol logic — codec, dedupe,
  staleness, pacing, applied-seq echo, telemetry, and the preview rate control
  (`PreviewBackoff`). It supervises the child: stdin-pipe EOF is the lifetime
  tether (no orphans), and a crashed child is respawned with 1→15 s backoff.
- The parent process **never imports aioquic**; only the child does. The
  `_connected` shadowing bug is fixed by the `_wt_connected` rename in
  `_quic_client`.

This split is what makes the QUIC path unit-testable without aioquic or a
network: the loopback framing (`_quic_ipc`), the parent's supervision + datagram
handling (a fake child played over a real loopback UDP socket), and a real
`-m ..._quic_proc` subprocess smoke test (hello heartbeat + stdin-EOF exit) are
each exercised offline; the live relay path is signed off on-robot.

## Consequences

- One extra process per active QUIC teleop session, plus a loopback UDP hop
  (~microseconds, kernel-local) on every control/video datagram. Cheap against
  the win of a handshake that structurally cannot be starved.
- The parent/child boundary is a hard contract: everything the child needs
  travels via env at spawn (`_quic_ipc.ENV_*`) and the cookie-pinned loopback
  socket; the child reports back only `hello` / `connected` / `disconnected` /
  `vstats` control messages. The video path is deliberately bicameral —
  mechanism (cap/TTL) in the child, rate policy (`PreviewBackoff`) in the parent
  — coupled only by the `vstats` counters.
- aioquic's own send-only-uni-stream leak GC is a separate concern, handled one
  layer down in the child's connection object — see ADR 0020.
- Any reader debugging a "QUIC handshake never completes" report should check
  GIL contention (is a driver thread starving the loop? — the child process is
  the structural fix) and attribute shadowing against aioquic internals first.
