# Nori keep-alive pump is liveness-tied, never unconditional

The Nori robot's on-Pi daemon (`NoriCoreAgent`, spoken to over the Nori-Protocol
v1 NDJSON contract on TCP `:7777`) has **no heartbeat message**: the control-frame
stream itself feeds its watchdog, and frame silence beyond `t_stop_ms` (disclosed
in the handshake ack, e.g. 500 ms LAN) safe-stops the robot. Interlatent's control
loop ticks at ~30 Hz but deliberately sends nothing on the hold path and skips
sends during DRTC gaps — so someone must keep frames flowing, and *how* they flow
decides whether the daemon's watchdog still means anything.

The `interlatent.adapters.nori` session client therefore runs an internal ~50 Hz
pump of motion-free `{"type":"control","seq":N}` frames that is **liveness-tied**:
it sends only while the control loop has called `get_observation()` within
roughly the daemon's `t_warn_ms` (capped at 500 ms, floored at two pump periods),
and only when no real control frame went out within a pump period. It never pumps
while disconnected, and `disconnect()` joins the pump thread so it provably
cannot outlive the session.

## Considered options

- **No pump** (each `send_action` is one frame). Rejected: every intentional hold
  and routine inference gap longer than `t_stop_ms` — DRTC cold start, cooldown,
  a slow camera read — would safe-stop the robot mid-session, and interlatent's
  hold semantics ("send nothing, servos hold") invert on Nori, where holding a
  pose requires *continuing to send* (CLIENTS.md).
- **Unconditional pump.** Rejected as a safety violation: if the control loop
  deadlocks while the pump thread survives, the robot stays armed forever under
  a dead brain — precisely the failure the daemon's watchdog exists to catch.
  A keep-alive that cannot fall silent *defeats* a robot-side safety feature,
  breaking this integration's prime constraint.

## Consequences

- The daemon watchdog's meaning is preserved at a coarser granularity: "client
  liveness" now means "the ~30 Hz control loop is ticking" rather than "a 50 Hz
  sender thread exists". A wedged loop stops the pump within ~`t_warn_ms` and
  the daemon safe-stops by `t_stop_ms`, exactly as designed. (`kill -STOP` on
  the node is the on-hardware acceptance test.)
- Two watchdogs, two jobs, neither bypassed: the daemon watchdog guards *client*
  liveness; the `SafetyGate` staleness hold (200 ms) guards *human-input*
  liveness. The pump feeds only the former.
- The pump lives entirely inside the adapter's session client; the generic
  control loop, DRTC client, and daemon plumbing are unaware of it. Cost: the
  loop's `get_observation()` call doubles as the liveness proof, so a native
  loop for Nori must call it every tick on every path (it does; it is the
  loop's first statement).
