# Operator e-stop rides the teleop frame; reset is human-only

Before this decision interlatent had **no operator hard stop at all**: the
teleop wire frame carried only `engaged`/`deadman` (release = soft hold), and
`SafetyGate.latch_estop` existed with zero callers. Integrating Nori made the
gap acute — while the node holds the Nori daemon's single `:7777` control-client
slot, Nori's own browser teleop (with its SPACE-bar e-stop) cannot connect, so
the adapter is the *only* software path to the daemon's e-stop during a session.

Decision, three layers:

1. **Wire**: an additive `estop: bool` field on `TeleopFrame` (the node-owned
   contract; absent → False, so legacy producers are untouched). Because a
   single datagram must not be droppable, both channels (WS `channel.py` and
   QUIC `quic_channel.py`) also latch a sticky `estop_seen` flag at decode
   time — before seq-dedupe, immune to the 250 ms staleness rule and to the
   disconnect frame-drop — consumed exactly once via `consume_estop()`.
2. **Node**: the control loop latches the `SafetyGate` (its dormant latch's
   first real caller) and short-circuits to a no-motion/no-capture/flush branch.
   The Nori native loop additionally forwards the daemon hard latch with the
   schema-canonical `{"type":"command","name":"estop"}` (not the legacy
   `{estop:true}` boolean form the TS `@nori/sdk` still sends), then ends the
   episode when the daemon reports `safety:"latched"`.
3. **Reset**: never automatic, never the loop's. For Nori it is
   `interlatent-act --robot nori --reset-latch [--token …]` — a fresh process
   sending the daemon-token-gated `reset_latch`. Where a latched gate and the
   daemon latch coexist in one process, the order is daemon first, gate second.

## Considered options

- **Adapter-API-only e-stop** (no operator button). Rejected: the human holding
  the deadman would have no panic path — deadman release only *holds*, and a
  held pose under a misbehaving policy is not a stop.
- **Producers must re-send `estop:true` until acknowledged** (no sticky latch).
  Rejected: a press during a >250 ms loop stall followed by a channel blip is
  simply lost; a panic button with a race is worse than none because it teaches
  false confidence.
- **Auto-clear on deadman re-press** (the old `SafetyConfig` comment's interim
  idea). Rejected: Nori deliberately token-gates `reset_latch` even in
  safe-stop; matching that friction end-to-end keeps one mental model — stops
  are cheap, resumes are deliberate.

## Consequences

- The estop→gate ingress lands in `node/control.py` and the Nori loop only.
  The YAM/Axol native loops do not yet check it — the known loop-drift hazard
  (FUTURE.md #13); promoting e-stop to a first-class adapter capability every
  loop gets for free is FUTURE.md #14.
- Producers (dashboard overlay, VR bridge) still need an e-stop control that
  sets the field; until one ships, the path is exercisable via a hand-sent
  frame.
- Fixed en passant: the teleop-takeover "drop queued policy chunks" call sites
  named a method (`client.flush_buffer()`) that `DRTCClient` never had and
  swallowed the `AttributeError` — takeover never actually flushed. Both call
  sites now use `client.schedule.flush()`, and the e-stop branches use the same.
