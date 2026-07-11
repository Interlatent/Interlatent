"""Optional vendor subpackage for the Nori robot (``--robot nori``).

Drives the on-Pi Nori daemon (NoriCoreAgent) over the Nori-Protocol v1 NDJSON
contract on TCP ``localhost:7777`` — not a motor driver, and deliberately not
the browser-only ``@nori/sdk``. All safety enforcement (range clamping, e-stop
hard latch, watchdog safe-stop) stays robot-side in the daemon; this adapter
discloses that state and feeds the daemon's watchdog with a liveness-tied
keep-alive pump (ADR 0015). See ADR 0011 for the vendor-subpackage pattern.

Requires the ``nori`` extra: ``pip install 'interlatent[nori]'`` (pyzmq for
the camera channel, opencv for JPEG decode). Vendor imports are lazy so the
base install never loads them.
"""
from .loop import control_loop

__all__ = ["control_loop"]
