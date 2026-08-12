"""Runs the embedded teleop relay alongside the coordinator's HTTP surface.

The coordinator's control plane is a ``ThreadingHTTPServer`` (blocking, one
thread per request) and the relay is aioquic (asyncio). Rather than convert
either, the relay gets its own thread with its own event loop — they share
nothing but the :class:`~.state.Coordinator` object, whose methods are already
lock-guarded because the HTTP side is multi-threaded anyway.

The relay is optional in three separate ways, and each degrades to "teleop is
off" rather than to a broken coordinator:

* ``aioquic`` may not be installed (it is an extra),
* ``cryptography`` may not be installed (needed to mint the certificate),
* the operator may simply not want it (``interlatent up --no-relay``).

When it is off the teleop-token route answers 404, which the node treats as
definitive and stops asking — see ``node/teleop/factory.py``.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_LOG = logging.getLogger("interlatent.coordinator.relay")

DEFAULT_RELAY_PORT = 4433


@dataclass
class RelayHandle:
    """What the HTTP surface needs to know about a running relay."""

    host: str
    port: int
    certificate_hashes: list[dict] = field(default_factory=list)
    _thread: Optional[threading.Thread] = None
    _loop: Optional[asyncio.AbstractEventLoop] = None

    @property
    def descriptor(self) -> dict:
        """What the teleop-token mint embeds in its response."""
        return {
            "base": f"https://{self.host}:{self.port}",
            "certificate_hashes": self.certificate_hashes,
        }

    def stop(self) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5.0)


def _advertise_host(explicit: str = "") -> str:
    """The address a *headset* can reach this machine at.

    Not ``0.0.0.0`` — that is a bind address, and a browser told to dial it
    goes nowhere. Probe the outbound interface, which on a normal LAN is the
    one the Quest is also on. No packet is sent; connect() on UDP just picks
    a route.
    """
    if explicit.strip():
        return explicit.strip()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))  # TEST-NET-1, deliberately unroutable
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def start_relay(
    *,
    coordinator,
    cert_dir: Path,
    host: str = "0.0.0.0",
    port: int = DEFAULT_RELAY_PORT,
    advertise: str = "",
) -> Optional[RelayHandle]:
    """Start the relay, or return None with a reason logged.

    Returning None is a supported outcome, not a failure: a coordinator with
    no relay is still a working control plane for inference.
    """
    try:
        import aioquic  # noqa: F401
    except ImportError:
        _LOG.info(
            "Teleop relay off: aioquic is not installed. "
            "Install it with: pip install 'interlatent[teleop-relay]'"
        )
        return None

    from . import certs

    advertised = _advertise_host(advertise)
    try:
        cert = certs.ensure(cert_dir, [advertised, "localhost", "127.0.0.1"])
    except certs.CertificateUnavailable as exc:
        _LOG.warning("Teleop relay off: %s", exc)
        return None

    handle = RelayHandle(
        host=advertised, port=port, certificate_hashes=cert.hashes_for_browser
    )
    ready = threading.Event()
    failure: list[BaseException] = []

    def _run() -> None:
        from .relay import serve_relay

        loop = asyncio.new_event_loop()
        handle._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(serve_relay(
                host=host,
                port=port,
                cert_file=str(cert.cert_path),
                key_file=str(cert.key_path),
                verify=coordinator.verify_teleop_token,
            ))
        except BaseException as exc:  # noqa: BLE001 - reported to the caller
            failure.append(exc)
            ready.set()
            return
        ready.set()
        try:
            loop.run_forever()
        finally:
            loop.close()

    thread = threading.Thread(target=_run, name="teleop-relay", daemon=True)
    handle._thread = thread
    thread.start()
    ready.wait(timeout=10.0)

    if failure:
        _LOG.warning("Teleop relay off: %s", failure[0])
        return None
    if not ready.is_set():
        _LOG.warning("Teleop relay off: did not come up within 10s")
        return None

    _LOG.info(
        "Teleop relay on https://%s:%d — certificate sha256 %s… "
        "(expires %s; rotated automatically)",
        advertised, port, cert.sha256[:16], cert.not_after.date(),
    )
    return handle
