"""Short-lived self-signed certificates for the embedded teleop relay.

There is no public CA for ``https://192.168.1.20:4433``, and WebTransport
requires TLS. Chromium's escape hatch is ``serverCertificateHashes``: dial a
self-signed cert if the page pins its SHA-256 up front. The constraints are
not ours to choose — they come from the WebTransport spec and Chromium's
implementation:

* validity **at most 14 days** (Chromium rejects longer outright),
* **ECDSA P-256** key,
* SHA-256 digest of the DER encoding.

The 14-day cap is why this module exists rather than a one-line
``openssl req``: something has to notice the cert is about to expire and mint
a new one, or teleop silently stops working a fortnight after setup. We rotate
when less than a quarter of the lifetime remains, and the mint response always
carries the *current* hash — the browser fetches a token immediately before
dialling, so it can never pin a hash we have already rotated away from.

The node does not use the hash: it verifies against the CA bundle like any
other client, so ``interlatent-node`` pins this certificate file directly (see
``node/teleop/_quic_client.py``) rather than being told to skip verification.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import ipaddress
import logging
import os
from dataclasses import dataclass
from pathlib import Path

_LOG = logging.getLogger("interlatent.coordinator.certs")

#: Chromium refuses a serverCertificateHashes cert valid for longer.
MAX_VALIDITY_DAYS = 14
#: Rotate with this much life left, so a running deployment never trips the cap.
ROTATE_WHEN_REMAINING_DAYS = 4


class CertificateUnavailable(RuntimeError):
    """`cryptography` is not installed, so no cert can be minted."""


@dataclass(frozen=True)
class RelayCert:
    cert_path: Path
    key_path: Path
    #: Lowercase hex SHA-256 of the DER cert — what the browser pins.
    sha256: str
    not_after: _dt.datetime

    @property
    def hashes_for_browser(self) -> list[dict]:
        """The ``serverCertificateHashes`` value a browser expects."""
        return [{"algorithm": "sha-256", "value": self.sha256}]


def _fingerprint(der: bytes) -> str:
    return hashlib.sha256(der).hexdigest()


def load(cert_path: Path, key_path: Path) -> RelayCert | None:
    """Read an existing pair, or None if absent/unreadable/expiring."""
    try:
        from cryptography import x509
    except ImportError:
        return None
    if not (cert_path.exists() and key_path.exists()):
        return None
    try:
        cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
    except Exception:
        _LOG.warning("Could not parse %s; minting a fresh one", cert_path)
        return None

    not_after = cert.not_valid_after_utc
    remaining = not_after - _dt.datetime.now(_dt.timezone.utc)
    if remaining < _dt.timedelta(days=ROTATE_WHEN_REMAINING_DAYS):
        _LOG.info(
            "Relay certificate expires in %s; rotating", remaining
        )
        return None
    from cryptography.hazmat.primitives.serialization import Encoding

    return RelayCert(
        cert_path=cert_path,
        key_path=key_path,
        sha256=_fingerprint(cert.public_bytes(Encoding.DER)),
        not_after=not_after,
    )


def mint(cert_path: Path, key_path: Path, hosts: list[str]) -> RelayCert:
    """Mint an ECDSA P-256 self-signed cert valid for :data:`MAX_VALIDITY_DAYS`."""
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.x509.oid import NameOID
    except ImportError as exc:  # pragma: no cover - dependency-gated
        raise CertificateUnavailable(
            "The embedded teleop relay needs a TLS certificate, which needs "
            "`cryptography`. Install it with: pip install 'interlatent[teleop-relay]'"
        ) from exc

    key = ec.generate_private_key(ec.SECP256R1())
    now = _dt.datetime.now(_dt.timezone.utc)
    not_after = now + _dt.timedelta(days=MAX_VALIDITY_DAYS)

    alt_names: list[x509.GeneralName] = []
    for host in hosts:
        host = host.strip()
        if not host:
            continue
        try:
            alt_names.append(x509.IPAddress(ipaddress.ip_address(host)))
        except ValueError:
            alt_names.append(x509.DNSName(host))
    if not alt_names:
        alt_names.append(x509.DNSName("localhost"))

    subject = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "interlatent-teleop-relay"),
    ])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        # Backdate a minute so a mildly-skewed headset clock does not reject a
        # cert that was valid the instant it was minted.
        .not_valid_before(now - _dt.timedelta(minutes=1))
        .not_valid_after(not_after)
        .add_extension(x509.SubjectAlternativeName(alt_names), critical=False)
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None), critical=True
        )
        .sign(key, hashes.SHA256())
    )

    cert_path.parent.mkdir(parents=True, exist_ok=True)
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    # The private key is a credential: create it 0600 rather than chmod after.
    fd = os.open(str(key_path), os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ))

    digest = _fingerprint(cert.public_bytes(serialization.Encoding.DER))
    _LOG.info(
        "Minted relay certificate for %s (sha256 %s…, expires %s)",
        ", ".join(hosts) or "localhost", digest[:16], not_after.date(),
    )
    return RelayCert(cert_path, key_path, digest, not_after)


def ensure(cert_dir: Path, hosts: list[str]) -> RelayCert:
    """Load a still-valid pair, or mint one. The rotation entry point."""
    cert_path = cert_dir / "relay-cert.pem"
    key_path = cert_dir / "relay-key.pem"
    existing = load(cert_path, key_path)
    if existing is not None:
        return existing
    return mint(cert_path, key_path, hosts)
