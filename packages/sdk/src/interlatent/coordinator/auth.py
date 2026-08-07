"""The coordinator is the authority for the keys it issues.

Three principals exist, distinguished by key prefix:

``ilop_``
    The **operator** key. Minted once by ``interlatent up``, stored 0600, and
    presented by the CLI, by ``interlatent-serve``, and by anyone
    administering the deployment. This is the root credential.
``ilnode_``
    A **node** token, minted at pair time. Scoped to exactly one node: node A's
    token is rejected on node B's routes.
``ilbox_``
    A **box** key, minted at registration. Scoped to one GPU box.

Deleted ADR 0001 shipped an unauthenticated ``/admin/*`` on ``0.0.0.0`` with
"the network is the trust boundary" as the rationale. ADR 0023 deliberately
reversed that stance for the GPU port, and ADR 0038 keeps the reversal: anyone
who can reach the port could otherwise assign a session and move your arm.

**Only hashes are persisted.** The coordinator's state file holds
``sha256(key)``; the sole plaintext secret on disk is the operator key itself,
0600, because the CLI has to present it.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "OPERATOR_PREFIX",
    "NODE_PREFIX",
    "BOX_PREFIX",
    "KIND_OPERATOR",
    "KIND_NODE",
    "KIND_BOX",
    "Principal",
    "default_operator_key_path",
    "ensure_operator_key",
    "load_operator_key",
    "mint_key",
    "hash_key",
    "key_matches",
]

OPERATOR_PREFIX = "ilop_"
NODE_PREFIX = "ilnode_"
BOX_PREFIX = "ilbox_"

KIND_OPERATOR = "operator"
KIND_NODE = "node"
KIND_BOX = "box"

_PREFIX_BY_KIND = {
    KIND_OPERATOR: OPERATOR_PREFIX,
    KIND_NODE: NODE_PREFIX,
    KIND_BOX: BOX_PREFIX,
}


@dataclass(frozen=True)
class Principal:
    """Who a presented key belongs to. ``None`` means "not a key we issued"."""

    kind: str
    node_id: str | None = None
    box_id: str | None = None

    @property
    def is_operator(self) -> bool:
        return self.kind == KIND_OPERATOR


def default_operator_key_path() -> Path:
    override = os.environ.get("INTERLATENT_OPERATOR_KEY_FILE", "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".interlatent" / "operator.key"


def mint_key(kind: str = KIND_OPERATOR) -> str:
    """A fresh key. 24 bytes of ``secrets`` entropy behind a kind prefix."""
    try:
        prefix = _PREFIX_BY_KIND[kind]
    except KeyError:
        raise ValueError(f"unknown principal kind {kind!r}") from None
    return prefix + secrets.token_hex(24)


def hash_key(key: str) -> str:
    """What gets persisted. Plain SHA-256: these are 192-bit random tokens,
    not passwords, so there is nothing for a KDF to slow down."""
    return hashlib.sha256(key.strip().encode()).hexdigest()


def key_matches(presented: str, expected_hash: str) -> bool:
    return hmac.compare_digest(hash_key(presented), expected_hash)


def load_operator_key(path: Path | None = None) -> str | None:
    path = Path(path) if path is not None else default_operator_key_path()
    try:
        value = path.read_text().strip()
    except (OSError, ValueError):
        return None
    return value or None


def ensure_operator_key(path: Path | None = None) -> tuple[str, bool]:
    """Return ``(key, created)``, minting one if the file does not exist.

    Written with ``O_CREAT | O_EXCL | O_WRONLY`` at mode 0600 rather than
    write-then-``chmod``: the latter leaves a window in which the key is
    world-readable, and ``O_EXCL`` additionally makes a concurrent second
    ``interlatent up`` lose the race cleanly instead of clobbering the key the
    first one just handed out.
    """
    path = Path(path) if path is not None else default_operator_key_path()
    existing = load_operator_key(path)
    if existing:
        return existing, False

    path.parent.mkdir(parents=True, exist_ok=True)
    key = mint_key(KIND_OPERATOR)
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        # Raced with another `up`, or the file existed but was empty/unreadable.
        raced = load_operator_key(path)
        if raced:
            return raced, False
        raise
    with os.fdopen(fd, "w") as fh:
        fh.write(key + "\n")
    return key, True
