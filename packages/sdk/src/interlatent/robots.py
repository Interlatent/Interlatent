"""Resolve installed robot embodiment data by ``robot_kind``.

Robot data (URDF, ``ik_config.json``, ``kinematic_spec.json``, ``meshes.lock``)
ships in the SDK wheel as ``interlatent_robots/<kind>/``, one data-only subpackage
per kind. This module is the read side: given a kind, find its installed data and
hand back paths or parsed JSON, uniformly, whether or not the wheel happens to be
unpacked on disk.

Why the distinct top-level ``interlatent_robots`` rather than data inside
``interlatent``: the SDK and the internal ``interlatent-engine`` are both the
top-level package ``interlatent`` and collide on install, so a name of our own
keeps the data reachable from either. It is also a PEP 420 namespace that the SDK
never claims (no ``interlatent_robots/__init__.py``), so a kind can move into its
own distribution later without changing a single import — the resolution below
walks the namespace and does not care which distribution provides a kind.

The data is ~18 KB per kind and ships for every kind, so ``pip install interlatent``
is enough to resolve one. The per-kind extras (``interlatent[yam]``) carry that
robot's *driver* dependencies, not its data.

IK needs no geometry (it is a function of the joint tree), so the shipped URDF is
kinematics-only and by default no meshes are involved anywhere — a kind ships no
``meshes.lock`` and :func:`has_meshes` is False for it. The mesh machinery remains
for a kind that genuinely needs STLs later (a 3D preview, sim, collision-aware
retargeting): if a ``meshes.lock`` is present, :func:`ensure_meshes` resolves it on
demand and verifies every file by ``sha256``, and :func:`ensure_bundle` lays out a
real-filesystem bundle dir (URDF + configs, plus ``assets/`` meshes only when a lock
exists).
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.resources as ir
import importlib.util
import json
import logging
import os
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_NAMESPACE = "interlatent_robots"
IK_CONFIG_FILENAME = "ik_config.json"
KINEMATIC_SPEC_FILENAME = "kinematic_spec.json"
MESHES_LOCK_FILENAME = "meshes.lock"


class RobotDataError(RuntimeError):
    """Robot data for a kind is missing, incomplete, or fails verification."""


def _pkg(kind: str) -> str:
    return f"{_NAMESPACE}.{kind}"


def is_installed(kind: str) -> bool:
    """True if ``interlatent_robots.<kind>`` is importable in this environment."""
    try:
        return importlib.util.find_spec(_pkg(kind)) is not None
    except (ImportError, ValueError):
        return False


def available() -> list[str]:
    """Every installed robot kind, sorted.

    Walks the ``interlatent_robots`` namespace across all distributions that
    contribute to it — empty (not an error) when none is installed.
    """
    try:
        ns = importlib.import_module(_NAMESPACE)
    except ImportError:
        return []
    kinds: set[str] = set()
    for entry in getattr(ns, "__path__", []):
        p = Path(entry)
        if not p.is_dir():
            continue
        for child in p.iterdir():
            if child.is_dir() and (child / "__init__.py").exists():
                kinds.add(child.name)
    return sorted(kinds)


def _require(kind: str):
    if not is_installed(kind):
        hint = ", ".join(available()) or "none"
        raise RobotDataError(
            f"no robot data for kind {kind!r}: install it with "
            f"`pip install interlatent[{kind}]` (installed: {hint})"
        )
    return ir.files(_pkg(kind))


def data_dir(kind: str) -> Path:
    """Filesystem dir holding ``kind``'s installed data files.

    Assumes a normally-unpacked install (wheels are, by default). Use the
    ``load_*`` helpers for content if you don't need real paths.
    """
    root = _require(kind)
    return Path(str(root))


def urdf_path(kind: str) -> Path:
    root = _require(kind)
    urdfs = sorted(f for f in root.iterdir() if f.name.endswith(".urdf"))
    if not urdfs:
        raise RobotDataError(f"robot data for {kind!r} ships no .urdf")
    if len(urdfs) > 1:
        raise RobotDataError(f"robot data for {kind!r} ships {len(urdfs)} .urdf files")
    return Path(str(urdfs[0]))


def _load_json(kind: str, name: str) -> dict:
    root = _require(kind)
    res = root / name
    if not res.is_file():
        raise RobotDataError(f"robot data for {kind!r} is missing {name}")
    return json.loads(res.read_text(encoding="utf-8"))


def load_ik_config(kind: str) -> dict:
    """Parsed ``ik_config.json`` (raw dict — the hand-authored tuning surface)."""
    return _load_json(kind, IK_CONFIG_FILENAME)


def load_kinematic_spec(kind: str) -> dict:
    """Parsed ``kinematic_spec.json`` (the browser IK serial-chain descriptor)."""
    return _load_json(kind, KINEMATIC_SPEC_FILENAME)


@dataclass(frozen=True)
class RobotData:
    kind: str
    urdf_path: Path
    ik_config: dict
    kinematic_spec: dict


def load(kind: str) -> RobotData:
    """Everything but the meshes, in one call."""
    return RobotData(
        kind=kind,
        urdf_path=urdf_path(kind),
        ik_config=load_ik_config(kind),
        kinematic_spec=load_kinematic_spec(kind),
    )


# ---------------------------------------------------------------------------
# Meshes: pinned by meshes.lock, fetched + sha256-verified on demand.
# ---------------------------------------------------------------------------

def _default_cache_root() -> Path:
    env = os.environ.get("INTERLATENT_ROBOT_CACHE")
    base = Path(env) if env else Path.home() / ".cache" / "interlatent" / "robots"
    return base


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def has_meshes(kind: str) -> bool:
    root = _require(kind)
    return (root / MESHES_LOCK_FILENAME).is_file()


def ensure_meshes(kind: str, dest: Optional[Path] = None) -> Path:
    """Fetch + verify ``kind``'s meshes into ``dest``; return the mesh dir.

    Reads ``meshes.lock``, downloads any file whose local ``sha256`` doesn't
    already match, and verifies every file after. Idempotent: a second call with
    the cache warm does no network I/O. Raises :class:`RobotDataError` on a hash
    mismatch — a corrupted or upstream-mutated mesh must not be trusted.

    ``dest`` defaults to ``$INTERLATENT_ROBOT_CACHE`` (or ``~/.cache/interlatent/
    robots``)``/<kind>``. The meshes land under the lock's ``dest_subdir`` (the
    URDF references them as e.g. ``assets/<name>.stl``).
    """
    lock = _load_json(kind, MESHES_LOCK_FILENAME)
    base_url = str(lock.get("base_url", "")).rstrip("/")
    subdir = str(lock.get("dest_subdir", "assets"))
    meshes = lock.get("meshes") or []
    if not base_url or not meshes:
        raise RobotDataError(f"{kind}/meshes.lock: needs base_url and a non-empty meshes list")

    dest = Path(dest) if dest else _default_cache_root() / kind
    mesh_dir = dest / subdir
    mesh_dir.mkdir(parents=True, exist_ok=True)

    for m in meshes:
        name = m["name"]
        want = m["sha256"]
        out = mesh_dir / name
        if _safe_name(name) != name:
            raise RobotDataError(f"{kind}/meshes.lock: unsafe mesh name {name!r}")
        if out.is_file() and _sha256(out) == want:
            continue
        url = f"{base_url}/{name}"
        log.info("fetching mesh %s for %s", name, kind)
        tmp = out.with_suffix(out.suffix + ".part")
        with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:  # noqa: S310
            shutil.copyfileobj(resp, f)
        got = _sha256(tmp)
        if got != want:
            tmp.unlink(missing_ok=True)
            raise RobotDataError(
                f"{kind} mesh {name}: sha256 mismatch (want {want[:12]}…, got {got[:12]}…) "
                f"from {url}"
            )
        os.replace(tmp, out)

    return mesh_dir


def _safe_name(name: str) -> str:
    # Mesh names are filenames, never paths — reject traversal/separators.
    return Path(name).name


def ensure_bundle(kind: str, dest: Optional[Path] = None) -> Path:
    """Materialize a complete on-disk bundle dir and return its path.

    Copies the URDF + both configs out of the (possibly read-only) install and
    fetches the verified meshes beside them, so the result is a self-contained
    directory a MuJoCo loader can open directly:

        <dest>/<robot>.urdf
        <dest>/ik_config.json
        <dest>/kinematic_spec.json
        <dest>/assets/*.stl

    This is the standalone-teleop / pod path. A node that only forwards the
    kinematic spec to the browser should use :func:`load_kinematic_spec`
    instead — it needs no meshes and no MuJoCo.
    """
    dest = Path(dest) if dest else _default_cache_root() / kind / "bundle"
    dest.mkdir(parents=True, exist_ok=True)
    root = _require(kind)
    shutil.copy2(str(urdf_path(kind)), dest / urdf_path(kind).name)
    for name in (IK_CONFIG_FILENAME, KINEMATIC_SPEC_FILENAME):
        res = root / name
        if res.is_file():
            (dest / name).write_text(res.read_text(encoding="utf-8"), encoding="utf-8")
    if has_meshes(kind):
        ensure_meshes(kind, dest=dest)
    return dest


__all__ = [
    "RobotData", "RobotDataError",
    "available", "is_installed",
    "data_dir", "urdf_path",
    "load", "load_ik_config", "load_kinematic_spec",
    "has_meshes", "ensure_meshes", "ensure_bundle",
    "IK_CONFIG_FILENAME", "KINEMATIC_SPEC_FILENAME", "MESHES_LOCK_FILENAME",
]
