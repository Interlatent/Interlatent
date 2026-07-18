"""interlatent.robots — resolving installed embodiment data by robot kind.

Network-free: a synthetic ``interlatent_robots.<kind>`` is planted on ``sys.path``
and meshes are served over ``file://``, so the sha256 fetch/verify path runs
without touching the real upstream.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

from interlatent import robots


def _plant_kind(root: Path, kind: str, *, urdf="arm.urdf",
                ik=None, spec=None, meshes_lock=None,
                include_ik: bool = True) -> None:
    """Write a data-only ``interlatent_robots/<kind>/`` under ``root`` (a sys.path entry).

    ``include_ik=True`` mimics a source checkout; ``False`` mimics a wheel
    install, where ik_config.json is excluded from package data (ADR 0017).
    """
    pkg = root / "interlatent_robots" / kind
    pkg.mkdir(parents=True)
    # No interlatent_robots/__init__.py: it must stay a namespace package.
    (pkg / "__init__.py").write_text(f"KIND = {kind!r}\n", encoding="utf-8")
    (pkg / urdf).write_text("<robot/>", encoding="utf-8")
    if include_ik:
        (pkg / "ik_config.json").write_text(json.dumps(ik or {"chains": {}}), encoding="utf-8")
    (pkg / "kinematic_spec.json").write_text(json.dumps(spec or {"version": 1}), encoding="utf-8")
    if meshes_lock is not None:
        (pkg / "meshes.lock").write_text(json.dumps(meshes_lock), encoding="utf-8")


@pytest.fixture
def planted(tmp_path, monkeypatch):
    """Install a synthetic robot kind 'testarm' importable for one test."""
    site = tmp_path / "site"
    site.mkdir()
    _plant_kind(site, "testarm", ik={"chains": {"left": {}, "right": {}}})
    monkeypatch.syspath_prepend(str(site))
    # Drop any cached view of the namespace so the new path entry is seen.
    for mod in [m for m in sys.modules if m == "interlatent_robots"
                or m.startswith("interlatent_robots.")]:
        del sys.modules[mod]
    importlib.invalidate_caches()
    yield site


def test_discovery_and_load(planted):
    assert "testarm" in robots.available()
    assert robots.is_installed("testarm")
    assert not robots.is_installed("nope")

    data = robots.load("testarm")
    assert data.kind == "testarm"
    assert data.urdf_path.name == "arm.urdf"
    assert list(data.ik_config["chains"]) == ["left", "right"]
    assert data.kinematic_spec["version"] == 1


def test_wheel_install_without_ik_config(tmp_path, monkeypatch):
    """A wheel install has no ik_config.json (repo-only curation source):
    load() degrades to ik_config=None, the spec still resolves, and
    load_ik_config raises with a pointer at the source repo."""
    site = tmp_path / "site"
    site.mkdir()
    _plant_kind(site, "wheelarm", include_ik=False)
    monkeypatch.syspath_prepend(str(site))
    for mod in [m for m in sys.modules if m.startswith("interlatent_robots")]:
        del sys.modules[mod]
    importlib.invalidate_caches()

    data = robots.load("wheelarm")
    assert data.ik_config is None
    assert data.kinematic_spec["version"] == 1
    with pytest.raises(robots.RobotDataError, match="repo-only"):
        robots.load_ik_config("wheelarm")


def test_missing_kind_names_the_extra(planted):
    with pytest.raises(robots.RobotDataError) as ei:
        robots.load("nope")
    assert "pip install interlatent[nope]" in str(ei.value)


def test_missing_spec_is_an_error(tmp_path, monkeypatch):
    site = tmp_path / "site"
    site.mkdir()
    pkg = site / "interlatent_robots" / "bare"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("KIND='bare'\n")
    (pkg / "arm.urdf").write_text("<robot/>")
    (pkg / "ik_config.json").write_text("{}")
    monkeypatch.syspath_prepend(str(site))
    for mod in [m for m in sys.modules if m.startswith("interlatent_robots")]:
        del sys.modules[mod]
    importlib.invalidate_caches()
    with pytest.raises(robots.RobotDataError, match="kinematic_spec.json"):
        robots.load_kinematic_spec("bare")


def test_ensure_meshes_fetches_and_verifies(tmp_path, monkeypatch):
    # A local "upstream" served over file://.
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    blob = b"solid stl\n" * 100
    (upstream / "part.stl").write_bytes(blob)
    digest = hashlib.sha256(blob).hexdigest()

    site = tmp_path / "site"
    site.mkdir()
    _plant_kind(site, "meshbot", meshes_lock={
        "lock_version": 1,
        "base_url": upstream.as_uri(),
        "dest_subdir": "assets",
        "meshes": [{"name": "part.stl", "size": len(blob), "sha256": digest}],
    })
    monkeypatch.syspath_prepend(str(site))
    for mod in [m for m in sys.modules if m.startswith("interlatent_robots")]:
        del sys.modules[mod]
    importlib.invalidate_caches()

    dest = tmp_path / "cache"
    mesh_dir = robots.ensure_meshes("meshbot", dest=dest)
    fetched = mesh_dir / "part.stl"
    assert fetched.read_bytes() == blob

    # Idempotent: a warm cache re-verifies rather than re-downloading. Point the
    # lock's upstream at nothing and it must still succeed from cache.
    (upstream / "part.stl").unlink()
    assert robots.ensure_meshes("meshbot", dest=dest) == mesh_dir


def test_ensure_meshes_rejects_tampered_hash(tmp_path, monkeypatch):
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    (upstream / "part.stl").write_bytes(b"real bytes")

    site = tmp_path / "site"
    site.mkdir()
    _plant_kind(site, "badhash", meshes_lock={
        "base_url": upstream.as_uri(),
        "dest_subdir": "assets",
        "meshes": [{"name": "part.stl", "size": 9, "sha256": "0" * 64}],
    })
    monkeypatch.syspath_prepend(str(site))
    for mod in [m for m in sys.modules if m.startswith("interlatent_robots")]:
        del sys.modules[mod]
    importlib.invalidate_caches()

    with pytest.raises(robots.RobotDataError, match="sha256 mismatch"):
        robots.ensure_meshes("badhash", dest=tmp_path / "cache")


# ---------------------------------------------------------------------------
# Every kind shipped in the SDK source tree is complete and well-named. This is
# the guard that used to live in the standalone wheel builder's _validate(); the
# data now ships in the SDK wheel, so it is checked against the real tree on
# every run instead of only at wheel-build time.
# ---------------------------------------------------------------------------

_SRC_KINDS_DIR = Path(__file__).resolve().parents[1] / "packages" / "sdk" / "src" / "interlatent_robots"
# A kind is a Python package-name component and the robot_kind the node reports,
# so hold it to what both accept.
_KIND_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _source_kinds() -> list[Path]:
    return sorted(p for p in _SRC_KINDS_DIR.iterdir()
                  if p.is_dir() and p.name != "__pycache__")


def test_source_tree_has_kinds():
    """Guard the guard: if the layout moves, the checks below must not silently
    pass by iterating an empty dir."""
    assert [p.name for p in _source_kinds()] == ["nori", "yam"]


@pytest.mark.parametrize("kind_dir", _source_kinds(), ids=lambda p: p.name)
def test_shipped_kind_is_complete(kind_dir):
    assert _KIND_RE.match(kind_dir.name), f"{kind_dir.name!r} must match {_KIND_RE.pattern}"
    names = {f.name for f in kind_dir.iterdir()}
    # A kind without kinematic_spec.json installs clean and then makes the arms
    # do nothing — the browser cannot build a solver from it. ik_config.json is
    # required in the repo tree (the curation source the spec is generated
    # from) even though it is excluded from the wheel (ADR 0017 amendment).
    for required in ("ik_config.json", "kinematic_spec.json", "__init__.py"):
        assert required in names, f"{kind_dir.name}/: missing {required}"
    urdfs = [f for f in kind_dir.iterdir() if f.suffix == ".urdf"]
    assert len(urdfs) == 1, f"{kind_dir.name}/: expected one .urdf, found {len(urdfs)}"
    for jf in ("ik_config.json", "kinematic_spec.json"):
        json.loads((kind_dir / jf).read_text(encoding="utf-8"))


def test_namespace_stays_unowned():
    """interlatent_robots must have no __init__.py: it is a PEP 420 namespace, so
    a kind can be split into its own distribution later without moving imports."""
    assert not (_SRC_KINDS_DIR / "__init__.py").exists()


def test_resolver_imports_clean_in_a_fresh_interpreter():
    """`interlatent.robots` must work in a process where nothing else has
    imported `importlib.util` first — pytest imports it, masking a missing
    `import importlib.util` that would otherwise crash `is_installed`."""
    import subprocess
    import sys as _sys
    r = subprocess.run(
        [_sys.executable, "-c",
         "from interlatent import robots; "
         "assert robots.is_installed('definitely-absent') is False; "
         "print('ok')"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert "ok" in r.stdout


def test_real_nori_data_resolves():
    """Robot data ships in the SDK wheel, so every kind resolves wherever the SDK
    is installed — no extra, no skip. `interlatent[nori]` adds the nori *driver*
    deps; the data is there either way. Install-agnostic: ik_config is absent on
    wheel installs, so the chains assertion reads the source tree directly."""
    assert robots.is_installed("nori")
    data = robots.load("nori")
    assert data.kinematic_spec
    # nori is kinematics-only: no meshes shipped or needed for IK.
    assert not robots.has_meshes("nori")
    src_ik = json.loads((_SRC_KINDS_DIR / "nori" / "ik_config.json").read_text(encoding="utf-8"))
    assert list(src_ik["chains"]) == ["left", "right"]
