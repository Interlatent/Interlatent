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
import sys
from pathlib import Path

import pytest

from interlatent import robots


def _plant_kind(root: Path, kind: str, *, urdf="arm.urdf",
                ik=None, spec=None, meshes_lock=None) -> None:
    """Write a data-only ``interlatent_robots/<kind>/`` under ``root`` (a sys.path entry)."""
    pkg = root / "interlatent_robots" / kind
    pkg.mkdir(parents=True)
    # No interlatent_robots/__init__.py: it must stay a namespace package.
    (pkg / "__init__.py").write_text(f"KIND = {kind!r}\n", encoding="utf-8")
    (pkg / urdf).write_text("<robot/>", encoding="utf-8")
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
# The build script's validation (runs without building anything).
# ---------------------------------------------------------------------------

def _load_builder():
    path = (Path(__file__).resolve().parents[1] / "packaging" / "build_robot_wheel.py")
    spec = importlib.util.spec_from_file_location("_build_robot_wheel", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_build_validation_requires_spec(tmp_path):
    b = _load_builder()
    src = tmp_path / "x"
    src.mkdir()
    (src / "arm.urdf").write_text("<robot/>")
    (src / "ik_config.json").write_text("{}")
    files = b._collect_data(src)
    with pytest.raises(SystemExit, match="kinematic_spec.json"):
        b._validate("x", src, files)


def test_build_validation_rejects_bad_kind(tmp_path):
    b = _load_builder()
    src = tmp_path / "X-Bad"
    src.mkdir()
    for n in ("arm.urdf", "ik_config.json", "kinematic_spec.json"):
        (src / n).write_text("{}" if n.endswith(".json") else "<robot/>")
    files = b._collect_data(src)
    with pytest.raises(SystemExit, match="must match"):
        b._validate("X-Bad", src, files)


def test_real_nori_wheel_installs_here_or_skips():
    """If interlatent[nori] is installed in this env, its data must resolve."""
    if not robots.is_installed("nori"):
        pytest.skip("interlatent-robot-nori not installed in this environment")
    data = robots.load("nori")
    assert list(data.ik_config["chains"]) == ["left", "right"]
    assert robots.has_meshes("nori")
