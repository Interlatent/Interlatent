#!/usr/bin/env python3
"""Build one ``interlatent-robot-<kind>`` wheel from a ``robots/<kind>/`` source dir.

The robot data can't live inside the ``interlatent`` package: the SDK and the
internal ``interlatent-engine`` are *both* the top-level import package
``interlatent`` and collide on install, so a pod running the engine could never
also ``pip install interlatent[<kind>]`` to get the data. Each robot kind is
therefore its own distribution, ``interlatent-robot-<kind>``, contributing to a
shared **namespace** package ``interlatent_robots`` (PEP 420 — no top-level
``__init__.py``), so any number of them coexist in one environment:

    interlatent_robots/            (namespace, owned by no single wheel)
        <kind>/                    (this wheel)
            __init__.py
            <robot>.urdf
            ik_config.json
            kinematic_spec.json
            meshes.lock

``pip install interlatent[<kind>]`` pulls the matching wheel via an extra; the
SDK's :mod:`interlatent.robots` resolver reads it back with
``importlib.resources``. Meshes are *not* vendored — ``meshes.lock`` pins them and
``interlatent.robots.ensure_meshes`` fetches on demand.

This is pure-data packaging: no compiled deps, no MuJoCo, so it builds anywhere.

Usage::

    python packaging/build_robot_wheel.py nori
    python packaging/build_robot_wheel.py nori --version 0.2.0 --outdir dist/
    python packaging/build_robot_wheel.py --all
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ROBOTS_DIR = REPO_ROOT / "robots"

# A kind is also a Python package-name component and a PyPI distribution suffix,
# so hold it to what all three accept: lowercase, dot/underscore-free.
_KIND_RE = re.compile(r"^[a-z][a-z0-9_]*$")

# Files carried into the wheel. The URDF basename varies per robot, so it is
# matched by suffix rather than name.
_DATA_GLOBS = ("*.urdf", "ik_config.json", "kinematic_spec.json", "meshes.lock")

_PYPROJECT = """\
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "interlatent-robot-{kind}"
version = "{version}"
description = "Robot data (URDF, IK config, kinematic spec) for the '{kind}' interlatent teleop embodiment."
requires-python = ">=3.9"
license = "Apache-2.0"
readme = "README.md"

[tool.setuptools.packages.find]
where = ["src"]
include = ["interlatent_robots*"]
namespaces = true

[tool.setuptools.package-data]
"interlatent_robots.{kind}" = ["*.urdf", "*.json", "*.lock"]
"""

_INIT = '''\
"""Robot data for the {kind!r} interlatent teleop embodiment.

A data-only subpackage under the ``interlatent_robots`` namespace. Do not import
it directly for paths — go through :func:`interlatent.robots.load` / ``data_dir``,
which resolve any installed kind uniformly via ``importlib.resources``.
"""
KIND = {kind!r}
'''

_README = """\
# interlatent-robot-{kind}

Robot data for the `{kind}` teleop embodiment: URDF, `ik_config.json`,
`kinematic_spec.json`, and `meshes.lock`. Installed via `pip install interlatent[{kind}]`
and read back through `interlatent.robots`. Collision meshes are fetched on demand
from `meshes.lock`, not shipped here. Built from `robots/{kind}/` by
`packaging/build_robot_wheel.py` — do not edit an installed copy.
"""


def _default_version(src: Path) -> str:
    """Version pinned in ``robots/<kind>/VERSION`` if present, else ``0.1.0``."""
    vf = src / "VERSION"
    if vf.exists():
        return vf.read_text(encoding="utf-8").strip()
    return "0.1.0"


def _collect_data(src: Path) -> list[Path]:
    files: list[Path] = []
    for pat in _DATA_GLOBS:
        files.extend(sorted(src.glob(pat)))
    return files


def _validate(kind: str, src: Path, files: list[Path]) -> None:
    if not _KIND_RE.match(kind):
        raise SystemExit(
            f"robot kind {kind!r} must match {_KIND_RE.pattern} "
            "(lowercase, digits, underscore; it becomes a package name)"
        )
    names = {f.name for f in files}
    urdfs = [f for f in files if f.suffix == ".urdf"]
    if not urdfs:
        raise SystemExit(f"robots/{kind}/: no .urdf found")
    if len(urdfs) > 1:
        raise SystemExit(f"robots/{kind}/: expected one .urdf, found {len(urdfs)}")
    for required in ("ik_config.json", "kinematic_spec.json"):
        if required not in names:
            raise SystemExit(
                f"robots/{kind}/: missing {required}. "
                + (
                    "Generate it with the MuJoCo kinematic_spec exporter — a bundle "
                    "without it makes the arms do nothing."
                    if required == "kinematic_spec.json"
                    else "It is the hand-authored IK/tuning config."
                )
            )
    # meshes.lock is optional (a mesh-free URDF is legal) but if present must parse.
    lock = src / "meshes.lock"
    if lock.exists():
        try:
            json.loads(lock.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise SystemExit(f"robots/{kind}/meshes.lock: invalid JSON: {e}")


def build_wheel(kind: str, *, version: str | None, outdir: Path) -> Path:
    src = ROBOTS_DIR / kind
    if not src.is_dir():
        raise SystemExit(f"no such robot source dir: robots/{kind}/")
    files = _collect_data(src)
    _validate(kind, src, files)
    version = version or _default_version(src)

    outdir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        build_root = Path(td)
        pkg = build_root / "src" / "interlatent_robots" / kind
        pkg.mkdir(parents=True)
        # No interlatent_robots/__init__.py — the namespace must stay unowned so
        # sibling interlatent-robot-* wheels can each contribute their own kind.
        (pkg / "__init__.py").write_text(_INIT.format(kind=kind), encoding="utf-8")
        for f in files:
            shutil.copy2(f, pkg / f.name)
        (build_root / "pyproject.toml").write_text(
            _PYPROJECT.format(kind=kind, version=version), encoding="utf-8"
        )
        (build_root / "README.md").write_text(_README.format(kind=kind), encoding="utf-8")

        # `pip wheel` drives the pyproject build backend without needing the
        # `build` package installed.
        subprocess.run(
            [sys.executable, "-m", "pip", "wheel", "--no-deps",
             "--wheel-dir", str(outdir), str(build_root)],
            check=True,
        )

    hits = sorted(outdir.glob(f"interlatent_robot_{kind}-{version}-*.whl"))
    if not hits:
        raise SystemExit(f"build produced no wheel for {kind} {version}")
    wheel = hits[-1]
    print(f"built {wheel.name}  ({', '.join(f.name for f in files)})")
    return wheel


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("kind", nargs="?", help="robot kind (a robots/<kind>/ dir)")
    ap.add_argument("--all", action="store_true", help="build every robots/<kind>/")
    ap.add_argument("--version", help="override wheel version (default: robots/<kind>/VERSION or 0.1.0)")
    ap.add_argument("--outdir", type=Path, default=REPO_ROOT / "dist",
                    help="wheel output dir (default: ./dist)")
    args = ap.parse_args()

    if args.all:
        kinds = sorted(p.name for p in ROBOTS_DIR.iterdir()
                       if p.is_dir() and (p / "ik_config.json").exists())
        if not kinds:
            raise SystemExit("no buildable robots/<kind>/ dirs found")
        for k in kinds:
            build_wheel(k, version=args.version, outdir=args.outdir)
    elif args.kind:
        build_wheel(args.kind, version=args.version, outdir=args.outdir)
    else:
        ap.error("give a robot kind or --all")


if __name__ == "__main__":
    main()
