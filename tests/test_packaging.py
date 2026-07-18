"""Regression: every console-script module must actually ship in the wheel.

The CI installs editable (`pip install -e`), which maps the whole src/
tree and silently masks subpackages that setuptools would exclude from a
real wheel for lack of an ``__init__.py``. That bug shipped once — the
``interlatent-node`` entry point pointed at a module that wasn't in the
wheel. This test runs the same package discovery setuptools uses at
build time.
"""
from pathlib import Path

from setuptools import find_packages

REPO = Path(__file__).resolve().parent.parent


def test_sdk_wheel_contains_all_entry_point_packages():
    pkgs = set(find_packages(str(REPO / "packages" / "sdk" / "src")))
    for needed in (
        "interlatent",
        "interlatent.node",                    # interlatent-node
        "interlatent.inference.integration",   # interlatent-rollout
        "interlatent.storage",
        "interlatent.cli",                     # interlatent
        "interlatent.adapters.yam",            # --robot yam native loop
        "interlatent.behaviors",               # interlatent.Robot / behavior ls|run
        "interlatent.adapters.nori",           # --robot nori native loop
        "interlatent.adapters.dimos",          # --robot dimos native loop
    ):
        assert needed in pkgs, f"{needed} missing from sdk wheel (no __init__.py?)"


def test_dimos_blueprint_entry_point_declared():
    """The dimos.blueprints entry point must stay in pyproject — dimos resolves
    `dimos run interlatent.xarm7` through it (namespace = distribution name)."""
    import tomllib

    pyproject = REPO / "packages" / "sdk" / "pyproject.toml"
    with open(pyproject, "rb") as fh:
        data = tomllib.load(fh)
    eps = data["project"]["entry-points"]["dimos.blueprints"]
    assert eps["xarm7"] == "interlatent.adapters.dimos.blueprints:xarm7"


def test_dimos_native_loop_registered():
    from interlatent.node.daemon import NodeDaemon

    assert (
        NodeDaemon._NATIVE_LOOPS["dimos"]
        == "interlatent.adapters.dimos:control_loop"
    )


def test_dimos_config_imports_without_dimos_installed():
    """config/kinds (and the adapter package itself) must never require the
    [dimos] extra at import time — the daemon imports lazily, and base
    installs list the loop in _NATIVE_LOOPS unconditionally."""
    import importlib

    for mod in (
        "interlatent.adapters.dimos",
        "interlatent.adapters.dimos.config",
        "interlatent.adapters.dimos.kinds",
        "interlatent.adapters.dimos.episode",
    ):
        importlib.import_module(mod)
