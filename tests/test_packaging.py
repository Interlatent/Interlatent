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
    ):
        assert needed in pkgs, f"{needed} missing from sdk wheel (no __init__.py?)"
