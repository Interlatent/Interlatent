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
        "interlatent.lerobot.sync_inference",  # interlatent-sync-rollout
        "interlatent.inference.integration",   # interlatent-rollout
        "interlatent.storage",
    ):
        assert needed in pkgs, f"{needed} missing from sdk wheel (no __init__.py?)"


def test_server_wheel_contains_all_packages():
    pkgs = set(find_packages(str(REPO / "packages" / "server" / "src")))
    for needed in (
        "interlatent_server.server",    # interlatent-serve
        "interlatent_server.protocol",
        "interlatent_server.storage",
    ):
        assert needed in pkgs, f"{needed} missing from server wheel (no __init__.py?)"


def test_teleop_wheel_contains_all_packages():
    pkgs = set(find_packages(str(REPO / "packages" / "teleop" / "src")))
    for needed in (
        "interlatent_teleop.laptop",    # interlatent-teleop-laptop / -keyboard
        "interlatent_teleop.pi",        # interlatent-teleop-pi
        "interlatent_teleop.protocol",
        "interlatent_teleop.common",
    ):
        assert needed in pkgs, f"{needed} missing from teleop wheel (no __init__.py?)"
