"""Every module in `interlatent_server` must import on a base install.

This is the test that was missing when the serving stack moved out of
`interlatent-engine` (ADR 0035, platform repo). The move kept the code byte-identical
but changed its package depth, and two relative imports written for the
old layout — `...cloud.box_status` and `...storage.lerobot_*`, valid as
`interlatent.cloud` / `interlatent.storage` — resolved past the top of
`interlatent_server` and raised ImportError at import time. The dist
built, installed, and passed `twine check`; it just could not run.

A full walk of the package catches that class of breakage, plus any
future module that grows a heavy top-level import. Base install only:
no torch, no lerobot. The policy backends must keep deferring those
(a box with no policy still has to record).
"""
from __future__ import annotations

import importlib
import pkgutil
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import interlatent_server  # noqa: E402


def _all_modules() -> list[str]:
    return sorted(
        m.name
        for m in pkgutil.walk_packages(
            interlatent_server.__path__, prefix="interlatent_server."
        )
    )


@pytest.mark.parametrize("module", _all_modules())
def test_module_imports(module: str) -> None:
    importlib.import_module(module)


def test_walk_found_the_expected_subpackages() -> None:
    """Guard the guard: if the walk silently found nothing, every
    parametrized case above would vacuously pass."""
    found = set(_all_modules())
    for expected in (
        "interlatent_server.cli",
        "interlatent_server.serve_gpu",
        "interlatent_server.credentials",
        "interlatent_server.box_status",
        "interlatent_server.protocol.messages_pb2",
        "interlatent_server.server.transport",
        "interlatent_server.server.recorder",
        "interlatent_server.storage.lerobot_rebuild",
        "interlatent_server.storage.lerobot_live",
    ):
        assert expected in found, f"{expected} not found by the package walk"


def test_no_torch_or_lerobot_at_import_time() -> None:
    """The heavy deps stay lazy. `interlatent_server.server` imports the
    policy backends to register them; that registration must not drag
    torch or lerobot in, or a recording-only box pays a multi-second
    import (and a bare install cannot start at all)."""
    for mod in list(sys.modules):
        if mod.startswith("interlatent_server"):
            del sys.modules[mod]
    for heavy in ("torch", "lerobot"):
        sys.modules.pop(heavy, None)

    importlib.import_module("interlatent_server.server")

    assert "torch" not in sys.modules, "importing interlatent_server.server pulled in torch"
    assert "lerobot" not in sys.modules, "importing interlatent_server.server pulled in lerobot"
