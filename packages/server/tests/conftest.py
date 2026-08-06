"""Keep the real-lerobot tests out of the default run.

Most of this suite tests the dataset writers with ``lerobot`` stubbed into
``sys.modules`` — the only way to cover them without dragging torch into
CI. ``test_lerobot_real_build.py`` and ``test_drtc_end_to_end.py`` do the
opposite: they import the real thing.

Those two stances cannot share a process. Once real lerobot (and torch) is
imported, ``test_import_surface``'s "nothing heavy at import time"
assertion is false by construction, the stub suites' ``sys.modules``
surgery operates on a half-real package, and torch raises internal
assertion failures when its module table is swapped underneath it. It has
to be prevented at *collection* — a skip is too late, because the
module-level imports have already run by then.

Three ways to get them:

    pytest packages/server/tests/ --real-lerobot        # whole dir
    pytest packages/server/tests/test_drtc_end_to_end.py  # named explicitly
    (CI: the `server-lerobot` job, which runs only these files)

Naming a file explicitly counts as asking for it. Collection-finish prints
what was excluded either way — an opt-in test that quietly never runs is
worse than no test, and CI additionally greps for the two assertions that
must have executed.
"""
from __future__ import annotations

from pathlib import Path

_FLAG = "--real-lerobot"

# Files that import the real lerobot at module scope.
REAL_LEROBOT_FILES = {
    "test_lerobot_real_build.py",
    "test_drtc_end_to_end.py",
}


def pytest_addoption(parser) -> None:
    parser.addoption(
        _FLAG,
        action="store_true",
        default=False,
        help="Collect the tests that import the real lerobot. Give them "
             "their own pytest process — they cannot share one with the "
             "suites that stub lerobot in sys.modules.",
    )


def _explicitly_requested(path: Path, config) -> bool:
    """True when the user named this file (or something inside it) on the
    command line, rather than sweeping a directory."""
    for arg in config.invocation_params.args:
        head = str(arg).split("::", 1)[0]
        if not head:
            continue
        try:
            if Path(head).resolve() == path.resolve():
                return True
        except OSError:  # pragma: no cover — malformed arg
            continue
    return False


def pytest_ignore_collect(collection_path, config):
    if collection_path.name not in REAL_LEROBOT_FILES:
        return None
    if config.getoption(_FLAG) or _explicitly_requested(collection_path, config):
        return None
    return True


def pytest_report_collectionfinish(config, items):
    if config.getoption(_FLAG):
        return f"real-lerobot tests ENABLED ({_FLAG})"
    collected = {Path(str(i.fspath)).name for i in items}
    excluded = sorted(REAL_LEROBOT_FILES - collected)
    if not excluded:
        return None
    return (
        f"NOT collected (need {_FLAG}, and a pytest process of their own): "
        + ", ".join(excluded)
    )
