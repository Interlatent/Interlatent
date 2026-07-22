"""Regression: every SDK module must import without the optional extras.

Module docstrings across the tree promise this — ``adapters/yam/robot.py``
("importing this module never requires the ``[yam]`` extra"),
``adapters/base.py`` and ``node/control.py`` ("importable on a barebones Pi"),
``adapters/axol/config.py``, ``node/smoothing.py``, and others. The daemon
relies on it: a Pi with only the base install must still be able to import
``interlatent.node.control`` and the adapter registry to decide *which* robot
loop to run, long before any extra is present.

Nothing verified that claim, and the CI smoke-import step only covered 4 of
the ~58 modules. One `import i2rt` hoisted out of a function and into module
scope would break every barebones node while CI stayed green.

The check imports each module in a subprocess with the extras-only packages
forced to look uninstalled, so it holds even on a developer machine that has
``lerobot`` or ``i2rt`` installed.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

# Top-level module names provided ONLY by an optional extra or a host-installed
# SDK. Importing one of these at module scope is what this test forbids.
EXTRAS_ONLY = [
    "i2rt",             # [yam]
    "pyrealsense2",     # [yam]  (x86_64-gated; absent on ARM by design)
    "cv2",              # [yam] / [axol]
    "lerobot",          # [lerobot]
    "huggingface_hub",  # [lerobot]
    "pyarrow",          # [lerobot]
    "PIL",              # [lerobot]
    "almond_axol",      # [axol]
    "pyroki",           # [axol]
    "aioquic",          # [teleop-quic]
    "pyzmq",            # [nori]
    "zmq",              # [nori]  (pyzmq's import name)
    "pyzed",            # host-installed ZED SDK, not on PyPI
]

# Modules that legitimately require an extra and make no extras-free claim.
# Keep this list short and justified — every entry is a module a barebones
# node can never import.
EXEMPT = {
    # THE isolation boundary for [teleop-quic]: this module exists so that
    # aioquic is imported in exactly one clearly-marked place, and only in the
    # QUIC child process (`_quic_proc.py` imports it inside a function).
    # factory.py gates on `importlib.util.find_spec("aioquic")` and the parent
    # process never touches it. Exempting it is the point of the file, not a
    # concession - the modules around it are what this suite protects.
    "interlatent.node.teleop._quic_client",
}

# Declared in [project].dependencies — always present in a real install, so a
# failure naming one of these is an environment problem, not an invariant
# violation. (grpcio and torch have no wheels for every platform a developer
# might run this on.)
BASE_DEPS = ["requests", "torch", "numpy", "httpx", "grpc", "grpc_tools",
             "google", "sonora", "websockets"]

_CHECKER = textwrap.dedent(
    '''
    import importlib, json, sys
    from pathlib import Path

    extras   = set(json.loads(sys.argv[1]))
    exempt   = set(json.loads(sys.argv[2]))
    base     = set(json.loads(sys.argv[3]))
    root_override = sys.argv[4] if len(sys.argv) > 4 else ""

    if root_override:
        sys.path.insert(0, root_override)

    import interlatent
    pkg_root = Path(root_override or interlatent.__path__[0])
    if root_override:
        pkg_root = pkg_root / "interlatent"

    # Discover by filesystem so discovery itself never imports anything.
    names = {"interlatent"}
    for path in pkg_root.rglob("*.py"):
        rel = path.relative_to(pkg_root.parent).with_suffix("")
        parts = list(rel.parts)
        if parts[-1] == "__init__":
            parts.pop()
        if not parts or any(p.startswith(".") for p in parts):
            continue
        names.add(".".join(parts))

    class Blocker:
        """Make the extras look uninstalled, wherever this runs."""
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in extras:
                raise ModuleNotFoundError(
                    "No module named %r [blocked: extras-free import check]" % name,
                    name=name.split(".")[0],
                )
            return None

    sys.meta_path.insert(0, Blocker())

    violations, env_limited = [], []
    for name in sorted(names):
        if name in exempt:
            continue
        for cached in [m for m in sys.modules if m.split(".")[0] in
                       ({"interlatent"} | extras)]:
            del sys.modules[cached]
        try:
            importlib.import_module(name)
        except ModuleNotFoundError as exc:
            missing = (getattr(exc, "name", "") or "").split(".")[0]
            if missing in base:
                env_limited.append((name, missing))
            else:
                violations.append([name, missing or str(exc), str(exc)[:200]])
        except Exception as exc:  # noqa: BLE001 - report, don't mask
            violations.append([name, type(exc).__name__, str(exc)[:200]])

    print(json.dumps({"checked": len(names), "violations": violations,
                      "env_limited": env_limited}))
    '''
)


def _run_check(root: str = "") -> dict:
    import json

    argv = [sys.executable, "-c", _CHECKER,
            json.dumps(EXTRAS_ONLY), json.dumps(sorted(EXEMPT)),
            json.dumps(BASE_DEPS)]
    if root:
        argv.append(root)
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, (
        "extras-free import checker crashed:\n%s\n%s" % (proc.stdout, proc.stderr)
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_every_module_imports_without_extras():
    """No module may need an optional extra at import time."""
    result = _run_check()
    assert result["checked"] > 40, (
        "module discovery found only %d modules — the walk is broken, so this "
        "test would pass vacuously" % result["checked"]
    )
    if result["violations"]:
        lines = [
            "%s\n      needs %r at import time\n      %s" % (name, missing, msg)
            for name, missing, msg in result["violations"]
        ]
        pytest.fail(
            "%d module(s) import an optional extra at module scope.\n\n%s\n\n"
            "These modules promise to import on a base install (barebones Pi).\n"
            "Move the import inside the function that uses it. Do NOT add the\n"
            "extra to [project].dependencies to make this pass."
            % (len(result["violations"]), "\n\n".join("  " + ln for ln in lines))
        )


def test_checker_detects_a_violation(tmp_path):
    """Guard the guard: a checker that cannot fail is worse than no checker.

    Builds a throwaway package tree with a deliberate top-level ``import i2rt``
    and asserts the checker flags it. Without this, a broken walk (wrong root,
    empty discovery) would let the test above pass silently forever.
    """
    pkg = tmp_path / "interlatent"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "offender.py").write_text("import i2rt\n", encoding="utf-8")

    result = _run_check(root=str(tmp_path))
    offenders = [v[0] for v in result["violations"]]
    assert "interlatent.offender" in offenders, (
        "checker failed to flag a deliberate top-level `import i2rt`; "
        "it is not actually enforcing anything. got: %r" % (result,)
    )
