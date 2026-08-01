"""Regression: every console-script module must actually ship in the wheel.

The CI installs editable (`pip install -e`), which maps the whole src/
tree and silently masks subpackages that setuptools would exclude from a
real wheel for lack of an ``__init__.py``. That bug shipped once — the
``interlatent-node`` entry point pointed at a module that wasn't in the
wheel. This test runs the same package discovery setuptools uses at
build time.
"""
from pathlib import Path

import pytest
from setuptools import find_packages

REPO = Path(__file__).resolve().parent.parent


def test_sdk_wheel_contains_all_entry_point_packages():
    pkgs = set(find_packages(str(REPO / "packages" / "sdk" / "src")))
    for needed in (
        "interlatent",
        "interlatent.node",                    # interlatent-node
        "interlatent.inference.integration",   # interlatent-rollout
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


def test_dimos_blueprints_actually_build_against_the_installed_dimos():
    """Every declared entry point must RESOLVE, not merely parse.

    The rest of the dimos-blueprint coverage here is ``ast``-only (it reads the
    source and never imports it), which is how ``blueprints.py`` shipped
    importing ``a1z_hardware`` — a name no released dimos exports; the real one
    is ``make_a1z_hardware`` — and calling ``HardwareComponent`` with
    ``gripper_open_position``/``gripper_closed_position``, fields that dataclass
    does not have. Both kinds live in one module, so the A1Z failure also took
    ``dimos run interlatent.xarm7`` down: the only visible symptom was the
    tier-2 integration test failing on an *xarm7* stack.

    Skipped without the [dimos] extra; the import guard in ``blueprints.py``
    means a base install can never run this.
    """
    import importlib
    import tomllib

    pytest.importorskip("dimos", reason="[dimos] extra not installed")

    pyproject = REPO / "packages" / "sdk" / "pyproject.toml"
    with open(pyproject, "rb") as fh:
        eps = tomllib.load(fh)["project"]["entry-points"]["dimos.blueprints"]

    assert eps, "no dimos.blueprints entry points declared"
    for name, target in eps.items():
        mod_name, _, attr = target.partition(":")
        blueprint = getattr(importlib.import_module(mod_name), attr)
        assert blueprint is not None, f"entry point {name!r} resolved to None"


def test_dimos_blueprint_failure_is_scoped_to_one_kind():
    """One kind's vendor incompatibility must not take the others down.

    A1Z was authored against a Galaxea-enabled dimos branch and imported
    `a1z_hardware` at module scope. On a stock release that raised — and since
    both kinds lived in one module, it also killed `dimos run
    interlatent.xarm7`, which shares nothing with A1Z. Per-kind builders behind
    ``__getattr__`` are the fix; this pins it.
    """
    pytest.importorskip("dimos", reason="[dimos] extra not installed")
    from interlatent.adapters.dimos import blueprints as bp

    def _boom():
        raise ImportError("cannot import name 'a1z_hardware' [simulated]")

    saved_builders, saved_cache = dict(bp._BUILDERS), dict(bp._CACHE)
    try:
        bp._BUILDERS["a1z"] = _boom
        bp._CACHE.pop("a1z", None)

        assert bp.xarm7 is not None, "a broken a1z must not affect xarm7"

        with pytest.raises(ImportError) as excinfo:
            bp.a1z
        assert "'a1z'" in str(excinfo.value), "the error must name the failing kind"
    finally:
        bp._BUILDERS.clear()
        bp._BUILDERS.update(saved_builders)
        bp._CACHE.clear()
        bp._CACHE.update(saved_cache)


def test_dimos_mock_hardware_matches_the_installed_component_shape():
    """``_mock_hardware`` must not pass fields this dimos does not declare.

    The Galaxea branch's ``HardwareComponent`` carries gripper open/closed
    range fields; stock 0.0.14b1 has none. Passing them unconditionally was a
    TypeError on every hardware-free start, so the range is feature-detected.
    """
    pytest.importorskip("dimos", reason="[dimos] extra not installed")
    from interlatent.adapters.dimos import blueprints as bp

    hw = bp._mock_hardware(
        "arm", 6, has_gripper=True,
        gripper_open_position=1.0, gripper_closed_position=0.0,
    )
    assert hw.adapter_type == "mock"
    assert hw.gripper_joints == ["arm/gripper"]
    # Applied iff this dimos declares them — never blindly, never dropped when
    # the build does support them.
    supported = bp._component_accepts(
        "gripper_open_position", "gripper_closed_position"
    )
    assert hasattr(hw, "gripper_open_position") == supported
    if supported:
        assert hw.gripper_open_position == 1.0


def test_dimos_extra_covers_the_shipped_blueprint():
    """`pip install 'interlatent[dimos]'` must be able to RUN `dimos run
    interlatent.xarm7`, not just import the adapter. That takes dimos's
    [manipulation] extra (Viser/ManipulationModule in the blueprint) plus three
    packages dimos 0.0.14b1's websocket_vis module imports but never declares
    (python-socketio, starlette, uvicorn). A bare `dimos` pin shipped once and
    the blueprint entry point died on ImportError for every fresh install."""
    import tomllib

    pyproject = REPO / "packages" / "sdk" / "pyproject.toml"
    with open(pyproject, "rb") as fh:
        data = tomllib.load(fh)
    extra = data["project"]["optional-dependencies"]["dimos"]

    assert any(req.startswith("dimos[manipulation]") for req in extra), extra
    for undeclared in ("python-socketio", "starlette", "uvicorn"):
        assert any(req.startswith(undeclared) for req in extra), (
            f"{undeclared} missing from the [dimos] extra — it is an undeclared "
            "import of dimos's camera/websocket_vis modules (as of 0.0.14b1)"
        )


def test_dimos_xarm7_blueprint_declares_manipulation_with_viser():
    """The reference stack must retain the hardware-free visual test path:
    DIMOS's planner/manipulation module renders mock coordinator state in
    Viser, while the exclusive servo task remains the only execution task."""
    import ast

    blueprint = (
        REPO
        / "packages"
        / "sdk"
        / "src"
        / "interlatent"
        / "adapters"
        / "dimos"
        / "blueprints.py"
    )
    tree = ast.parse(blueprint.read_text(encoding="utf-8"))
    planner_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "planner"
    ]
    assert len(planner_calls) == 1

    keywords = {keyword.arg: keyword.value for keyword in planner_calls[0].keywords}
    visualization = ast.literal_eval(keywords["visualization"])
    assert visualization == {"backend": "viser"}

    source = blueprint.read_text(encoding="utf-8")
    assert "make_xarm7_model_config" in source
    assert 'update={"coordinator_task_name": None}' in source


def test_dimos_native_loop_registered():
    """`--robot dimos` must reach the dimos shim, not the LeRobot wrapper.

    This asserted ``NodeDaemon._NATIVE_LOOPS`` — one of the four disagreeing
    maps ADR 0022 collapsed into ``adapters._NATIVE_KINDS`` — so it broke open
    on the attribute rather than on the registration, and hid the fact that
    dimos was never added to the surviving table: ``native_loop_path("dimos")``
    returned None and the daemon fell through to ``lerobot_control_loop``.
    """
    from interlatent.adapters import native_kind, native_loop_path

    assert native_kind("dimos") == "dimos"
    assert (
        native_loop_path("dimos") == "interlatent.adapters.dimos.loop:control_loop"
    )
    # The `dimos_<embodiment>` sugar resolves as a native kind (so the act CLI
    # does not demand a --port) but carries NO session loop: the shim builds its
    # config straight from --robot-arg, so a driving session names the canonical
    # kind. Same rule as the yam_left/yam_right variants.
    assert native_kind("dimos_xarm7") == "dimos"
    assert native_loop_path("dimos_xarm7") is None


def test_dimos_config_imports_without_dimos_installed():
    """config/kinds (and the adapter package itself) must never require the
    [dimos] extra at import time — the daemon imports lazily, and base
    installs list the loop in _NATIVE_KINDS unconditionally."""
    import importlib

    for mod in (
        "interlatent.adapters.dimos",
        "interlatent.adapters.dimos.config",
        "interlatent.adapters.dimos.kinds",
        "interlatent.adapters.dimos.episode",
    ):
        importlib.import_module(mod)
