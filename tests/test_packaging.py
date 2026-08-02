"""Regression: every console-script module must actually ship in the wheel.

The CI installs editable (`pip install -e`), which maps the whole src/
tree and silently masks subpackages that setuptools would exclude from a
real wheel for lack of an ``__init__.py``. That bug shipped once — the
``interlatent-node`` entry point pointed at a module that wasn't in the
wheel. This test runs the same package discovery setuptools uses at
build time.
"""
import ast
from pathlib import Path

import pytest
from setuptools import find_packages

REPO = Path(__file__).resolve().parent.parent


def _parent_map(tree: ast.AST) -> dict:
    return {
        child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)
    }


def _enclosing_function(parents: dict, node: ast.AST) -> ast.FunctionDef | None:
    """Innermost ``def`` containing ``node`` (ast nodes carry no parent link)."""
    while node in parents:
        node = parents[node]
        if isinstance(node, ast.FunctionDef):
            return node
    return None


def _function_named(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"no def {name}() in blueprints.py")


def _param_names(fn: ast.FunctionDef) -> set:
    args = fn.args
    return {a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)}


def _nulls_coordinator_task(fn: ast.FunctionDef) -> bool:
    """True if ``fn`` returns ``<its own param>.model_copy(update={...: None})``.

    Structural, not textual: the copy must be taken off the parameter the
    function was handed and handed straight back as a return value, so a
    version that nulls a *different* model — or computes the copy and drops it
    on the floor — does not count.
    """
    if not fn.args.args:
        return False
    param = fn.args.args[0].arg
    for node in ast.walk(fn):
        if not isinstance(node, ast.Return) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        func = call.func
        if not (isinstance(func, ast.Attribute) and func.attr == "model_copy"):
            continue
        if not (isinstance(func.value, ast.Name) and func.value.id == param):
            continue
        for keyword in call.keywords:
            if keyword.arg != "update":
                continue
            try:
                update = ast.literal_eval(keyword.value)
            except ValueError:
                continue
            if update == {"coordinator_task_name": None}:
                return True
    return False


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
    Viser, while the exclusive servo task remains the only execution task.

    The exclusivity half is checked structurally, by walking the model
    expression from ``planner(robots=[...])`` back to where it is bound. String
    assertions ("does ``coordinator_task_name`` appear anywhere in the file")
    cannot tell a nulled model that reaches the planner from a nulled model
    assigned to some unrelated variable while the planner gets a fresh one —
    and that refactor is exactly how a second trajectory task would come back
    to fight the policy invisibly, corrupting the recorded control_source.
    """
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
    parents = _parent_map(tree)
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

    # 1. The planner plans for exactly one robot, and that robot expression is
    #    a call — the sanitizer — not a bare model handed straight through.
    robots = keywords["robots"]
    assert isinstance(robots, ast.List) and len(robots.elts) == 1, ast.dump(robots)
    sanitizer_call = robots.elts[0]
    assert isinstance(sanitizer_call, ast.Call) and isinstance(
        sanitizer_call.func, ast.Name
    ), f"planner robots[0] must be a sanitizing call: {ast.dump(sanitizer_call)}"

    # 2. That sanitizer genuinely nulls coordinator_task_name on the model it
    #    was handed, and returns it.
    sanitizer = _function_named(tree, sanitizer_call.func.id)
    assert _nulls_coordinator_task(sanitizer), (
        f"{sanitizer.name}() must return "
        'model.model_copy(update={"coordinator_task_name": None}) — otherwise the '
        "planner attaches a trajectory task claiming the servo task's joints"
    )

    # 3. The model it sanitizes is the one the blueprint was given, not a fresh
    #    factory call: it must be a plain parameter of the enclosing builder,
    #    never rebound between the signature and the planner call.
    assert len(sanitizer_call.args) == 1 and not sanitizer_call.keywords
    sanitized = sanitizer_call.args[0]
    assert isinstance(sanitized, ast.Name), (
        "the sanitized model must be the builder's own model parameter, not an "
        f"inline expression: {ast.dump(sanitized)}"
    )
    builder = _enclosing_function(parents, planner_calls[0])
    assert builder is not None
    assert sanitized.id in _param_names(builder), (
        f"{sanitized.id} is not a parameter of {builder.name}() — the planner is "
        "being handed a model built somewhere the sanitizer never saw"
    )
    rebinds = [
        node
        for node in ast.walk(builder)
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.NamedExpr))
        for target in (
            node.targets if isinstance(node, ast.Assign) else [node.target]
        )
        if isinstance(target, ast.Name) and target.id == sanitized.id
    ]
    assert not rebinds, (
        f"{builder.name}() rebinds {sanitized.id} before the planner sees it "
        f"(line {rebinds[0].lineno}) — the sanitized model may not be the one that "
        "reaches the planner"
    )

    # 4. And the xarm7 kind reaches that builder with the real model config.
    xarm7_builder = _function_named(tree, "_build_xarm7")
    models = [
        keyword.value
        for node in ast.walk(xarm7_builder)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == builder.name
        for keyword in node.keywords
        if keyword.arg == sanitized.id
    ]
    assert len(models) == 1, f"_build_xarm7 must pass one {sanitized.id}= : {models}"
    assert (
        isinstance(models[0], ast.Call)
        and isinstance(models[0].func, ast.Name)
        and models[0].func.id == "make_xarm7_model_config"
    ), f"expected make_xarm7_model_config(...), got {ast.dump(models[0])}"


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
