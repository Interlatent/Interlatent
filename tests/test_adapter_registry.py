"""The one native-kind registry (``interlatent.adapters._NATIVE_KINDS``).

Four ``robot_kind`` maps used to disagree (``daemon._NATIVE_LOOPS``, an inline
ladder in ``act_cli``, a yam-only check in ``resolve_adapter``); ADR 0022
collapsed them onto one table. These tests pin the collapsed contract: which
kinds are native, which get a native session loop, and that ``resolve_adapter``
dispatches every native kind (with its variant defaults) instead of silently
falling through to ``LeRobotAdapter`` — the pre-collapse Nori/Axol bug.
"""
from __future__ import annotations

import pytest

from interlatent import adapters
from interlatent.adapters import native_kind, native_loop_path, resolve_adapter
from interlatent.node.control import import_callable

_CANONICAL = ("yam", "nori", "axol")


def test_canonical_kinds_resolve_importable_shim_loops():
    for kind in _CANONICAL:
        path = native_loop_path(kind)
        assert path == f"interlatent.adapters.{kind}.loop:control_loop"
        assert callable(import_callable(path)), (
            "%s's registry entry points at nothing importable — the daemon "
            "would crash at session start" % kind
        )


def test_variants_and_lerobot_kinds_use_the_bundled_wrapper():
    """YAM variants are CLI-side conveniences: their arm defaults live in
    resolve_adapter, not the session shim, so dispatching them to the native
    loop would drive the wrong arms. LeRobot kinds fall through by design."""
    for kind in ("yam_left", "yam_right", "yam_bimanual", "so101", "", None):
        assert native_loop_path(kind) is None, kind
    for kind in ("yam_left", "yam_right", "yam_bimanual"):
        assert native_kind(kind) == "yam", kind
    assert native_kind("so101") is None


@pytest.fixture
def dispatch(monkeypatch):
    """Patch every adapter class to a recorder so construction is observable
    without hardware, vendor SDKs, or sockets."""
    import interlatent.adapters.axol.config as axol_cfg
    import interlatent.adapters.axol.robot as axol_robot
    import interlatent.adapters.lerobot.robot as lerobot_robot
    import interlatent.adapters.nori.config as nori_cfg
    import interlatent.adapters.nori.robot as nori_robot
    import interlatent.adapters.yam.config as yam_cfg
    import interlatent.adapters.yam.robot as yam_robot

    for cfg_mod in (yam_cfg, nori_cfg, axol_cfg):
        monkeypatch.setattr(cfg_mod, "build_adapter_config", lambda e, c: dict(e))
    monkeypatch.setattr(yam_robot, "YAMNativeRobot", lambda cfg: ("yam", cfg))
    monkeypatch.setattr(nori_robot, "NoriNativeRobot", lambda cfg: ("nori", cfg))
    monkeypatch.setattr(axol_robot, "AxolNativeRobot", lambda cfg: ("axol", cfg))
    monkeypatch.setattr(
        lerobot_robot, "LeRobotAdapter",
        lambda kind, *, port=None, extra=None, cameras=None: ("lerobot", kind, port),
    )
    return adapters


def test_resolve_adapter_dispatches_every_native_kind(dispatch):
    assert resolve_adapter("nori") == ("nori", {})
    assert resolve_adapter("axol") == ("axol", {})
    assert resolve_adapter("yam") == ("yam", {"auto_home": "false"})


def test_resolve_adapter_applies_yam_variant_defaults(dispatch):
    kind, cfg = resolve_adapter("yam_left")
    assert kind == "yam"
    assert cfg == {"auto_home": "false", "arms": "left"}
    _, cfg = resolve_adapter("yam_right")
    assert cfg["arms"] == "right"
    # Explicit user args always beat the variant defaults.
    _, cfg = resolve_adapter("yam_left", extra={"arms": "right", "auto_home": "true"})
    assert cfg == {"arms": "right", "auto_home": "true"}


def test_resolve_adapter_falls_through_for_lerobot_kinds(dispatch):
    assert resolve_adapter("so101", port="/dev/ttyACM0") == (
        "lerobot", "so101", "/dev/ttyACM0",
    )
