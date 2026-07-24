"""Robot integration adapters for the Interlatent node.

Each submodule adapts a specific robot family to the duck-typed LeRobot
``Robot`` interface that the built-in ``lerobot_control_loop`` drives:

- ``interlatent.adapters.lerobot`` — LeRobot-native rollout/record helpers.
- ``interlatent.adapters.axol`` — Almond Axol dual-arm robot (``interlatent[axol]``).

These are optional and dependency-heavy, so they are **not** imported here;
import the specific submodule you need (the node does so lazily). "Adapter"
here means a *robot* adapter — distinct from a server-side *policy backend*
(``policy_backend``), a collection ``--loop`` adapter, or a LoRA adapter.

:func:`resolve_adapter` is the one shared way to turn a ``--robot`` kind string into a
constructed (not-yet-connected) adapter — used by ``interlatent-act`` and the
:class:`~interlatent.robot.Robot` behaviors facade so both resolve robots identically.
Heavy adapter modules are still imported lazily inside it.
"""
from __future__ import annotations

from typing import Any, Optional

# The one native-kind registry (ADR 0011, amended by ADR 0022). A robot kind
# listed here is constructed from its own adapter config — never through
# LeRobotAdapter — and, when the kind is canonical (equal to its subpackage
# name), its DRTC sessions run the subpackage's ``loop:control_loop`` shim over
# the shared runner. The non-canonical YAM variants are CLI-side conveniences
# (arm selection defaults); a driving session must name the canonical kind.
#
# This table used to be four disagreeing maps (``daemon._NATIVE_LOOPS``, an
# inline ladder in ``act_cli``, and a yam-only check here); everything now
# resolves through it. ``teleop/robot_profile._PROFILES`` stays separate — it
# is the teleop-embodiment map and also covers LeRobot kinds.
_NATIVE_KINDS: dict[str, str] = {
    "axol": "axol",
    "yam": "yam",
    "yam_bimanual": "yam",
    "yam_left": "yam",
    "yam_right": "yam",
    "nori": "nori",
}


def native_kind(kind: Optional[str]) -> Optional[str]:
    """The native subpackage for a ``--robot`` kind, or ``None`` for LeRobot
    serial arms (which construct through :class:`LeRobotAdapter`)."""
    return _NATIVE_KINDS.get((kind or "").lower().strip())


def native_loop_path(kind: Optional[str]) -> Optional[str]:
    """The ``module:function`` control-loop entry point for a canonical native
    kind, or ``None`` when the bundled LeRobot wrapper should run.

    Deliberately ``None`` for the YAM variants (``yam_left`` …): their arm
    defaults are applied by :func:`resolve_adapter`, not by the session shim,
    so dispatching them to the native loop would drive the wrong arms.
    """
    k = (kind or "").lower().strip()
    pkg = _NATIVE_KINDS.get(k)
    if pkg is None or pkg != k:
        return None
    return f"interlatent.adapters.{pkg}.loop:control_loop"


def resolve_adapter(
    kind: str,
    *,
    port: Optional[str] = None,
    extra: Optional[dict[str, str]] = None,
    cameras: Optional[dict[str, str]] = None,
) -> Any:
    """Construct the robot adapter for a ``--robot`` kind (not yet connected).

    Kinds in ``_NATIVE_KINDS`` are built from their own adapter config (YAM over
    CAN, Nori over the daemon wire, Axol over CAN); every other kind is a
    :class:`~interlatent.adapters.lerobot.robot.LeRobotAdapter`. Adapter modules
    (and their heavy deps) are imported lazily here, so importing this package
    stays cheap. ``interlatent-act`` and the behaviors facade both resolve
    through here, so the CLI and the API can never disagree on a kind.

    ``auto_home`` defaults to off for YAM: opening an adapter to run a *named* behavior
    (or a one-shot move) should not itself home the arm the instant you connect.
    """
    extra = dict(extra or {})
    k = (kind or "").lower().strip()
    pkg = _NATIVE_KINDS.get(k)
    if pkg == "yam":
        extra.setdefault("auto_home", "false")
        if k == "yam_left":
            extra.setdefault("arms", "left")
        elif k == "yam_right":
            extra.setdefault("arms", "right")
        from .yam.config import build_adapter_config
        from .yam.robot import YAMNativeRobot

        return YAMNativeRobot(build_adapter_config(extra, cameras))
    if pkg == "nori":
        from .nori.config import build_adapter_config
        from .nori.robot import NoriNativeRobot

        return NoriNativeRobot(build_adapter_config(extra, cameras))
    if pkg == "axol":
        from .axol.config import build_adapter_config
        from .axol.robot import AxolNativeRobot

        return AxolNativeRobot(build_adapter_config(extra, cameras))

    from .lerobot.robot import LeRobotAdapter

    return LeRobotAdapter(kind, port=port, extra=extra, cameras=cameras)


__all__ = ["native_kind", "native_loop_path", "resolve_adapter"]
