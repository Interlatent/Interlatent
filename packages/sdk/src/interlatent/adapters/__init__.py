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

# Native-CAN kinds are built directly from their own config; everything else is a
# LeRobot serial arm built through the LeRobot adapter.
_YAM_KINDS = ("yam", "yam_bimanual", "yam_left", "yam_right")


def resolve_adapter(
    kind: str,
    *,
    port: Optional[str] = None,
    extra: Optional[dict[str, str]] = None,
    cameras: Optional[dict[str, str]] = None,
) -> Any:
    """Construct the robot adapter for a ``--robot`` kind (not yet connected).

    Mirrors the resolution ``interlatent-act`` uses: native-CAN YAM kinds are built
    from :class:`~interlatent.adapters.yam.config.YAMAdapterConfig`; every other kind
    is a :class:`~interlatent.adapters.lerobot.robot.LeRobotAdapter`. Adapter modules
    (and their heavy deps) are imported lazily here, so importing this package stays
    cheap.

    ``auto_home`` defaults to off for YAM: opening an adapter to run a *named* behavior
    (or a one-shot move) should not itself home the arm the instant you connect.
    """
    extra = dict(extra or {})
    k = (kind or "").lower().strip()
    if k in _YAM_KINDS:
        extra.setdefault("auto_home", "false")
        if k == "yam_left":
            extra.setdefault("arms", "left")
        elif k == "yam_right":
            extra.setdefault("arms", "right")
        from .yam.config import build_adapter_config
        from .yam.robot import YAMNativeRobot

        return YAMNativeRobot(build_adapter_config(extra, cameras))

    if k == "dimos" or k.startswith("dimos_"):
        # `--robot dimos --robot-arg kind=xarm7`, or the `dimos_<kind>` sugar
        # (`--robot dimos_xarm7`). Binds to a RUNNING dimos stack as a bus
        # peer; connect() runs the fail-closed declare-then-verify check.
        if k.startswith("dimos_"):
            extra.setdefault("kind", k[len("dimos_"):])
        from .dimos.config import build_adapter_config as build_dimos_config
        from .dimos.robot import DimosNativeRobot

        return DimosNativeRobot(build_dimos_config(extra, cameras))

    from .lerobot.robot import LeRobotAdapter

    return LeRobotAdapter(kind, port=port, extra=extra, cameras=cameras)


__all__ = ["resolve_adapter"]
