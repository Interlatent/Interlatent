"""Almond Axol dual-arm robot support for the Interlatent node (``interlatent[axol]``).

Optional, vendor-specific subpackage. Selected via ``--robot axol`` (the daemon
maps the ``axol`` robot kind to :func:`control_loop` through its native-loop
registry), so the base ``interlatent`` install never requires ``almond-axol``.
Install with ``pip install 'interlatent[axol]'``.

This adapter drives the **native async Axol SDK** (``almond_axol.robot.Axol``)
directly and opens the GMSL-attached ZED cameras **onboard the Jetson by serial
number** through ``almond_axol.lerobot.camera`` (so the camera path depends on
lerobot, which ships in the ``[axol]`` extra). See
``docs/adr/0011-vendor-robot-subpackage-via-robot-kind.md``.
"""

from __future__ import annotations

from .loop import control_loop

__all__ = ["control_loop"]
