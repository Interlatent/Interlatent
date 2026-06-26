"""I2RT YAM bimanual arm support for the Interlatent node (``interlatent[yam]``).

Optional, vendor-specific subpackage. Selected via ``--robot yam`` (the daemon maps
the ``yam`` robot kind to :func:`control_loop` through its native-loop registry), so
the base ``interlatent`` install never requires ``i2rt``. Install with
``pip install 'interlatent[yam]'``.

This adapter drives the **I2RT ``i2rt`` driver directly** (``get_yam_robot`` →
``command_joint_pos`` / ``get_joint_pos``) — the same thin path raiden's
``scripts/read_arm_poses.py`` uses — and captures RGB from RealSense / ZED cameras
(:mod:`.cameras`). It deliberately does NOT depend on the raiden package or its
teleop / IK / depth / serving stack. Joint-space only (ADR 0013). See
``docs/adr/0011-vendor-robot-subpackage-via-robot-kind.md``.
"""

from __future__ import annotations

from .loop import control_loop

__all__ = ["control_loop"]
