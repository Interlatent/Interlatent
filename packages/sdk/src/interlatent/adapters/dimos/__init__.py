"""Dimos robot adapter: drive robots managed by a running dimos stack.

Unlike every other adapter, this one owns no motor driver — the "vendor SDK" is
the dimos process itself. The adapter binds to it as an external bus peer
(LCM/Zenoh): it subscribes ``coordinator_joint_state`` + camera Image topics,
publishes ``joint_command`` to a dimos servo task, and calls the coordinator's
RPC layer for the gripper. Identity is declare-then-verify: the operator states
``--robot-arg kind=<kind>`` and connect() fail-closes if the live stack
disagrees (see :mod:`.verify`). The delta clamp here is the ONLY limit in the
whole path — dimos applies none to streamed joint commands.

See ADR 0018 and :doc:`CONFIG.md` for the blueprint contract
(``dimos run interlatent.<kind>`` ships a known-good session blueprint).
Requires the ``[dimos]`` extra (python 3.11–3.12) at runtime; this package and
``.config``/``.kinds`` import without it.
"""
from __future__ import annotations

__all__ = ["control_loop"]


def __getattr__(name: str):  # lazy: keep base import dimos-free
    if name == "control_loop":
        from .loop import control_loop

        return control_loop
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
