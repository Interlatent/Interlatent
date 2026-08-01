"""Per-kind static declarations for dimos-mediated embodiments.

A :class:`DimosKind` is the bridge between two naming worlds: dimos joint names
(``arm/joint1`` — hardware-id-prefixed, ``/``-separated) and the SDK's
``<name>.pos`` feature-key convention, which has never carried ``/`` (recording
schemas, policies, and the profile registry all assume it). The map is
``dimos_name.replace("/", "_") + ".pos"``, owned here in both directions; the
dimos wire only ever sees dimos names, the SDK only ever sees feature keys.

The kind's ``profile_name`` keys the :mod:`~interlatent.node.teleop.robot_profile`
registry (``dimos_xarm7`` — the bare ``xarm7`` name stays free for a possible
future direct-xArm adapter). Profile ``joint_names`` order must equal
``feature_keys`` order minus ``.pos``; ``base.py`` raises if they diverge.

No dimos imports here — this module must stay importable without the ``[dimos]``
extra.
"""
from __future__ import annotations

from dataclasses import dataclass, field


def feature_key_for(dimos_name: str) -> str:
    """``arm/joint1`` -> ``arm_joint1.pos`` (the SDK-facing action feature)."""
    return dimos_name.replace("/", "_") + ".pos"


@dataclass(frozen=True)
class DimosKind:
    """Static declaration of one dimos-mediated embodiment.

    - ``dimos_arm_joints``: dimos joint names in dimos hardware order — the order
      the coordinator reports and the servo task claims. This order IS the action
      vector order (gripper appended last, yam/nori precedent).
    - ``gripper_hardware_id``: first argument to the coordinator's
      ``set_gripper_position`` RPC (dimos hardware id, e.g. ``"arm"``).
    """

    name: str
    profile_name: str
    dimos_arm_joints: tuple[str, ...]
    dimos_gripper_joint: str | None = None
    gripper_hardware_id: str | None = None
    # Derived; do not pass.
    feature_keys: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        dimos_names = self.dimos_arm_joints + (
            (self.dimos_gripper_joint,) if self.dimos_gripper_joint else ()
        )
        keys = tuple(feature_key_for(n) for n in dimos_names)
        if len(set(keys)) != len(keys) or len(set(dimos_names)) != len(dimos_names):
            raise ValueError(
                f"kind {self.name!r}: joint names {dimos_names} do not map to "
                "unique feature keys — the '/'->'_' map must stay bijective"
            )
        object.__setattr__(self, "feature_keys", keys)

    @property
    def dimos_joint_names(self) -> tuple[str, ...]:
        """All dimos joint names in action order (arm joints, then gripper)."""
        if self.dimos_gripper_joint:
            return self.dimos_arm_joints + (self.dimos_gripper_joint,)
        return self.dimos_arm_joints

    def dimos_name_for(self, feature_key: str) -> str:
        """Inverse map, restricted to this kind's declared joints."""
        try:
            return self._feature_to_dimos[feature_key]
        except KeyError:
            raise KeyError(
                f"kind {self.name!r} has no joint for feature {feature_key!r}; "
                f"known: {list(self.feature_keys)}"
            ) from None

    @property
    def _feature_to_dimos(self) -> dict[str, str]:
        return dict(zip(self.feature_keys, self.dimos_joint_names))


# UFACTORY xArm7 behind a dimos ControlCoordinator. Joint names come from
# dimos's `make_joints("arm", 7)` (dimos/control/components.py) with the
# hardware id the reference blueprint uses; the gripper is declared separately
# (`gripper_joints=["arm/gripper"]`) and driven via the coordinator's
# `set_gripper_position("arm", ...)` RPC, not the joint_command stream.
XARM7 = DimosKind(
    name="xarm7",
    profile_name="dimos_xarm7",
    dimos_arm_joints=tuple(f"arm/joint{i}" for i in range(1, 8)),
    dimos_gripper_joint="arm/gripper",
    gripper_hardware_id="arm",
)

# go2_base (velocity virtual joints go2/vx, go2/vy, go2/wz) is planned for v1.5 —
# see CONFIG.md; the seam is identical but profile semantics are velocity bounds.
KNOWN_KINDS: dict[str, DimosKind] = {XARM7.name: XARM7}


def get_kind(name: str) -> DimosKind:
    """Resolve a declared kind or raise with the known list (fail closed)."""
    try:
        return KNOWN_KINDS[str(name).strip().lower()]
    except KeyError:
        raise ValueError(
            f"unknown dimos kind {name!r}; known kinds: {sorted(KNOWN_KINDS)}. "
            "Pass --robot-arg kind=<kind> matching the running dimos stack."
        ) from None


__all__ = ["DimosKind", "XARM7", "KNOWN_KINDS", "get_kind", "feature_key_for"]
