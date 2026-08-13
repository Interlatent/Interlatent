"""``ik_config.json`` — the robot-specific half of VR IK retargeting.

Everything vr-teleop-kit hardcoded for the DK1 (body names, tool0 offset,
rest pose, gripper range, damping) is authored per robot kind instead, and
lives next to the URDF in that kind's data bundle:

    packages/sdk/src/interlatent_robots/<kind>/
        <robot>.urdf
        ik_config.json
        kinematic_spec.json

Nothing at runtime reads this file — it is the hand-authored *source* the
spec exporter (:mod:`kinematic_spec`) digests, and it is excluded from the
wheel for exactly that reason (ADR 0017, amended 2026-07-18). So this module
ships under ``packaging/`` with the other maintainer tools rather than in the
SDK, and stays dependency-free: importing it must not need MuJoCo.

The config also owns the units seam. The node speaks robot-native units
(SO101: degrees + gripper 0..100 open; YAM: radians + gripper 0..1 open) in
``action_features`` order; the solvers speak radians over the IK joints.
``robot_units_to_rad`` is the per-IK-joint affine ``rad = scale * unit +
offset`` (inverted on the way out), and ``ik_joint_action_indices`` /
``gripper_action_index`` map IK-space vectors into the action vector.
Action indices covered by neither are passthrough — filled from the seed
(e.g. the undriven arm of a bimanual YAM).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

SOLVER_TYPES = ("decoupled_6dof", "so101_5dof", "weighted_dls")

IK_CONFIG_FILENAME = "ik_config.json"

# Canonical ``damping`` tunables per solver_type: the single table the solvers
# and the browser-spec exporter all read, so a default can't be changed in one
# place and silently stay stale in another.
#
# A key absent here is not a knob — it is a typo. Nothing reads it, so a bundle
# carrying one runs stock damping while looking tuned. That is not theoretical:
# the live ``nori`` bundle shipped ``{"default": 0.1}`` on both chains and every
# reader ignored it. See :func:`unknown_damping_keys`.
DAMPING_DEFAULTS: dict[str, dict[str, float]] = {
    # so101_5dof (browser DLS solver) — one weighted 6D task over the chain.
    "so101_5dof": {
        "lam_pos": 0.05,
        "lam0": 0.15,
        "w0": 0.05,
        "mu": 0.02,
        "w_rot": 0.1,
        "rot_err_hold": 2.2,
    },
    # weighted_dls — the same generic browser solver as so101_5dof, named for
    # what it is rather than for the first arm that ran it. Kinds authored
    # after the SO101 (xarm6, xarm7) declare this; the knobs are identical
    # because the solver is.
    "weighted_dls": {
        "lam_pos": 0.05,
        "lam0": 0.15,
        "w0": 0.05,
        "mu": 0.02,
        "w_rot": 0.1,
        "rot_err_hold": 2.2,
    },
    # decoupled_6dof — position sub-solve over joints 1-3, wrist sub-solve
    # over 4-6, each with its own damping ramp. No w_rot: the tasks are split
    # rather than weighted against each other.
    "decoupled_6dof": {
        "lam_pos": 0.05,
        "lam0": 0.15,
        "w0": 0.05,
        "mu": 0.02,
        "lam_rot": 0.05,
        "lam0_rot": 0.4,
        "w0_rot": 0.5,
        "rot_err_hold": 2.2,
    },
}


def unknown_damping_keys(solver_type: str, damping: dict) -> tuple[str, ...]:
    """``damping`` keys no solver of this type reads, in sorted order."""
    known = DAMPING_DEFAULTS.get(solver_type)
    if known is None:
        return ()
    return tuple(sorted(k for k in damping if k not in known))


def resolve_damping(solver_key: str, damping: dict | None) -> dict[str, float]:
    """``damping`` filled in over ``solver_key``'s defaults.

    Callers name the table they want rather than reading it off the config:
    the browser-spec exporter resolves against ``so101_5dof`` for *every*
    bundle, because the browser ships one generic weighted-DLS solver — a
    ``decoupled_6dof`` bundle's wrist knobs have no meaning there.

    Unknown keys pass through untouched; :func:`unknown_damping_keys` is what
    reports them.
    """
    return {**DAMPING_DEFAULTS[solver_key], **(damping or {})}


@dataclass(frozen=True)
class JointAffine:
    """``rad = scale * robot_unit + offset``."""
    scale: float = 1.0
    offset: float = 0.0

    def to_rad(self, unit: float) -> float:
        return self.scale * unit + self.offset

    def from_rad(self, rad: float) -> float:
        return (rad - self.offset) / self.scale


@dataclass(frozen=True)
class IkConfig:
    """Parsed robot bundle IK configuration.

    Fields:
        solver_type: one of ``SOLVER_TYPES``.
        urdf: bundle-relative path of the URDF file.
        urdf_joint_names: the IK-driven joints, in IK order, by URDF joint
            name. The model wrapper resolves qpos/dof addresses from these —
            never positional qpos assumptions.
        ee_body: URDF body the ``tool0`` EE site is attached to.
        tool0_offset_xyz / tool0_offset_rpy: site pose on ``ee_body``.
        anchor_body / anchor_offset_xyz: wrist-invariant position-task anchor
            (``decoupled_6dof`` only; the importer-collapsed equivalent of
            vr-teleop-kit's ``j4_anchor``).
        q_rest: rest pose over the IK joints, radians.
        ik_joint_action_indices: action-vector index of each IK joint.
        gripper_action_index: action-vector index of the gripper, or None.
        gripper_range: ``[open_cmd, closed_cmd]`` in robot units; the wire
            ``pinch`` (0 open .. 1 closed) lerps between them.
        robot_units_to_rad: per-IK-joint affine (see module docstring).
        webxr_to_base_R: 3x3 rotation taking WebXR/Quest world vectors into
            the arm-base frame; shipped to the browser mapper via the
            teleop-token ``ik_hints``.
        scale_translation / scale_rotation / pos_reach_limit /
            rot_reach_limit: browser-mapper defaults, also via ``ik_hints``.
        damping: solver damping tunables (defaults match vr-teleop-kit).
        max_dq_per_joint: per-IK-joint per-solve Δq cap, radians.
    """

    solver_type: str
    urdf: str
    urdf_joint_names: tuple[str, ...]
    ee_body: str
    tool0_offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    tool0_offset_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    anchor_body: Optional[str] = None
    anchor_offset_xyz: Optional[tuple[float, float, float]] = None
    q_rest: tuple[float, ...] = ()
    ik_joint_action_indices: tuple[int, ...] = ()
    gripper_action_index: Optional[int] = None
    gripper_range: tuple[float, float] = (1.0, 0.0)
    robot_units_to_rad: tuple[JointAffine, ...] = ()
    webxr_to_base_R: tuple[tuple[float, float, float], ...] = (
        (0.0, 0.0, -1.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    )
    scale_translation: float = 1.0
    scale_rotation: float = 1.0
    pos_reach_limit: float = 0.25
    rot_reach_limit: float = 0.6
    damping: dict = field(default_factory=dict)
    max_dq_per_joint: Optional[tuple[float, ...]] = None

    def __post_init__(self) -> None:
        if self.solver_type not in SOLVER_TYPES:
            raise ValueError(
                f"ik_config: unknown solver_type {self.solver_type!r} "
                f"(known: {SOLVER_TYPES})"
            )
        n = len(self.urdf_joint_names)
        if n == 0:
            raise ValueError("ik_config: urdf_joint_names must be non-empty")
        if len(self.ik_joint_action_indices) != n:
            raise ValueError(
                "ik_config: ik_joint_action_indices length "
                f"{len(self.ik_joint_action_indices)} != {n} IK joints"
            )
        if self.q_rest and len(self.q_rest) != n:
            raise ValueError(f"ik_config: q_rest length != {n} IK joints")
        if self.robot_units_to_rad and len(self.robot_units_to_rad) != n:
            raise ValueError(
                f"ik_config: robot_units_to_rad length != {n} IK joints"
            )
        if self.max_dq_per_joint is not None and len(self.max_dq_per_joint) != n:
            raise ValueError(
                f"ik_config: max_dq_per_joint length != {n} IK joints"
            )
        if self.solver_type == "decoupled_6dof":
            if n != 6:
                raise ValueError("decoupled_6dof requires exactly 6 IK joints")
            if not self.anchor_body or self.anchor_offset_xyz is None:
                raise ValueError(
                    "decoupled_6dof requires anchor_body + anchor_offset_xyz"
                )

    @property
    def n_ik_joints(self) -> int:
        return len(self.urdf_joint_names)

    def affine(self, i: int) -> JointAffine:
        if self.robot_units_to_rad:
            return self.robot_units_to_rad[i]
        return JointAffine()

    def ik_hints(self) -> dict:
        """Browser-mapper hints surfaced through the teleop-token response."""
        return {
            "webxr_to_base_R": [list(row) for row in self.webxr_to_base_R],
            "scale_translation": self.scale_translation,
            "scale_rotation": self.scale_rotation,
            "pos_reach_limit": self.pos_reach_limit,
            "rot_reach_limit": self.rot_reach_limit,
        }


def _f3(raw: object, name: str) -> tuple[float, float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError(f"ik_config: {name} must be a 3-vector")
    return (float(raw[0]), float(raw[1]), float(raw[2]))


def parse_ik_config(obj: dict, *, strict: bool = False) -> IkConfig:
    """Build an :class:`IkConfig` from a parsed ``ik_config.json`` dict.

    ``strict`` decides what an unreadable ``damping`` key costs. A caller that
    can still refuse — a curation step with a human present — passes
    ``strict=True`` so a typo fails there and gets fixed. A caller reading a
    bundle that already shipped leaves it False and only warns: raising would
    turn a cosmetic typo into a robot that won't teleop at all, and the config
    is live by then either way. Loud at the seam that can say no; fail-open at
    the seam that can't.
    """
    if not isinstance(obj, dict):
        raise ValueError("ik_config: top level must be an object")

    affines: list[JointAffine] = []
    for entry in obj.get("robot_units_to_rad") or []:
        if isinstance(entry, dict):
            affines.append(JointAffine(
                scale=float(entry.get("scale", 1.0)),
                offset=float(entry.get("offset", 0.0)),
            ))
        else:  # bare scale shorthand
            affines.append(JointAffine(scale=float(entry)))

    R_raw = obj.get("webxr_to_base_R")
    if R_raw is not None:
        if (not isinstance(R_raw, (list, tuple)) or len(R_raw) != 3
                or any(len(r) != 3 for r in R_raw)):
            raise ValueError("ik_config: webxr_to_base_R must be 3x3")
        R = tuple(tuple(float(x) for x in row) for row in R_raw)
    else:
        R = IkConfig.webxr_to_base_R  # type: ignore[assignment]

    anchor_xyz = obj.get("anchor_offset_xyz")
    grip_range = obj.get("gripper_range") or [1.0, 0.0]
    max_dq = obj.get("max_dq_per_joint")

    cfg = IkConfig(
        solver_type=str(obj.get("solver_type", "")),
        urdf=str(obj.get("urdf", "")),
        urdf_joint_names=tuple(str(n) for n in obj.get("urdf_joint_names") or []),
        ee_body=str(obj.get("ee_body", "")),
        tool0_offset_xyz=_f3(obj.get("tool0_offset_xyz") or [0, 0, 0], "tool0_offset_xyz"),
        tool0_offset_rpy=_f3(obj.get("tool0_offset_rpy") or [0, 0, 0], "tool0_offset_rpy"),
        anchor_body=(str(obj["anchor_body"]) if obj.get("anchor_body") else None),
        anchor_offset_xyz=(_f3(anchor_xyz, "anchor_offset_xyz") if anchor_xyz else None),
        q_rest=tuple(float(x) for x in obj.get("q_rest") or []),
        ik_joint_action_indices=tuple(int(i) for i in obj.get("ik_joint_action_indices") or []),
        gripper_action_index=(
            int(obj["gripper_action_index"])
            if obj.get("gripper_action_index") is not None else None
        ),
        gripper_range=(float(grip_range[0]), float(grip_range[1])),
        robot_units_to_rad=tuple(affines),
        webxr_to_base_R=R,
        scale_translation=float(obj.get("scale_translation", 1.0)),
        scale_rotation=float(obj.get("scale_rotation", 1.0)),
        pos_reach_limit=float(obj.get("pos_reach_limit", 0.25)),
        rot_reach_limit=float(obj.get("rot_reach_limit", 0.6)),
        damping=dict(obj.get("damping") or {}),
        max_dq_per_joint=(
            tuple(float(x) for x in max_dq) if max_dq is not None else None
        ),
    )

    # solver_type is already validated by __post_init__, so the lookup is safe.
    unknown = unknown_damping_keys(cfg.solver_type, cfg.damping)
    if unknown:
        msg = (
            f"ik_config: damping key(s) {', '.join(repr(k) for k in unknown)} "
            f"are not read by solver_type {cfg.solver_type!r} — those knobs run "
            f"at their stock defaults. Known keys: "
            f"{', '.join(sorted(DAMPING_DEFAULTS[cfg.solver_type]))}"
        )
        if strict:
            raise ValueError(msg)
        log.warning("%s", msg)

    return cfg


def load_ik_config(bundle_dir: str | Path, *, strict: bool = False) -> IkConfig:
    """Load ``ik_config.json`` from a downloaded bundle directory."""
    path = Path(bundle_dir) / IK_CONFIG_FILENAME
    with open(path, "r", encoding="utf-8") as f:
        return parse_ik_config(json.load(f), strict=strict)


def parse_ik_config_bundle(
    obj: dict, *, strict: bool = False
) -> IkConfig | dict[str, IkConfig]:
    """Bundle-aware ``ik_config.json`` parse: flat or ``chains``-wrapped.

    A flat document is a single :class:`IkConfig` and is parsed exactly as
    :func:`parse_ik_config` always has. A wrapped one puts one such document
    per arm under a top-level ``"chains"`` key, each section independently
    valid per :func:`parse_ik_config` (own ``urdf``, ``ee_body``,
    ``ik_joint_action_indices`` etc. — the browser runs one solver per chain
    into one shared action vector).

    The wrapper does **not** mean bimanual: the number of populated sides
    does. Every shipped bundle is wrapped, and four of the six (``a1z``,
    ``so101``, ``xarm6``, ``xarm7``) carry only ``"right"``. The browser
    already reads it that way — see ``resolveSpecBundle`` in
    ``teleop/teleop-web/src/lib/teleop/kinematics.ts`` — so requiring both
    sides here would only make the exporter refuse bundles the runtime
    accepts.
    """
    if not isinstance(obj, dict):
        raise ValueError("ik_config: top level must be an object")
    chains = obj.get("chains")
    if chains is None:
        return parse_ik_config(obj, strict=strict)
    if (not isinstance(chains, dict) or not chains
            or not set(chains) <= {"left", "right"}):
        raise ValueError(
            'ik_config: "chains" must be a non-empty object keyed by "left" '
            'and/or "right"'
        )
    return {
        side: parse_ik_config(cfg, strict=strict) for side, cfg in chains.items()
    }


__all__ = [
    "IkConfig", "JointAffine", "parse_ik_config", "parse_ik_config_bundle",
    "load_ik_config", "SOLVER_TYPES", "IK_CONFIG_FILENAME",
    "DAMPING_DEFAULTS", "unknown_damping_keys", "resolve_damping",
]
