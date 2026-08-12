"""The Interlatent Coordinator Protocol — the SDK's one control-plane contract.

A **coordinator** is whatever HTTP service a node, a GPU box, the CLI, and the
teleop web app talk to in order to be assigned work. ``interlatent up`` is the
coordinator this repo ships, and implementing this contract is the only thing
that makes something else one. The SDK reaches a coordinator by address and
must never branch on which one is on the other end — that dual-mode fork is
what collapsed the 2026-06 stack (see ADR 0038, superseding 0023).

**One tier.** Every route below is part of the contract; a coordinator serves
all of them or it is not a coordinator. The table used to carry three tiers,
which only ever described what a *second* implementation did not serve —
``coordinator-only`` was literally "the hosted control plane 404s these". That
control plane is gone, so the tiers collapsed and the thirteen routes nothing
will ever serve were deleted rather than left advertised (ADR 0039).

This module is the machine-readable half of ``docs/coordinator-protocol.md``.
The two are pinned to each other by ``tests/test_coordinator_protocol.py``, so
the table in the doc cannot drift from the tuples here.

Deliberately stdlib-only and import-cheap: it is read by the node daemon, the
CLI, and ``interlatent-serve``'s twin, none of which want a dependency for it.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "PROTOCOL_VERSION",
    "API_PREFIX",
    "Route",
    "ROUTES",
    "TIER_MANDATORY",
    "MANDATORY",
    "by_tier",
]

#: Advertised by ``GET /api/v1/capabilities`` and by ``interlatent up``.
#: Bump the integer only for a breaking change. ``/2`` is one, deliberately:
#: ADR 0039 **removed** thirteen routes and collapsed the tier model, which
#: breaks the additive-only compatibility rule this constant used to promise
#: (the same rule ``proto/messages.proto`` follows). Advertising a surface no
#: coordinator serves was judged worse than a version bump.
PROTOCOL_VERSION = "interlatent.coordinator/2"

#: Every route hangs off this. A coordinator address is a **bare origin** with
#: no trailing slash and no ``/api/v1`` suffix (``https://host`` /
#: ``http://host:8900``); callers append this prefix themselves. Two conflicting
#: conventions used to coexist in this repo and were reconciled at runtime in
#: three separate places — one convention now, resolved in one place.
API_PREFIX = "/api/v1"

#: The only tier. A coordinator MUST serve every route in :data:`ROUTES`;
#: absent any one of them something concrete breaks — a node cannot pair, a box
#: refuses to boot, every gRPC call is rejected, or an operator verb has no
#: spelling. Teleop is the one surface with a runtime condition on top (it
#: needs a relay), and ``GET /capabilities`` is how a caller asks about that.
TIER_MANDATORY = "mandatory"


@dataclass(frozen=True)
class Route:
    """One endpoint in the contract.

    ``path`` is a template relative to :data:`API_PREFIX`, using ``{name}``
    placeholders — it is documentation, not a router; nothing here parses it.
    """

    method: str
    path: str
    tier: str
    summary: str

    @property
    def full_path(self) -> str:
        return f"{API_PREFIX}{self.path}"

    def __str__(self) -> str:  # pragma: no cover - debugging affordance
        return f"{self.method} {self.full_path}"


# ----------------------------------------------------------------------
# The contract
# ----------------------------------------------------------------------
#
# Ordering is meaningful: it is the order of the table in
# docs/coordinator-protocol.md, and the doc/code equality test compares
# sequences, not sets, so a reordering shows up as a diff to review.

ROUTES: tuple[Route, ...] = (
    # -- Node plane ----------------------------------------------------
    Route("POST", "/nodes", TIER_MANDATORY,
          "Pair a node; mints its node id and node token."),
    Route("POST", "/nodes/{node_id}/heartbeat", TIER_MANDATORY,
          "Liveness plus the node's recording-spool and safety telemetry."),
    Route("GET", "/nodes/{node_id}/poll", TIER_MANDATORY,
          "Long-poll for the node's current assignment."),
    Route("POST", "/nodes/{node_id}/hardware", TIER_MANDATORY,
          "Report robot type, port, cameras and robot args."),
    Route("POST", "/nodes/{node_id}/robot-features", TIER_MANDATORY,
          "Report feature element names and the teleop profile."),
    # -- Box plane -----------------------------------------------------
    Route("POST", "/compute/boxes/register", TIER_MANDATORY,
          "Register a GPU box; idempotent on box_id."),
    Route("POST", "/compute/boxes/{box_id}/status", TIER_MANDATORY,
          "Box self-reports ready/running/uploading/stopped."),
    Route("GET", "/compute/boxes/{box_id}/warmup-target", TIER_MANDATORY,
          "Policy and camera keys to pre-warm; 404 means 'no target'."),
    Route("GET", "/compute/boxes/{box_id}/authz", TIER_MANDATORY,
          "Per-RPC authorization probe for the box's gRPC port."),
    # -- Operator plane ------------------------------------------------
    Route("GET", "/nodes", TIER_MANDATORY,
          "List nodes."),
    Route("GET", "/gpus", TIER_MANDATORY,
          "List GPU boxes available to the caller."),
    Route("GET", "/environments", TIER_MANDATORY,
          "List environments. Doubles as the box auth probe."),
    Route("POST", "/environments", TIER_MANDATORY,
          "Create an environment."),
    Route("GET", "/environments/{env_id}/config", TIER_MANDATORY,
          "Observation schema: action_dim, camera_names, num_cameras."),
    Route("GET", "/inference/sessions", TIER_MANDATORY,
          "List inference sessions."),
    Route("POST", "/inference/sessions", TIER_MANDATORY,
          "Create an inference session and assign it to a node."),
    Route("DELETE", "/inference/sessions/{session_id}", TIER_MANDATORY,
          "Stop a session by unassigning it. MUST NOT kill the node."),
    # -- Capability discovery ------------------------------------------
    Route("GET", "/capabilities", TIER_MANDATORY,
          "Protocol version and which conditional surfaces are live."),
    # -- Teleop --------------------------------------------------------
    Route("GET", "/teleop-recordings", TIER_MANDATORY,
          "List teleop recordings."),
    Route("POST", "/teleop-recordings", TIER_MANDATORY,
          "Create a teleop recording and assign it to a node."),
    Route("POST", "/teleop-recordings/{recording_id}/stop", TIER_MANDATORY,
          "Stop a teleop recording."),
    Route("POST", "/inference/sessions/{session_id}/teleop-token", TIER_MANDATORY,
          "Mint a teleop token for role=node|browser."),
    Route("POST", "/teleop-recordings/{recording_id}/teleop-token", TIER_MANDATORY,
          "Mint a teleop token for role=node|browser."),
    # -- Coordinator administration ------------------------------------
    Route("PUT", "/coordinator/recording", TIER_MANDATORY,
          "Set the recording destination stamped onto every session."),
)


def by_tier(tier: str) -> tuple[Route, ...]:
    """Every route in ``tier``, in declaration order.

    One tier exists (:data:`TIER_MANDATORY`), so this returns either the whole
    table or nothing. It survives the collapse because the tier is still a
    field on :class:`Route` and a column in the doc table, and deriving
    :data:`MANDATORY` through it is what makes a mistyped tier show up as a
    missing route rather than as nothing at all.
    """
    return tuple(r for r in ROUTES if r.tier == tier)


MANDATORY: tuple[Route, ...] = by_tier(TIER_MANDATORY)
