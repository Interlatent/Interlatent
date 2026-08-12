"""The Interlatent Coordinator Protocol — the SDK's one control-plane contract.

A **coordinator** is whatever HTTP service a node, a GPU box, the CLI, and the
teleop web app talk to in order to be assigned work. The hosted Interlatent
dashboard is one implementation; ``interlatent up`` ships another. The SDK does
not know which it is talking to and must never branch on it — that dual-mode
fork is what collapsed the 2026-06 stack (see ADR 0038, superseding 0023).

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
    "MANDATORY",
    "OPTIONAL",
    "COORDINATOR_ONLY",
    "by_tier",
]

#: Advertised by ``GET /api/v1/capabilities`` and by ``interlatent up``.
#: Bump the integer only for a breaking change; the compatibility rule is the
#: same one ``proto/messages.proto`` follows — additive changes only.
PROTOCOL_VERSION = "interlatent.coordinator/1"

#: Every route hangs off this. A coordinator address is a **bare origin** with
#: no trailing slash and no ``/api/v1`` suffix (``https://host`` /
#: ``http://host:8900``); callers append this prefix themselves. Two conflicting
#: conventions used to coexist in this repo and were reconciled at runtime in
#: three separate places — one convention now, resolved in one place.
API_PREFIX = "/api/v1"

#: A coordinator MUST serve these or nothing works.
TIER_MANDATORY = "mandatory"
#: A coordinator MAY serve these; every SDK caller degrades on 404.
TIER_OPTIONAL = "optional"
#: Served only by a self-hosted coordinator. The hosted dashboard 404s these,
#: and the CLI says so by name rather than reporting a bare 404.
TIER_COORDINATOR_ONLY = "coordinator-only"


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
    # -- Inbox plane ---------------------------------------------------
    Route("POST", "/episodes", TIER_OPTIONAL,
          "Register an episode row. 409 means 'already exists', tolerated."),
    Route("POST", "/episodes/{episode_id}/upload-urls", TIER_OPTIONAL,
          "Exchange file keys for presigned PUT urls."),
    Route("POST", "/episodes/{episode_id}/upload-complete", TIER_OPTIONAL,
          "Signal the inbox that every file landed."),
    # -- Capability discovery ------------------------------------------
    Route("GET", "/capabilities", TIER_OPTIONAL,
          "Protocol version and which optional tiers are served."),
    # -- Teleop --------------------------------------------------------
    Route("GET", "/teleop-recordings", TIER_OPTIONAL,
          "List teleop recordings."),
    Route("POST", "/teleop-recordings", TIER_OPTIONAL,
          "Create a teleop recording and assign it to a node."),
    Route("POST", "/teleop-recordings/{recording_id}/stop", TIER_OPTIONAL,
          "Stop a teleop recording."),
    Route("POST", "/inference/sessions/{session_id}/teleop-token", TIER_OPTIONAL,
          "Mint a teleop token for role=node|browser."),
    Route("POST", "/teleop-recordings/{recording_id}/teleop-token", TIER_OPTIONAL,
          "Mint a teleop token for role=node|browser."),
    # -- Hosted analysis and dataset surfaces --------------------------
    Route("GET", "/environments/{env_id}/episodes", TIER_OPTIONAL,
          "List an environment's episodes."),
    Route("POST", "/environments/{env_id}/process", TIER_OPTIONAL,
          "Kick the hosted merge pipeline."),
    Route("GET", "/environments/{env_id}/processing-status", TIER_OPTIONAL,
          "Poll the hosted merge pipeline."),
    Route("POST", "/environments/{env_id}/cancel-processing", TIER_OPTIONAL,
          "Cancel the hosted merge pipeline."),
    Route("POST", "/environments/{env_id}/analyze", TIER_OPTIONAL,
          "Request hosted policy analysis."),
    Route("GET", "/episodes/{episode_id}", TIER_OPTIONAL,
          "Fetch one episode row."),
    Route("GET", "/episodes/{episode_id}/status", TIER_OPTIONAL,
          "Poll an episode's processing status."),
    Route("GET", "/episodes/{episode_id}/results", TIER_OPTIONAL,
          "Fetch analysis results."),
    Route("GET", "/episodes/{episode_id}/meta", TIER_OPTIONAL,
          "Fetch episode metadata."),
    Route("GET", "/episodes/{episode_id}/chunks/{chunk}", TIER_OPTIONAL,
          "Fetch one dataset chunk."),
    Route("POST", "/episodes/{episode_id}/inbox-gc", TIER_OPTIONAL,
          "Drop a partially uploaded inbox session."),
    # -- Coordinator-only ----------------------------------------------
    Route("PUT", "/coordinator/recording", TIER_COORDINATOR_ONLY,
          "Set the recording destination stamped onto every session."),
)


def by_tier(tier: str) -> tuple[Route, ...]:
    """Every route in ``tier``, in declaration order."""
    return tuple(r for r in ROUTES if r.tier == tier)


MANDATORY: tuple[Route, ...] = by_tier(TIER_MANDATORY)
OPTIONAL: tuple[Route, ...] = by_tier(TIER_OPTIONAL)
COORDINATOR_ONLY: tuple[Route, ...] = by_tier(TIER_COORDINATOR_ONLY)
