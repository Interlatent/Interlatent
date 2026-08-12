"""The coordinator protocol doc and the protocol module must agree.

``docs/coordinator-protocol.md`` is normative: it is what a third party writes
a coordinator against. ``interlatent.coordinator.protocol`` is what this repo's
code reads. A contract that exists twice drifts unless something fails when it
does — that is this file's whole job.

The route-table assertion compares *sequences*, not sets, so reordering the
table shows up as a reviewable diff rather than passing silently.

The companion "every mandatory route actually answers" test lives with the
coordinator server implementation; this file pins the contract itself.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from interlatent.coordinator import protocol as proto

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs" / "coordinator-protocol.md"

# `POST` | `/api/v1/nodes` | mandatory | Pair a node; ...
_ROW = re.compile(
    r"^\|\s*`(?P<method>[A-Z]+)`\s*"
    r"\|\s*`(?P<path>/[^`]*)`\s*"
    r"\|\s*(?P<tier>[a-z-]+)\s*"
    r"\|\s*(?P<summary>.*?)\s*\|$"
)


def _doc_rows() -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    for line in DOC.read_text().splitlines():
        m = _ROW.match(line.strip())
        if m:
            rows.append(
                (m["method"], m["path"], m["tier"], m["summary"])
            )
    return rows


def _module_rows() -> list[tuple[str, str, str, str]]:
    return [(r.method, r.full_path, r.tier, r.summary) for r in proto.ROUTES]


# ----------------------------------------------------------------------
# The freeze
# ----------------------------------------------------------------------


def test_doc_table_is_the_module_table() -> None:
    doc, mod = _doc_rows(), _module_rows()
    assert doc, f"no route rows parsed out of {DOC} — did the table format change?"
    assert doc == mod


def test_doc_table_is_not_accidentally_empty_or_truncated() -> None:
    """Guards the failure mode the equality test cannot see: a regex that stops
    matching returns [] from both sides only if the module is also empty."""
    assert len(_doc_rows()) == len(proto.ROUTES) > 20


# ----------------------------------------------------------------------
# Internal consistency of the contract
# ----------------------------------------------------------------------


def test_every_route_carries_a_known_tier() -> None:
    assert {r.tier for r in proto.ROUTES} == {proto.TIER_MANDATORY}


def test_the_one_tier_is_the_whole_table() -> None:
    """ADR 0039 collapsed three tiers into one. A route that falls out of
    ``MANDATORY`` now has a typo'd tier, not an optional one."""
    assert proto.MANDATORY == proto.ROUTES
    assert proto.by_tier("optional") == ()
    assert proto.by_tier("coordinator-only") == ()


def test_no_duplicate_method_and_path() -> None:
    seen = [(r.method, r.path) for r in proto.ROUTES]
    dupes = {x for x in seen if seen.count(x) > 1}
    assert not dupes, f"duplicated routes: {sorted(dupes)}"


def test_paths_are_relative_and_get_the_prefix_exactly_once() -> None:
    for r in proto.ROUTES:
        assert r.path.startswith("/"), r
        assert not r.path.startswith(proto.API_PREFIX), (
            f"{r.path} already carries {proto.API_PREFIX}; paths are relative to it"
        )
        assert r.full_path == f"{proto.API_PREFIX}{r.path}"


def test_every_route_has_a_summary() -> None:
    for r in proto.ROUTES:
        assert r.summary.strip() and r.summary.rstrip().endswith("."), r


def test_routes_are_immutable() -> None:
    """The contract is a constant. Freezing it means a caller cannot quietly
    add a route at runtime and have the doc test still pass."""
    assert isinstance(proto.ROUTES, tuple)
    with pytest.raises(Exception):
        proto.ROUTES[0].method = "PATCH"  # type: ignore[misc]


# ----------------------------------------------------------------------
# Specific guarantees other parts of the plan depend on
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "method,path",
    [
        # Node plane — without these a node cannot pair or converge.
        ("POST", "/nodes"),
        ("GET", "/nodes/{node_id}/poll"),
        ("POST", "/nodes/{node_id}/heartbeat"),
        # Box plane — registration failure is fatal in interlatent-serve, and
        # /authz gates every single gRPC call on the box.
        ("POST", "/compute/boxes/register"),
        ("GET", "/compute/boxes/{box_id}/authz"),
        # The GPU server validates any presented key by calling /environments.
        ("GET", "/environments"),
        # Stopping a session must be an unassign, so it must be a route.
        ("DELETE", "/inference/sessions/{session_id}"),
    ],
)
def test_load_bearing_routes_are_mandatory(method: str, path: str) -> None:
    match = [r for r in proto.MANDATORY if r.method == method and r.path == path]
    assert match, f"{method} {path} must be mandatory"


@pytest.mark.parametrize(
    "path",
    [
        # The inbox plane. Recording goes to the box's own sink now; a
        # coordinator stamps the destination onto the session instead.
        "/episodes",
        "/episodes/{episode_id}/upload-urls",
        "/episodes/{episode_id}/upload-complete",
        "/episodes/{episode_id}/inbox-gc",
        # Analysis and dataset reads: product surface, never protocol, and
        # nothing shipping ever served them.
        "/environments/{env_id}/analyze",
        "/environments/{env_id}/process",
        "/environments/{env_id}/processing-status",
        "/environments/{env_id}/episodes",
        "/episodes/{episode_id}",
        "/episodes/{episode_id}/results",
    ],
)
def test_the_deleted_surfaces_stay_deleted(path: str) -> None:
    """ADR 0039 deleted these rather than leave them advertised. Re-adding one
    means re-adding an implementation of it, not just a row."""
    assert not [r for r in proto.ROUTES if r.path == path], (
        f"{path} came back; see ADR 0039"
    )


def test_the_recording_destination_is_in_the_contract() -> None:
    """The operator verb that replaced the inbox. It is the coordinator's own
    administration surface and belongs to neither the node nor the box plane,
    but with one implementation left it is served like everything else."""
    match = [r for r in proto.MANDATORY if r.path == "/coordinator/recording"]
    assert [r.method for r in match] == ["PUT"]


def test_the_warmup_target_survived() -> None:
    """Kept deliberately (ADR 0039): it reads as upstream code, but it is how a
    box pre-compiles the policy `session start --policy X` is about to give it,
    and how it learns the environment's camera keys."""
    assert [
        r for r in proto.MANDATORY
        if r.path == "/compute/boxes/{box_id}/warmup-target"
    ]


def test_protocol_version_is_the_documented_one() -> None:
    assert proto.PROTOCOL_VERSION == "interlatent.coordinator/2"
    assert f"`{proto.PROTOCOL_VERSION}`" in DOC.read_text()


def test_api_prefix_has_no_trailing_slash() -> None:
    assert proto.API_PREFIX == "/api/v1"
