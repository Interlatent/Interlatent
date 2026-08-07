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
    known = {
        proto.TIER_MANDATORY,
        proto.TIER_OPTIONAL,
        proto.TIER_COORDINATOR_ONLY,
    }
    assert {r.tier for r in proto.ROUTES} <= known


def test_tiers_partition_the_route_table() -> None:
    assert (
        len(proto.MANDATORY) + len(proto.OPTIONAL) + len(proto.COORDINATOR_ONLY)
        == len(proto.ROUTES)
    )


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
        # Teleop is disabled for the session on 404 (node/teleop/factory.py
        # treats 401/403/404 as definitive), so it must not be mandatory.
        "/inference/sessions/{session_id}/teleop-token",
        # A coordinator that stamps a recording destination onto every session
        # needs no inbox at all.
        "/episodes",
        # Hosted analysis has no self-hosted counterpart.
        "/environments/{env_id}/analyze",
    ],
)
def test_degradable_surfaces_are_optional(path: str) -> None:
    tiers = {r.tier for r in proto.ROUTES if r.path == path}
    assert tiers == {proto.TIER_OPTIONAL}, f"{path} should be optional, got {tiers}"


def test_recording_destination_is_coordinator_only() -> None:
    """The hosted dashboard 404s this one; the CLI must be able to say so by
    name rather than reporting a bare 404."""
    assert [r.path for r in proto.COORDINATOR_ONLY] == ["/coordinator/recording"]


def test_protocol_version_is_the_documented_one() -> None:
    assert proto.PROTOCOL_VERSION == "interlatent.coordinator/1"
    assert f"`{proto.PROTOCOL_VERSION}`" in DOC.read_text()


def test_api_prefix_has_no_trailing_slash() -> None:
    assert proto.API_PREFIX == "/api/v1"
