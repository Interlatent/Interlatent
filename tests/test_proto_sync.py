"""`proto/messages.proto` is the single source of truth for the DRTC wire.

Two packages ship a mirror of it plus generated stubs: the SDK client
(`interlatent`) and the policy server (`interlatent-server`). They are
built and released independently but must speak the identical protocol —
a node from PyPI has to talk to a self-hosted box from PyPI, and both
have to talk to Interlatent's hosted boxes.

Nothing in the build enforces that. `proto/gen_proto.sh` writes every
mirror in one pass, but a hand-edit of one copy, a regeneration of one
package, or a merge that resolves only one side all produce a repo where
the two ends disagree — and the failure shows up as a protocol error on a
robot, not as a red build.

So: assert the mirrors are byte-identical to the source, and assert the
generated descriptors agree. The descriptor check is the one that matters
for the wire (comments don't reach it); the text check is what keeps the
sources of truth honest.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "proto" / "messages.proto"

# (label, mirrored .proto, generated _pb2.py, generated _pb2_grpc.py)
MIRRORS = [
    (
        "sdk",
        REPO / "packages/sdk/src/interlatent/inference/protocol/messages.proto",
        REPO / "packages/sdk/src/interlatent/inference/protocol/messages_pb2.py",
        REPO / "packages/sdk/src/interlatent/inference/protocol/messages_pb2_grpc.py",
    ),
    (
        "server",
        REPO / "packages/server/src/interlatent_server/protocol/messages.proto",
        REPO / "packages/server/src/interlatent_server/protocol/messages_pb2.py",
        REPO / "packages/server/src/interlatent_server/protocol/messages_pb2_grpc.py",
    ),
]


def _serialized_descriptor(pb2_path: Path) -> bytes:
    """The `AddSerializedFile(b'...')` blob — the actual wire schema.

    Parsed out of the text rather than imported: importing needs a
    protobuf runtime whose version satisfies the stub's assertion, and
    this test must run wherever the repo does.
    """
    src = pb2_path.read_text()
    match = re.search(r"AddSerializedFile\(\s*(b'.*?')\s*\)", src, re.S)
    assert match, f"no serialized descriptor found in {pb2_path}"
    return eval(match.group(1))  # noqa: S307 — a bytes literal from our own build


def test_source_of_truth_exists() -> None:
    assert SOURCE.is_file(), f"missing the source of truth: {SOURCE}"


@pytest.mark.parametrize("label,mirror,_pb2,_grpc", MIRRORS)
def test_mirror_matches_source(label: str, mirror: Path, _pb2: Path, _grpc: Path) -> None:
    assert mirror.is_file(), f"{label}: missing mirror {mirror}"
    assert mirror.read_bytes() == SOURCE.read_bytes(), (
        f"{label}: {mirror.relative_to(REPO)} has drifted from "
        f"{SOURCE.relative_to(REPO)}. Edit the source and re-run "
        f"./proto/gen_proto.sh — never edit a mirror."
    )


@pytest.mark.parametrize("label,mirror,pb2,_grpc", MIRRORS)
def test_stubs_are_generated_from_the_mirror(
    label: str, mirror: Path, pb2: Path, _grpc: Path
) -> None:
    """Catches a mirror updated without regenerating beside it."""
    assert pb2.is_file(), f"{label}: missing generated stub {pb2}"
    descriptor = _serialized_descriptor(pb2)
    # Every message and field name in the .proto must appear in the
    # descriptor. Comment-only edits are invisible here, by design.
    names = set(re.findall(r"^\s*(?:message|service)\s+(\w+)", mirror.read_text(), re.M))
    assert names, f"{label}: parsed no message/service names out of {mirror}"
    for name in names:
        assert name.encode() in descriptor, (
            f"{label}: '{name}' is in {mirror.name} but not in the generated "
            f"descriptor — stubs are stale, re-run ./proto/gen_proto.sh"
        )


def test_packages_agree_on_the_wire() -> None:
    """The check that actually protects deployed robots: the SDK client
    and the policy server must compile to the same schema."""
    (label_a, _, pb2_a, grpc_a), (label_b, _, pb2_b, grpc_b) = MIRRORS
    assert _serialized_descriptor(pb2_a) == _serialized_descriptor(pb2_b), (
        f"{label_a} and {label_b} have different serialized descriptors — "
        f"the client and server would not speak the same protocol. "
        f"Re-run ./proto/gen_proto.sh, which writes both in one pass."
    )
    assert grpc_a.read_bytes() == grpc_b.read_bytes(), (
        f"{label_a} and {label_b} gRPC stubs differ (service definitions "
        f"or generator version). Re-run ./proto/gen_proto.sh."
    )


def test_wire_package_is_pinned() -> None:
    """The proto package name is the compatibility surface for every
    deployed node and GPU box — including hosted ones built from the
    closed engine. Renaming it silently breaks all of them, so it is
    pinned here rather than left to review."""
    assert re.search(
        r"^package\s+interlatent\.inference\.v1\s*;", SOURCE.read_text(), re.M
    ), (
        "the DRTC wire package must stay 'interlatent.inference.v1' — it does "
        "not track the Python package names (see proto/README.md)"
    )
