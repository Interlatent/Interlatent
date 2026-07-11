"""Nori-Protocol conformance: the adapter's wire codec vs the vendored contract.

Two directions, mirroring how the daemon/NoriLeLab consume the protocol repo:

- OUTBOUND: every frame `adapters.nori.protocol` can emit must validate against
  the strict vendored schemas (Draft 2020-12, additionalProperties:false).
  `bye` has no schema upstream, so it gets an exact-shape assertion.
- INBOUND: every golden fixture must replay through `parse_line` without
  raising; daemon->client types yield typed frames, client-bound and PROPOSED
  types yield None (runtime leniency rule from the protocol README).

Vendored contract lives at tests/fixtures/nori_protocol/ (see its README for
the source commit + re-sync procedure). jsonschema is a dev-only dependency:
absent -> skip, never fail.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

jsonschema = pytest.importorskip("jsonschema")

from interlatent.adapters.nori import protocol as np_  # noqa: E402 (after importorskip)
from interlatent.node.teleop.robot_profile import get_profile  # noqa: E402

CONTRACT = Path(__file__).parent / "fixtures" / "nori_protocol"
SCHEMAS = {p.stem: json.loads(p.read_text()) for p in sorted((CONTRACT / "schema").glob("*.json"))}
FIXTURES = sorted((CONTRACT / "fixtures").glob("*.json"))


def _validate(frame: dict) -> None:
    kind = frame["type"]
    assert kind in SCHEMAS, f"no vendored schema for type={kind!r}"
    jsonschema.Draft202012Validator(SCHEMAS[kind]).validate(frame)


# --------------------------------------------------------------------------- #
# Outbound builders                                                            #
# --------------------------------------------------------------------------- #


def test_vendored_version_matches_codec():
    assert int((CONTRACT / "VERSION").read_text().strip()) == np_.PROTOCOL_VERSION


def test_schemas_are_themselves_valid():
    for name, schema in SCHEMAS.items():
        jsonschema.Draft202012Validator.check_schema(schema)


@pytest.mark.parametrize("token", [None, "sekrit-token"])
def test_hello_validates(token):
    _validate(np_.make_hello(token=token))


def test_hello_carries_decided_fields():
    hello = np_.make_hello(token="t")
    assert hello["input_mode"] == "vr" and hello["mode"] == "lan"
    assert hello["protocol_version"] == 1 and hello["bus_choice"] == "3"


def test_control_action_validates():
    profile = get_profile("nori")
    action = {f"{n}.pos": 1.5 for n in profile.joint_names}
    frame = np_.make_control_action(7, action)
    _validate(frame)
    assert len(frame["action"]) == 12


def test_keepalive_validates_and_is_motion_free():
    frame = np_.make_keepalive(123)
    _validate(frame)
    assert set(frame) == {"type", "seq"}  # no jog/action/leader/reset keys


def test_estop_is_schema_canonical():
    frame = np_.make_estop()
    _validate(frame)
    # The load-bearing assertion: canonical name-enum form, not the legacy
    # boolean form ({"estop": true}) the TS SDK sends.
    assert frame == {"type": "command", "name": "estop"}


def test_reset_latch_requires_token():
    _validate(np_.make_reset_latch("tok"))
    with pytest.raises(np_.NoriProtocolError):
        np_.make_reset_latch("")
    # And the schema itself enforces the conditional: name=reset_latch without
    # a token must be rejected by the vendored contract.
    bare = {"type": "command", "name": "reset_latch"}
    with pytest.raises(jsonschema.ValidationError):
        _validate(bare)


def test_bye_exact_shape():
    # No bye.json exists upstream (CLIENTS.md-only message) — exact shape.
    assert np_.make_bye() == {"type": "bye"}


def test_encode_frame_is_ndjson():
    raw = np_.encode_frame(np_.make_keepalive(1))
    assert raw.endswith(b"\n") and raw.count(b"\n") == 1
    json.loads(raw.decode("utf-8"))


# --------------------------------------------------------------------------- #
# Inbound golden replay                                                        #
# --------------------------------------------------------------------------- #

# Client->daemon and PROPOSED-but-unconsumed types must parse to None without
# raising; daemon->client types must yield their typed frame.
_EXPECT_TYPED = {"ack": np_.Ack, "telemetry": np_.Telemetry, "error": np_.ErrorFrame,
                 "action_status": np_.ActionStatus}


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_golden_fixture_replay(path):
    obj = json.loads(path.read_text())
    result = np_.parse_line(path.read_text())
    expected = _EXPECT_TYPED.get(obj["type"])
    if expected is None:
        assert result is None
    else:
        assert isinstance(result, expected)


def test_golden_ack_fields():
    ack = np_.parse_line((CONTRACT / "fixtures" / "ack.json").read_text())
    assert ack.accepted is True and ack.protocol_version == 1
    assert ack.norm_mode == "range_m100_100"
    assert ack.watchdog == np_.WatchdogProfile(t_warn_ms=300, t_stop_ms=1000)
    assert len(ack.joints) == 12 and ack.cameras == ("front", "right_wrist")
    assert ack.ranges["left_arm_gripper.pos"] == (0.0, 100.0)
    assert ack.initial_state["right_arm_shoulder_pan.pos"] == pytest.approx(12.4)


def test_golden_telemetry_fields():
    tel = np_.parse_line((CONTRACT / "fixtures" / "telemetry_periodic.json").read_text())
    assert tel.status["safety"] == "ok" and tel.status["watchdog"] == "ok"
    assert tel.state["right_arm_shoulder_pan.pos"] == pytest.approx(12.4)
    assert tel.currents == {"right_arm_gripper": 90}


def test_golden_fatal_error():
    err = np_.parse_line(
        (CONTRACT / "fixtures" / "error_version_mismatch.json").read_text()
    )
    assert err.code == "version_mismatch" and err.fatal is True


def test_parse_line_never_raises_on_garbage():
    for garbage in (b"", b"not json\n", b"[1,2]", b'{"no_type":1}',
                    b'{"type":"perception","ts_ns":1,"objects":[]}', b"\xff\xfe"):
        assert np_.parse_line(garbage) is None
