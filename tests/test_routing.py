"""Routing-method seam: descriptors, the direct method, and extensibility."""
import pytest

from interlatent import routing


def test_direct_resolve_and_connect():
    d = routing.make_descriptor("host:50051")
    assert d == {"method": "direct", "address": "host:50051"}
    r = routing.resolve(d)
    assert r["method"] == "direct" and r["address"] == "host:50051"
    # Node-side: the connector yields exactly what connect_drtc needs.
    assert routing.connect_params(r) == {"server_address": "host:50051"}


def test_direct_is_a_known_method():
    assert "direct" in routing.known_methods()


def test_unknown_method_raises_both_sides():
    with pytest.raises(ValueError):
        routing.resolve({"method": "relay", "address": "x"})
    with pytest.raises(ValueError):
        routing.connect_params({"method": "relay", "address": "x"})


def test_register_custom_method():
    routing.register_method(
        "test_upper",
        resolver=lambda d: {"method": "test_upper", "address": d["address"].upper()},
        connector=lambda r: {"server_address": r["address"]},
    )
    assert "test_upper" in routing.known_methods()
    r = routing.resolve({"method": "test_upper", "address": "abc:1"})
    assert r["address"] == "ABC:1"
    assert routing.connect_params(r)["server_address"] == "ABC:1"
