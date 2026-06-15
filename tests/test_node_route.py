"""Node-side route resolution precedence (env > session route > legacy > cfg)."""
from interlatent.node.daemon import NodeDaemon, NodeDaemonConfig


def _daemon(drtc_url=None):
    return NodeDaemon(
        NodeDaemonConfig(node_id="n", token="t", api_base="http://x", drtc_url=drtc_url)
    )


def test_session_route_used():
    d = _daemon()
    s = {"route": {"method": "direct", "address": "gpu:1"}, "drtc_endpoint": "gpu:1"}
    assert d._resolve_route(s) == {"method": "direct", "address": "gpu:1"}
    assert d._resolve_endpoint(s) == "gpu:1"


def test_legacy_drtc_endpoint_fallback():
    d = _daemon()
    r = d._resolve_route({"drtc_endpoint": "legacy:2"})
    assert r["method"] == "direct" and r["address"] == "legacy:2"


def test_cfg_drtc_url_fallback():
    d = _daemon(drtc_url="cfg:3")
    assert d._resolve_route({})["address"] == "cfg:3"


def test_none_when_nothing_resolves():
    assert _daemon()._resolve_route({}) is None
    assert _daemon()._resolve_endpoint({}) == ""


def test_env_var_overrides_session_route(monkeypatch):
    monkeypatch.setenv("INTERLATENT_DRTC_URL", "env:9")
    d = _daemon(drtc_url="cfg:3")
    s = {"route": {"method": "direct", "address": "gpu:1"}}
    assert d._resolve_route(s) == {"method": "direct", "address": "env:9"}
