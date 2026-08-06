"""make_teleop_channel probe classification (node/teleop/factory.py).

The one-shot probe mint must only disable teleop for the session on a
*definitive* answer (deployment not QUIC-configured, auth refusal). A 409
while the session row isn't ``active`` yet — the normal race at session start —
or any other transient failure must build the channel optimistically, because
the channel's child process re-mints with its own retry/backoff and never
reads the probe's result. Returning ``None`` on a startup 409 used to disable
teleop (and therefore intervention) for the entire session.
"""
from __future__ import annotations

from unittest import mock

import httpx
import pytest

from interlatent.node.teleop import factory


def _mk(**kw):
    return factory.make_teleop_channel(
        session_id="sess-1",
        api_base="https://api.example.test",
        api_key="k",
        **kw,
    )


def _status_error(code: int) -> httpx.HTTPStatusError:
    req = httpx.Request("POST", "https://api.example.test/t")
    resp = httpx.Response(code, request=req)
    return httpx.HTTPStatusError(f"HTTP {code}", request=req, response=resp)


@pytest.fixture
def channel_cls():
    """Stub the QuicTeleopChannel constructor + the aioquic find_spec probe."""
    with mock.patch.object(factory, "mint_teleop_token") as mint, \
            mock.patch("importlib.util.find_spec", return_value=object()):
        with mock.patch(
            "interlatent.node.teleop.quic_channel.QuicTeleopChannel"
        ) as cls:
            cls.return_value = mock.sentinel.channel
            yield mint, cls


def test_quic_deployment_builds_channel(channel_cls):
    mint, _ = channel_cls
    mint.return_value = {
        "transport": "quic", "webtransport_url": "https://relay/teleop",
    }
    assert _mk() is mock.sentinel.channel


def test_non_quic_transport_disables_teleop(channel_cls):
    mint, _ = channel_cls
    mint.return_value = {"transport": "ws", "webtransport_url": None}
    assert _mk() is None


def test_definitive_refusal_disables_teleop(channel_cls):
    mint, _ = channel_cls
    for code in (401, 403, 404):
        mint.side_effect = _status_error(code)
        assert _mk() is None, f"HTTP {code} must disable teleop"


def test_transient_statuses_build_channel_optimistically(channel_cls):
    mint, _ = channel_cls
    for code in (409, 408, 425, 429, 500, 503):
        mint.side_effect = _status_error(code)
        assert _mk() is mock.sentinel.channel, (
            f"HTTP {code} is transient — the channel's own mint loop retries"
        )


def test_network_error_builds_channel_optimistically(channel_cls):
    mint, _ = channel_cls
    mint.side_effect = httpx.ConnectError("no route")
    assert _mk() is mock.sentinel.channel
