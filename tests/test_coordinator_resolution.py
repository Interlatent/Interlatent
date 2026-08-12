"""Coordinator address resolution — one convention, one precedence order.

Two implementations exist on purpose: ADR 0023 makes ``interlatent-server``
installable without the SDK, so a GPU box cannot import
``interlatent._coordinator``. Duplication is only safe if something asserts the
two agree, which is most of what this file does.

There is no default. An unconfigured caller gets a
``CoordinatorNotConfigured`` naming its own flag, not a silent connection to
somebody else's control plane.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from interlatent import _coordinator as sdk

REPO = Path(__file__).resolve().parents[1]
_SERVER_SRC = REPO / "packages" / "server" / "src"

# Load the server twin the same way the GPU box would: as its own dist, with
# no SDK on the path. Importing it by package name would work here only
# because this repo happens to hold both.
if str(_SERVER_SRC) not in sys.path:
    sys.path.insert(0, str(_SERVER_SRC))
_spec = importlib.util.spec_from_file_location(
    "il_server_coordinator", _SERVER_SRC / "interlatent_server" / "coordinator.py"
)
srv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(srv)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("INTERLATENT_COORDINATOR", raising=False)
    monkeypatch.delenv("INTERLATENT_API_BASE", raising=False)


BOTH = pytest.mark.parametrize("mod", [sdk, srv], ids=["sdk", "server"])


# ----------------------------------------------------------------------
# normalize: one convention, both spellings accepted
# ----------------------------------------------------------------------


@BOTH
@pytest.mark.parametrize(
    "given,expected",
    [
        ("https://interlatent.com", "https://interlatent.com"),
        ("https://interlatent.com/", "https://interlatent.com"),
        ("https://interlatent.com/api/v1", "https://interlatent.com"),
        ("https://interlatent.com/api/v1/", "https://interlatent.com"),
        ("http://10.0.0.5:8900", "http://10.0.0.5:8900"),
        ("http://10.0.0.5:8900/api/v1", "http://10.0.0.5:8900"),
        ("  https://x.test/  ", "https://x.test"),
    ],
)
def test_normalize_collapses_both_historical_conventions(mod, given, expected):
    assert mod.normalize(given) == expected


@BOTH
def test_normalize_is_idempotent(mod):
    once = mod.normalize("https://x.test/api/v1/")
    assert mod.normalize(once) == once


def test_api_v1_is_idempotent_on_the_server_side():
    """The recorder builds urls by appending to this, and used to double the
    prefix when the base already carried it — a 405 on /episodes."""
    assert srv.api_v1("https://x.test") == "https://x.test/api/v1"
    assert srv.api_v1("https://x.test/api/v1") == "https://x.test/api/v1"
    assert srv.api_v1(srv.api_v1("https://x.test")) == "https://x.test/api/v1"


# ----------------------------------------------------------------------
# Precedence — identical in both implementations
# ----------------------------------------------------------------------


@BOTH
def test_explicit_beats_everything(mod, monkeypatch):
    monkeypatch.setenv("INTERLATENT_COORDINATOR", "http://env.test")
    assert mod.resolve("http://flag.test") == "http://flag.test"


@BOTH
def test_env_var_beats_the_default(mod, monkeypatch):
    monkeypatch.setenv("INTERLATENT_COORDINATOR", "http://env.test/api/v1")
    assert mod.resolve() == "http://env.test"


@BOTH
def test_legacy_env_var_still_read_but_warns(mod, monkeypatch):
    monkeypatch.setenv("INTERLATENT_API_BASE", "http://old.test")
    with pytest.warns(DeprecationWarning, match="INTERLATENT_API_BASE"):
        assert mod.resolve() == "http://old.test"


@BOTH
def test_new_env_var_wins_over_the_legacy_one(mod, monkeypatch):
    monkeypatch.setenv("INTERLATENT_COORDINATOR", "http://new.test")
    monkeypatch.setenv("INTERLATENT_API_BASE", "http://old.test")
    assert mod.resolve() == "http://new.test"


@BOTH
def test_blank_values_do_not_count_as_configured(mod, monkeypatch):
    monkeypatch.setenv("INTERLATENT_COORDINATOR", "   ")
    with pytest.raises(mod.CoordinatorNotConfigured):
        mod.resolve("")


# ----------------------------------------------------------------------
# The fallback, and its removal
# ----------------------------------------------------------------------


@BOTH
def test_an_unconfigured_caller_raises_with_a_remedy(mod):
    """The whole point of the step. With a default, "one code path" is an
    aspiration nobody can test; without one, pointing at a coordinator is the
    only way anything runs."""
    with pytest.raises(mod.CoordinatorNotConfigured) as exc:
        mod.resolve()
    assert "INTERLATENT_COORDINATOR" in str(exc.value)


@BOTH
def test_no_address_is_named_as_somewhere_to_point_at(mod):
    """There is one coordinator implementation and you run it, so no
    remediation sentence may name a host to point at — a suggested address
    is a default with extra steps (ADR 0039)."""
    for purpose_error in _every_remedy(mod):
        assert "interlatent.com" not in purpose_error
        assert "dashboard" not in purpose_error.lower()


def _every_remedy(mod) -> list[str]:
    """The remediation sentence(s) a module can produce. The SDK keys them
    by ``purpose``; the server twin has exactly one."""
    if not hasattr(mod, "_REMEDIES"):
        return [mod._REMEDY]
    return [mod._remedy(p) for p in mod._REMEDIES]


def test_sdk_error_names_the_flag_the_caller_is_actually_looking_at():
    """'set INTERLATENT_COORDINATOR' is unhelpful when the user is staring at
    interlatent-serve and the flag is --coordinator."""
    seen = {}
    for purpose in ("node", "serve", "cli", "connect", "preflight", "client"):
        with pytest.raises(sdk.CoordinatorNotConfigured) as exc:
            sdk.resolve(purpose=purpose)
        seen[purpose] = str(exc.value)

    assert "interlatent-node" in seen["node"]
    assert "interlatent-serve" in seen["serve"]
    assert "connect_drtc()" in seen["connect"]
    assert "interlatent-preflight" in seen["preflight"]
    assert "base_url=" in seen["client"]
    assert "interlatent up" in seen["cli"]
    # Distinct messages, not one generic string wearing six hats.
    assert len(set(seen.values())) == len(seen)


# ----------------------------------------------------------------------
# The two implementations must not drift
# ----------------------------------------------------------------------


def test_both_implementations_agree_on_the_public_constants():
    assert sdk.ENV_VAR == srv.ENV_VAR
    assert sdk.LEGACY_ENV_VAR == srv.LEGACY_ENV_VAR
    # Neither twin carries an address of its own: the pair went from
    # "no implicit default, but here is the hosted one" to "no address".
    assert not hasattr(sdk, "HOSTED_COORDINATOR")
    assert not hasattr(srv, "HOSTED_COORDINATOR")


@pytest.mark.parametrize(
    "given",
    [
        "https://interlatent.com",
        "https://interlatent.com/api/v1/",
        "http://10.0.0.5:8900/",
        "  http://x.test  ",
    ],
)
def test_both_implementations_normalize_identically(given):
    assert sdk.normalize(given) == srv.normalize(given)


def test_server_twin_needs_no_sdk_on_the_path():
    """A GPU box installs interlatent-server alone. If the twin ever grows an
    `interlatent.` import this fails, which is the point."""
    source = (_SERVER_SRC / "interlatent_server" / "coordinator.py").read_text()
    assert "import interlatent." not in source
    assert "from interlatent." not in source
