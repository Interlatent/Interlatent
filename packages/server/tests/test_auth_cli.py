"""Functional tests for the self-hosted server's auth + CLI plumbing.

Covers the pieces that don't need a GPU, a policy, or a live backend:

  - credentials.resolve()          — identity precedence (admin > owner > none)
  - auth.validate_key_for_box      — authz-probe URL/headers/status mapping
  - auth.build_box_key_validator   — TTL cache behavior
  - auth.wrap_servicer_with_auth   — every RPC gated, UNAUTHENTICATED on bad key
  - box_status.report_status       — reportable-state + system-"stopped" filtering
  - cli._resolve_box_id            — mint/persist/override semantics
  - cli._register                  — payload shape + failure modes

Run:  python packages/server/tests/test_auth_cli.py
Needs grpcio + httpx importable (no torch, no lerobot).
"""
from __future__ import annotations

import asyncio
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest.mock as mock
import urllib.error
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import interlatent_server.credentials as credentials  # noqa: E402
import interlatent_server.box_status as box_status  # noqa: E402
import interlatent_server.cli as cli  # noqa: E402

# Load auth.py directly by file path: importing interlatent_server.server
# would pull in the policy backends (torch etc.), which these tests
# deliberately avoid.
_auth_spec = importlib.util.spec_from_file_location(
    "il_server_auth", SRC / "interlatent_server" / "server" / "auth.py"
)
auth = importlib.util.module_from_spec(_auth_spec)
_auth_spec.loader.exec_module(auth)

PASSED = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASSED
    if not cond:
        print(f"  [FAIL] {name}" + (f" — {detail}" if detail else ""))
        sys.exit(1)
    PASSED += 1
    print(f"  [PASS] {name}")


class _Env:
    """Temporarily set/clear the box-identity env vars."""

    VARS = (
        "INTERLATENT_BOX_ID",
        "INTERLATENT_API_BASE",
        "INTERLATENT_ADMIN_KEY",
        "INTERLATENT_API_KEY",
    )

    def __init__(self, **kv: str):
        self.kv = kv

    def __enter__(self):
        self.saved = {v: os.environ.get(v) for v in self.VARS}
        for v in self.VARS:
            os.environ.pop(v, None)
        os.environ.update(self.kv)

    def __exit__(self, *exc):
        for v, old in self.saved.items():
            if old is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = old


# ---------------------------------------------------------------- credentials
print("credentials.resolve()")

with _Env():
    check("no identity -> None", credentials.resolve() is None)

with _Env(INTERLATENT_BOX_ID="b1", INTERLATENT_API_BASE="https://x.com/"):
    check("box id + base but no key -> None", credentials.resolve() is None)

with _Env(
    INTERLATENT_BOX_ID="b1",
    INTERLATENT_API_BASE="https://x.com/",
    INTERLATENT_ADMIN_KEY="sys-secret",
    INTERLATENT_API_KEY="ilat_owner",
):
    c = credentials.resolve()
    check("admin key wins over owner key", c.api_key == "sys-secret" and c.is_system)
    check("api_root strips trailing slash", c.api_root == "https://x.com")

with _Env(
    INTERLATENT_BOX_ID="b1",
    INTERLATENT_API_BASE="https://x.com",
    INTERLATENT_API_KEY="ilat_owner",
):
    c = credentials.resolve()
    check("owner key -> is_system=False", c.api_key == "ilat_owner" and not c.is_system)


# ---------------------------------------------------- auth.validate_key_for_box
print("auth.validate_key_for_box()")


def _fake_get(status_code):
    resp = mock.Mock(status_code=status_code)
    return mock.Mock(return_value=resp)


# auth imports httpx lazily inside the function, so patching httpx.get
# on the real module is enough.
with mock.patch("httpx.get", _fake_get(200)) as g:
    ok = auth.validate_key_for_box("ilat_k", box_id="boxA", api_base="https://x.com")
    check("200 -> True", ok is True)
    url = g.call_args.args[0]
    check(
        "bare api_base gets /api/v1 + box path",
        url == "https://x.com/api/v1/compute/boxes/boxA/authz",
        url,
    )
    check(
        "key travels as X-Api-Key header",
        g.call_args.kwargs["headers"] == {"X-Api-Key": "ilat_k"},
    )

with mock.patch("httpx.get", _fake_get(200)) as g:
    auth.validate_key_for_box("k", box_id="b", api_base="https://x.com/api/v1")
    check(
        "api_base already ending in /api/v1 not doubled",
        g.call_args.args[0] == "https://x.com/api/v1/compute/boxes/b/authz",
        g.call_args.args[0],
    )

for code in (401, 403, 404):
    with mock.patch("httpx.get", _fake_get(code)):
        check(
            f"{code} -> False",
            auth.validate_key_for_box("k", box_id="b", api_base="https://x.com") is False,
        )

import httpx as _httpx  # noqa: E402

with mock.patch("httpx.get", mock.Mock(side_effect=_httpx.ConnectError("down"))):
    check(
        "network failure -> False (closed)",
        auth.validate_key_for_box("k", box_id="b", api_base="https://x.com") is False,
    )

with mock.patch("httpx.get", _fake_get(200)) as g:
    check(
        "empty token -> False without probing",
        auth.validate_key_for_box("", box_id="b", api_base="https://x.com") is False
        and g.call_count == 0,
    )


# ------------------------------------------------- auth.build_box_key_validator
print("auth.build_box_key_validator() cache")

with mock.patch("httpx.get", _fake_get(200)) as g:
    checkfn = auth.build_box_key_validator(box_id="b", api_base="https://x.com")
    r1, r2 = checkfn("ilat_k"), checkfn("ilat_k")
    check("both calls True", r1 is True and r2 is True)
    check("second call served from cache (1 probe)", g.call_count == 1)
    checkfn("ilat_other")
    check("different key probes separately", g.call_count == 2)

# TTL expiry: freeze auth's clock, validate, advance past TTL, expect re-probe.
_real_time = auth.time
try:
    now = {"t": 1000.0}
    auth.time = mock.Mock(time=lambda: now["t"])
    with mock.patch("httpx.get", _fake_get(200)) as g:
        checkfn = auth.build_box_key_validator(
            box_id="b", api_base="https://x.com", ttl_s=60.0
        )
        checkfn("k")
        now["t"] += 59.0
        checkfn("k")
        check("within TTL -> cached", g.call_count == 1)
        now["t"] += 2.0
        checkfn("k")
        check("past TTL -> re-probed", g.call_count == 2)
    # A denial is cached too (a revoked key stays locked out, and a burst of
    # bad-key RPCs can't hammer the backend).
    with mock.patch("httpx.get", _fake_get(403)) as g:
        checkfn = auth.build_box_key_validator(box_id="b", api_base="https://x.com")
        checkfn("bad")
        checkfn("bad")
        check("denial cached (1 probe for 2 rejections)", g.call_count == 1)
finally:
    auth.time = _real_time


# ------------------------------------------------- auth.wrap_servicer_with_auth
print("auth.wrap_servicer_with_auth()")

import grpc  # noqa: E402


class AbortError(Exception):
    def __init__(self, code, detail):
        self.code, self.detail = code, detail


class FakeContext:
    def __init__(self, key: str | None):
        self.key = key

    def invocation_metadata(self):
        return [("x-api-key", self.key)] if self.key is not None else []

    async def abort(self, code, detail):
        raise AbortError(code, detail)


class FakeServicer:
    """Unary RPCs + the bidi Stream, mirroring InferenceServicer's surface.
    RecordTick intentionally absent — the wrapper must tolerate that."""

    def __init__(self):
        self.calls = []

    async def OpenSession(self, request, context):
        self.calls.append("OpenSession")
        return "open-ok"

    async def Infer(self, request, context):
        self.calls.append("Infer")
        return "infer-ok"

    async def RecordTicks(self, request, context):
        self.calls.append("RecordTicks")
        return "ticks-ok"

    async def CloseSession(self, request, context):
        self.calls.append("CloseSession")
        return "close-ok"

    async def Stream(self, request_iterator, context):
        self.calls.append("Stream")
        yield "s1"
        yield "s2"


async def _run_wrap_tests():
    sv = FakeServicer()
    auth.wrap_servicer_with_auth(sv, check_token=lambda t: t == "good")

    out = await sv.OpenSession("req", FakeContext("good"))
    check("valid key: unary passes through", out == "open-ok")

    for name in ("OpenSession", "Infer", "RecordTicks", "CloseSession"):
        sv2_calls_before = list(sv.calls)
        try:
            await getattr(sv, name)("req", FakeContext("wrong"))
            check(f"bad key: {name} aborted", False, "no abort raised")
        except AbortError as e:
            check(
                f"bad key: {name} -> UNAUTHENTICATED",
                e.code is grpc.StatusCode.UNAUTHENTICATED,
            )
        check(f"bad key: {name} handler never ran", sv.calls == sv2_calls_before)

    try:
        await getattr(sv, "OpenSession")("req", FakeContext(None))
        check("missing key metadata aborted", False, "no abort raised")
    except AbortError:
        check("missing key metadata aborted", True)

    got = [x async for x in sv.Stream(iter(()), FakeContext("good"))]
    check("valid key: Stream yields", got == ["s1", "s2"])
    try:
        async for _ in sv.Stream(iter(()), FakeContext("wrong")):
            pass
        check("bad key: Stream aborted before first yield", False, "no abort")
    except AbortError as e:
        check(
            "bad key: Stream aborted before first yield",
            e.code is grpc.StatusCode.UNAUTHENTICATED,
        )


asyncio.run(_run_wrap_tests())


# ------------------------------------------------------ box_status.report_status
print("box_status.report_status()")

_posts = []
_real_post = box_status._post
box_status._post = lambda *a: _posts.append(a)
try:
    OWNER = dict(
        INTERLATENT_BOX_ID="b1",
        INTERLATENT_API_BASE="https://x.com",
        INTERLATENT_API_KEY="ilat_owner",
    )
    SYSTEM = dict(
        INTERLATENT_BOX_ID="b1",
        INTERLATENT_API_BASE="https://x.com",
        INTERLATENT_ADMIN_KEY="sys-secret",
    )

    with _Env(**OWNER):
        box_status.report_status("ready", wait=True)
        check("owner 'ready' posted", _posts[-1][0] == "ready")
        box_status.report_status("stopped", detail="bye", wait=True)
        check("owner 'stopped' posted (graceful BYO exit)", _posts[-1][0] == "stopped")
        n = len(_posts)
        box_status.report_status("warming_up", wait=True)
        box_status.report_status("error", wait=True)
        check("backend-owned states never posted", len(_posts) == n)

    with _Env(**SYSTEM):
        n = len(_posts)
        box_status.report_status("stopped", wait=True)
        check("system-key 'stopped' ignored", len(_posts) == n)
        box_status.report_status("running", wait=True)
        check("system-key 'running' posted", _posts[-1][0] == "running")

    with _Env():
        n = len(_posts)
        box_status.report_status("ready", wait=True)
        check("no identity -> no-op", len(_posts) == n)
finally:
    box_status._post = _real_post


# ---------------------------------------------------------- cli._resolve_box_id
print("cli._resolve_box_id()")

with tempfile.TemporaryDirectory() as td:
    box_path = Path(td) / ".interlatent" / "box-id"
    with mock.patch.object(cli, "BOX_ID_PATH", box_path):
        check("explicit override wins", cli._resolve_box_id("  my-id  ") == "my-id")
        check("override does not persist", not box_path.exists())

        minted = cli._resolve_box_id("")
        check(
            "mints a uuid and persists it",
            box_path.read_text().strip() == minted and len(minted) == 36,
        )
        check("restart reuses the persisted id", cli._resolve_box_id("") == minted)

        box_path.write_text("   \n")
        reminted = cli._resolve_box_id("")
        check(
            "blank persisted file re-mints",
            reminted != minted and box_path.read_text().strip() == reminted,
        )


# ---------------------------------------------------------------- cli._register
print("cli._register()")

REG = dict(
    api_base="https://x.com/",
    api_key="ilat_k",
    box_id="box-1",
    name="my-gpu",
    endpoint="203.0.113.7:50051",
    gpu_model="RTX 4090",
    warmup_policy=None,
)


def _ok_response(body: dict):
    resp = mock.MagicMock()
    resp.__enter__.return_value = resp
    resp.read.return_value = json.dumps(body).encode()
    return resp


with mock.patch(
    "urllib.request.urlopen",
    mock.Mock(return_value=_ok_response({"name": "my-gpu", "status": "ready"})),
) as uo:
    cli._register(**REG)
    req = uo.call_args.args[0]
    check(
        "registration URL",
        req.full_url == "https://x.com/api/v1/compute/boxes/register",
        req.full_url,
    )
    check("authenticates with x-api-key", req.get_header("X-api-key") == "ilat_k")
    payload = json.loads(req.data.decode())
    check(
        "payload: byo provider + box identity",
        payload["provider"] == "byo"
        and payload["box_id"] == "box-1"
        and payload["endpoint"] == "203.0.113.7:50051"
        and payload["gpu_model"] == "RTX 4090"
        and payload["warmup_policy"] is None,
    )

http_401 = urllib.error.HTTPError(
    "https://x.com", 401, "unauth", {}, io.BytesIO(b'{"detail":"Invalid API key"}')
)
with mock.patch("urllib.request.urlopen", mock.Mock(side_effect=http_401)):
    try:
        cli._register(**REG)
        check("HTTP 401 -> SystemExit", False, "no exit raised")
    except SystemExit as e:
        msg = str(e)
        check(
            "HTTP 401 -> SystemExit with actionable message",
            "HTTP 401" in msg and "Invalid API key" in msg and "INTERLATENT_API_KEY" in msg,
        )

with mock.patch(
    "urllib.request.urlopen", mock.Mock(side_effect=urllib.error.URLError("refused"))
):
    try:
        cli._register(**REG)
        check("network failure -> SystemExit", False, "no exit raised")
    except SystemExit as e:
        check("network failure -> SystemExit naming the URL", "Could not reach" in str(e))


print(f"All {PASSED} server auth/CLI checks passed.")


def test_all_checks_passed() -> None:
    """Pytest handle for the module-level checks above.

    Every check in this file runs at import time (the file doubles as a
    standalone script) and ``check()`` exits non-zero on the first
    failure, so under pytest a failure surfaces as a collection error.
    This turns the *passing* case into a reported test rather than
    "no tests ran" — which is indistinguishable from the file having
    been deleted.
    """
    assert PASSED > 0
