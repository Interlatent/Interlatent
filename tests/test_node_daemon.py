"""Tests for the node daemon (``interlatent/node/daemon.py``).

The daemon is what makes a robot reachable: it heartbeats, long-polls for
its assignment, and converges the local control loop onto whatever the
coordinator says it should be running. It had no tests beyond
``test_node_route.py``'s route-precedence checks — so the convergence table
in its own module docstring, the ADR 0023 spool gate, and the whole
OpenSession metadata assembly were unverified.

No network, no robot, no GPU: ``_http`` is a fake that returns scripted
responses, ``connect_drtc`` and the teleop factory are stubs, and the
control loop is a function that records its kwargs and returns.

Route resolution itself lives in ``test_node_route.py`` and is not repeated
here.
"""
from __future__ import annotations

import asyncio
import sys
import types

import pytest

from interlatent.node import daemon as dmod
from interlatent.node.daemon import NodeDaemon, NodeDaemonConfig, _ControlLoopHandle


class _StopLoop(BaseException):
    """Breaks out of a `while True:` daemon loop.

    Deliberately a BaseException: the loops catch `Exception` and retry, so
    anything narrower would just be swallowed as a network blip.
    """


def _cfg(**over) -> NodeDaemonConfig:
    base = dict(
        node_id="node-1",
        token="node-token",
        api_base="https://interlatent.test",
        drtc_api_key="ilat_user",
        # Zero the timers so the loops don't actually wait.
        heartbeat_period_s=0.0,
        reconnect_backoff_s=0.0,
        max_backoff_s=0.0,
    )
    base.update(over)
    return NodeDaemonConfig(**base)


def _daemon(**over) -> NodeDaemon:
    return NodeDaemon(_cfg(**over))


class _FakeResponse:
    def __init__(self, status_code: int = 200, body: dict | None = None) -> None:
        self.status_code = status_code
        self._body = body if body is not None else {}
        self.text = str(self._body)

    def json(self) -> dict:
        return self._body


class _FakeHttp:
    """Scripted stand-in for the daemon's httpx.AsyncClient.

    Each scripted entry is returned in order; running off the end raises
    _StopLoop, which is how a `while True:` loop under test terminates.
    """

    def __init__(self, script: list) -> None:
        self.script = list(script)
        self.posts: list[tuple[str, dict]] = []
        self.gets: list[tuple[str, dict]] = []

    def _next(self):
        if not self.script:
            raise _StopLoop
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    async def post(self, url, json=None):
        self.posts.append((url, json or {}))
        return self._next()

    async def get(self, url, params=None):
        self.gets.append((url, params or {}))
        return self._next()

    async def aclose(self):
        pass


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------


def test_client_carries_the_node_token() -> None:
    plain = _daemon()
    assert plain._http.headers["x-api-key"] == "node-token"


def test_reachable_addresses_runs_on_any_host() -> None:
    """Informational only, but it runs at every daemon start — it must not
    raise on a box with no route or no resolvable hostname."""
    host, ips = dmod._reachable_addresses()
    assert isinstance(host, str) and host
    assert all(isinstance(ip, str) for ip in ips)


def test_reachable_addresses_drops_loopback_and_duplicates(monkeypatch) -> None:
    import socket

    monkeypatch.setattr(socket, "gethostname", lambda: "robot-pi")
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [
        (None, None, None, None, ("127.0.0.1", 0)),
        (None, None, None, None, ("::1", 0)),
        (None, None, None, None, ("10.0.0.5", 0)),
        (None, None, None, None, ("10.0.0.5", 0)),
        (None, None, None, None, ("192.168.1.9", 0)),
    ])
    host, ips = dmod._reachable_addresses()
    assert host == "robot-pi"
    assert "127.0.0.1" not in ips and "::1" not in ips
    assert ips.count("10.0.0.5") == 1
    assert "192.168.1.9" in ips


def test_reachable_addresses_tolerates_a_host_that_cannot_resolve(monkeypatch) -> None:
    import socket

    def boom(*a, **k):
        raise OSError("no DNS")

    monkeypatch.setattr(socket, "gethostname", lambda: "robot-pi")
    monkeypatch.setattr(socket, "getaddrinfo", boom)
    host, ips = dmod._reachable_addresses()
    assert host == "robot-pi"
    assert isinstance(ips, list)


# ----------------------------------------------------------------------
# Heartbeat payload
# ----------------------------------------------------------------------


def test_recording_state_aggregates_the_spool_backlog(monkeypatch) -> None:
    """ADR 0023: the backend is supposed to gate the next session launch on
    drain_done, so the totals have to cover every spool dir, not just the
    active one."""
    from interlatent.inference.client import spool

    monkeypatch.setattr(spool, "orphan_sessions", lambda: [
        {"session_id": "a", "pending_count": 3, "pending_bytes": 300},
        {"session_id": "b", "pending_count": 4, "pending_bytes": 700},
    ])
    d = _daemon()
    state = d._recording_state()
    assert state["spool_pending"] == 7
    assert state["spool_bytes"] == 1000
    assert state["drain_done"] is False
    assert state["blocked"] is False


def test_recording_state_reports_drain_done_on_an_empty_spool(monkeypatch) -> None:
    from interlatent.inference.client import spool

    monkeypatch.setattr(spool, "orphan_sessions", lambda: [])
    assert _daemon()._recording_state() == {
        "blocked": False, "spool_pending": 0, "spool_bytes": 0, "drain_done": True,
    }


def test_recording_state_surfaces_the_active_clients_hard_stop(monkeypatch) -> None:
    from interlatent.inference.client import spool

    monkeypatch.setattr(spool, "orphan_sessions", lambda: [])
    d = _daemon()

    class _Client:
        recording_blocked = True

    d._active = _ControlLoopHandle(client=_Client())
    assert d._recording_state()["blocked"] is True


def test_recording_state_survives_a_broken_spool(monkeypatch) -> None:
    """Telemetry must never take the heartbeat down — a node that stops
    heartbeating loses its session assignment."""
    from interlatent.inference.client import spool

    def boom():
        raise OSError("spool dir vanished")

    monkeypatch.setattr(spool, "orphan_sessions", boom)
    state = _daemon()._recording_state()
    assert state["spool_pending"] == 0
    assert state["drain_done"] is True


def test_safety_state_reports_the_delta_clamp(monkeypatch) -> None:
    """ADR 0037 (platform repo): the backend refuses a world-model launch when the per-tick
    clamp is off, so max_step_set must distinguish unset from set."""
    off = _daemon(robot_extra={})._safety_state()
    assert off["max_step"] is None
    assert off["max_step_set"] is False
    assert off["context_ring"] is True

    on = _daemon(robot_extra={"max_step": "0.05"})._safety_state()
    assert on["max_step"] == pytest.approx(0.05)
    assert on["max_step_set"] is True

    # A non-positive / unparseable value is "not set", not "set to garbage".
    assert _daemon(robot_extra={"max_step": "0"})._safety_state()["max_step_set"] is False
    assert _daemon(
        robot_extra={"max_step": "banana"}
    )._safety_state()["max_step_set"] is False


def test_safety_state_uses_the_same_parser_the_clamp_enforces() -> None:
    """Reported value and enforced value must come from one function, or the
    backend can gate on a limit the loop isn't applying."""
    from interlatent.node.control import _parse_max_step

    extra = {"max_step": "0.25"}
    assert _daemon(robot_extra=extra)._safety_state()["max_step"] == _parse_max_step(extra)


def test_heartbeat_posts_recording_and_safety(monkeypatch) -> None:
    from interlatent.inference.client import spool

    monkeypatch.setattr(spool, "orphan_sessions", lambda: [])
    d = _daemon()
    http = _FakeHttp([_FakeResponse(200)])
    d._http = http

    with pytest.raises(_StopLoop):
        asyncio.run(d._heartbeat_loop())

    url, body = http.posts[0]
    assert url == "/api/v1/nodes/node-1/heartbeat"
    assert body["recording"]["drain_done"] is True
    assert "max_step_set" in body["safety"]


def test_heartbeat_retries_a_server_error_instead_of_giving_up(monkeypatch) -> None:
    from interlatent.inference.client import spool

    monkeypatch.setattr(spool, "orphan_sessions", lambda: [])
    d = _daemon()
    d._http = _FakeHttp([
        _FakeResponse(503),
        ConnectionError("network blip"),
        _FakeResponse(200),
    ])
    with pytest.raises(_StopLoop):
        asyncio.run(d._heartbeat_loop())
    assert len(d._http.posts) == 4  # 3 scripted + the one that ends the loop


def test_report_hardware_posts_the_attached_devices() -> None:
    d = _daemon(
        robot_kind="yam",
        robot_port="/dev/ttyUSB0",
        robot_cameras={"wrist": "/dev/video0", "top": "/dev/video2"},
        robot_extra={"kind": "yam_left"},
    )
    d._http = _FakeHttp([_FakeResponse(200)])
    asyncio.run(d._report_hardware())

    url, body = d._http.posts[0]
    assert url == "/api/v1/nodes/node-1/hardware"
    assert body["robot_type"] == "yam"
    assert body["robot_port"] == "/dev/ttyUSB0"
    assert body["cameras"] == [
        {"name": "wrist", "device": "/dev/video0"},
        {"name": "top", "device": "/dev/video2"},
    ]
    assert body["robot_args"] == {"kind": "yam_left"}


def test_report_hardware_never_raises() -> None:
    """A failed report just leaves the coordinator without hardware details;
    it must not stop the daemon from coming up."""
    d = _daemon()
    d._http = _FakeHttp([ConnectionError("down")])
    asyncio.run(d._report_hardware())  # no raise

    d2 = _daemon()
    d2._http = _FakeHttp([_FakeResponse(500)])
    asyncio.run(d2._report_hardware())


# ----------------------------------------------------------------------
# Poll loop
# ----------------------------------------------------------------------


def _poll_once(daemon: NodeDaemon, body: dict) -> list[tuple]:
    converged: list[tuple] = []
    daemon._converge = lambda payload, kind: converged.append((payload, kind))  # type: ignore[assignment]
    daemon._http = _FakeHttp([_FakeResponse(200, body)])
    with pytest.raises(_StopLoop):
        asyncio.run(daemon._poll_loop())
    return converged


def test_poll_echoes_what_it_knows_so_the_backend_can_wake_it() -> None:
    d = _daemon()
    d._known_session_id = "sess-1"
    d._known_endpoint = "gpu-a:50051"
    _poll_once(d, {"changed": False})
    _url, params = d._http.gets[0]
    assert _url == "/api/v1/nodes/node-1/poll"
    assert params["known_session_id"] == "sess-1"
    assert params["known_endpoint"] == "gpu-a:50051"
    assert params["wait"] == d.cfg.poll_wait_s


def test_poll_does_not_converge_when_nothing_changed() -> None:
    assert _poll_once(_daemon(), {"changed": False, "session": {"id": "s"}}) == []


def test_poll_dispatches_an_inference_session() -> None:
    got = _poll_once(_daemon(), {"changed": True, "session": {"id": "sess-1"}})
    assert got == [({"id": "sess-1"}, "inference_session")]


def test_poll_dispatches_a_teleop_recording_from_the_envelope() -> None:
    """A recording is a different resource with a different payload; routing
    it as an inference session would start the policy loop against it."""
    got = _poll_once(_daemon(), {
        "changed": True,
        "assignment": {"type": "teleop_recording", "recording": {"id": "rec-9"}},
    })
    assert got == [({"id": "rec-9"}, "teleop_recording")]


def test_poll_reads_a_session_nested_in_the_envelope() -> None:
    got = _poll_once(_daemon(), {
        "changed": True,
        "assignment": {"type": "inference_session", "session": {"id": "sess-2"}},
    })
    assert got == [({"id": "sess-2"}, "inference_session")]


def test_poll_converges_to_none_when_the_assignment_is_cleared() -> None:
    assert _poll_once(_daemon(), {"changed": True, "session": None}) == [
        (None, "inference_session")
    ]


def test_poll_retries_after_an_error_response() -> None:
    d = _daemon()
    d._converge = lambda payload, kind: None  # type: ignore[assignment]
    d._http = _FakeHttp([_FakeResponse(500), ValueError("bad json")])
    with pytest.raises(_StopLoop):
        asyncio.run(d._poll_loop())
    assert len(d._http.gets) == 3


# ----------------------------------------------------------------------
# Convergence
# ----------------------------------------------------------------------


@pytest.fixture
def converging(monkeypatch):
    """A daemon whose _start_loop is recorded rather than run, with a
    healthy spool."""
    from interlatent.inference.client import spool

    monkeypatch.setattr(spool, "disk_pressure", lambda: None)
    d = _daemon()
    started: list[tuple] = []
    d._start_loop = lambda session, kind="inference_session": started.append(  # type: ignore[assignment]
        (session, kind)
    )
    stopped: list[int] = []
    real_stop = d._stop_active_loop
    d._stop_active_loop = lambda: (stopped.append(1), real_stop())[1]  # type: ignore[assignment]
    return d, started, stopped


def test_converge_starts_a_newly_assigned_session(converging) -> None:
    d, started, _stopped = converging
    d._converge({"id": "sess-1", "drtc_endpoint": "gpu-a:1"})
    assert started == [({"id": "sess-1", "drtc_endpoint": "gpu-a:1"},
                        "inference_session")]
    assert d._known_session_id == "sess-1"
    assert d._known_endpoint == "gpu-a:1"
    assert d._active_endpoint == "gpu-a:1"


def test_converge_forwards_the_assignment_kind(converging) -> None:
    d, started, _stopped = converging
    d._converge({"id": "rec-1", "drtc_endpoint": "gpu-a:1"}, "teleop_recording")
    assert started[0][1] == "teleop_recording"


def test_converge_is_a_noop_for_the_same_session_and_endpoint(converging) -> None:
    d, started, stopped = converging
    d._known_session_id = "sess-1"
    d._active_endpoint = "gpu-a:1"
    d._active = _ControlLoopHandle(client=object())

    d._converge({"id": "sess-1", "drtc_endpoint": "gpu-a:1"})
    assert started == [] and stopped == []
    # Sync to the server's view so this diff stops waking the long-poll.
    assert d._known_endpoint == "gpu-a:1"


def test_converge_restarts_when_the_endpoint_moves_under_a_live_session(
    converging,
) -> None:
    """The env's compute box can be swapped while a session runs; the loop
    is still dialled at the old address and has to be rebuilt."""
    d, started, stopped = converging
    d._known_session_id = "sess-1"
    d._active_endpoint = "gpu-a:1"
    d._active = _ControlLoopHandle(client=object())

    d._converge({"id": "sess-1", "drtc_endpoint": "gpu-b:2"})
    assert stopped == [1]
    assert started[0][0]["drtc_endpoint"] == "gpu-b:2"
    assert d._active_endpoint == "gpu-b:2"


def test_converge_ignores_a_server_endpoint_change_while_pinned(
    converging, monkeypatch
) -> None:
    """With INTERLATENT_DRTC_URL set, the address we dial doesn't move — so
    a server-reported change must not thrash the session."""
    monkeypatch.setenv("INTERLATENT_DRTC_URL", "pinned:9")
    d, started, stopped = converging
    d._known_session_id = "sess-1"
    d._active_endpoint = "pinned:9"
    d._active = _ControlLoopHandle(client=object())

    d._converge({"id": "sess-1", "drtc_endpoint": "gpu-b:2"})
    assert started == [] and stopped == []
    assert d._known_endpoint == "gpu-b:2"


def test_converge_stops_the_loop_when_the_assignment_clears(converging) -> None:
    d, started, stopped = converging
    d._known_session_id = "sess-1"
    d._known_endpoint = "gpu-a:1"
    d._active_endpoint = "gpu-a:1"
    d._active = _ControlLoopHandle(client=object())

    d._converge(None)
    assert stopped == [1] and started == []
    assert (d._known_session_id, d._known_endpoint, d._active_endpoint) == ("", "", "")


def test_converge_swaps_a_running_session_for_a_different_one(converging) -> None:
    d, started, stopped = converging
    d._known_session_id = "sess-1"
    d._active_endpoint = "gpu-a:1"
    d._active = _ControlLoopHandle(client=object())

    d._converge({"id": "sess-2", "drtc_endpoint": "gpu-a:1"})
    assert stopped == [1]
    assert started[0][0]["id"] == "sess-2"
    assert d._known_session_id == "sess-2"


def test_converge_refuses_an_assignment_under_disk_pressure(
    converging, monkeypatch
) -> None:
    """ADR 0023: accepting would hard-stop on the first capture. Refusing
    keeps the node retrying, so it auto-resumes once the spool drains —
    nothing is dropped to make room."""
    from interlatent.inference.client import spool

    monkeypatch.setattr(spool, "disk_pressure", lambda: "spool backlog at cap")
    d, started, _stopped = converging
    d._converge({"id": "sess-1", "drtc_endpoint": "gpu-a:1"})
    assert started == []
    assert d._known_session_id == ""  # so the next poll retries


def test_converge_tolerates_a_spool_that_cannot_be_inspected(
    converging, monkeypatch
) -> None:
    from interlatent.inference.client import spool

    def boom():
        raise OSError("no spool root")

    monkeypatch.setattr(spool, "disk_pressure", boom)
    d, started, _stopped = converging
    d._converge({"id": "sess-1", "drtc_endpoint": "gpu-a:1"})
    assert started  # an unreadable spool is not a refusal


def test_converge_clears_state_when_the_loop_fails_to_start(converging) -> None:
    """Leaving _known_session_id set would make the node believe it is
    running a session it never started."""
    d, _started, _stopped = converging

    def boom(session, kind="inference_session"):
        raise RuntimeError("robot not found")

    d._start_loop = boom  # type: ignore[assignment]
    d._converge({"id": "sess-1", "drtc_endpoint": "gpu-a:1"})
    assert (d._known_session_id, d._known_endpoint, d._active_endpoint) == ("", "", "")


# ----------------------------------------------------------------------
# Loop-function resolution
# ----------------------------------------------------------------------


def test_loop_override_wins_and_is_resolved_once(monkeypatch) -> None:
    from interlatent.node import control

    calls: list[str] = []
    sentinel = object()
    monkeypatch.setattr(control, "import_callable",
                        lambda spec: (calls.append(spec), sentinel)[1])

    d = _daemon(loop_override="my.pkg:drive", robot_kind="yam")
    assert d._resolve_loop_fn() is sentinel
    assert d._resolve_loop_fn() is sentinel  # cached
    assert calls == ["my.pkg:drive"]


def test_a_native_robot_kind_uses_its_own_loop(monkeypatch) -> None:
    from interlatent import adapters
    from interlatent.node import control

    monkeypatch.setattr(adapters, "native_loop_path", lambda kind: "native.mod:run")
    seen: list[str] = []
    sentinel = object()
    monkeypatch.setattr(control, "import_callable",
                        lambda spec: (seen.append(spec), sentinel)[1])

    assert _daemon(robot_kind="nori")._resolve_loop_fn() is sentinel
    assert seen == ["native.mod:run"]


def test_a_non_native_kind_falls_back_to_the_bundled_lerobot_loop(monkeypatch) -> None:
    from interlatent import adapters
    from interlatent.node import control

    monkeypatch.setattr(adapters, "native_loop_path", lambda kind: None)
    assert _daemon(robot_kind="so101")._resolve_loop_fn() is control.lerobot_control_loop


def test_no_robot_and_no_override_is_a_clear_error(monkeypatch) -> None:
    from interlatent import adapters

    monkeypatch.setattr(adapters, "native_loop_path", lambda kind: None)
    with pytest.raises(RuntimeError, match="--robot"):
        _daemon()._resolve_loop_fn()


# ----------------------------------------------------------------------
# _start_loop
# ----------------------------------------------------------------------


class _StubClient:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


class _StubChannel:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.started = 0
        self.stopped = 0

    def start(self) -> None:
        self.started += 1

    def stop(self) -> None:
        self.stopped += 1


@pytest.fixture
def starting(monkeypatch):
    """Everything _start_loop reaches for, stubbed. Returns (make, calls)."""
    calls: dict = {"connect": [], "teleop": [], "loop": [], "clients": [],
                   "channels": []}

    def _connect_drtc(**kwargs):
        calls["connect"].append(kwargs)
        c = _StubClient(**kwargs)
        calls["clients"].append(c)
        return c

    stub_connect = types.ModuleType("interlatent.inference.integration.connect")
    stub_connect.connect_drtc = _connect_drtc
    monkeypatch.setitem(
        sys.modules, "interlatent.inference.integration.connect", stub_connect,
    )

    from interlatent.node.teleop import factory

    def _make_channel(**kwargs):
        calls["teleop"].append(kwargs)
        ch = _StubChannel(**kwargs)
        calls["channels"].append(ch)
        return ch

    monkeypatch.setattr(factory, "make_teleop_channel", _make_channel)

    def make(**cfg_over) -> NodeDaemon:
        d = _daemon(**cfg_over)
        d._resolve_loop_fn = lambda: (  # type: ignore[assignment]
            lambda **kw: calls["loop"].append(kw)
        )
        return d

    return make, calls


def _run_start(d: NodeDaemon, session: dict, kind: str = "inference_session") -> None:
    d._start_loop(session, kind=kind)
    d._stop_active_loop()  # join the control-loop thread


def test_start_loop_wires_the_session_into_connect_drtc(starting) -> None:
    make, calls = starting
    d = make(robot_kind="so101", robot_cameras={"wrist": "/dev/video0"})
    _run_start(d, {
        "id": "sess-1",
        "drtc_endpoint": "gpu-a:50051",
        "policy_uri": "lerobot/smolvla_base",
        "policy_backend": "lerobot",
        "task": "pick the cube",
        "task_id": "task-7",
        "chunk_size": 32,
        "action_dim": 7,
        "fps": 25.0,
        "environment_id": "env-3",
        "collection_context": {"env_slug": "pick-cube"},
    })

    kw = calls["connect"][0]
    assert kw["server_address"] == "gpu-a:50051"
    assert kw["api_key"] == "ilat_user"  # the USER key, not the node token
    assert kw["environment"] == "pick-cube"
    assert kw["policy_uri"] == "lerobot/smolvla_base"
    assert (kw["chunk_size"], kw["action_dim"], kw["fps"]) == (32, 7, 25.0)
    assert kw["task_id"] == "task-7"
    assert kw["episode_id"] == "sess-1"
    assert kw["env_id"] == "env-3"
    assert kw["record"] is True

    loop_kw = calls["loop"][0]
    assert loop_kw["policy_enabled"] is True
    assert loop_kw["robot_kind"] == "so101"
    assert callable(loop_kw["should_stop"])
    # The runner always closes the client and the channel on exit.
    assert calls["clients"][0].closed == 1
    assert calls["channels"][0].stopped >= 1


def test_start_loop_defaults_a_session_with_almost_nothing_in_it(starting) -> None:
    make, calls = starting
    d = make(robot_kind="so101")
    _run_start(d, {"id": "sess-1", "drtc_endpoint": "gpu:1"})
    kw = calls["connect"][0]
    assert kw["environment"] == "default"
    assert (kw["chunk_size"], kw["action_dim"], kw["fps"]) == (50, 6, 30.0)
    assert kw["task_id"] is None


def test_start_loop_forwards_camera_keys_as_observation_image_keys(starting) -> None:
    """MolmoAct2 and friends can't name their own cameras — the node is the
    authority, so the keys have to ride OpenSession metadata."""
    make, calls = starting
    d = make(robot_kind="yam", robot_cameras={"wrist": "/dev/video0",
                                              "top": "/dev/video2"})
    _run_start(d, {"id": "s", "drtc_endpoint": "gpu:1"})
    md = calls["connect"][0]["metadata"]
    assert md["image_keys"] == "observation.images.wrist,observation.images.top"
    assert md["robot_kind"] == "yam"
    assert md["inference_action_mode"] == "continuous"


def test_start_loop_sends_no_metadata_when_there_is_nothing_to_say(starting) -> None:
    make, calls = starting
    d = make()
    _run_start(d, {"id": "s", "drtc_endpoint": "gpu:1"})
    assert calls["connect"][0]["metadata"] is None


def test_start_loop_defaults_image_resize_only_for_molmoact(starting) -> None:
    """Molmo's processor downsamples anyway, so 640x480 just burns uplink."""
    make, calls = starting
    d = make(robot_kind="so101")
    _run_start(d, {"id": "s", "drtc_endpoint": "gpu:1",
                   "policy_uri": "allenai/MolmoAct2-7B"})
    assert calls["loop"][0]["image_resize"] == 256

    calls["loop"].clear()
    _run_start(make(robot_kind="so101"),
               {"id": "s", "drtc_endpoint": "gpu:1", "policy_uri": "lerobot/act"})
    assert calls["loop"][0]["image_resize"] is None

    calls["loop"].clear()
    _run_start(make(robot_kind="so101", image_resize=128),
               {"id": "s", "drtc_endpoint": "gpu:1",
                "policy_uri": "allenai/MolmoAct2-7B"})
    assert calls["loop"][0]["image_resize"] == 128  # explicit config wins


def test_start_loop_forwards_num_inference_steps_only_when_set(starting) -> None:
    make, calls = starting
    _run_start(make(robot_kind="so101"), {"id": "s", "drtc_endpoint": "gpu:1"})
    assert "num_inference_steps" not in (calls["connect"][0]["metadata"] or {})

    calls["connect"].clear()
    _run_start(make(robot_kind="so101", num_inference_steps=3),
               {"id": "s", "drtc_endpoint": "gpu:1"})
    assert calls["connect"][0]["metadata"]["num_inference_steps"] == "3"


def test_start_loop_passes_the_recording_block_through_opaquely(starting) -> None:
    """The node doesn't interpret the sink config — the GPU recorder does.
    Nulls are dropped so the map stays clean."""
    make, calls = starting
    d = make(robot_kind="so101")
    _run_start(d, {
        "id": "s", "drtc_endpoint": "gpu:1",
        "recording": {"s3_uri": "s3://bucket/x", "output_dir": None, "fps": 30},
    })
    md = calls["connect"][0]["metadata"]
    assert md["s3_uri"] == "s3://bucket/x"
    assert md["fps"] == "30"
    assert "output_dir" not in md


def test_start_loop_runs_a_teleop_recording_without_a_policy(starting) -> None:
    """A recording has no policy loaded: stepping it would drive the robot
    with the echo backend's sinusoid."""
    make, calls = starting
    d = make(robot_kind="yam")
    _run_start(d, {"id": "rec-1", "drtc_endpoint": "gpu:1",
                   "policy_uri": "should-be-ignored"},
               kind="teleop_recording")

    kw = calls["connect"][0]
    assert kw["policy_uri"] == "teleop-recording"
    assert kw["policy_backend"] == "echo"
    assert calls["loop"][0]["policy_enabled"] is False
    # Recordings mint their teleop token against their own route (ADR 0020).
    assert calls["teleop"][0]["token_path"] == (
        "/api/v1/teleop-recordings/rec-1/teleop-token"
    )


def test_start_loop_mints_an_inference_teleop_channel_with_no_token_path(
    starting,
) -> None:
    make, calls = starting
    _run_start(make(robot_kind="yam"), {"id": "s", "drtc_endpoint": "gpu:1"})
    assert calls["teleop"][0]["token_path"] is None
    assert calls["teleop"][0]["api_key"] == "ilat_user"
    assert calls["channels"][0].started == 1


def test_start_loop_resolves_the_dimos_embodiment_for_teleop(starting) -> None:
    """`--robot dimos` covers kinematically distinct arms; the QUIC channel
    needs the specific embodiment to find its kinematic_spec."""
    make, calls = starting
    d = make(robot_kind="dimos", robot_extra={"kind": "xarm7"})
    _run_start(d, {"id": "s", "drtc_endpoint": "gpu:1"})
    assert calls["teleop"][0]["robot_kind"] == "xarm7"

    calls["teleop"].clear()
    _run_start(make(robot_kind="dimos", robot_extra={}),
               {"id": "s", "drtc_endpoint": "gpu:1"})
    assert calls["teleop"][0]["robot_kind"] == "dimos"


def test_start_loop_skips_teleop_without_a_user_key(starting) -> None:
    """The relay rejects the node token — the teleop endpoint is owned by
    the user."""
    make, calls = starting
    d = make(robot_kind="yam", drtc_api_key=None)
    _run_start(d, {"id": "s", "drtc_endpoint": "gpu:1"})
    assert calls["teleop"] == []
    assert calls["loop"][0]["teleop_channel"] is None


def test_start_loop_tolerates_a_deployment_without_quic(starting, monkeypatch) -> None:
    make, calls = starting
    from interlatent.node.teleop import factory

    # The factory returns None when the deployment isn't QUIC-configured.
    monkeypatch.setattr(factory, "make_teleop_channel", lambda **kw: None)
    d = make(robot_kind="yam")
    _run_start(d, {"id": "s", "drtc_endpoint": "gpu:1"})
    assert calls["loop"][0]["teleop_channel"] is None


def test_synchronous_is_the_or_of_operator_and_backend(starting) -> None:
    """ADR 0037: a world-action model REQUIRES sync mode — in async mode
    every late chunk falls below the cursor and the arm never moves."""
    make, calls = starting
    _run_start(make(robot_kind="so101"), {"id": "s", "drtc_endpoint": "gpu:1"})
    assert calls["connect"][0]["synchronous"] is False

    calls["connect"].clear()
    _run_start(make(robot_kind="so101"),
               {"id": "s", "drtc_endpoint": "gpu:1", "synchronous": True})
    assert calls["connect"][0]["synchronous"] is True

    calls["connect"].clear()
    _run_start(make(robot_kind="so101", synchronous=True),
               {"id": "s", "drtc_endpoint": "gpu:1"})
    assert calls["connect"][0]["synchronous"] is True


def test_start_loop_refuses_a_session_with_no_route(starting) -> None:
    """Better a logged refusal than a client hanging against an empty
    address forever."""
    make, calls = starting
    d = make(robot_kind="so101")
    d._start_loop({"id": "sess-1"})
    assert calls["connect"] == []
    assert d._active is None


def test_start_loop_refuses_an_unroutable_method(starting) -> None:
    make, calls = starting
    d = make(robot_kind="so101")
    d._start_loop({"id": "sess-1", "route": {"method": "carrier-pigeon",
                                             "address": "coop:1"}})
    assert calls["connect"] == []
    assert d._active is None


def test_a_crashing_control_loop_still_closes_the_client(starting) -> None:
    make, calls = starting
    d = make(robot_kind="so101")

    def boom(**kw):
        raise RuntimeError("robot exploded")

    d._resolve_loop_fn = lambda: boom  # type: ignore[assignment]
    _run_start(d, {"id": "s", "drtc_endpoint": "gpu:1"})
    assert calls["clients"][0].closed == 1
    assert calls["channels"][0].stopped >= 1


# ----------------------------------------------------------------------
# Teardown
# ----------------------------------------------------------------------


class _WedgedThread:
    """A control-loop thread whose robot teardown never returns."""

    def __init__(self) -> None:
        self.joins: list[float | None] = []

    def join(self, timeout=None) -> None:
        self.joins.append(timeout)

    def is_alive(self) -> bool:
        return True


def test_stop_force_closes_the_client_when_the_thread_wedges() -> None:
    """A wedged teardown never runs its finally, so CloseSession — and the
    server-side upload — would never fire and the recording would be lost."""
    d = _daemon()
    client = _StubClient()
    channel = _StubChannel()
    thread = _WedgedThread()
    d._active = _ControlLoopHandle(client=client, teleop_channel=channel,
                                   thread=thread)

    d._stop_active_loop()

    assert d._active is None
    assert thread.joins == [10.0]
    assert client.closed == 1
    assert channel.stopped == 1


def test_stop_is_a_noop_when_nothing_is_running() -> None:
    d = _daemon()
    d._stop_active_loop()
    assert d._active is None


def test_stop_survives_a_channel_and_client_that_raise_on_close() -> None:
    class _Angry:
        def stop(self):
            raise RuntimeError("nope")

        def close(self):
            raise RuntimeError("nope")

    d = _daemon()
    d._active = _ControlLoopHandle(client=_Angry(), teleop_channel=_Angry(),
                                   thread=_WedgedThread())
    d._stop_active_loop()  # must not raise
    assert d._active is None


def test_main_scans_orphan_spools_then_runs_both_loops(monkeypatch) -> None:
    """A spool left by a crashed run is real un-uploaded data (ADR 0023) —
    startup must surface it, GC the expired ones, and get on with it."""
    from interlatent.inference.client import spool

    gc_calls: list[int] = []
    monkeypatch.setattr(spool, "orphan_sessions", lambda: [
        {"session_id": "old", "pending_count": 12, "pending_bytes": 5_000_000,
         "dir": "/tmp/spool/old"},
        {"session_id": "drained", "pending_count": 0, "pending_bytes": 0,
         "dir": "/tmp/spool/drained"},
    ])
    monkeypatch.setattr(spool, "gc_orphans", lambda: gc_calls.append(1))

    d = _daemon()
    ran: list[str] = []

    async def _hb():
        ran.append("heartbeat")

    async def _poll():
        ran.append("poll")

    reported: list[int] = []

    async def _hw():
        reported.append(1)

    d._heartbeat_loop = _hb  # type: ignore[assignment]
    d._poll_loop = _poll  # type: ignore[assignment]
    d._report_hardware = _hw  # type: ignore[assignment]
    closed: list[int] = []
    d._http = _FakeHttp([])
    d._http.aclose = lambda: _noop(closed)  # type: ignore[assignment]

    asyncio.run(d._main())

    assert gc_calls == [1]
    assert reported == [1]
    assert sorted(ran) == ["heartbeat", "poll"]
    assert closed == [1]


def _noop(sink: list):
    async def _run():
        sink.append(1)

    return _run()


def test_main_starts_even_when_the_spool_scan_blows_up(monkeypatch) -> None:
    from interlatent.inference.client import spool

    def boom():
        raise OSError("spool root unreadable")

    monkeypatch.setattr(spool, "orphan_sessions", boom)

    d = _daemon()
    ran: list[str] = []

    async def _hb():
        ran.append("heartbeat")

    async def _poll():
        ran.append("poll")

    async def _hw():
        pass

    d._heartbeat_loop = _hb  # type: ignore[assignment]
    d._poll_loop = _poll  # type: ignore[assignment]
    d._report_hardware = _hw  # type: ignore[assignment]
    d._http = _FakeHttp([])

    asyncio.run(d._main())
    assert sorted(ran) == ["heartbeat", "poll"]


def test_run_forever_stops_the_robot_on_ctrl_c() -> None:
    """Ctrl-C must not leave the arm driven by an orphaned control loop."""
    d = _daemon()
    client = _StubClient()
    d._active = _ControlLoopHandle(client=client, thread=_WedgedThread())

    async def _boom():
        raise KeyboardInterrupt

    d._main = _boom  # type: ignore[assignment]
    d.run_forever()  # swallows the interrupt

    assert d._active is None
    assert client.closed == 1


def test_the_handles_stop_flag_drives_should_stop() -> None:
    h = _ControlLoopHandle(client=object())
    assert h.should_stop() is False
    h.stop_flag.set()
    assert h.should_stop() is True
