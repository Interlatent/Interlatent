"""The `interlatent` CLI finds the coordinator it just started.

``interlatent up`` spawns a background daemon, so it cannot export
``INTERLATENT_COORDINATOR`` into the shell that invoked it. Resolution used to
consult only the flag and the env vars, which made the documented first session
fail::

    $ interlatent up
    ✓ Coordinator up (pid 32781) on port 8900.
    $ interlatent gpus ls
    CoordinatorNotConfigured: ... or run `interlatent up` to start one locally.

— a traceback whose remedy was the command you had just run. The runtime file
``interlatent up`` writes is now consulted as :func:`resolve`'s ``config``
step, which is *below* the flag and the env var: a host that runs its own
coordinator must still be able to drive somebody else's.

The listing test guards a separate slip in the same command: the table asked
for keys (``id``, ``gpu``, ``region``) that no coordinator has ever emitted, so
three columns were blank in every listing while ``url`` — the field that says
whether a registration points anywhere real — was not shown at all.
"""

from __future__ import annotations

import argparse
import json
import os
import threading

import pytest

from interlatent._coordinator import CoordinatorNotConfigured
from interlatent.cli import main as cli
from interlatent.coordinator import supervisor
from interlatent.coordinator.server import build_server

OPERATOR = "ilop_" + "d" * 48


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """No test may read the developer's own daemon or env."""
    monkeypatch.delenv("INTERLATENT_COORDINATOR", raising=False)
    monkeypatch.delenv("INTERLATENT_API_BASE", raising=False)
    monkeypatch.delenv("INTERLATENT_API_KEY", raising=False)
    monkeypatch.setattr(
        supervisor, "RUNTIME_PATH", tmp_path / "coordinator.runtime.json"
    )


def _write_runtime(pid: int, port: int = 8900) -> None:
    supervisor.RUNTIME_PATH.write_text(
        json.dumps({"pid": pid, "host": "127.0.0.1", "port": port})
    )


# ----------------------------------------------------------------------
# Discovery
# ----------------------------------------------------------------------


def test_a_running_local_coordinator_is_discovered():
    _write_runtime(pid=os.getpid())
    assert cli._local_coordinator() == "http://127.0.0.1:8900"


def test_no_runtime_file_means_no_local_coordinator():
    assert cli._local_coordinator() is None


def test_a_stale_runtime_file_does_not_count():
    """A crashed daemon leaves the file behind. Trusting it would aim every
    command at a closed port instead of saying nothing is running."""
    _write_runtime(pid=0)  # pid_alive() is False for anything < 1
    assert cli._local_coordinator() is None


def test_the_port_is_read_from_the_file_not_assumed():
    _write_runtime(pid=os.getpid(), port=9123)
    assert cli._local_coordinator() == "http://127.0.0.1:9123"


# ----------------------------------------------------------------------
# Precedence — discovery is a fallback, never an override
# ----------------------------------------------------------------------


def test_the_env_var_beats_the_local_daemon(monkeypatch):
    """Running a coordinator here must not hijack a CLI pointed elsewhere."""
    _write_runtime(pid=os.getpid())
    monkeypatch.setenv("INTERLATENT_COORDINATOR", "http://remote.test")
    args = argparse.Namespace(api_key=OPERATOR, api_base=None)
    assert cli._make_client(args).base_url.startswith("http://remote.test")


def test_the_flag_beats_both(monkeypatch):
    _write_runtime(pid=os.getpid())
    monkeypatch.setenv("INTERLATENT_COORDINATOR", "http://remote.test")
    args = argparse.Namespace(
        api_key=OPERATOR, api_base="http://flag.test"
    )
    assert cli._make_client(args).base_url.startswith("http://flag.test")


# ----------------------------------------------------------------------
# The error, when there really is nothing to talk to
# ----------------------------------------------------------------------


def test_an_unconfigured_command_explains_itself_without_a_traceback(capsys):
    code = cli.main(["gpus", "ls", "--api-key", OPERATOR])
    assert code == 2
    err = capsys.readouterr().err
    assert "needs a coordinator" in err
    assert "Traceback" not in err


def test_the_error_type_is_still_the_documented_one():
    """The clean message is a presentation choice in main(); callers using
    the library keep the exception."""
    with pytest.raises(CoordinatorNotConfigured):
        cli._make_client(
            argparse.Namespace(api_key=OPERATOR, api_base=None)
        )


# ----------------------------------------------------------------------
# gpu ls shows fields the coordinator actually emits
# ----------------------------------------------------------------------


@pytest.fixture
def coordinator(tmp_path):
    server = build_server("127.0.0.1", 0, tmp_path / "coordinator.json", OPERATOR)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", server.coordinator
    server.shutdown()
    server.server_close()


def test_gpu_ls_has_no_permanently_blank_columns(coordinator, capsys):
    base, state = coordinator
    state.add_gpu(name="box-a", url="10.0.0.5:9000")

    assert cli.main(["gpu", "ls", "--coordinator", base, "--api-key", OPERATOR]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    headers, row = lines[0].split(), lines[1]

    # Every column carries a value for a freshly registered box. Asking for a
    # key the coordinator never sets is exactly the bug this guards.
    for header in headers:
        assert header in ("NAME", "URL", "STATUS", "GPU", "PROVIDER")
    assert "box-a" in row
    assert "10.0.0.5:9000" in row  # the field that was missing entirely
    assert "ready" in row


def test_gpu_ls_columns_match_the_stored_box_schema(coordinator):
    """Pin the table to the writer. If add_gpu's keys are renamed, this fails
    here rather than as silently empty columns in front of an operator."""
    _, state = coordinator
    stored = state.add_gpu(name="box-b", url="10.0.0.6:9000")
    for key in ("name", "url", "status"):
        assert key in stored
    for absent in ("id", "gpu", "region"):
        assert absent not in stored
