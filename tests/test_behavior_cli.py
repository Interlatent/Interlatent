"""`interlatent behavior ls|validate|run` CLI — offline, no API key."""
from __future__ import annotations


from behavior_fakes import FakeAdapter

import interlatent.adapters as adapters_mod
import interlatent.robot as robot_mod
from interlatent.behaviors import arbitration as arb
from interlatent.cli.main import main


def test_ls_lists_home_and_hello(capsys):
    rc = main(["behavior", "ls", "--robot", "so101"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "home" in out and "hello" in out and "trajectory" in out


def test_ls_json(capsys):
    rc = main(["behavior", "ls", "--robot", "so101", "--json"])
    assert rc == 0
    assert '"name"' in capsys.readouterr().out


def test_validate_ok(tmp_path, capsys):
    p = tmp_path / "b.toml"
    p.write_text("[wave]\ntype='pose'\nduration=0.5\nwrist_roll=20.0\n")
    rc = main(["behavior", "validate", str(p), "--robot", "so101"])
    assert rc == 0
    assert "valid" in capsys.readouterr().out


def test_validate_reports_error(tmp_path, capsys):
    p = tmp_path / "b.toml"
    p.write_text("[bad]\ntype='pose'\nduration=0.5\nshoulder_pan=999.0\n")
    rc = main(["behavior", "validate", str(p), "--robot", "so101"])
    assert rc == 1
    assert "shoulder_pan" in capsys.readouterr().err


def test_run_home(monkeypatch, capsys):
    monkeypatch.setattr(
        adapters_mod, "resolve_adapter",
        lambda *a, **k: FakeAdapter(),
    )
    monkeypatch.setattr(robot_mod, "acquire_bus_lock", lambda *a, **k: arb.BusLock(None))
    rc = main(["behavior", "run", "home", "--robot", "so101", "--port", "/dev/null",
               "--control-hz", "2000"])
    assert rc == 0
    assert "reached" in capsys.readouterr().out
