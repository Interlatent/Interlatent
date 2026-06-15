"""BYO-box dashboard self-registration: id persistence, the no-key gate,
and the register/status request wire format (URL + payload + auth header).

All network is stubbed with a localhost ``http.server`` that captures the
request — no real backend, no GPU, no torch required.
"""
import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from interlatent_server.server import box_report
from interlatent_server.server.box_report import BoxReporter, build_reporter


# ----------------------------------------------------------------------
# box id: env override, mint-once, persistence
# ----------------------------------------------------------------------


def test_box_id_env_override_wins(tmp_path, monkeypatch):
    monkeypatch.setattr(box_report, "_BOX_ID_PATH", tmp_path / "box-id")
    monkeypatch.setenv("INTERLATENT_BOX_ID", "pinned-123")
    assert box_report.box_id() == "pinned-123"
    # Env override must not write a file.
    assert not (tmp_path / "box-id").exists()


def test_box_id_mints_once_and_persists(tmp_path, monkeypatch):
    path = tmp_path / "box-id"
    monkeypatch.setattr(box_report, "_BOX_ID_PATH", path)
    monkeypatch.delenv("INTERLATENT_BOX_ID", raising=False)

    first = box_report.box_id()
    assert first and path.read_text().strip() == first
    # A second call (and a fresh process reading the same file) is stable.
    assert box_report.box_id() == first


def test_detect_gpu_never_raises():
    # No torch/CUDA in the test env -> "unknown", but it must always be a str.
    assert isinstance(box_report.detect_gpu(), str)


# ----------------------------------------------------------------------
# build_reporter gate
# ----------------------------------------------------------------------


def test_build_reporter_disabled_without_key():
    assert build_reporter(api_base="http://x", api_key="", endpoint="h:1") is None
    assert build_reporter(api_base="http://x", api_key="   ", endpoint="h:1") is None


def test_build_reporter_enabled_with_key(tmp_path, monkeypatch):
    monkeypatch.setattr(box_report, "_BOX_ID_PATH", tmp_path / "box-id")
    monkeypatch.delenv("INTERLATENT_BOX_ID", raising=False)
    r = build_reporter(
        api_base="http://x", api_key="ilat_abc", endpoint="1.2.3.4:50051",
        warmup_policy="lerobot/smolvla_base",
    )
    assert isinstance(r, BoxReporter)
    assert r.box_id  # minted


# ----------------------------------------------------------------------
# wire format against a capturing stub server
# ----------------------------------------------------------------------


class _Capture(BaseHTTPRequestHandler):
    requests: list = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b"{}"
        type(self).requests.append({
            "path": self.path,
            "api_key": self.headers.get("x-api-key"),
            "body": json.loads(body or b"{}"),
        })
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *a):  # silence
        pass


def _stub_server():
    _Capture.requests = []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Capture)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def _reporter(port):
    return BoxReporter(
        api_base=f"http://127.0.0.1:{port}",
        api_key="ilat_secret",
        box_id="box-xyz",
        name="lab-rig",
        endpoint="1.2.3.4:50051",
        gpu_model="NVIDIA A100 · 80GB",
        warmup_policy="lerobot/smolvla_base",
    )


def test_register_request_shape():
    srv, port = _stub_server()
    try:
        ok = asyncio.run(_reporter(port).register())
    finally:
        srv.shutdown()
        srv.server_close()
    assert ok is True
    assert len(_Capture.requests) == 1
    req = _Capture.requests[0]
    assert req["path"] == "/api/v1/compute/boxes/register"
    assert req["api_key"] == "ilat_secret"
    body = req["body"]
    assert body["box_id"] == "box-xyz"
    assert body["provider"] == "byo"
    assert body["endpoint"] == "1.2.3.4:50051"
    assert body["gpu_model"] == "NVIDIA A100 · 80GB"
    assert body["warmup_policy"] == "lerobot/smolvla_base"


def test_status_request_shape_and_gate():
    srv, port = _stub_server()
    try:
        r = _reporter(port)
        # Non-reportable status: no request leaves.
        r.report_status("warming_up", block=True)
        assert _Capture.requests == []
        # Valid transition (block=True -> synchronous, deterministic for tests).
        r.report_status("running", block=True)
    finally:
        srv.shutdown()
        srv.server_close()
    assert len(_Capture.requests) == 1
    req = _Capture.requests[0]
    assert req["path"] == "/api/v1/compute/boxes/box-xyz/status"
    assert req["api_key"] == "ilat_secret"
    assert req["body"] == {"status": "running", "endpoint": "1.2.3.4:50051"}
