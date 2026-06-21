"""Unit coverage for the SpatialVLA and RDT-1B VLA backends.

No GPU, no transformers, no model weights: the heavy model + processor are
faked, so these tests exercise exactly the in-repo glue we own —
registration, offline URI routing, observation -> model-input parsing, the
unified-action slicing, and the (chunk_size, action_dim) float32 output
contract. The real model load / inference path needs a GPU and is out of
CI scope (same boundary as the molmoact2 backend; see TESTING.md).
"""
import numpy as np
import pytest

import interlatent_server.server  # noqa: F401  (registers all backends)
from interlatent_server.server.policy_runtime import (
    _BACKENDS,
    as_action_chunk,
    resolve_backend,
)
from interlatent_server.server.rdt_backend import RDTBackend, is_rdt
from interlatent_server.server.spatialvla_backend import SpatialVLABackend, is_spatialvla


# ---------------------------------------------------------------------------
# Registration + routing
# ---------------------------------------------------------------------------
def test_both_backends_registered():
    assert "spatialvla" in _BACKENDS
    assert "rdt" in _BACKENDS


@pytest.mark.parametrize("uri", [
    "IPEC-COMMUNITY/spatialvla-4b-mix-224-pt",
    "IPEC-COMMUNITY/spatialvla-4b-224-pt",
    "/local/checkpoints/SpatialVLA-finetuned",
])
def test_spatialvla_routing(uri):
    assert is_spatialvla(uri)
    assert resolve_backend("lerobot", uri) == "spatialvla"


@pytest.mark.parametrize("uri", [
    "robotics-diffusion-transformer/rdt-1b",
    "thu-ml/rdt-1b-finetune",
    "/ckpts/rdt-1b",
])
def test_rdt_routing(uri):
    assert is_rdt(uri)
    assert resolve_backend("lerobot", uri) == "rdt"


def test_routing_declines_unrelated_uris():
    # A normal lerobot policy is left untouched, and the backend name is
    # only ever changed for an explicit "lerobot" request.
    assert resolve_backend("lerobot", "lerobot/smolvla_base") == "lerobot"
    assert resolve_backend("echo", "") == "echo"
    assert resolve_backend("rdt", "robotics-diffusion-transformer/rdt-1b") == "rdt"
    assert not is_spatialvla("lerobot/smolvla_base")
    assert not is_rdt("lerobot/act_so101")


# ---------------------------------------------------------------------------
# as_action_chunk helper
# ---------------------------------------------------------------------------
def test_as_action_chunk_shapes_and_dtype():
    # drop batch dim, keep dtype
    out = as_action_chunk(np.zeros((1, 8, 7), dtype=np.float64), chunk_size=16)
    assert out.shape == (8, 7) and out.dtype == np.float32
    # promote a single action vector to a length-1 chunk
    assert as_action_chunk(np.ones(7), chunk_size=4).shape == (1, 7)
    # truncate horizon to chunk_size
    assert as_action_chunk(np.zeros((64, 7)), chunk_size=16).shape == (16, 7)
    # pad / truncate action_dim
    assert as_action_chunk(np.zeros((4, 5)), 4, action_dim=7).shape == (4, 7)
    assert as_action_chunk(np.zeros((4, 9)), 4, action_dim=7).shape == (4, 7)


# ---------------------------------------------------------------------------
# SpatialVLA forward (faked model + processor)
# ---------------------------------------------------------------------------
class _FakeProcessor:
    def __init__(self, action):
        self._action = action
        self.seen_text = None

    def __call__(self, images, text, return_tensors):
        self.seen_text = text
        return {"pixel_values": "x"}  # no .to() -> forward skips device move

    def decode_actions(self, generation, unnorm_key):
        assert generation == "GEN"
        self.seen_unnorm = unnorm_key
        return self._action


class _FakeSpatialModel:
    def predict_action(self, inputs):
        return "GEN"


def _make_spatialvla(monkeypatch, action, **meta):
    import torch

    def fake_load(self, policy_uri, device, dtype):
        self._torch = torch
        self._device = torch.device("cpu")
        self._dtype = torch.float32
        self._processor = _FakeProcessor(action)
        self._model = _FakeSpatialModel()

    monkeypatch.setattr(SpatialVLABackend, "_load", fake_load)
    return SpatialVLABackend(policy_uri="x/spatialvla", session_metadata=meta)


def test_spatialvla_forward_returns_fit_chunk(monkeypatch):
    be = _make_spatialvla(monkeypatch, np.zeros((4, 7), dtype=np.float32))
    obs = {
        "observation.images.top": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation.state": np.zeros(6, dtype=np.float32),  # ignored by SpatialVLA
        "task": "pick up the red cube",
    }
    out = be.forward(obs, None)
    assert out.shape == (be.chunk_size, be.action_dim)
    assert out.dtype == np.float32 and np.isfinite(out).all()
    assert be._processor.seen_text == "pick up the red cube"
    assert be._processor.seen_unnorm == "bridge_orig/1.0.0"


def test_spatialvla_unnorm_key_and_dict_decode(monkeypatch):
    # decode_actions may return a dict; unnorm_key comes from metadata.
    be = _make_spatialvla(
        monkeypatch, {"actions": np.ones((4, 7), dtype=np.float32)},
        unnorm_key="fractal20220817_data/0.1.0",
    )
    obs = {"cam": np.zeros((100, 100, 3), dtype=np.uint8), "task": "x"}
    out = be.forward(obs, None)
    assert out.shape == (be.chunk_size, be.action_dim)
    assert be._processor.seen_unnorm == "fractal20220817_data/0.1.0"


def test_spatialvla_requires_image(monkeypatch):
    be = _make_spatialvla(monkeypatch, np.zeros((4, 7), dtype=np.float32))
    with pytest.raises(ValueError, match="camera image"):
        be.forward({"observation.state": np.zeros(6, dtype=np.float32)}, None)


# ---------------------------------------------------------------------------
# RDT forward (faked model + text embeds)
# ---------------------------------------------------------------------------
class _FakeRDT:
    def __init__(self, out):
        self._out = out
        self.last_images = None

    def step(self, proprio, images, text_embeds):
        self.last_images = images
        return self._out


def _make_rdt(monkeypatch, out, action_dim=7, **meta):
    import torch

    def fake_load(self, policy_uri, device, dtype):
        self._torch = torch
        self._device = torch.device("cpu")
        self._dtype = torch.float32
        self._model = _FakeRDT(out)

    monkeypatch.setattr(RDTBackend, "_load", fake_load)
    # Skip T5: feed a dummy embedding so forward never touches transformers.
    monkeypatch.setattr(RDTBackend, "_text_embeds", lambda self, instr: "EMB")
    return RDTBackend(
        policy_uri="x/rdt-1b", action_dim=action_dim, session_metadata=meta
    )


def test_rdt_requires_action_dim(monkeypatch):
    monkeypatch.setattr(RDTBackend, "_load", lambda *a, **k: None)
    with pytest.raises(ValueError, match="positive action_dim"):
        RDTBackend(policy_uri="x/rdt-1b", action_dim=0)


def test_rdt_forward_slices_unified_to_robot_dofs(monkeypatch):
    # RDT emits a (64, 128) unified chunk; backend must return (64, 7).
    unified = np.tile(np.arange(128, dtype=np.float32), (64, 1))
    be = _make_rdt(monkeypatch, unified, action_dim=7)
    obs = {
        "observation.images.exterior": np.zeros((50, 50, 3), dtype=np.uint8),
        "observation.state": np.zeros(14, dtype=np.float32),
        "task": "fold the towel",
    }
    out = be.forward(obs, None)
    assert out.shape == (64, 7) and out.dtype == np.float32
    # Default slice is the first action_dim columns (0..6).
    assert np.allclose(out[0], np.arange(7))
    # RDT gets 6 image slots (3 prev + 3 cur).
    assert len(be._model.last_images) == 6


def test_rdt_action_indices_metadata(monkeypatch):
    unified = np.tile(np.arange(128, dtype=np.float32), (64, 1))
    be = _make_rdt(monkeypatch, unified, action_dim=3, action_indices="10,20,30")
    obs = {"cam": np.zeros((40, 40, 3), dtype=np.uint8), "task": "x"}
    out = be.forward(obs, None)
    assert out.shape == (64, 3)
    assert np.allclose(out[0], [10, 20, 30])


def test_rdt_image_history_advances(monkeypatch):
    unified = np.zeros((64, 128), dtype=np.float32)
    be = _make_rdt(monkeypatch, unified, action_dim=7)
    frame1 = np.full((40, 40, 3), 1, dtype=np.uint8)
    frame2 = np.full((40, 40, 3), 2, dtype=np.uint8)
    be.forward({"cam": frame1, "task": "x"}, None)
    # First tick: no history yet -> prev slots == current frame.
    assert be._model.last_images[0] is not None
    be.forward({"cam": frame2, "task": "x"}, None)
    # Second tick: previous-timestep slot holds frame1, current holds frame2.
    assert int(be._model.last_images[0][0, 0, 0]) == 1
    assert int(be._model.last_images[3][0, 0, 0]) == 2
