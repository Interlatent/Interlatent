"""Tests for the video-codec compatibility shim (``storage/lerobot_codec.py``).

Regression cover for a silent data-loss bug: both dataset writers passed
``vcodec=`` to ``LeRobotDataset.create()`` unconditionally, but upstream
renamed that parameter to ``camera_encoder`` on 2026-05-14 and to
``rgb_encoder`` on 2026-06-27, and ``create()`` takes no ``**kwargs``. On
any lerobot from mid-May 2026 — including the ref ``docker/Dockerfile``
pins — the call raised ``TypeError``, the writers' broad excepts turned
that into "rebuild failed; aborting upload", and the staged episode was
deleted. The robot side saw a clean close.

Each generation is exercised against a stand-in whose ``create``
signature matches the real one, so all three are covered without
installing three lerobots. ``test_lerobot_real_build.py`` then proves the
resolved kwargs are actually accepted by whichever lerobot is installed.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from interlatent_server.storage import lerobot_codec  # noqa: E402
from interlatent_server.storage.lerobot_codec import video_encoder_kwargs  # noqa: E402


# --- stand-ins for the three upstream generations of create() -----------


def create_v0_4_4(*, repo_id, fps, features, root, robot_type, use_videos,
                  vcodec=None):
    """lerobot v0.4.4 – v0.5.1."""


def create_pinned(*, repo_id, fps, features, root, robot_type, use_videos,
                  camera_encoder=None, streaming_encoding=False):
    """lerobot after bd9619df (2026-05-14) — what docker/Dockerfile pins."""


def create_v0_6(*, repo_id, fps, features, root, robot_type, use_videos,
                rgb_encoder=None, depth_encoder=None):
    """lerobot after 3dd19d04 (2026-06-27) / v0.6.x."""


def create_future(*, repo_id, fps, features, root, robot_type, use_videos,
                  encoder_settings=None):
    """A hypothetical fourth rename."""


@dataclass
class _Cfg:
    vcodec: str


@pytest.fixture(autouse=True)
def _config_classes(monkeypatch):
    """Stand in for lerobot.configs.{VideoEncoderConfig,RGBEncoderConfig}."""
    import types

    mod = types.ModuleType("lerobot.configs")
    mod.VideoEncoderConfig = _Cfg
    mod.RGBEncoderConfig = _Cfg
    pkg = types.ModuleType("lerobot")
    pkg.configs = mod
    monkeypatch.setitem(sys.modules, "lerobot", pkg)
    monkeypatch.setitem(sys.modules, "lerobot.configs", mod)
    # Reset the once-per-process error latch so each test sees it fresh.
    monkeypatch.setattr(lerobot_codec, "_WARNED", False)


def test_old_lerobot_takes_the_plain_vcodec_kwarg() -> None:
    assert video_encoder_kwargs(create_v0_4_4, "h264") == {"vcodec": "h264"}


def test_the_pinned_lerobot_takes_a_camera_encoder_config() -> None:
    """This is the generation the deployed image is built against — the
    one the old code raised TypeError on."""
    kwargs = video_encoder_kwargs(create_pinned, "h264")
    assert set(kwargs) == {"camera_encoder"}
    assert kwargs["camera_encoder"].vcodec == "h264"


def test_lerobot_0_6_takes_an_rgb_encoder_config() -> None:
    kwargs = video_encoder_kwargs(create_v0_6, "h264")
    assert set(kwargs) == {"rgb_encoder"}
    assert kwargs["rgb_encoder"].vcodec == "h264"
    # depth_encoder is lerobot's business; we must not set it.
    assert "depth_encoder" not in kwargs


@pytest.mark.parametrize(
    "create_fn", [create_v0_4_4, create_pinned, create_v0_6, create_future]
)
def test_the_resolved_kwargs_are_always_accepted_by_that_signature(create_fn) -> None:
    """The point of the shim: whatever it returns must bind. A wrong name
    is a TypeError that costs an episode."""
    import inspect

    kwargs = video_encoder_kwargs(create_fn, "h264")
    inspect.signature(create_fn).bind(
        repo_id="r", fps=30, features={}, root="/tmp/x",
        robot_type="t", use_videos=True, **kwargs,
    )


@pytest.mark.parametrize("vcodec", [None, ""])
def test_no_codec_preference_means_no_kwargs(vcodec) -> None:
    """None keeps lerobot's own default — the caller said it doesn't care."""
    assert video_encoder_kwargs(create_v0_6, vcodec) == {}


def test_an_unknown_encoder_api_degrades_loudly_but_still_records(caplog) -> None:
    """Raising here would discard the episode. An AV1 shard is recoverable
    (codec-agnostic canonical, ADR 0021); a deleted episode is not — so
    log an ERROR and let lerobot choose."""
    with caplog.at_level("ERROR"):
        assert video_encoder_kwargs(create_future, "h264") == {}
    assert "no known video encoder parameter" in caplog.text
    assert "encoder_settings" in caplog.text  # names what it did find


def test_the_unknown_api_error_is_logged_once_per_process(caplog) -> None:
    """A box on an unsupported lerobot must not log this every session."""
    with caplog.at_level("ERROR"):
        for _ in range(5):
            video_encoder_kwargs(create_future, "h264")
    assert caplog.text.count("no known video encoder parameter") == 1


def test_a_newer_parameter_wins_when_two_are_present() -> None:
    """Upstream kept the old name around for a deprecation window more than
    once; prefer the newer contract rather than the legacy alias."""

    def create_overlap(*, repo_id, fps, features, root, robot_type, use_videos,
                       rgb_encoder=None, camera_encoder=None, vcodec=None):
        pass

    assert set(video_encoder_kwargs(create_overlap, "h264")) == {"rgb_encoder"}


def test_a_signature_that_cannot_be_read_degrades_instead_of_raising() -> None:
    # Some C-implemented callables have no introspectable signature.
    assert video_encoder_kwargs(print, "h264") == {}


def test_config_class_falls_back_to_the_submodule(monkeypatch) -> None:
    """lerobot re-exports the config from ``lerobot.configs``, but older
    layouts only have ``lerobot.configs.video``."""
    import types

    bare = types.ModuleType("lerobot.configs")  # no RGBEncoderConfig
    video = types.ModuleType("lerobot.configs.video")
    video.RGBEncoderConfig = _Cfg
    monkeypatch.setitem(sys.modules, "lerobot.configs", bare)
    monkeypatch.setitem(sys.modules, "lerobot.configs.video", video)

    kwargs = video_encoder_kwargs(create_v0_6, "h264")
    assert kwargs["rgb_encoder"].vcodec == "h264"
