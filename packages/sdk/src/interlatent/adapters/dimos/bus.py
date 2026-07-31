"""The dimos bus layer: latest-wins subscription caches + the command publisher.

Everything that touches dimos transports lives here, behind an injectable
``transport_factory`` so unit tests never need dimos installed. The factory
signature is dimos's own ``make_transport(topic, msg_type)`` — ``msg_type=None``
selects the pickled transport (used for episode markers).

Latest-wins is the contract on BOTH sides of this file: dimos image/pointcloud
QoS drops stale frames, and the servo task consumes the newest ``joint_command``
at its tick — which is exactly ``send_action``'s fire-and-forget semantics.

Staleness is gated on LOCAL ARRIVAL time (``time.monotonic()`` at callback), not
the message's producer ``ts``: same-host loopback makes arrival time the honest
signal, and it is immune to clock skew if the stack ever spans hosts (the
producer ``ts`` is still exposed for recording/debug).
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from .config import DimosAdapterConfig

_logger = logging.getLogger(__name__)

# transport_factory(topic, msg_type_or_None) -> object with:
#   subscribe(callback) -> unsubscribe_fn ; broadcast(None, msg) ; stop()
TransportFactory = Callable[[str, Optional[type]], Any]


@dataclass(frozen=True)
class CachedMsg:
    msg: Any
    arrival_monotonic: float
    producer_ts: float | None


def _default_transport_factory() -> TransportFactory:
    from dimos.core.transport_factory import make_transport  # heavy; [dimos] extra

    return make_transport


def _resolve_msg_types() -> tuple[type, type]:
    from dimos.msgs.sensor_msgs.Image import Image
    from dimos.msgs.sensor_msgs.JointState import JointState

    return JointState, Image


def image_to_rgb(msg: Any) -> np.ndarray:
    """dimos ``Image`` -> ``uint8 HxWx3`` RGB (the adapter observation contract).

    numpy-only on purpose — no cv2 import in the adapter. Format names follow
    dimos's ``ImageFormat`` enum values.
    """
    data = np.asarray(msg.data)
    fmt = getattr(getattr(msg, "format", None), "value", "RGB")
    if data.ndim == 2:  # GRAY / GRAY16 / DEPTH*
        if data.dtype != np.uint8:  # normalize 16-bit to 8 for the policy eye
            span = float(data.max()) or 1.0
            data = (data.astype(np.float32) / span * 255.0).astype(np.uint8)
        return np.repeat(data[:, :, None], 3, axis=2)
    if fmt in ("BGR", "BGRA"):
        data = data[:, :, 2::-1]  # reverse first three channels
    elif fmt == "RGBA":
        data = data[:, :, :3]
    else:  # RGB (or unknown 3-channel: pass through)
        data = data[:, :, :3]
    if data.dtype != np.uint8:
        data = np.clip(data, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(data)


class DimosBus:
    """Bus peer: subscriptions with latest-wins caches, plus publishers.

    Lifecycle: ``open()`` builds transports and subscribes; ``close()`` tears
    down. Not reentrant.
    """

    def __init__(
        self,
        cfg: DimosAdapterConfig,
        transport_factory: TransportFactory | None = None,
        joint_state_cls: type | None = None,
        image_cls: type | None = None,
    ) -> None:
        self._cfg = cfg
        self._factory = transport_factory
        self._joint_state_cls = joint_state_cls
        self._image_cls = image_cls
        self._lock = threading.Lock()
        self._joint_state: CachedMsg | None = None
        self._images: dict[str, CachedMsg] = {}
        self._transports: list[Any] = []
        self._unsubs: list[Callable[[], None]] = []
        self._cmd_transport: Any = None
        self._episode_transport: Any = None
        self._open = False

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        if self._open:
            return
        if self._factory is None:
            self._factory = _default_transport_factory()
        if self._joint_state_cls is None or self._image_cls is None:
            js_cls, img_cls = _resolve_msg_types()
            self._joint_state_cls = self._joint_state_cls or js_cls
            self._image_cls = self._image_cls or img_cls

        js = self._factory(self._cfg.joint_state_topic, self._joint_state_cls)
        self._transports.append(js)
        self._unsubs.append(js.subscribe(self._on_joint_state))

        for name, topic in self._cfg.cameras.items():
            t = self._factory(topic, self._image_cls)
            self._transports.append(t)
            self._unsubs.append(t.subscribe(self._make_image_cb(name)))

        self._cmd_transport = self._factory(
            self._cfg.joint_command_topic, self._joint_state_cls
        )
        self._transports.append(self._cmd_transport)
        # Pickled transport (msg_type=None): EpisodeMarker is an SDK-side class.
        self._episode_transport = self._factory(self._cfg.episode_topic, None)
        self._transports.append(self._episode_transport)
        self._open = True

    def close(self) -> None:
        for unsub in self._unsubs:
            try:
                unsub()
            except Exception:  # noqa: BLE001 - teardown is best-effort
                pass
        self._unsubs.clear()
        for t in self._transports:
            try:
                t.stop()
            except Exception:  # noqa: BLE001
                pass
        self._transports.clear()
        self._cmd_transport = None
        self._episode_transport = None
        self._open = False

    # ------------------------------------------------------------------
    # subscription caches
    # ------------------------------------------------------------------

    def _on_joint_state(self, msg: Any) -> None:
        cached = CachedMsg(msg, time.monotonic(), getattr(msg, "ts", None))
        with self._lock:
            self._joint_state = cached

    def _make_image_cb(self, name: str) -> Callable[[Any], None]:
        def _cb(msg: Any) -> None:
            cached = CachedMsg(msg, time.monotonic(), getattr(msg, "ts", None))
            with self._lock:
                self._images[name] = cached

        return _cb

    def latest_joint_state(self) -> CachedMsg | None:
        with self._lock:
            return self._joint_state

    def latest_image(self, name: str) -> CachedMsg | None:
        with self._lock:
            return self._images.get(name)

    def joint_state_age_ms(self) -> float | None:
        """Age of the newest joint state, or None if none arrived yet."""
        cached = self.latest_joint_state()
        if cached is None:
            return None
        return (time.monotonic() - cached.arrival_monotonic) * 1000.0

    def image_age_ms(self, name: str) -> float | None:
        cached = self.latest_image(name)
        if cached is None:
            return None
        return (time.monotonic() - cached.arrival_monotonic) * 1000.0

    # ------------------------------------------------------------------
    # publishers
    # ------------------------------------------------------------------

    def publish_joint_command(
        self, names: list[str], positions: list[float]
    ) -> None:
        """Publish one joint-position command (dimos names, radians).

        Fire-and-forget: the servo task consumes the newest at its own tick.
        """
        assert self._cmd_transport is not None, "bus not open"
        msg = self._joint_state_cls(name=list(names), position=list(positions))
        self._cmd_transport.broadcast(None, msg)

    def publish_episode_marker(self, marker: Any) -> None:
        """Best-effort marker publish on the pickled episode topic."""
        if self._episode_transport is None:
            return
        try:
            self._episode_transport.broadcast(None, marker)
        except Exception:  # noqa: BLE001 - markers must never break the loop
            _logger.warning("episode marker publish failed", exc_info=True)


__all__ = ["DimosBus", "CachedMsg", "image_to_rgb", "TransportFactory"]
