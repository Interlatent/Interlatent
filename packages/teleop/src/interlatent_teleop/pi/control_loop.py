"""Pi-side control loop: pulls latest target via SafetyGate, writes to driver.

Runs in its own thread at a fixed rate. The gRPC server thread feeds
TeleopTargets into the SafetyGate; this loop owns the only writes to
the hardware and the only reads of the current pose used in TeleopAck.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .safety import SafetyGate
from .so101_driver import RobotDriver

_LOG = logging.getLogger("interlatent_teleop.pi.loop")


@dataclass
class LoopState:
    current_joints: np.ndarray
    last_status: str
    last_commanded_ts_ns: int
    estopped: bool


class ControlLoop:
    def __init__(
        self,
        driver: RobotDriver,
        gate: SafetyGate,
        control_hz: int,
        on_tick: Optional[Callable[[LoopState], None]] = None,
    ) -> None:
        self.driver = driver
        self.gate = gate
        self.control_hz = control_hz
        self.dt = 1.0 / control_hz
        self._on_tick = on_tick
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._state = LoopState(
            current_joints=np.zeros(len(driver.joint_names), dtype=np.float32),
            last_status="init",
            last_commanded_ts_ns=0,
            estopped=False,
        )

    @property
    def state(self) -> LoopState:
        with self._lock:
            return LoopState(
                current_joints=self._state.current_joints.copy(),
                last_status=self._state.last_status,
                last_commanded_ts_ns=self._state.last_commanded_ts_ns,
                estopped=self._state.estopped,
            )

    def start(self) -> None:
        if self._thread is not None:
            return
        # Seed current pose from the driver before commanding anything.
        initial = self.driver.read_joints()
        with self._lock:
            self._state.current_joints = initial.astype(np.float32).copy()
            self._state.last_status = "started"
        self._thread = threading.Thread(target=self._run, name="teleop-control", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        next_tick = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            current = self.driver.read_joints()
            commanded, status = self.gate.step(current, now=now)
            try:
                self.driver.write_joints(commanded)
            except Exception as e:  # noqa: BLE001
                _LOG.exception("driver write failed: %s", e)
                self.gate.latch_estop("driver_write_failed")
                status = "driver_error"
            with self._lock:
                self._state.current_joints = current
                self._state.last_status = status
                self._state.last_commanded_ts_ns = time.monotonic_ns()
                self._state.estopped = self.gate.config.estop_latched
            if self._on_tick is not None:
                try:
                    self._on_tick(self._state)
                except Exception:  # noqa: BLE001
                    _LOG.exception("on_tick callback raised")

            next_tick += self.dt
            sleep = next_tick - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                # Fell behind — resync rather than spin.
                next_tick = time.monotonic()
