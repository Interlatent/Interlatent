"""Node daemon — long-poll + heartbeat + control-loop dispatch.

The daemon owns three concurrent tasks:

1. **Heartbeat** (10s): POST /nodes/{id}/heartbeat. Lets the dashboard
   show the node as online.
2. **Poll** (long-poll, up to ~25s per request): GET /nodes/{id}/poll.
   Returns the current desired session for this node (or null).
   Whenever the assignment changes, we converge.
3. **Control loop** (when assigned): spawned in a background thread
   that owns the `connect_drtc()` client and the robot I/O.

Convergence rules:
    desired None  + actual None     -> noop
    desired None  + actual running  -> stop the loop, close client
    desired X     + actual None     -> open client + start loop with X
    desired X     + actual running Y -> stop, then start with X
                                        (shouldn't happen unless the
                                         backend was edited; backend
                                         refuses to overwrite a busy
                                         node — but we handle it
                                         anyway for robustness)
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import httpx

_LOG = logging.getLogger("interlatent.node.daemon")


@dataclass
class NodeDaemonConfig:
    node_id: str
    token: str
    api_base: str
    # Interlatent user API key (ilat_...). The node `token` authenticates
    # heartbeat/poll against the nodes API, but the DRTC server validates
    # against `/environments` (require_auth), which rejects node tokens —
    # so DRTC inference must use the user API key instead.
    drtc_api_key: Optional[str] = None
    # DRTC inference endpoint (gRPC-Web URL). Captured at `pair` time
    # and persisted to ~/.interlatent/node.toml. The daemon refuses to
    # start a session if this is unset everywhere (env var, this field,
    # session payload) — there is no hosted default.
    drtc_url: Optional[str] = None
    robot_kind: Optional[str] = None
    robot_port: Optional[str] = None
    robot_extra: dict[str, str] = field(default_factory=dict)
    # {camera_name: device} — name becomes the observation.images.<name> key.
    robot_cameras: dict[str, str] = field(default_factory=dict)
    loop_override: Optional[str] = None  # "module:callable"

    # Flow-matching denoising steps used by VLA policies (currently
    # MolmoAct2). None lets the GPU pick its default (5 for MolmoAct2).
    # Lower = faster compute, slightly noisier actions; 3-5 is the
    # usable range for SO100/101.
    num_inference_steps: Optional[int] = None
    # Pre-encode resize target (pixels per side) for camera frames.
    # None keeps native resolution; explicit int squares the frame.
    # Auto-defaulted to 256 for MolmoAct2 sessions (its image processor
    # downsamples anyway, so sending native 640x480 burns bandwidth).
    image_resize: Optional[int] = None

    heartbeat_period_s: float = 10.0
    poll_wait_s: int = 25
    reconnect_backoff_s: float = 2.0
    max_backoff_s: float = 30.0


# ---------------------------------------------------------------------------
# Daemon
# ---------------------------------------------------------------------------


class NodeDaemon:
    def __init__(self, cfg: NodeDaemonConfig) -> None:
        self.cfg = cfg
        self._http = httpx.AsyncClient(
            base_url=cfg.api_base,
            headers={"x-api-key": cfg.token},
            timeout=httpx.Timeout(cfg.poll_wait_s + 10),
        )
        self._known_session_id: str = ""  # what we've executed against
        # Server's most recently reported DRTC endpoint for our active
        # session. We echo this back on each poll so the backend can
        # wake us if the env's attached compute box (and therefore the
        # GPU endpoint) changes — without this the long-poll would only
        # fire on session_id changes and a box attached after the
        # session was created would never reach us.
        self._known_endpoint: str = ""
        # The endpoint our active loop is actually connected to (post
        # INTERLATENT_DRTC_URL / cfg.drtc_url override resolution).
        # Different from _known_endpoint when an operator override is
        # in play — we only restart when *this* changes.
        self._active_endpoint: str = ""
        # Active control loop (None when idle)
        self._active: Optional[_ControlLoopHandle] = None
        # Resolved at start_session time; lazy so non-lerobot uses don't
        # eat the import.
        self._loop_fn: Optional[Callable[..., None]] = None

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------

    def run_forever(self) -> None:
        try:
            asyncio.run(self._main())
        except KeyboardInterrupt:
            _LOG.info("Shutdown requested, stopping any active session...")
            self._stop_active_loop()

    async def _main(self) -> None:
        _LOG.info(
            "Node daemon online: node_id=%s api_base=%s",
            self.cfg.node_id, self.cfg.api_base,
        )
        # Best-effort: tell the dashboard what hardware is attached so
        # the user can examine it (robot type, USB port, cameras). Never
        # fatal — a failed report just leaves the panel empty.
        await self._report_hardware()
        try:
            await asyncio.gather(
                self._heartbeat_loop(),
                self._poll_loop(),
            )
        finally:
            await self._http.aclose()
            self._stop_active_loop()

    async def _report_hardware(self) -> None:
        """POST the node's known hardware to the dashboard for examination.

        Only what the daemon knows at launch — robot type, USB serial
        port, cameras (name -> device), extra robot args. State/action
        dims aren't known until the robot connects inside the control
        loop, so they're reported later via /robot-features.
        """
        payload = {
            "robot_type": self.cfg.robot_kind,
            "robot_port": self.cfg.robot_port,
            "cameras": [
                {"name": name, "device": device}
                for name, device in self.cfg.robot_cameras.items()
            ],
            "robot_args": dict(self.cfg.robot_extra),
        }
        try:
            r = await self._http.post(
                f"/api/v1/nodes/{self.cfg.node_id}/hardware", json=payload
            )
            if r.status_code >= 400:
                _LOG.warning("Hardware report %s: %s", r.status_code, r.text)
        except Exception as e:
            _LOG.warning("Hardware report failed (non-fatal): %s", e)

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------

    async def _heartbeat_loop(self) -> None:
        backoff = self.cfg.reconnect_backoff_s
        while True:
            try:
                r = await self._http.post(
                    f"/api/v1/nodes/{self.cfg.node_id}/heartbeat"
                )
                if r.status_code >= 400:
                    _LOG.warning("Heartbeat %s: %s", r.status_code, r.text)
                    await asyncio.sleep(min(backoff, self.cfg.max_backoff_s))
                    backoff = min(backoff * 2, self.cfg.max_backoff_s)
                    continue
                backoff = self.cfg.reconnect_backoff_s
            except Exception as e:  # network blip
                _LOG.warning("Heartbeat failed: %s", e)
                await asyncio.sleep(min(backoff, self.cfg.max_backoff_s))
                backoff = min(backoff * 2, self.cfg.max_backoff_s)
                continue
            await asyncio.sleep(self.cfg.heartbeat_period_s)

    # ------------------------------------------------------------------
    # Long-poll + convergence
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        backoff = self.cfg.reconnect_backoff_s
        while True:
            try:
                r = await self._http.get(
                    f"/api/v1/nodes/{self.cfg.node_id}/poll",
                    params={
                        "known_session_id": self._known_session_id,
                        "known_endpoint": self._known_endpoint,
                        "wait": self.cfg.poll_wait_s,
                    },
                )
                if r.status_code >= 400:
                    _LOG.warning("Poll %s: %s", r.status_code, r.text)
                    await asyncio.sleep(min(backoff, self.cfg.max_backoff_s))
                    backoff = min(backoff * 2, self.cfg.max_backoff_s)
                    continue
                backoff = self.cfg.reconnect_backoff_s
                data = r.json()
                if data.get("changed"):
                    # _converge runs a blocking connect_drtc() (the DRTC
                    # OpenSession can take 30s+ on a cold container). Run
                    # it off the event loop so the heartbeat coroutine
                    # keeps firing — otherwise the node looks offline and
                    # the backend drops its session assignment.
                    await asyncio.to_thread(self._converge, data.get("session"))
            except Exception as e:
                _LOG.warning("Poll failed: %s", e)
                await asyncio.sleep(min(backoff, self.cfg.max_backoff_s))
                backoff = min(backoff * 2, self.cfg.max_backoff_s)

    def _resolve_endpoint(self, session: dict) -> str:
        """Pick the DRTC endpoint we'd actually dial for this session.

        Mirrors the precedence in _start_loop so _converge can tell
        whether a server-reported endpoint change actually affects us
        (when we're pinned via env var / cfg, it doesn't).
        """
        return (
            os.environ.get("INTERLATENT_DRTC_URL")
            or (session.get("drtc_endpoint") or None)
            or self.cfg.drtc_url
            or ""
        )

    def _converge(self, session: Optional[dict]) -> None:
        desired_id = (session or {}).get("id", "") if session else ""
        server_endpoint = (session or {}).get("drtc_endpoint", "") or ""
        desired_endpoint = self._resolve_endpoint(session or {}) if session else ""

        same_session = (
            desired_id == self._known_session_id and self._active is not None
        )
        endpoint_changed = (
            same_session and desired_endpoint != self._active_endpoint
        )

        if same_session and not endpoint_changed:
            # Same session, same resolved endpoint — noop. Sync
            # _known_endpoint to whatever the server now reports so
            # we stop being woken on this diff.
            self._known_endpoint = server_endpoint
            return

        # Always stop the prior loop before starting a new one — either
        # the session changed or the endpoint moved underneath us.
        if self._active is not None:
            if endpoint_changed:
                _LOG.info(
                    "DRTC endpoint changed (%s → %s); reconnecting session %s",
                    self._active_endpoint, desired_endpoint, self._known_session_id,
                )
            else:
                _LOG.info("Stopping current session %s", self._known_session_id)
            self._stop_active_loop()

        if session is None:
            self._known_session_id = ""
            self._known_endpoint = ""
            self._active_endpoint = ""
            return

        try:
            self._start_loop(session)
            self._known_session_id = desired_id
            self._known_endpoint = server_endpoint
            self._active_endpoint = desired_endpoint
        except Exception as e:
            _LOG.exception("Failed to start session %s: %s", desired_id, e)
            # Don't update _known_session_id — next poll will retry once
            # the backend assignment changes (or the user fixes the issue
            # and re-assigns).
            self._known_session_id = ""
            self._known_endpoint = ""
            self._active_endpoint = ""

    # ------------------------------------------------------------------
    # Control-loop lifecycle
    # ------------------------------------------------------------------

    def _resolve_loop_fn(self) -> Callable[..., None]:
        """Pick the control-loop function exactly once.

        --loop module:fn wins. Otherwise the built-in LeRobot wrapper.
        """
        if self._loop_fn is not None:
            return self._loop_fn

        if self.cfg.loop_override:
            from .control import import_callable
            self._loop_fn = import_callable(self.cfg.loop_override)
        else:
            if not self.cfg.robot_kind:
                raise RuntimeError(
                    "No control loop available: pass --robot <name> for "
                    "the bundled LeRobot wrapper, or --loop "
                    "module:function for a custom adapter."
                )
            from .control import lerobot_control_loop
            self._loop_fn = lerobot_control_loop
        return self._loop_fn

    def _start_loop(self, session: dict) -> None:
        from interlatent.inference.integration.connect import connect_drtc

        # DRTC endpoint resolution, in priority order:
        #   1. INTERLATENT_DRTC_URL env var — operator override per run
        #   2. session's `drtc_endpoint` — the dashboard now stamps this
        #      per session from whichever compute box is attached to
        #      the parent env. This is the normal path for self-serve
        #      compute — the user picks a box in the workspace and the
        #      node inherits the endpoint automatically.
        #   3. node config (`drtc_url` in ~/.interlatent/node.toml) —
        #      legacy fallback for fleet-wide / operator-set endpoints.
        # If all three are empty we refuse to start; the daemon would
        # otherwise hang against an empty address with a cryptic gRPC
        # error. The usual fix: attach a running compute box to the
        # env in the dashboard.
        drtc_endpoint = self._resolve_endpoint(session)
        if not drtc_endpoint:
            _LOG.error(
                "No DRTC endpoint for session %s. Attach a running "
                "compute box to this env in the dashboard (Compute → "
                "Spin up, then select it in the env Workspace), or set "
                "INTERLATENT_DRTC_URL to a fixed endpoint.",
                session.get("id", "?"),
            )
            return
        _LOG.info("DRTC endpoint for session %s: %s",
                  session.get("id", "?"), drtc_endpoint)

        # Pull recording context out of the session payload. The backend
        # populates ``collection_context`` from the parent Environment,
        # so by the time we get here we already know the env slug + task
        # + fps and can ask the GPU container to record + upload the
        # episode without any Pi-side staging.
        ctx = session.get("collection_context") or {}
        env_slug = ctx.get("env_slug") or "default"

        # MolmoAct2 (and any policy whose released checkpoint omits the
        # robot I/O contract) needs the camera observation keys at
        # session-open. The node is the authority: each `--camera
        # name=device` becomes the `observation.images.<name>` key the
        # policy sees, so forward those keys through OpenSession.metadata.
        # The GPU backend ignores them for self-describing checkpoints.
        session_metadata: dict[str, str] = {}
        if self.cfg.robot_cameras:
            image_keys = [
                f"observation.images.{name}" for name in self.cfg.robot_cameras
            ]
            session_metadata["image_keys"] = ",".join(image_keys)
            # Continuous flow-matching is the default for action_mode=both
            # checkpoints (smoother trajectories). Harmless for backends
            # that don't read it.
            session_metadata["inference_action_mode"] = "continuous"

        # MolmoAct2 latency-tuning knobs — only forwarded when set, so
        # non-VLA backends see a clean metadata map. Detection is by
        # policy_uri substring because that's the only thing the node
        # knows about the GPU-side model at session-open.
        policy_uri = str(session.get("policy_uri", "")).lower()
        is_molmoact = "molmoact" in policy_uri
        if self.cfg.num_inference_steps is not None:
            session_metadata["num_inference_steps"] = str(
                int(self.cfg.num_inference_steps)
            )
        # Pre-encode resize: lets us avoid uploading 640x480 frames when
        # the policy's image processor will immediately downsample. The
        # node, not the GPU, applies the resize (saves uplink bandwidth),
        # but the *default* is picked here so callers don't have to know
        # which policies care. 256 leaves headroom above Molmo's 224
        # tokenizer input.
        image_resize = self.cfg.image_resize
        if image_resize is None and is_molmoact:
            image_resize = 256

        _LOG.info(
            "Session policy=%s molmoact=%s num_inference_steps=%s image_resize=%s",
            session.get("policy_uri", ""), is_molmoact,
            session_metadata.get("num_inference_steps", "default"),
            image_resize if image_resize is not None else "native",
        )

        client = connect_drtc(
            # DRTC auth needs the ilat_ user key; the node token is
            # rejected by the server's /environments probe.
            api_key=self.cfg.drtc_api_key or self.cfg.token,
            environment=env_slug,
            policy_uri=session.get("policy_uri", ""),
            policy_backend=session.get("policy_backend", "lerobot"),
            task=session.get("task", ""),
            chunk_size=int(session.get("chunk_size", 50) or 50),
            action_dim=int(session.get("action_dim", 6) or 6),
            fps=float(session.get("fps", 30.0) or 30.0),
            server_address=drtc_endpoint,
            metadata=session_metadata or None,
            # Server-side recorder builds + uploads the dataset; the Pi
            # streams per-tick captures via RecordTick so the recorder
            # gets 30 Hz coverage instead of Infer-rate coverage.
            record=True,
            episode_id=session.get("id"),
            env_id=session.get("environment_id"),
        )

        # Browser-driven DAgger teleop channel. The TeleopChannel owns a
        # background WS to the GPU box for the lifetime of the session
        # (idle when no dashboard is engaged). The control loop reads
        # the latest frame and overrides the policy when engaged. We
        # use ``drtc_api_key`` here because the teleop-token endpoint
        # is owned by the user, not the node — the node token is
        # rejected by ``require_auth``.
        from .teleop_channel import TeleopChannel

        teleop_channel: Optional[TeleopChannel] = None
        teleop_api_key = self.cfg.drtc_api_key or ""
        if teleop_api_key and session.get("id"):
            teleop_channel = TeleopChannel(
                session_id=session["id"],
                api_base=self.cfg.api_base,
                api_key=teleop_api_key,
            )
            teleop_channel.start()

        loop_fn = self._resolve_loop_fn()
        handle = _ControlLoopHandle(client=client, teleop_channel=teleop_channel)
        kwargs = {
            "client": client,
            "session": session,
            "should_stop": handle.should_stop,
            "robot_kind": self.cfg.robot_kind,
            "robot_port": self.cfg.robot_port,
            "robot_extra": self.cfg.robot_extra,
            "robot_cameras": self.cfg.robot_cameras,
            "api_key": self.cfg.token,
            "api_base": self.cfg.api_base,
            "teleop_channel": teleop_channel,
            "node_id": self.cfg.node_id,
            "image_resize": image_resize,
        }

        def _runner():
            try:
                loop_fn(**kwargs)
            except Exception:
                _LOG.exception("Control loop crashed")
            finally:
                try:
                    client.close()
                except Exception:
                    pass
                if teleop_channel is not None:
                    try:
                        teleop_channel.stop()
                    except Exception:
                        pass

        handle.thread = threading.Thread(
            target=_runner, name=f"node-loop-{session.get('id','')[:8]}", daemon=True
        )
        handle.thread.start()
        self._active = handle
        _LOG.info("Started session %s (policy=%s task=%r)",
                  session.get("id"), session.get("policy_uri"), session.get("task"))

    def _stop_active_loop(self) -> None:
        if self._active is None:
            return
        h = self._active
        self._active = None
        h.stop_flag.set()
        # The control-loop runner stops the teleop channel inside its
        # finally clause, but if the loop thread is stuck the teleop
        # channel would never get torn down — close it here too so the
        # WS connection isn't leaked.
        if h.teleop_channel is not None:
            try:
                h.teleop_channel.stop()
            except Exception:
                pass
        if h.thread is not None:
            h.thread.join(timeout=10.0)
            if h.thread.is_alive():
                _LOG.warning("Control-loop thread did not exit within 10s")


@dataclass
class _ControlLoopHandle:
    client: Any
    teleop_channel: Any = None
    stop_flag: threading.Event = field(default_factory=threading.Event)
    thread: Optional[threading.Thread] = None

    def should_stop(self) -> bool:
        return self.stop_flag.is_set()
