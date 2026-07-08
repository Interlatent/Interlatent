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

from .. import routing

_LOG = logging.getLogger("interlatent.node.daemon")


def _reachable_addresses() -> tuple[str, list[str]]:
    """Best-effort list of this host's non-loopback addresses (no extra deps).

    The node is outbound-only today (it dials the dashboard + GPU), so this
    is informational — useful for debugging and for future routing methods
    where the node must advertise an address.
    """
    import socket

    ips: list[str] = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))  # no packets sent — just selects a route
            ips.append(s.getsockname()[0])
        finally:
            s.close()
    except OSError:
        pass
    host = socket.gethostname()
    try:
        for info in socket.getaddrinfo(host, None):
            ip = info[4][0]
            if ip not in ips and not ip.startswith("127.") and ip != "::1":
                ips.append(ip)
    except OSError:
        pass
    return host, ips


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
    # Protection-bypass secret for a protected preview/test domain (e.g. a
    # Vercel branch deployment). Sent as x-vercel-protection-bypass on every
    # heartbeat/poll so the daemon reaches the same domain the node paired
    # against. None on production.
    bypass_key: Optional[str] = None
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
    # Sequential (request-response) chunking for every session this node runs:
    # one fully-drained chunk per observation, no async overlap. Diagnostic
    # fallback for high-latency policies whose overlapping plans thrash the robot
    # (MolmoAct2). Node-level today; the per-Session home is the dashboard payload
    # (see the SDK CONTEXT.md flagged ambiguity).
    synchronous: bool = False

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
        _headers = {"x-api-key": cfg.token}
        if (cfg.bypass_key or "").strip():
            # Protected test domains (Vercel preview deployments) challenge
            # un-bypassed requests; carry the automation bypass secret.
            _headers["x-vercel-protection-bypass"] = cfg.bypass_key.strip()
        self._http = httpx.AsyncClient(
            base_url=cfg.api_base,
            headers=_headers,
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
        _host, _ips = _reachable_addresses()
        if _ips:
            _LOG.info(
                "Node host addresses: %s (hostname=%s)", ", ".join(_ips), _host
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
                    # Assignment envelope (typed: inference_session |
                    # teleop_recording). Legacy backends only send the
                    # bare ``session`` field.
                    envelope = data.get("assignment") or {}
                    if envelope.get("type") == "teleop_recording":
                        payload = envelope.get("recording")
                        kind = "teleop_recording"
                    else:
                        payload = data.get("session") or envelope.get("session")
                        kind = "inference_session"
                    # _converge runs a blocking connect_drtc() (the DRTC
                    # OpenSession can take 30s+ on a cold container). Run
                    # it off the event loop so the heartbeat coroutine
                    # keeps firing — otherwise the node looks offline and
                    # the backend drops its session assignment.
                    await asyncio.to_thread(self._converge, payload, kind)
            except Exception as e:
                _LOG.warning("Poll failed: %s", e)
                await asyncio.sleep(min(backoff, self.cfg.max_backoff_s))
                backoff = min(backoff * 2, self.cfg.max_backoff_s)

    def _resolve_route(self, session: dict) -> Optional[dict]:
        """Pick the route descriptor we'd actually use for this session.

        Precedence (preserves the legacy endpoint precedence):
          1. ``INTERLATENT_DRTC_URL`` env var — operator override (direct)
          2. the session's ``route`` block stamped by the dashboard
          3. the session's legacy ``drtc_endpoint`` (direct)
          4. node config ``drtc_url`` — fixed fallback (direct)
        Returns ``None`` when nothing resolves.
        """
        env = os.environ.get("INTERLATENT_DRTC_URL")
        if env:
            return routing.make_descriptor(env)
        route = session.get("route")
        if route and route.get("address"):
            return route
        ep = session.get("drtc_endpoint")
        if ep:
            return routing.make_descriptor(ep)
        if self.cfg.drtc_url:
            return routing.make_descriptor(self.cfg.drtc_url)
        return None

    def _resolve_endpoint(self, session: dict) -> str:
        """The address we'd dial — used by _converge to detect endpoint changes.

        Mirrors the precedence in _resolve_route so _converge can tell
        whether a server-reported endpoint change actually affects us
        (when we're pinned via env var / cfg, it doesn't).
        """
        route = self._resolve_route(session)
        return route.get("address", "") if route else ""

    def _converge(
        self, session: Optional[dict], kind: str = "inference_session"
    ) -> None:
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
            self._start_loop(session, kind=kind)
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

    # Robot kinds that ship their own native (non-LeRobot) control loop, mapped
    # to the "module:function" the daemon imports. A native loop talks to its
    # robot's own SDK and reuses only the LeRobot-free DRTC wire helpers — so the
    # bundled LeRobot wrapper (and lerobot itself) is never imported for it. This
    # is the single place a vendor robot registers a native loop.
    # See docs/adr/0011-vendor-robot-subpackage-via-robot-kind.md.
    _NATIVE_LOOPS: dict[str, str] = {
        "axol": "interlatent.adapters.axol:control_loop",
        "yam": "interlatent.adapters.yam:control_loop",
    }

    def _resolve_loop_fn(self) -> Callable[..., None]:
        """Pick the control-loop function exactly once.

        --loop module:fn wins. Else a robot kind with a registered native loop
        uses that. Else the bundled LeRobot wrapper.
        """
        if self._loop_fn is not None:
            return self._loop_fn

        from .control import import_callable

        kind = (self.cfg.robot_kind or "").lower().strip()
        if self.cfg.loop_override:
            self._loop_fn = import_callable(self.cfg.loop_override)
        elif kind in self._NATIVE_LOOPS:
            self._loop_fn = import_callable(self._NATIVE_LOOPS[kind])
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

    def _start_loop(self, session: dict, kind: str = "inference_session") -> None:
        from interlatent.inference.integration.connect import connect_drtc

        # Full VR-teleop recording (no policy): the "session" payload is a
        # TeleopRecordingOut. The loop runs with policy_enabled=False —
        # never client.step() (the echo backend would drive the robot with
        # a sinusoid) — and all motion comes from the teleop channel's
        # mode="targets" frames; every tick still records via RecordTick.
        is_recording = kind == "teleop_recording"

        # DRTC route resolution (see _resolve_route for precedence): env-var
        # override > dashboard-stamped ``route`` > legacy ``drtc_endpoint`` >
        # node-config ``drtc_url``. The route's ``method`` selects a connector
        # (only ``direct`` today); the connector yields the address to dial.
        # If nothing resolves we refuse to start rather than hang against an
        # empty address. The usual fix: register/select a reachable GPU box.
        route = self._resolve_route(session)
        if not route or not route.get("address"):
            _LOG.error(
                "No DRTC endpoint for session %s. Attach a reachable compute "
                "pod to this session in the Interlatent dashboard, or set "
                "INTERLATENT_DRTC_URL to a fixed endpoint.",
                session.get("id", "?"),
            )
            return
        try:
            drtc_endpoint = routing.connect_params(route)["server_address"]
        except ValueError as exc:
            _LOG.error("Cannot route session %s: %s", session.get("id", "?"), exc)
            return
        if not drtc_endpoint:
            _LOG.error("Route for session %s resolved to an empty address",
                       session.get("id", "?"))
            return
        _LOG.info("DRTC route for session %s: method=%s endpoint=%s",
                  session.get("id", "?"), route.get("method", "direct"), drtc_endpoint)

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
        # The pod-side teleop retarget stage picks the robot bundle
        # (URDF + ik_config) by robot kind. Without this an engaged VR
        # producer gets ee_state{ready:false, reason:"no_robot_kind"}.
        if self.cfg.robot_kind:
            session_metadata["robot_kind"] = str(self.cfg.robot_kind)
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

        # Recording destination is configured on the dashboard and rides in
        # the session payload's ``recording`` block. The node forwards it
        # verbatim into OpenSession metadata; the GPU container's recorder
        # interprets the keys (output_dir / s3_uri / s3_*) to pick a sink.
        # Opaque to the node.
        recording_cfg = session.get("recording") or {}
        for _k, _v in recording_cfg.items():
            if _v is not None:
                session_metadata[str(_k)] = str(_v)

        client = connect_drtc(
            # DRTC auth needs the ilat_ user key; the node token is
            # rejected by the server's /environments probe.
            api_key=self.cfg.drtc_api_key or self.cfg.token,
            environment=env_slug,
            # Recording sessions load no policy — the echo backend is a
            # placeholder the loop never steps (policy_enabled=False).
            policy_uri=("teleop-recording" if is_recording
                        else session.get("policy_uri", "")),
            policy_backend=("echo" if is_recording
                            else session.get("policy_backend", "lerobot")),
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
            synchronous=self.cfg.synchronous,
        )

        # Hosted DAgger teleop receiver. The TeleopChannel owns a background WS
        # to the GPU-box relay for the session lifetime (idle when no producer
        # is engaged); the control loop reads the latest frame and overrides the
        # policy when engaged. We use ``drtc_api_key`` because the teleop-token
        # endpoint is owned by the user, not the node — the node token is
        # rejected by the relay's auth. Skipped (teleop disabled) when no user
        # key or session id is available.
        # The factory picks WS vs QUIC/WebTransport from the backend's
        # ``transport`` flag (returned in the teleop-token response). Both
        # channels expose the same surface, so the control loop is unchanged.
        from .teleop.factory import make_teleop_channel

        teleop_channel = None
        teleop_api_key = self.cfg.drtc_api_key or ""
        if teleop_api_key and session.get("id"):
            teleop_channel = make_teleop_channel(
                session_id=session["id"],
                api_base=self.cfg.api_base,
                api_key=teleop_api_key,
                # Recordings mint against their own route (the relay lives
                # in the Modal container behind a TLS tunnel, not on a
                # tailnet box). 409s while the pod provisions are absorbed
                # by the channel's retry loop.
                token_path=(
                    f"/api/v1/teleop-recordings/{session['id']}/teleop-token"
                    if is_recording else None
                ),
                bypass_key=self.cfg.bypass_key,
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
            "bypass_key": self.cfg.bypass_key,
            "image_resize": image_resize,
            # False for teleop recordings: the loop must never client.step()
            # (no policy is loaded; the echo backend returns sinusoids).
            "policy_enabled": not is_recording,
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
        # The control-loop runner stops the teleop channel in its finally
        # clause, but if the loop thread is stuck the channel would never get
        # torn down — close it here too so the WS connection isn't leaked.
        if h.teleop_channel is not None:
            try:
                h.teleop_channel.stop()
            except Exception:
                pass
        if h.thread is not None:
            h.thread.join(timeout=10.0)
            if h.thread.is_alive():
                # The control loop wedged in its teardown — e.g. a robot
                # disconnect blocked on a downed bus (i2rt YAM arm.close()
                # selecting on a closed CAN fd). Its finally never ran, so
                # client.close() — and thus CloseSession + the server-side
                # upload — never fired, and the recording would be lost to
                # the idle-GC. Force the close ourselves. client.close() is
                # idempotent, so if the thread later unwedges its own finally
                # is a harmless no-op.
                _LOG.warning(
                    "Control-loop thread did not exit within 10s (robot "
                    "teardown wedged); force-closing DRTC client to flush "
                    "the recorder and trigger upload"
                )
                try:
                    h.client.close()
                except Exception:  # noqa: BLE001
                    _LOG.warning("Force client.close() failed", exc_info=True)


@dataclass
class _ControlLoopHandle:
    client: Any
    teleop_channel: Any = None
    stop_flag: threading.Event = field(default_factory=threading.Event)
    thread: Optional[threading.Thread] = None

    def should_stop(self) -> bool:
        return self.stop_flag.is_set()
