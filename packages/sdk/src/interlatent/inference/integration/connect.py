"""One-call connection to a DRTC policy server.

Point it at the GPU box you run (``interlatent-serve``) with the key your
coordinator issued and the policy you want to run; ``connect_drtc(...)``
does the rest.

Example:

    from interlatent.inference.integration.connect import connect_drtc

    client = connect_drtc(
        api_key=os.environ["INTERLATENT_API_KEY"],
        environment="smolvla-pickup",
        policy_uri="lerobot/smolvla_base",
        policy_backend="lerobot",
        fps=30,
    )
    try:
        while running:
            action = client.step(observation_npz_bytes, codec="npz")
            if action is not None:
                robot.apply(action)
    finally:
        client.close()

There is no default server address: pass ``server_address=`` (or set
``INTERLATENT_DRTC_URL``) with your box's advertised gRPC address. A node
started by a coordinator is handed that address per session instead.
"""

from __future__ import annotations

import os
from typing import Optional

from ..client import DRTCClient, DRTCConfig

# There is no default GPU endpoint. A DRTC address is either handed to you
# per-session by a coordinator, or you name one yourself with
# ``server_address=`` / ``INTERLATENT_DRTC_URL``. The old default pointed at
# one specific hosted deployment, which is exactly the coupling ADR 0038
# removes.
DEFAULT_DRTC_URL = None


def connect_drtc(
    *,
    api_key: Optional[str] = None,
    environment: str,
    policy_uri: str = "",
    policy_backend: str = "lerobot",
    server_address: Optional[str] = None,
    chunk_size: int = 50,
    action_dim: int = 6,
    min_execution_horizon: int = 12,
    cooldown_steps: int = 16,
    fps: float = 30.0,
    payload_codec: str = "npz",
    task: str = "",
    stats_interval_s: float = 5.0,
    synchronous: bool = False,
    metadata: Optional[dict[str, str]] = None,
    # Server-side episode recording (DRTC node path). When ``record=True``
    # the GPU container persists every Infer observation + the returned
    # action chunk's leading row, builds a LeRobot dataset on close, and
    # writes it to the destination that box was configured with — its
    # ``--output-dir`` or the S3 target on the session. The Pi never stages
    # bytes locally.
    record: bool = False,
    episode_id: Optional[str] = None,
    env_id: Optional[str] = None,
    task_id: Optional[str] = None,
) -> DRTCClient:
    """Open a DRTC session against your GPU box.

    Returns an already-opened ``DRTCClient`` ready for ``step()``.

    Auth:
        Sends ``api_key`` (or the ``INTERLATENT_API_KEY`` env var) as
        a Bearer token. The box validates it against the coordinator it
        registered with on first contact and caches the result
        per-container.

    Args mirror ``DRTCConfig``; ``fps`` is converted to
    ``control_period_s`` for convenience.

    Recording:
        Pass ``record=True`` together with ``episode_id`` (typically
        the coordinator's session id) to have the GPU container record
        the episode and write it to that box's configured destination.
        ``task`` and ``fps`` are reused as the LeRobot dataset's task
        string and frame rate. The Pi remains stateless for storage —
        nothing is staged locally.
    """
    key = api_key or os.environ.get("INTERLATENT_API_KEY", "")
    url = server_address or os.environ.get("INTERLATENT_DRTC_URL") or ""
    if not url:
        raise ValueError(
            "No DRTC endpoint. A GPU address is provided per-session by your "
            "coordinator (that is what `interlatent-node` does); to dial one "
            "directly, pass server_address=... or set INTERLATENT_DRTC_URL."
        )

    # Natural-language task (e.g. SmolVLA instruction) flows via
    # OpenSession metadata. The server pulls `task` and passes it as
    # default_task to the policy backend; LeRobotBackend then injects
    # it into every batch automatically.
    md = dict(metadata or {})
    if task:
        md.setdefault("task", task)

    # Recording metadata — only added when the user opts in, so older
    # servers that ignore unknown metadata keys still see a clean
    # OpenSession.
    if record:
        md.setdefault("record", "1")
        if episode_id:
            md.setdefault("episode_id", episode_id)
        if environment:
            md.setdefault("env_slug", environment)
        if env_id:
            md.setdefault("env_id", env_id)
        # Durable Task link — the recorder echoes it back when registering
        # the episode so the catalog row attributes without a lookup.
        if task_id:
            md.setdefault("task_id", task_id)
        # ``fps`` is a float here but the server expects an integer
        # string; round to the nearest control rate.
        md.setdefault("fps", str(int(round(fps)) if fps > 0 else 30))

    # Sequential (request-response) chunking. The behavior is entirely client-side
    # (see DRTCConfig.synchronous / controller.step); the server needs no change
    # because RTC in-painting and crossfade self-disable when chunks stop
    # overlapping. We still flag it in OpenSession metadata so the GPU-side log
    # records which cadence a session ran.
    if synchronous:
        md.setdefault("synchronous", "1")

    # The DRTC wire protocol still names this field ``model_id`` (out of
    # scope for the SDK model_id retirement — renaming it is a wire-protocol
    # change). We pass the env slug through it so the server can identify
    # which env this DRTC session belongs to.
    cfg = DRTCConfig(
        server_address=url,
        api_key=key,
        model_id=environment,
        policy_uri=policy_uri,
        policy_backend=policy_backend,
        chunk_size=chunk_size,
        action_dim=action_dim,
        min_execution_horizon=min_execution_horizon,
        cooldown_steps=cooldown_steps,
        control_period_s=1.0 / fps if fps > 0 else 1.0 / 30,
        payload_codec=payload_codec,
        # `use_grpc_web` is inferred from URL scheme — plain gRPC is
        # host:port; an endpoint behind an HTTP proxy is an http(s) URL.
        use_grpc_web=url.startswith(("http://", "https://")),
        stats_interval_s=stats_interval_s,
        synchronous=synchronous,
        metadata=md,
    )
    client = DRTCClient(cfg)
    client.open()
    return client


__all__ = ["connect_drtc", "DEFAULT_DRTC_URL"]
