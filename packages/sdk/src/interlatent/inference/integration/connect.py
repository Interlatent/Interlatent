"""One-call connection to Interlatent's hosted DRTC server.

Users do not deploy anything. They just call ``connect_drtc(...)``
with their Interlatent API key and the policy they want to run.

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

The server address defaults to Interlatent's hosted Modal app. Pass
``server_address=`` to override (useful for staging deploys or local
gRPC servers used during SDK development).
"""

from __future__ import annotations

import os
from typing import Optional

from ..client import DRTCClient, DRTCConfig

# Production URL for Interlatent's hosted DRTC server. Replace with
# the URL printed by `modal deploy` once the app is live.
# Override at runtime via env var INTERLATENT_DRTC_URL or via the
# ``server_address`` arg to connect_drtc().
DEFAULT_DRTC_URL = "https://interlatent--interlatent-drtc-inference-web.modal.run"


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
    # uploads it through the same inbox protocol the legacy SDK upload
    # path uses. The Pi never stages bytes locally.
    record: bool = False,
    episode_id: Optional[str] = None,
    env_id: Optional[str] = None,
) -> DRTCClient:
    """Open a DRTC session against Interlatent's hosted server.

    Returns an already-opened ``DRTCClient`` ready for ``step()``.

    Auth:
        Sends ``api_key`` (or the ``INTERLATENT_API_KEY`` env var) as
        a Bearer token. The server validates against the Interlatent
        backend on first contact and caches the result per-container.

    Args mirror ``DRTCConfig``; ``fps`` is converted to
    ``control_period_s`` for convenience.

    Recording:
        Pass ``record=True`` together with ``episode_id`` (typically
        the dashboard's ``InferenceSession.id``) to have the GPU
        container record + upload the episode. ``task`` and ``fps``
        are reused as the LeRobot dataset's task string and frame
        rate. The Pi remains stateless for storage — nothing is
        staged locally.
    """
    key = api_key or os.environ.get("INTERLATENT_API_KEY", "")
    url = server_address or os.environ.get("INTERLATENT_DRTC_URL") or DEFAULT_DRTC_URL
    if not key and url == DEFAULT_DRTC_URL:
        # The hosted endpoint requires an account.
        raise ValueError(
            "An Interlatent API key is required for the hosted endpoint. "
            "Pass api_key=... or set INTERLATENT_API_KEY in your "
            "environment — or pass server_address= to dial an explicit "
            "compute endpoint."
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
    # scope for the SDK model_id retirement — that contract is owned by
    # Modal). We pass the env slug through it so the server can identify
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
        # `use_grpc_web` is inferred from URL scheme — local plain-gRPC
        # uses host:port; the hosted Modal endpoint is an https URL.
        use_grpc_web=url.startswith(("http://", "https://")),
        stats_interval_s=stats_interval_s,
        synchronous=synchronous,
        metadata=md,
    )
    client = DRTCClient(cfg)
    client.open()
    return client


__all__ = ["connect_drtc", "DEFAULT_DRTC_URL"]
