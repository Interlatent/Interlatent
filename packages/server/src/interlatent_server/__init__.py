"""Interlatent inference engine — server side.

Hosted policy inference for action-chunking policies. Implements the
server half of Distributed Real-Time Chunking (DRTC). The client half
lives in the SDK package
(`packages/sdk/src/interlatent/inference/`) because it
runs on the robot.

References:
    - https://jackvial.com/posts/distributed-real-time-chunking.html
    - https://github.com/jackvial/drtc
    - Real-Time Execution of Action Chunking Flow Policies (RTC)

================================================================
Deployment shape
================================================================

Target host:
    A persistent GPU box on a private network. The reference
    deployment is a Prime Intellect VM joined to Tailscale; any Linux
    host with a CUDA GPU works the same way. The server is a long-
    running asyncio process — no serverless cold starts, no per-call
    container churn — which is what lets the SmolVLA-class policies'
    multi-minute torch.compile happen exactly once per process.

Transport: native gRPC over HTTP/2
    Bidirectional streaming fits the observation/action duplex. The
    long OpenSession (policy load + torch.compile warmup) is a single
    long-lived RPC — no HTTP proxy, no redirect storms.

Session state: in-process
    The server runs in a single process, so per-session state
    (PolicyRuntime + ChunkBuffer + SessionRecorder) lives in plain
    Python dicts. ChunkBuffer is exposed as a Protocol in case a
    multi-process deployment ever needs an external KV store, but
    today there is only one implementation
    (:class:`InMemoryChunkBuffer`).

Episode recording: on the server
    When the OpenSession metadata carries ``record=1`` the server
    persists per-step observations + the policy's returned action
    chunk + raw camera JPEG bytes into a per-session working dir.
    On CloseSession (or after a 60s idle timeout) the server builds
    a LeRobot v3.0 dataset and uploads it to the Interlatent backend
    via the same inbox protocol the SDK upload path uses. The Pi
    never stages bytes locally.

================================================================
Module layout (this package — server only)
================================================================

    server/                 DRTC policy server
        app.py                  Module entrypoint that serves on
                                localhost (used by tests + smoke runs);
                                the production launcher is
                                ``interlatent.cloud.serve_gpu``.
        policy_runtime.py       Policy load/warm/forward;
                                RTC in-painting
        chunk_buffer.py         ChunkBuffer interface + in-process impl
        schedule.py             Action-schedule reconstruction from
                                client-provided spans
        transport.py            grpc.aio servicer
        recorder.py             SessionRecorder + RecorderStepSource
                                (server-side episode capture/upload)
        auth.py                 ``x-api-key`` gRPC validation against
                                the Interlatent backend
        lerobot_backend.py      LeRobot policy adapter (SmolVLA, ACT,
                                etc.) — also used for warmup

    protocol/               Wire format — source of truth
        messages.proto          gRPC service + message definitions
        messages_pb2.py         Generated stubs (committed)
        messages_pb2_grpc.py    Generated stubs (committed)
        timestamps.py           Monotonic control-timestamp helpers

The SDK package mirrors `protocol/` (generated stubs only) so it can
import the same wire types without depending on interlatent-engine
(the two packages collide on the `interlatent` module name and the
engine is internal). A small `proto/gen_proto.sh` regenerates stubs
into both locations from this `.proto` source.

================================================================
What this is solving
================================================================
    1. Inference latency > control rate -> gaps between action chunks.
    2. Distributed client/server means unreliable transport (drops,
       reorder, duplicates) and variable RTT.
    3. RTC handles in-painting locally; DRTC extends it to the
       network-separated case with cooldown + LWW merge so the client
       degrades gracefully even when the server hits a slow tick.
"""
