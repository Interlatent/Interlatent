"""Interlatent SDK inference — client side.

Robot-side half of Distributed Real-Time Chunking (DRTC). Streams
observations to a cloud-managed GPU pod over the gRPC contract in
`proto/messages.proto` and merges the action chunks it returns.

References:
    - https://jackvial.com/posts/distributed-real-time-chunking.html
    - https://github.com/jackvial/drtc
    - Real-Time Execution of Action Chunking Flow Policies (RTC)

================================================================
What lives here
================================================================

    client/                 DRTC client
        controller.py           Main control loop + action scheduler
        sender.py               Observation sender thread
        receiver.py             Action receiver thread
        cooldown.py             Inference-request cooldown counter
        latency.py              Jacobson-Karels latency estimator
        merge.py                LWW register / semilattice join for
                                the action schedule

    protocol/               Mirrored from interlatent-engine
        messages.proto          Copy of source-of-truth proto
        messages_pb2.py         Generated stubs (committed)
        messages_pb2_grpc.py    Generated stubs (committed)
        messages.py             Pydantic wrappers / typed helpers
        timestamps.py           Monotonic control-timestamp helpers

    integration/            Glue to existing SDK surfaces
        sdk_adapter.py          Wires DRTC client into the public
                                `Interlatent.watch()` / `tick()` path
        rollout.py              Backend used by the existing
                                `interlatent-rollout` LeRobot entry
                                point in
                                `lerobot/async_inference/async_rollout.py`
                                (that module stays where it is and
                                becomes a thin wrapper over this)

================================================================
Why the client lives in the SDK
================================================================

The DRTC client runs on the robot and is a public surface that SDK
users consume. The policy forward pass runs on a cloud-managed GPU
pod; the two halves share only the protobuf wire format, regenerated
into `protocol/` from `proto/messages.proto` (via `proto/gen_proto.sh`).

================================================================
Activation capture (v2)
================================================================

Today's SDK hook path attaches to the policy locally. Once a run uses
DRTC, the policy forward pass happens on the GPU pod, so activations
will be captured pod-side. v1 skips capture; v2 will ship activations
back alongside action chunks (or async to S3 keyed by session).
"""
