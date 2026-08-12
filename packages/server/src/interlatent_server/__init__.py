"""interlatent-server — the open-source Interlatent DRTC policy server.

Run it on a GPU machine of your own: `interlatent-serve` registers the
box with the coordinator you run (your key, your hardware) and serves
policies to your robot nodes over native gRPC.

Layout:
  - :mod:`interlatent_server.cli`         — the ``interlatent-serve`` entry point
  - :mod:`interlatent_server.serve_gpu`   — the server launcher (warmup, CPU
    isolation, gRPC bind, status reporting)
  - :mod:`interlatent_server.server`      — servicer, policy runtime, backends,
    recorder, auth
  - :mod:`interlatent_server.protocol`    — the DRTC wire protocol
    (``interlatent.inference.v1``) — identical bytes to the SDK client's copy
  - :mod:`interlatent_server.credentials` — box identity (hosted admin key vs
    self-hosted owner key)

The wire protocol package name is ``interlatent.inference.v1`` on purpose:
this server is drop-in compatible with every existing `interlatent` SDK
client.
"""

__all__ = ["credentials", "box_status", "serve_gpu"]
