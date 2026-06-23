# Interlatent DRTC client — Context

The robot-side stack for running robot policies on cloud GPUs: the
[Interlatent dashboard](https://interlatent.com) assigns sessions and provisions a
managed **GPU pod** per session, a **Node** drives the robot and connects to the
pod, and the pod loads policies and serves action chunks over the DRTC gRPC
protocol. The thin `interlatent` CLI lists pods/nodes and starts/stops sessions
against the dashboard.

## Language

**Policy**:
A trained model (an HF repo id or local checkpoint) that maps observations to
action chunks. Identified by a **policy URI**.
_Avoid_: model (overloaded — used for the recorded-dataset "Model layer" too).

**Node**:
The long-running `interlatent-node` daemon on the robot. It pairs to the account
with an API key, polls the dashboard, and converges to whatever inference session
the dashboard assigns it. The DRTC GPU endpoint is provided per-session by the
dashboard. _Avoid_: calling this a "coordinator" — there is no self-hosted control
plane; the dashboard is the control plane.

**Session**:
A live binding of a node (or a hand-written `connect_drtc()` loop) to a policy URI
running on a managed **GPU pod**. Created from the dashboard or via
`interlatent session start --node … --pod … --policy …`; stopping it closes the
DRTC link and triggers any recorded dataset to be built/published.

**GPU pod**:
A managed cloud GPU that loads a policy and serves action chunks over the DRTC
gRPC protocol. Pods are provisioned and warm-pooled by the dashboard, not
self-hosted. List the pods available to your account with `interlatent pods ls`.

## Relationships

- A **Node** is paired once and may be assigned many **Sessions** over its life.
- A **Session** pins one **policy URI** on one **GPU pod** for its lifetime.
- The **dashboard** assigns sessions and provisions the GPU pod, returning the
  DRTC endpoint to the node/client per-session.

## Flagged ambiguities

- "warmup" historically meant both *pre-warm* (loading a policy before a session,
  a cloud-side latency optimization) and *correct compilation*. On the robot side
  neither is a concern — the client simply waits for the first action chunk; pod
  warm-pooling is handled by the dashboard.
