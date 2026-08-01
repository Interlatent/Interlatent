# 18. Dimos adapter binds to the ControlCoordinator as an external bus peer

Date: 2026-07-17

## Status

Accepted

## Context

We want to run cloud policies (DRTC sessions) and collect data on robots
already managed by the dimos stack (Unitree Go2/G1, xArm manipulators). Dimos
is a reactive module system: everything is a `Module` with typed streams over a
pluggable LCM/Zenoh transport, wired into "blueprints", with a request/reply
RPC layer riding the same bus. Facts that shaped this decision (verified
against dimos `0.0.14b1`):

- Dimos's `ControlCoordinator` normalizes embodiments behind one seam:
  `coordinator_joint_state` out, `joint_command` in (consumed by pre-declared
  tasks at a 100 Hz tick), gripper via RPC. Even the Go2 base is modeled as
  virtual velocity joints (`go2/vx,vy,wz`) through this seam.
- **Every dimos stream is on the wire by default** (unassigned streams get a
  `/<name>` topic), and out-of-blueprint bus peers are an in-tree dimos
  pattern (`RPCClient(None, ControlCoordinator)`, `make_transport`).
- **Robot identity is not on the bus.** `robot_model` is authoring-time
  config; the blueprint name lives in a local file. Runtime discovery yields
  modules, hardware, tasks, and joint names — never "this is an xarm7".
- **Dimos enforces no limits on streamed joint commands** — no joint ranges,
  no velocity clamps; arbitration is per-joint task priority with
  first-writer-wins ties. Its capability mutual-exclusion (CAP_MOVEMENT)
  lives in the MCP agent layer, which a bus peer never touches.
- **A stock dimos coordinator blueprint configures no servo task** and the
  coordinator does not even subscribe `joint_command` without one — streaming
  to it is silently ignored. Tasks cannot be created over RPC.
- **Servo tasks accept only FULL vectors**: a `joint_command` missing even one
  claimed joint is rejected without updating (`set_target_by_name` returns
  False), and dimos's per-tick hardware write re-sends `_last_commanded` for
  gripper joints — together forcing the gripper onto the stream and every
  published command to carry the complete claimed joint set.
- Third-party pip packages can register dimos blueprints via the
  `"dimos.blueprints"` entry-point group (`dimos run <dist>.<name>`).
- Dimos has **no gRPC**; its RPC is request/reply over the same bus. The
  SDK's DRTC gRPC link is unrelated and stays robot↔pod only.
- Dimos also has its own recording pipeline (memory2 `Recorder` → SQLite →
  dataprep → LeRobot v3), parallel to the SDK's local-first staging.

## Decision

1. **External bus peer, never embedded.** `interlatent.adapters.dimos`
   (`interlatent[dimos]`, python 3.11–3.12) binds to a *running* dimos stack
   over its own transports: subscribe `coordinator_joint_state` + camera
   topics (latest-wins caches, arrival-time staleness), publish
   `joint_command` (latest-wins == `send_action`'s contract) carrying **all
   joints including the gripper**. The gripper deliberately does NOT use the
   coordinator's `set_gripper_position` RPC on the command path: dimos's
   per-tick hardware write re-sends its last-commanded gripper value whenever
   any task streams to the hardware, so out-of-band gripper RPCs are stomped
   at tick rate (verified empirically); the RPC is read-only territory here.
   We do not import dimos modules into the node's process or manage its
   lifecycle.
2. **Declare-then-verify identity, fail-closed.** The operator declares the
   embodiment (`--robot-arg kind=xarm7`; per-embodiment kinds, profile
   `dimos_xarm7`). `connect()` verifies against live evidence — coordinator
   ping, `list_modules`, `list_joints`, task claims, first `JointState`
   names+order, gripper hardware — accumulating every mismatch into one raise
   (the nori pattern). A servo task must claim **exactly** the declared
   joints — arm AND gripper (an unclaimed gripper is stomped, see decision 1)
   — with a non-zero timeout, and **no other task may claim them** (strict
   exclusivity, v1): preemption would fight the policy invisibly and corrupt
   recorded `control_source` provenance. Task introspection falls back to a
   no-op `joint_command` probe watching `get_active_tasks` when live task
   objects don't survive the pickled RPC (empirically they don't — servo
   tasks hold thread locks); in probe mode exclusivity and exact claims are
   unverifiable, so single-task stacks are the only ones it can pass, and
   anything else fails with guidance. (A dimos-side `get_task_info` RPC would
   remove this fallback — worth contributing upstream.)
3. **The adapter clamp is the only clamp — and we ship the safety envelope.**
   Dimos has no curated limits to reuse (its xarm limits are placeholders;
   model-config limits are unset on the streaming path), so the hand-written
   `RobotProfile` plus the adapter's `max_step_rad` delta clamp are the entire
   safety model. "Next to the motors" here means *the last hand that touches
   the command before the bus* — same host, different process.
4. **Session blueprints: ship AND document.** The SDK registers reference
   blueprints via the `dimos.blueprints` entry point (`dimos run
   interlatent.xarm7`: coordinator + servo task with explicit
   priority/timeout + camera + mock/sim fallbacks), and documents the
   blueprint contract for operator-authored stacks; connect-time verification
   enforces the same contract either way. Entry points are metadata-only for
   base installs; the target module import-guards the missing extra.
5. **Collection role partition, joined by episode markers.** The interlatent
   node records the episode of record (unchanged pipeline). The adapter
   publishes a small custom `EpisodeMarker` (pickled) at episode start/stop —
   deliberately NOT dimos's `EpisodeStatus`, which is the control signal of
   dimos's own recording state machine. A dimos-side memory2 recorder may
   record low-level streams locally and segment them by marker + same-host
   timestamps. No recorder-to-recorder forwarding; no stream lands in two
   episode datasets.

## Consequences

- Zero dimos-repo changes required; everything rides public extension points.
- xArm7 first; `go2_base` (velocity virtual joints via `JointSpec`
  `control_mode="velocity"`) is a follow-up; G1 wholebody (per-joint
  `kp/kd/tau`) is out of scope for this contract.
- Running the session inside agentic/teleop dimos blueprints is unsupported
  in v1 (strict exclusivity); dimos-side takeover aligns with the future
  teleop-intervention work, not this ADR.
- The `[dimos]` extra is the SDK's heaviest (open3d, rerun, opencv, pinocchio
  transitively) and caps at python 3.12; documented on the extra.
- No vendored wire fixtures: the "protocol" is dimos's Python API, pinned by
  the extra's version bound; a snapshot would drift instantly. Conformance is
  the integration suite against `dimos run interlatent.xarm7` (mock/sim).
