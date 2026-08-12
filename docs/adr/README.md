# Architecture decision records

Records of decisions that shaped this SDK, kept because the reasoning is
harder to recover than the code. They are historical: an ADR is amended or
superseded, never rewritten.

## This repo's ADRs

| # | Decision |
|---|---|
| [0011](0011-vendor-robot-subpackage-via-robot-kind.md) | Vendor robot support as a subpackage selected by robot kind |
| [0012](0012-teleop-receiver-stub-open-core-boundary.md) | Teleop: a thin client receiver stub, engine off the node |
| [0013](0013-manual-action-interface-below-schedule.md) | Manual action interface: a final actuator below the schedule |
| [0014](0014-yam-via-i2rt-direct-joint-space.md) | YAM arms via the i2rt driver directly, joint-space only |
| [0015](0015-nori-liveness-tied-keepalive.md) | Nori keep-alive pump is liveness-tied, never unconditional |
| [0016](0016-teleop-estop-ingress-human-only-reset.md) | Operator e-stop rides the teleop frame; reset is human-only |
| [0017](0017-robot-data-ships-in-the-sdk.md) | Robot embodiment data ships in the SDK wheel, per-kind |
| [0018](0018-collection-verbs-removed-streaming-only.md) | Collection verbs removed: recording is streaming-only |
| [0018](0018-dimos-adapter-external-bus-peer.md) † | Dimos adapter binds to the ControlCoordinator as an external bus peer |
| [0019](0019-nvjpeg-ctypes-jpeg-backend.md) | CUDA JPEG encode via ctypes bindings (nvJPEG + GPUJPEG) |
| [0020](0020-aioquic-uni-stream-discard.md) | Per-frame QUIC uni streams must be manually discarded (aioquic leak) |
| [0021](0021-quic-teleop-child-process.md) | QUIC teleop runs in a dumb-pipe child process |
| [0022](0022-command-bus-owns-the-motion-path.md) | The command bus owns the motion path; adapters declare guards |
| [0023](0023-self-hosted-policy-server-returns.md) | The self-hosted policy server returns: `interlatent-server` |

† **Two ADRs share the number 0018.** Both were written against the same
sequence and the collision was not caught. Neither has been renumbered,
because the numbers are cited from code comments and tests across both
packages, and renumbering would silently invalidate those citations. Link to
these two by **filename**, not by number.

## Numbers that are not in this table

Some code comments and tests cite ADR numbers you will not find here — they
belong to Interlatent's closed platform monorepo, which keeps its own ADR
sequence. The two sequences overlap, so the same number can mean two
different things depending on which repo you are in:

| Cited number | In the platform repo it means | In *this* repo the same number is |
|---|---|---|
| 0022 | collection is streaming-first | the command bus owns the motion path |
| 0023 | node spool / lossless uplink | the self-hosted policy server |
| 0024 | recorder tick dedupe | — |
| 0034 | intervention vs. teleop labelling | — |
| 0035 | the policy server split out of the engine | — |
| 0037 | world-action models (DreamZero) | — |

Citations of 0024/0034/0035/0037 are marked **(platform repo)** at their first
mention in each file, since those numbers cannot refer to anything here.
Citations of 0022 and 0023 are genuinely ambiguous and are **not** marked —
read them in context: collection and spool topics mean the platform's, motion
path and self-hosting mean this repo's.

The public-surface consequences of the platform decisions are recorded here
where they affect SDK users — [0018 (collection verbs
removed)](0018-collection-verbs-removed-streaming-only.md) is the SDK-side
record of the platform's streaming-first decision, and is the one to cite from
this repo.
