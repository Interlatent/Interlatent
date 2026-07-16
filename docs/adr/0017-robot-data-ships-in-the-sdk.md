# 0017 — Robot embodiment data ships in the SDK, per-kind, via public PyPI

- Status: Accepted
- Date: 2026-07-16
- Supersedes part of: [0012](0012-teleop-receiver-stub-open-core-boundary.md) (the open-core boundary — see below)
- Extends: [0011](0011-vendor-robot-subpackage-via-robot-kind.md) (vendor subpackage selected by robot kind)

## Context

A robot kind's teleop data — the URDF, its collision meshes, the hand-authored
`ik_config.json`, and the generated `kinematic_spec.json` — lived only in the
platform's admin-curated S3 library at `urdf/{robot_kind}/{version}/`. That is
fine for the hosted pod, which resolves a bundle at session time, but it means:

- **Onboarding a robot is a private, high-friction ritual.** A contributor needs
  S3 write credentials, an arm64 MuJoCo environment, a hand-authored `ik_config`,
  and a manual `aws s3 cp` — none of it visible or reviewable in a PR. There is no
  "add a URDF and open a PR" path.
- **An operator with the open-source SDK cannot teleop their own robot.** The data
  they need is behind our S3 and an `ilat_` key.

[0012](0012-teleop-receiver-stub-open-core-boundary.md) deliberately drew the
open-core line so that "the SDK no longer ships kinematics, IK, or retargeting."
That kept the differentiating *solver* off the client — a boundary we still want.
But it also, as a side effect, kept the *embodiment description* (a URDF is a
published hardware fact, not our IP) off the client, which is the thing blocking
both problems above.

A naive fix — vendor the files into the `interlatent` package — does not work:
the SDK and the internal `interlatent-engine` are **both** the top-level import
package `interlatent` and collide on install (last install wins), so data buried
in `interlatent` could never reach a pod running the engine, and would bloat the
117 KB base wheel with ~15 MB of STL per robot for users who drive one arm.

## Decision

Robot data ships as a **separate distribution per kind**, `interlatent-robot-<kind>`,
each contributing `interlatent_robots/<kind>/` to a shared **PEP 420 namespace**
package `interlatent_robots` (no top-level `__init__.py`, so any number coexist).
Source of truth is the **public SDK repo** under `robots/<kind>/`; wheels publish
to **public PyPI**; `pip install interlatent[<kind>]` pulls the matching wheel via
the kind's existing extra. The SDK's `interlatent.robots` module resolves an
installed kind by `robot_kind` (`load`, `ensure_bundle`, `ensure_meshes`).

1. **Distinct top-level name (`interlatent_robots`), not `interlatent`.** This is
   the crux: it collides with neither the SDK nor the engine, so a pod can install
   the engine *and* a robot wheel in one environment. Verified by a coexistence
   install test.

2. **Meshes are not vendored.** `meshes.lock` pins each STL by `sha256` against a
   stable upstream (`TheRobotStudio/SO-ARM100` for SO-101); `ensure_meshes` fetches
   and verifies on demand into a cache. Keeps ~15 MB of geometry out of every wheel
   and out of git, and makes the previously-ad-hoc `curl` reproducible. The node's
   forward-the-spec-to-browser path needs no meshes and never fetches; the engine/pod
   (MuJoCo) path calls `ensure_bundle`.

3. **What crosses the open-core line, and what does not.** Robot data (URDF,
   `ik_config`, `kinematic_spec`) now ships publicly. The **solver** (`so101_dls.py`
   /`decoupled_ik.py`, the retarget stage) stays in the internal engine — 0012's
   real intent. So this supersedes 0012 only on "the SDK ships no kinematic *data*",
   not on "the SDK ships no *solver*." The tuning in `ik_config.json` becomes public;
   that was an explicit, accepted trade for standalone-teleop and reviewable
   onboarding.

4. **`kinematic_spec.json` is generated but committed.** The MuJoCo exporter lives
   in the internal engine; public-repo CI cannot run it. For now a maintainer
   regenerates the spec and commits it alongside a contributor's URDF + `ik_config`
   — honest about the fact that a new robot needs maintainer review anyway
   (`webxr_to_base_R`/`tool0_offset` encode rig geometry no file carries).

5. **The kind is the key, everywhere.** `robots/<kind>/` dir = `robot_kind` the node
   reports = wheel kind = S3 bundle key. The canonical dual-SO-101 kind is `nori`;
   `so101_bimanual` (a stale S3 mis-upload) is retired.

## Consequences

- Adding a robot approaches "add `robots/<kind>/` and open a PR"; the wheel build
  (`packaging/build_robot_wheel.py`) is pure-data and runs anywhere.
- An operator can `pip install interlatent[<kind>]` and get a real, verifiable
  embodiment — the standalone-teleop story 0012 had foreclosed.
- The S3 bundle path for the hosted pod is **unchanged**; this adds a second
  distribution channel, it does not retire the first. Two sources of the same data
  now exist (S3 + wheel); keeping them from drifting is a follow-up (single build
  that publishes both).
- Public tuning: `ik_config` values (and the `damping`/`w_rot` embedded in the
  spec) are now visible. Accepted.
- Open question deferred: who runs the MuJoCo exporter on an untrusted public PR
  (private CI on approved PRs vs. publishing the exporter). Interim: maintainer
  commits the spec (point 4).
