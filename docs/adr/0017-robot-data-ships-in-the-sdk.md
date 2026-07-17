# 0017 — Robot embodiment data ships in the SDK wheel, per-kind

- Status: Accepted
- Date: 2026-07-16
- Amended: 2026-07-16 — robot data ships **in the `interlatent` wheel**, not as a
  separate `interlatent-robot-<kind>` distribution per kind. See "Amendment" below.
  The rest of the decision (public, per-kind, in-repo, mesh-free) is unchanged.
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

Robot data ships **in the `interlatent` wheel** as `interlatent_robots/<kind>/`, one
data-only subpackage per kind under a **PEP 420 namespace** package
`interlatent_robots` (no top-level `__init__.py`). Source of truth is the **public
SDK repo** under `packages/sdk/src/interlatent_robots/<kind>/`. Every kind ships with
every install — `pip install interlatent` is enough to resolve one; the per-kind
extras carry that robot's **driver** deps, not its data. The SDK's
`interlatent.robots` module resolves an installed kind by `robot_kind` (`load`,
`ensure_bundle`, `ensure_meshes`).

1. **Distinct top-level name (`interlatent_robots`), not `interlatent`.** It collides
   with neither the SDK nor the engine, and — as an unowned PEP 420 namespace —
   resolution walks the namespace without caring which distribution provides a kind.
   So a kind can be split into its own distribution later without changing an import
   or a line of `robots.py`. That is the option this decision keeps open; see the
   Amendment for why it is not exercised now.

2. **Meshes are off the critical path entirely.** IK is a function of the joint
   tree (origins, axes, limits, tool0), not geometry, so the shipped URDF is
   **kinematics-only** — `<visual>`/`<collision>` stripped — and MuJoCo compiles it
   with zero mesh assets. No `meshes.lock`, nothing fetched, ~0 MB of STL in the wheel
   or git. The browser proves the principle: it runs full IK from `kinematic_spec.json`
   with no URDF, no meshes, no MuJoCo. The mesh machinery (`meshes.lock` +
   `ensure_meshes`, sha256-pinned) is retained but **unused by default**, for a kind
   that genuinely needs STLs later (a 3D preview, sim, collision-aware retargeting);
   a full mesh-referencing URDF stays reconstructable (meshes upstream, calibration in
   the kept joints). This supersedes the earlier "meshes are vendored via a lock but
   not in the wheel" position — they are now simply not part of the IK story.

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

5. **The kind is the key, everywhere.** `interlatent_robots/<kind>/` dir =
   `robot_kind` the node reports = subpackage name. The canonical dual-SO-101 kind is
   `nori`; `so101_bimanual` (a stale S3 mis-upload) is retired.

## Amendment (2026-07-16): one distribution, not one per kind

As first written, this ADR shipped each kind as its own `interlatent-robot-<kind>`
distribution on PyPI. That rested on one argument — the SDK and the engine are both
the import package `interlatent` and collide, so **a pod running the engine could
never `pip install` data buried in `interlatent`**. The wheels were never published,
which made `pip install interlatent[yam]` unresolvable (the extra required a registry
package that did not exist). Revisiting it, the argument does not hold:

- **No pod ever consumes the data by pip.** The engine has zero references to
  `interlatent_robots`; it resolved bundles from the backend/S3 (`retarget/bundle_cache.py`),
  and that path is being retired. The new design sends `kinematic_spec` to the browser
  over QUIC from the node, and the browser runs IK — so the pod needs no robot data at
  all. The only pip consumer is the **node** (`node/teleop/quic_channel.py`), which
  installs the SDK and therefore has no collision to avoid.
- **The size argument died with the meshes.** The "~15 MB of STL per robot" that
  motivated separation was eliminated by point 2 above. The real payload is ~18 KB per
  kind, so gating it behind an extra saves nothing worth a distribution boundary.
- **Separation was not what bought the collision fix anyway.** The collision is on the
  *import* name `interlatent`; `interlatent_robots` is a different top-level name, and
  a single wheel can ship both.

What that design cost was real and recurring: a PyPI project and a manual pending-publisher
step per kind (in an ADR whose goal was reviewable, PR-only onboarding), a publish
workflow, and version skew between the SDK and its data.

Reversing is cheap to undo. Because point 1 keeps `interlatent_robots` an unowned PEP
420 namespace, splitting a kind back into its own distribution is a packaging change
with no import changes — worth doing **if** a pod ever needs to `pip install` robot
data, and not before.

## Consequences

- Adding a robot is "add `packages/sdk/src/interlatent_robots/<kind>/`, list it in
  `package-data`, open a PR" — no publish step, no PyPI project, no release lever.
  `tests/test_robots.py` fails a kind that is incomplete or mis-named (the guard that
  previously lived in the wheel builder's `_validate`).
- An operator can `pip install interlatent` and get a real, verifiable embodiment —
  the standalone-teleop story 0012 had foreclosed.
- Robot data now versions **with the SDK**, so the two cannot drift apart, and a kind
  is either in your `interlatent` version or it is not. Shipping a robot means
  releasing the SDK — acceptable while robot data is ~18 KB/kind and changes rarely.
- Public tuning: `ik_config` values (and the `damping`/`w_rot` embedded in the
  spec) are now visible. Accepted.
- Open question deferred: who runs the MuJoCo exporter on an untrusted public PR
  (private CI on approved PRs vs. publishing the exporter). Interim: maintainer
  commits the spec (point 4).
