# `robots/` — teleop embodiment data, one directory per robot kind

Each subdirectory is the **source** for one `interlatent-robot-<kind>` wheel: the
robot data an operator's node and the browser IK need, keyed by the `robot_kind`
string the node reports (`--robot <kind>`). Publishing is `packaging/build_robot_wheel.py`.

Adding a robot is meant to be "drop a URDF in and open a PR":

```
robots/<kind>/
    <robot>.urdf          # the arm(s). Joint <limit> tags are authoritative.
    ik_config.json        # hand-authored IK/tuning — the retarget-stage config.
    kinematic_spec.json    # GENERATED from urdf+ik_config by the MuJoCo exporter.
    meshes.lock           # {name,url,sha256} per collision mesh — NOT the STLs.
```

## The two configs are different jobs — do not confuse them

- **`ik_config.json`** — hand-authored. The tuning surface (damping, reach limits,
  scales, `webxr_to_base_R`). Read pod-side by the retarget stage and by the backend
  to build the browser's `ik_hints`.
- **`kinematic_spec.json`** — **generated**, do not hand-edit. The compact serial-chain
  descriptor the in-browser solver walks. Produced from the URDF + `ik_config.json` by
  `interlatent.inference.server.retarget.kinematic_spec` (needs MuJoCo). A bundle
  missing it makes the arms do **nothing** — the browser can't build a solver.

Regenerate the spec after any `ik_config.json` or URDF change, or the browser solver
and the pod solver silently disagree.

## Meshes are not vendored

The STLs (~15 MB/rig) stay out of the wheel and out of git. `meshes.lock` pins each
by `sha256` against a stable upstream; `interlatent.robots.ensure_meshes(kind, dest)`
fetches + verifies them on demand (engine/pod side, where MuJoCo needs geometry). The
node's forward-to-browser path needs only `kinematic_spec.json`, so it never fetches.

## Naming

The directory name **is** the `robot_kind` and **is** the wheel's kind — it must equal
the string the live node reports. `nori` is the dual-SO-101 rig (historically also
mis-labelled `so101_bimanual` in an early S3 upload; that name is retired).
