# `interlatent_robots/` — teleop embodiment data, one subpackage per robot kind

Each subdirectory is the robot data an operator's node and the browser IK need,
keyed by the `robot_kind` string the node reports (`--robot <kind>`). It ships in
the `interlatent` wheel (~18 KB/kind) and is read back through `interlatent.robots`
— never by path from here.

`interlatent_robots` is a **PEP 420 namespace**: it has no `__init__.py`, and must
not gain one. The SDK does not own the name, so a kind can be split into its own
distribution later without changing an import.

Adding a robot is meant to be "drop a URDF in and open a PR":

```
packages/sdk/src/interlatent_robots/<kind>/
    __init__.py           # data-only marker; copy an existing kind's.
    <robot>.urdf          # KINEMATICS-ONLY: links + joints + inertials, no
                          #   <visual>/<collision> geometry. Joint <limit>/<origin>/
                          #   <axis> are authoritative; there are no mesh refs.
    ik_config.json        # hand-authored IK/tuning — the retarget-stage config.
    kinematic_spec.json    # GENERATED from urdf+ik_config by the MuJoCo exporter.
    meshes.lock           # OPTIONAL: {name,url,sha256} per mesh. Omit it (default)
                          #   — IK needs no geometry. Add only if a kind ships meshes.
```

Also add the kind to `[tool.setuptools.package-data]` in `packages/sdk/pyproject.toml`,
or its data files will not ship. `tests/test_robots.py` fails on a kind that is
incomplete, mis-named, or missing from the tree.

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

## Meshes are not used (IK needs no geometry)

Inverse kinematics is a function of the joint tree alone — origins, axes, limits,
tool0. The collision/visual STLs carry none of that, so the shipped URDF is
**kinematics-only** (visual/collision stripped) and MuJoCo compiles it with zero mesh
assets. No `meshes.lock`, nothing fetched, no `~15 MB/rig` anywhere. The browser's
in-solver proves the point: it runs full IK from `kinematic_spec.json` with no URDF,
no meshes, no MuJoCo.

Geometry is off the critical path, not forbidden: `meshes.lock` +
`interlatent.robots.ensure_meshes()` still exist for a kind that genuinely needs STLs
later (a 3D preview, sim, collision-aware retargeting). Until such a feature lands,
leave the lock out. If you want a full (mesh-referencing) URDF for a viewer, it is
reconstructable — the meshes are upstream (e.g. `TheRobotStudio/SO-ARM100` for SO-101)
and the calibration lives in the joints kept here.

## Verifying a URDF (needs MuJoCo)

`packaging/verify_urdf.py` is a maintainer/CI check (not shipped in the wheel — the
SDK has no MuJoCo). Run it after editing a URDF/`ik_config`, regenerating a spec, or
stripping meshes:

```bash
pip install mujoco numpy
python packaging/verify_urdf.py packages/sdk/src/interlatent_robots/nori
python packaging/verify_urdf.py --all   # every kind
```

It (1) compiles the URDF exactly as the engine does (`MjSpec.from_file` + `compile`)
— this is the "builds with no STLs" check — (2) confirms `ik_config.json` resolves
(`ee_body` + every joint), and (3) runs **FK parity** between the compiled model and
`kinematic_spec.json` over random configs, so a spec that drifted from the URDF fails
loudly. Exit status is nonzero on any failure.

## Naming

The directory name **is** the `robot_kind` and **is** the kind's subpackage name — it must equal
the string the live node reports. `nori` is the dual-SO-101 rig (historically also
mis-labelled `so101_bimanual` in an early S3 upload; that name is retired).
