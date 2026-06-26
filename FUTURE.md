# Future directions

Forward-looking work that isn't scheduled yet. Each item is a direction, not a spec.

## Adapters should consume robot URDFs directly

Today a robot's kinematic facts — joint names, order, limits, velocity caps, rest
pose — are hand-transcribed into static `RobotProfile` literals in
[`robot_profile.py`](packages/sdk/src/interlatent/node/teleop/robot_profile.py). That
is a transcription step that drifts from the hardware: the YAM profile shipped with a
conservative placeholder envelope, and the real limits only landed once we pulled the
joint `<limit>` values out of the i2rt YAM URDF by hand. The URDF is the manufacturer's
source of truth; the adapter should read it rather than restate it.

**Direction:** let an adapter derive its profile (and eventually FK/collision data)
from the robot's URDF, so limits/order/rest-pose come from one authoritative file.

**What we know already:**
- I2RT ships a real YAM URDF at `i2rt/robot_models/arm/yam/yam.urdf` (joints listed
  reversed vs i2rt command order; `joint1..joint6` map to our `joint_0..joint_5`).
  The arm `joint_limits` in our profile are now transcribed from it; `max_velocity`
  and the gripper range are still hand-chosen (the gripper is combined in separately
  from the `LINEAR_4310` model, so it is not in `yam.urdf`).
- Axol has no URDF in the picture yet — needs investigation before this generalizes.

**Open design questions (resolve before building):**
- Parse the URDF at build time into a static profile (keeps the current convention,
  no runtime parse-dep) vs. at `connect()` (always matches the installed driver, adds
  a `yourdfpy`-style dependency on the import path)?
- Vendor the URDF + meshes into the adapter, or read it from the installed vendor
  package (e.g. i2rt's `ARM_YAM_XML_PATH`)? Meshes/asset paths complicate vendoring.
- How does URDF joint order reconcile with `action_features` ordering (the policy
  binds to order, not names)? The reversed YAM ordering shows this needs an explicit
  mapping, not a blind import.
- Keep the static literal as a hand-verified fallback / safety-tightened override, or
  treat the URDF as canonical? URDF limits are mechanical max — we currently inset
  velocity below them on purpose, which a naive import would lose.
