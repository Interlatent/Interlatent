"""Robot integration adapters for the Interlatent node.

Each submodule adapts a specific robot family to the duck-typed LeRobot
``Robot`` interface that the built-in ``lerobot_control_loop`` drives:

- ``interlatent.adapters.lerobot`` — LeRobot-native rollout/record helpers.
- ``interlatent.adapters.axol`` — Almond Axol dual-arm robot (``interlatent[axol]``).

These are optional and dependency-heavy, so they are **not** imported here;
import the specific submodule you need (the node does so lazily). "Adapter"
here means a *robot* adapter — distinct from a server-side *policy backend*
(``policy_backend``), a collection ``--loop`` adapter, or a LoRA adapter.
"""
