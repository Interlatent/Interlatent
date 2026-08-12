# Examples

Ordered by how much hardware you need. (Numbering has gaps.)

| # | Example | Needs |
|---|---|---|
| 03 | [`03_run_on_so101.py`](03_run_on_so101.py) — an SO-101-shaped DRTC loop against your GPU box, on synthesized observations | a GPU box + `lerobot` (no arm) |
| 04 | [`04_manual_action.py`](04_manual_action.py) — manual `action()`: center the arm, sweep the base, work the gripper | an arm + `--port` |
| 07 | [`07_named_behaviors.py`](07_named_behaviors.py) — named behaviors (`home`, `hello`, `move`) offline | none (fake arm) or an arm |
