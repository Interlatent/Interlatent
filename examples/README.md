# Examples

Ordered by how much hardware you need. (Numbering has gaps; these four are all of them.)

| # | Example | Needs |
|---|---|---|
| 03 | [`03_run_on_so101.py`](03_run_on_so101.py) — an SO-101-shaped DRTC loop against a cloud GPU pod, on synthesized observations | API key + `lerobot` (no arm) |
| 04 | [`04_manual_action.py`](04_manual_action.py) — manual `action()`: center the arm, sweep the base, work the gripper | an arm + `--port` |
| 06 | [`06_connect_hosted.py`](06_connect_hosted.py) — the minimal cloud connect | an API key |
| 07 | [`07_named_behaviors.py`](07_named_behaviors.py) — named behaviors (`home`, `hello`, `move`) offline | none (fake arm) or an arm |
