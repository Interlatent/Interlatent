# Examples

Ordered by how much hardware you need — start at 01 with nothing but a laptop.

| # | Example | Needs |
|---|---|---|
| 01 | [`01_loopback_no_hardware.py`](01_loopback_no_hardware.py) — full client↔server DRTC loop with the built-in test backend | nothing |
| 02 | [`02_serve_policy.md`](02_serve_policy.md) — serve SmolVLA / Pi0 / MolmoAct2 on your own GPU | CUDA GPU |
| 03 | [`03_run_on_so101.py`](03_run_on_so101.py) — drive an SO-101 against your server (synthesizes observations if you don't have the arm) | GPU server; arm optional |
| 04 | [`04_teleop_record.md`](04_teleop_record.md) — teleoperate from your laptop (keyboard or hand tracking), record demonstrations | SO-101 + Pi (mock driver available) |
| 05 | [`05_collect_dataset.py`](05_collect_dataset.py) — collect a LeRobot v3.0 dataset locally from any gym env | nothing |
| 06 | [`06_connect_hosted.py`](06_connect_hosted.py) — the one-argument upgrade to Interlatent Cloud | an API key |
| 07 | [`07_offline_no_dashboard.md`](07_offline_no_dashboard.md) — run node sessions + record datasets with the local coordinator, no account | GPU server + node (mock OK) |
