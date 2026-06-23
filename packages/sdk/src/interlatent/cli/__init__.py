"""`interlatent` umbrella CLI — a thin client for the Interlatent dashboard.

Inference runs in the cloud through the dashboard; this CLI is a small utility
view of it. List your GPU pods (``interlatent pods ls``) and robot nodes
(``interlatent nodes ls``), and drive cloud inference sessions with
``interlatent session start/stop/ls``. Authenticate with an Interlatent API
key (``--api-key`` / ``INTERLATENT_API_KEY``). Nodes pair directly with the
dashboard via ``interlatent-node pair --api-key ilat_…``.
"""
