"""`interlatent` umbrella CLI — manage the local coordinator and sessions.

A thin client + daemon manager over :mod:`interlatent.coordinator`. Run
``interlatent up`` to start the coordinator (background), register GPU boxes
with ``interlatent gpu add``, and drive inference sessions with
``interlatent session start/stop``. Nodes self-register via
``interlatent-node pair --api-base http://<host>:<port>``.
"""
