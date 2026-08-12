"""`interlatent` umbrella CLI — run a coordinator, and drive it from a terminal.

``interlatent up`` starts the coordinator on this machine; every other command
is a client of it. List the GPU boxes registered with it
(``interlatent gpus ls``) and the robot nodes paired to it
(``interlatent nodes ls``), and drive inference sessions with
``interlatent session start/stop/ls``.

Auth: pass ``--api-key`` or set ``INTERLATENT_API_KEY``; with neither, the
operator key (``ilop_…``) ``interlatent up`` wrote to disk is used. There is no
default coordinator address — name yours with ``--coordinator`` or
``INTERLATENT_COORDINATOR``. Nodes pair with the same coordinator via
``interlatent-node pair --coordinator <url> --api-key ilop_…``.

The command implementations live in ``cli/main.py``.
"""
