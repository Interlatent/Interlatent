"""Robot data for the 'xarm6' interlatent teleop embodiment (UFACTORY xArm 6,
driven through the dimos adapter -- see interlatent.adapters.dimos, --robot-arg
kind=xarm6).

A data-only subpackage under the ``interlatent_robots`` namespace. Do not import
it directly for paths — go through :func:`interlatent.robots.load` / ``data_dir``,
which resolve any installed kind uniformly via ``importlib.resources``.
"""
KIND = "xarm6"
