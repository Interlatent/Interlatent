"""Robot data for the 'a1z' interlatent teleop embodiment (Galaxea A1Z, driven
through the dimos adapter -- see interlatent.adapters.dimos, --robot-arg
kind=a1z).

A data-only subpackage under the ``interlatent_robots`` namespace. Do not import
it directly for paths — go through :func:`interlatent.robots.load` / ``data_dir``,
which resolve any installed kind uniformly via ``importlib.resources``.
"""
KIND = "a1z"
