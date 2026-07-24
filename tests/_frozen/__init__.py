"""Frozen pre-migration control loops, kept only for equivalence testing.

Each module here is a copy of a control loop taken immediately before it was
migrated onto ``node/looprunner.py`` + ``CommandBus.drive()``, so
``tests/test_loop_equivalence.py`` can drive old and new through identical
fakes and assert the emitted trace matches tick-for-tick. Delete a frozen
module once its migration has soaked on hardware and the equivalence test for
it is retired.

Not shipped: this package lives under ``tests/`` and is never importable from
``interlatent``.
"""
