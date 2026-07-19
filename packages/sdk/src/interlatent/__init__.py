from ._client import Interlatent
from ._exceptions import APIError, AuthenticationError, InterlatentError, NotFoundError

__all__ = [
    "Interlatent",
    "InterlatentError",
    "APIError",
    "AuthenticationError",
    "NotFoundError",
    # Named, offline behaviors (interlatent.behaviors / interlatent.robot).
    "Robot",
    "behavior",
    "BehaviorValidationError",
    "BehaviorExecutionError",
    "RobotBusyError",
]


def __getattr__(name):
    """Lazy-load behavior types on first access.

    Behaviors live behind this lazy seam so ``import interlatent`` stays cheap and
    never pulls in a robot adapter until you actually construct a :class:`Robot`.
    (The collection types — ActivationEvent, RunInfo, Watcher, auto_metrics —
    were removed with client-side collection; see ADR 0022.)
    """
    _lazy = {
        "Robot": ".robot",
        "behavior": ".behaviors",
        "BehaviorValidationError": ".behaviors",
        "BehaviorExecutionError": ".behaviors",
        "RobotBusyError": ".behaviors",
    }
    if name in _lazy:
        import importlib
        return getattr(importlib.import_module(_lazy[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
