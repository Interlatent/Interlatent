from ._client import Interlatent
from ._exceptions import APIError, AuthenticationError, InterlatentError, NotFoundError

__all__ = [
    "Interlatent",
    "InterlatentError",
    "APIError",
    "AuthenticationError",
    "NotFoundError",
]


def __getattr__(name):
    """Lazy-load collection types on first access."""
    _lazy = {
        "ActivationEvent": "._schema",
        "RunInfo": "._schema",
        "auto_metrics": "._metrics",
        "Watcher": "._watcher",
    }
    if name in _lazy:
        import importlib
        return getattr(importlib.import_module(_lazy[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
