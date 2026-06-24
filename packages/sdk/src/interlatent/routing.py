"""Routing methods — how a node reaches a GPU box for a session.

A **route descriptor** is a small dict describing how to connect, tagged by
``method``::

    {"method": "direct", "address": "100.x.y.z:50051"}

Today only ``direct`` exists: the user supplies an address (any reachable
``host:port`` or ``http(s)://…`` URL — no lock-in to Tailscale or any single
transport; the gRPC vs gRPC-web choice is inferred downstream from the address
scheme). The registries below are the seam for adding NAT-traversal relays,
MagicDNS resolution, tunnels, etc. **without touching the node control flow**
— a new method just registers a resolver + a connector.

Two pluggable points:

* **resolver** (control-plane side): ``descriptor -> resolved descriptor``. For
  ``direct`` this is identity. A future ``relay`` resolver might allocate a
  rendezvous session and return its address.
* **connector** (node-side): ``route -> {"server_address": str}`` — what the
  node feeds to ``connect_drtc``. For ``direct`` it returns the address. A
  future ``relay`` connector would establish its hop and return a local
  address to dial.
"""

from __future__ import annotations

from typing import Any, Callable

DEFAULT_METHOD = "direct"

# method -> (resolver, connector)
_RESOLVERS: dict[str, Callable[[dict], dict]] = {}
_CONNECTORS: dict[str, Callable[[dict], dict]] = {}


def register_method(
    name: str,
    *,
    resolver: Callable[[dict], dict],
    connector: Callable[[dict], dict],
) -> None:
    """Register a routing method's control-plane resolver + node connector."""
    _RESOLVERS[name] = resolver
    _CONNECTORS[name] = connector


def known_methods() -> list[str]:
    return sorted(_RESOLVERS)


def make_descriptor(address: str, *, method: str = DEFAULT_METHOD, **extra: Any) -> dict:
    """Build a route descriptor (stored on a GPU registration)."""
    return {"method": method, "address": address, **extra}


def resolve(descriptor: dict) -> dict:
    """Control-plane side: turn a stored descriptor into a session ``route``.

    Raises ``ValueError`` for an unknown method.
    """
    method = (descriptor or {}).get("method", DEFAULT_METHOD)
    resolver = _RESOLVERS.get(method)
    if resolver is None:
        raise ValueError(
            f"unknown routing method {method!r}; known: {known_methods()}"
        )
    return resolver(descriptor)


def connect_params(route: dict) -> dict:
    """Node-side: turn a session ``route`` into ``connect_drtc`` params.

    Returns at least ``{"server_address": str}``. Raises ``ValueError`` for an
    unknown method.
    """
    method = (route or {}).get("method", DEFAULT_METHOD)
    connector = _CONNECTORS.get(method)
    if connector is None:
        raise ValueError(
            f"unknown routing method {method!r}; known: {known_methods()}"
        )
    return connector(route)


# ----------------------------------------------------------------------
# Built-in: direct (a user-supplied address)
# ----------------------------------------------------------------------


def _direct_resolver(descriptor: dict) -> dict:
    address = (descriptor or {}).get("address") or ""
    # Identity, carrying through any extra keys a caller stored.
    out = {k: v for k, v in (descriptor or {}).items() if k not in ("method", "address")}
    out.update({"method": "direct", "address": address})
    return out


def _direct_connector(route: dict) -> dict:
    # Transport (plain gRPC vs gRPC-web) is inferred by connect_drtc from the
    # address scheme, so the address is all the node needs.
    return {"server_address": (route or {}).get("address") or ""}


register_method("direct", resolver=_direct_resolver, connector=_direct_connector)


__all__ = [
    "DEFAULT_METHOD",
    "register_method",
    "known_methods",
    "make_descriptor",
    "resolve",
    "connect_params",
]
