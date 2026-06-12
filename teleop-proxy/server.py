"""Teleop WebSocket proxy.

Sits between the browser/dashboard and the GPU box's teleop relay so that
browsers on networks that block Tailscale Funnel (corporate firewalls,
SNI-filtered ISPs, some home routers) can still reach the relay.

Data path:

    Browser
      | wss://teleop.interlatent.com/<box_short>/teleop/<role>/<sid>?token=...
      v
    THIS proxy (on Fly.io, joined to our tailnet at boot)
      | ws://il-drtc-<box_short>.<tailnet>:50052/teleop/<role>/<sid>?token=...
      v
    GPU box teleop_relay

The browser sees only `*.interlatent.com` (a real LE cert from Fly), so
the SNI block disappears. The upstream hop is plain ws:// over the
tailnet — Tailscale encrypts the underlay.

The proxy does NOT verify the token. The token is opaque here and gets
passed through unmodified; the box's teleop_relay is the source of truth
for auth (HMAC-signed by the Vercel backend with a secret we don't have
to share with this proxy). All we authenticate is the URL shape.

Required env:
    TELEOP_TAILNET   — e.g. "tail285014.ts.net" (the trailing tailnet suffix)
    PORT             — listen port (Fly.io sets this automatically)

Optional env:
    TELEOP_BOX_PORT  — upstream box port (default 50052)
"""
from __future__ import annotations

import asyncio
import logging
import os
import time

import websockets
from websockets.asyncio.client import connect as ws_connect
from websockets.asyncio.server import ServerConnection, serve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("teleop-proxy")


TAILNET = os.environ.get("TELEOP_TAILNET", "")
if not TAILNET:
    raise SystemExit(
        "TELEOP_TAILNET is required (e.g. 'tail285014.ts.net'). "
        "Set it with `fly secrets set TELEOP_TAILNET=...` or export it locally."
    )
BOX_PORT = int(os.environ.get("TELEOP_BOX_PORT", "50052"))
LISTEN_PORT = int(os.environ.get("PORT", "8080"))
TS_SOCKET = os.environ.get("TS_SOCKET", "/var/run/tailscale/tailscaled.sock")


# ----------------------------------------------------------------------
# Tailscale name resolution
# ----------------------------------------------------------------------
# Fly.io's resolv.conf doesn't include Tailscale's MagicDNS resolver, so
# `il-drtc-XXX.<tailnet>` falls through to public DNS and resolves to
# Tailscale's Funnel anycast IPs (port 443 only). We instead shell out
# to `tailscale ip <hostname>` which talks directly to the local
# tailscaled and returns the peer's 100.x address. Cache for 60s so
# repeat connects don't fork a subprocess every time.

_ip_cache: dict[str, tuple[float, str]] = {}
_IP_TTL_S = 60.0


async def _tailnet_ip(hostname: str) -> str | None:
    now = time.monotonic()
    cached = _ip_cache.get(hostname)
    if cached is not None and now - cached[0] < _IP_TTL_S:
        return cached[1]

    proc = await asyncio.create_subprocess_exec(
        "tailscale", f"--socket={TS_SOCKET}", "ip", "-4", hostname,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=3.0)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        log.warning("tailscale ip lookup timed out: %s", hostname)
        return None

    if proc.returncode != 0:
        log.warning(
            "tailscale ip lookup failed: %s (rc=%d): %s",
            hostname, proc.returncode, stderr.decode(errors="replace").strip(),
        )
        return None

    ip = stdout.decode().strip().splitlines()[0] if stdout else ""
    if not ip:
        return None
    _ip_cache[hostname] = (now, ip)
    return ip


async def _pump(src, dst, label: str) -> None:
    """Forward frames src→dst until one side closes."""
    try:
        async for msg in src:
            await dst.send(msg)
    except websockets.ConnectionClosed:
        pass
    except Exception:
        log.exception("pump %s errored", label)


async def handle(client: ServerConnection) -> None:
    raw_path = client.request.path
    path_only, _, qs = raw_path.partition("?")
    parts = [p for p in path_only.split("/") if p]

    # Expected shape: /<box_short>/teleop/<role>/<sid>
    # box_short is the 6-char box-id suffix used in the box's tailnet hostname
    # (e.g. "90e118" → "il-drtc-90e118.<tailnet>"). See site/app/routers/compute.py.
    if (
        len(parts) != 4
        or parts[1] != "teleop"
        or parts[2] not in ("browser", "node")
    ):
        log.warning("bad path: %s", raw_path)
        await client.close(4404, "bad_path")
        return

    box_short, _, role, sid = parts

    # Reject anything that's not a 6-char hex id. Prevents SSRF — without this
    # a crafted URL could make the proxy dial arbitrary tailnet hosts.
    if not (len(box_short) == 6 and all(c in "0123456789abcdef" for c in box_short)):
        log.warning("bad box_short: %s", box_short)
        await client.close(4404, "bad_box_id")
        return

    upstream_host = f"il-drtc-{box_short}"
    # Resolve to a tailnet 100.x address via local tailscaled. Bypasses
    # Fly's resolv.conf (which would return Funnel anycast IPs from
    # public DNS — those only listen on :443, so 50052 would time out).
    upstream_ip = await _tailnet_ip(upstream_host)
    if upstream_ip is None:
        log.warning("no tailnet IP for %s", upstream_host)
        await client.close(4502, "upstream_dns")
        return

    # Pass the original Host header so any name-based logic on the box
    # still sees the friendly hostname. websockets uses ``server_hostname``
    # for SNI and ``Host`` header value if we set both via URL — easier
    # to just keep the URL hostname and let the kernel reach the IP via
    # the route Tailscale installs (MagicDNS not required for routing,
    # only for lookup).
    upstream_url = f"ws://{upstream_ip}:{BOX_PORT}/teleop/{role}/{sid}"
    if qs:
        upstream_url += f"?{qs}"

    log.info(
        "client connected role=%s sid=%s → %s (%s):%d",
        role, sid, upstream_host, upstream_ip, BOX_PORT,
    )
    try:
        async with ws_connect(
            upstream_url,
            open_timeout=10,
            close_timeout=5,
            ping_interval=20,
        ) as upstream:
            await asyncio.gather(
                _pump(client, upstream, "browser→box"),
                _pump(upstream, client, "box→browser"),
            )
    except (OSError, asyncio.TimeoutError, websockets.exceptions.InvalidStatus) as e:
        log.warning("upstream connect failed: %s: %s", type(e).__name__, e)
        try:
            await client.close(4502, f"upstream: {type(e).__name__}")
        except Exception:
            pass


async def main() -> None:
    log.info(
        "teleop proxy listening 0.0.0.0:%d (upstream tailnet=%s box_port=%d)",
        LISTEN_PORT, TAILNET, BOX_PORT,
    )
    async with serve(handle, "0.0.0.0", LISTEN_PORT, ping_interval=20):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
