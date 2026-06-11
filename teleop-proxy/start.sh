#!/bin/sh
# Bring up Tailscale then exec the proxy. Single-shot; on failure we let
# the container exit so Fly's restart policy can reschedule us cleanly.
set -eu

mkdir -p /var/lib/tailscale /var/run/tailscale

# Background tailscaled. State lives on the (ephemeral) container disk —
# we re-auth with the authkey on every cold start, so a reusable
# authkey on a tagged machine is the right kind of credential.
tailscaled \
    --state=/var/lib/tailscale/tailscaled.state \
    --socket=/var/run/tailscale/tailscaled.sock \
    >/var/log/tailscaled.log 2>&1 &

# Wait for the control socket — tailscale CLI blocks on it otherwise.
i=0
while [ ! -S /var/run/tailscale/tailscaled.sock ] && [ "$i" -lt 30 ]; do
    sleep 0.5
    i=$((i + 1))
done
if [ ! -S /var/run/tailscale/tailscaled.sock ]; then
    echo "[start] FATAL: tailscaled socket never appeared" >&2
    tail -n 50 /var/log/tailscaled.log >&2 || true
    exit 1
fi

tailscale --socket=/var/run/tailscale/tailscaled.sock up \
    --authkey="${TS_AUTHKEY}" \
    --hostname="${TS_HOSTNAME:-teleop-proxy}" \
    --accept-routes

echo "[start] tailnet joined as ${TS_HOSTNAME:-teleop-proxy}"

exec python -u /app/server.py
