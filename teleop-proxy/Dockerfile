# Fly.io image for the teleop WS proxy.
#
# Two processes live in this container:
#   1. tailscaled  — joins our tailnet at boot (authkey from $TS_AUTHKEY)
#   2. server.py   — the WS proxy that browsers connect to
#
# Tailscale uses kernel TUN. Fly.io machines support TUN by default in
# Machines (v2) mode, so we don't need userspace networking / SOCKS5.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        gnupg \
        iproute2 \
        iptables \
    && rm -rf /var/lib/apt/lists/*

# Tailscale's official Debian apt repo. Stable channel; the proxy is
# small enough that we don't pin a specific version.
RUN curl -fsSL https://pkgs.tailscale.com/stable/debian/bookworm.noarmor.gpg \
        > /usr/share/keyrings/tailscale-archive-keyring.gpg \
    && curl -fsSL https://pkgs.tailscale.com/stable/debian/bookworm.tailscale-keyring.list \
        > /etc/apt/sources.list.d/tailscale.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends tailscale \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "websockets>=12,<14"

WORKDIR /app
COPY server.py /app/server.py
COPY start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]
